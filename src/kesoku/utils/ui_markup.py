"""Structural parsing helpers for interactive UI markup in assistant replies.

Supported markup (platform-neutral; adapters decide how to render it):

* A fenced ```ui block containing YAML::

      ```ui
      type: select          # select | multiselect
      prompt: Which one?
      options:
        - First
        - Second [note]
      max: 2                # multiselect only, defaults to len(options)
      ```

* `<details><summary>TITLE</summary>BODY</details>` collapsible sections.

Multiselect answers are posted back as a single user message `选择：A；B；C` (see `format_multi_selection`).
"""

import re
from dataclasses import dataclass
from typing import Any

import yaml

from kesoku.logger import setup_logger

logger = setup_logger(__name__)

UI_TYPE_SELECT = "select"
UI_TYPE_MULTISELECT = "multiselect"
UI_TYPES = (UI_TYPE_SELECT, UI_TYPE_MULTISELECT)

MULTI_SELECT_PREFIX = "选择："
MULTI_SELECT_SEPARATOR = "；"

# Fenced ```ui block: opening fence line, body, closing fence line (structural parsing only).
UI_BLOCK_RE = re.compile(r"^[ \t]*```ui[ \t]*\r?\n(?P<body>.*?)^[ \t]*```[ \t]*$", re.MULTILINE | re.DOTALL)
# `<details>` with a leading `<summary>` (attributes such as `open` are tolerated).
DETAILS_RE = re.compile(
    r"<details(?:\s[^>]*)?>\s*<summary>(?P<title>.*?)</summary>(?P<body>.*?)</details>",
    re.DOTALL | re.IGNORECASE,
)

# Lenient fallback parsing of a YAML-ish body: top-level `key: value` lines and `- item` list lines.
_TOP_KEY_RE = re.compile(r"^(?P<key>[A-Za-z_]+)\s*:\s*(?P<value>.*)$")
_LIST_ITEM_RE = re.compile(r"^\s+-\s*(?P<item>.*)$|^-\s+(?P<item0>.*)$")


@dataclass(frozen=True)
class UiQuestion:
    """A validated interactive question parsed from a ```ui block."""

    prompt: str
    options: list[str]
    multi: bool
    max: int

    def to_segment(self) -> dict[str, Any]:
        """Convert to the gateway question segment shape."""
        return {
            "type": "question",
            "question": self.prompt,
            "choices": list(self.options),
            "multi": self.multi,
            "max": self.max,
        }


def _strip_quotes(value: str) -> str:
    """Remove one pair of matching surrounding quotes from a scalar."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _lenient_parse(body: str) -> dict[str, Any]:
    """Parse a YAML-ish ui body line by line, keeping option items verbatim.

    Used when strict YAML fails (e.g. `: ` inside the prompt) or yields non-string options
    (e.g. an option starting with `[`).
    """
    data: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in body.splitlines():
        if not raw_line.strip():
            continue
        item_match = _LIST_ITEM_RE.match(raw_line.rstrip())
        if item_match and current_list_key:
            item = item_match.group("item") if item_match.group("item") is not None else item_match.group("item0")
            data.setdefault(current_list_key, []).append(_strip_quotes(item or ""))
            continue
        key_match = _TOP_KEY_RE.match(raw_line.rstrip())
        if key_match:
            key = key_match.group("key").lower()
            value = key_match.group("value").split(" #", 1)[0].strip()
            if value:
                if key == "options" and value.startswith("[") and value.endswith("]"):
                    # Inline flow list: split items verbatim on commas.
                    data[key] = [_strip_quotes(v) for v in value[1:-1].split(",")]
                else:
                    data[key] = _strip_quotes(value)
                current_list_key = None
            else:
                current_list_key = key
                data[key] = []
    return data


def _strict_parse(body: str) -> dict[str, Any] | None:
    """Parse the ui body with YAML; returns None when it is not a mapping with string options."""
    try:
        data = yaml.safe_load(body)
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    options = data.get("options")
    if not isinstance(options, list) or not all(isinstance(o, str) for o in options):
        return None
    if not isinstance(data.get("prompt"), str):
        return None
    return {str(k).lower(): v for k, v in data.items()}


def _validate(data: dict[str, Any]) -> UiQuestion | None:
    """Validate parsed fields and build a UiQuestion (None when required fields are missing or invalid)."""
    raw_type = data.get("type")
    ui_type = str(raw_type).strip().lower() if raw_type is not None else ""
    if not ui_type:
        ui_type = UI_TYPE_SELECT
    elif ui_type not in UI_TYPES:
        return None
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    raw_options = data.get("options")
    if not isinstance(raw_options, list):
        return None
    options = [o.strip() for o in raw_options if isinstance(o, str) and o.strip()]
    if not options:
        return None
    multi = ui_type == UI_TYPE_MULTISELECT
    max_choices = 1
    if multi:
        max_choices = len(options)
        raw_max = data.get("max")
        if raw_max is not None:
            try:
                max_choices = int(str(raw_max).strip())
            except ValueError:
                return None
            max_choices = max(1, min(max_choices, len(options)))
    return UiQuestion(prompt=prompt.strip(), options=options, multi=multi, max=max_choices)


def parse_ui_block_body(body: str) -> UiQuestion | None:
    """Parse the YAML body of a ```ui block into a UiQuestion, or None if invalid.

    Strict YAML is tried first; a lenient line-based parser is used as a fallback so that option text
    containing brackets or colons is kept verbatim.
    """
    strict = _strict_parse(body)
    if strict is not None:
        question = _validate(strict)
        if question is not None:
            return question
    return _validate(_lenient_parse(body))


def find_first_ui_block(text: str | None) -> tuple[re.Match[str] | None, UiQuestion | None]:
    """Return the first ```ui block match in text and its parsed question (None when invalid)."""
    if not text or "```ui" not in text:
        return None, None
    match = UI_BLOCK_RE.search(text)
    if not match:
        return None, None
    return match, parse_ui_block_body(match.group("body"))


def extract_ui_options(text: str | None) -> tuple[list[str], bool]:
    """Return (options, multi) of the interactive (first) ```ui block in text, or ([], False)."""
    _, question = find_first_ui_block(text)
    if question is None:
        return [], False
    return list(question.options), question.multi


def format_multi_selection(choices: list[str]) -> str:
    """Format a multiselect answer as the single user message posted back to the agent."""
    return MULTI_SELECT_PREFIX + MULTI_SELECT_SEPARATOR.join(choices)


def parse_multi_selection(text: str | None) -> list[str] | None:
    """Parse a `选择：A；B` multiselect answer into its parts (None when the text is not in that format)."""
    if not text:
        return None
    stripped = text.strip()
    if not stripped.startswith(MULTI_SELECT_PREFIX):
        return None
    parts = [p.strip() for p in stripped[len(MULTI_SELECT_PREFIX) :].split(MULTI_SELECT_SEPARATOR)]
    parts = [p for p in parts if p]
    return parts or None


def collapse_ui_blocks(text: str) -> str:
    """Collapse every valid ```ui block in text to its plain prompt text (stripping fences and options).

    Leaving half-stripped ```ui fences in history causes the LLM to imitate incomplete ```ui blocks
    (e.g., omitting `type: select`), and the user's chosen answer already appears as the next user message.
    Invalid blocks are left untouched. Output is deterministic for prefix-caching stability.

    Args:
        text: Assistant message text.

    Returns:
        Text with valid ```ui blocks replaced by their prompt text.
    """
    if not text or "```ui" not in text:
        return text

    def _replace(match: re.Match[str]) -> str:
        question = parse_ui_block_body(match.group("body"))
        if question is None:
            return match.group(0)
        return question.prompt

    return UI_BLOCK_RE.sub(_replace, text)


def render_details_as_text(title: str, body: str) -> str:
    """Plain-text fallback rendering of a `<details>` section."""
    title = title.strip()
    body = body.strip()
    if not title:
        return body
    if not body:
        return f"**{title}**"
    return f"**{title}**\n\n{body}"


def render_question_as_text(question: str, choices: list[str], multi: bool = False, max_choices: int = 1) -> str:
    """Plain-text fallback rendering of a question for platforms without interactive components."""
    lines = [question.strip()]
    if multi:
        lines.append(f"(select up to {max_choices})")
    lines.extend(f"{idx}. {choice}" for idx, choice in enumerate(choices, start=1))
    return "\n".join(lines)
