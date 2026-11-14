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


# ---------------------------------------------------------------------------------------------------------------
# Rich fenced blocks: ```bilingual and ```card (platform-neutral; generic structure only, no domain semantics)
# ---------------------------------------------------------------------------------------------------------------

RICH_BLOCK_RE = re.compile(
    r"^[ \t]*```(?P<kind>bilingual|card)[ \t]*\r?\n(?P<body>.*?)^[ \t]*```[ \t]*\r?$",
    re.MULTILINE | re.DOTALL,
)
CARD_KIND_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_MEDIA_TAG_RE = re.compile(r"^\s*\[(?P<kind>file|voice):\s*(?P<path>[^\]]+?)\s*\]\s*$")
# Inline ruby notations: `漢字(かな)` / `漢字（かな）` (base = preceding kanji run) and `{base|reading}`.
_KANJI_CLASS = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3005\u3006\u30f6\u30f5"
_INLINE_RUBY_RE = re.compile(
    r"\{(?P<b1>[^{}|\n]+)\|(?P<r1>[^{}|\n]+)\}"
    rf"|(?P<b2>[{_KANJI_CLASS}]+)[(（](?P<r2>[\u3040-\u309f\u30a0-\u30ffー・\s]+)[)）]"
)
CARD_FALLBACK_MAX_LINES = 12
CARD_FALLBACK_VALUE_MAX_CHARS = 200
_CARD_TITLE_KEYS = ("title", "name", "banner", "headline", "source")
_NAME_LANG_KEYS = ("zh", "ja", "en", "local")


def _html_escape(text: str) -> str:
    """Escape text for safe inclusion in a minimal HTML string."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _parts_from_inline(text: str) -> list[dict[str, str]]:
    """Split inline ruby notation into `[{base, rt?}]` parts (plain runs have no `rt`)."""
    parts: list[dict[str, str]] = []
    last = 0
    for m in _INLINE_RUBY_RE.finditer(text):
        if m.start() > last:
            parts.append({"base": text[last : m.start()]})
        base = m.group("b1") if m.group("b1") is not None else m.group("b2")
        rt = m.group("r1") if m.group("r1") is not None else m.group("r2")
        parts.append({"base": base, "rt": rt.strip()})
        last = m.end()
    if last < len(text):
        parts.append({"base": text[last:]})
    return parts


def _parts_from_pairs(plain: str, pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
    """Annotate `plain` sequentially with `(base, reading)` pairs; unmatched pairs are skipped."""
    parts: list[dict[str, str]] = []
    cursor = 0
    for base, rt in pairs:
        if not base:
            continue
        idx = plain.find(base, cursor)
        if idx < 0:
            continue
        if idx > cursor:
            parts.append({"base": plain[cursor:idx]})
        parts.append({"base": base, "rt": rt})
        cursor = idx + len(base)
    if cursor < len(plain):
        parts.append({"base": plain[cursor:]})
    return parts


def _parts_from_mapping(plain: str, mapping: dict[str, str]) -> list[dict[str, str]]:
    """Annotate every occurrence of mapping keys in `plain` (longest key first at each position)."""
    keys = sorted((k for k in mapping if k), key=len, reverse=True)
    parts: list[dict[str, str]] = []
    buf = ""
    i = 0
    while i < len(plain):
        hit = next((k for k in keys if plain.startswith(k, i)), None)
        if hit is None:
            buf += plain[i]
            i += 1
            continue
        if buf:
            parts.append({"base": buf})
            buf = ""
        parts.append({"base": hit, "rt": str(mapping[hit])})
        i += len(hit)
    if buf:
        parts.append({"base": buf})
    return parts


def _merge_plain_parts(parts: list[dict[str, str]]) -> list[dict[str, str]]:
    """Merge adjacent unannotated parts and drop empty ones."""
    merged: list[dict[str, str]] = []
    for p in parts:
        if not p.get("base"):
            continue
        if "rt" not in p and merged and "rt" not in merged[-1]:
            merged[-1] = {"base": merged[-1]["base"] + p["base"]}
        else:
            merged.append(p)
    return merged


def _coerce_media_tag(value: Any) -> Any:
    """Map YAML's reading of an unquoted `[file: p]` / `[voice: p]` (`[{"file": p}]`) back to the tag string."""
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], dict)
        and len(value[0]) == 1
    ):
        (key, inner), = value[0].items()
        if key in ("file", "voice") and isinstance(inner, (str, int, float)) and str(inner).strip():
            return f"[{key}: {str(inner).strip()}]"
    return value


def _media_ref(value: Any) -> dict[str, Any] | None:
    """Normalize a voice/file reference (`[voice: p]`, bare path, or `{ref|path, duration_ms}`)."""
    value = _coerce_media_tag(value)
    if isinstance(value, str) and value.strip():
        m = _MEDIA_TAG_RE.match(value)
        return {"ref": (m.group("path") if m else value).strip()}
    if isinstance(value, dict):
        ref = value.get("ref") or value.get("path")
        if not isinstance(ref, str) or not ref.strip():
            return None
        m = _MEDIA_TAG_RE.match(ref)
        out: dict[str, Any] = {"ref": (m.group("path") if m else ref).strip()}
        dur = value.get("duration_ms")
        if isinstance(dur, (int, float)) and dur >= 0:
            out["duration_ms"] = int(dur)
        return out
    return None


def parse_bilingual_block_body(body: str) -> dict[str, Any] | None:
    """Parse a ```bilingual YAML body (`ja`, `ruby`, `zh`, `voice`, `tokens`, optional `speaker`).

    Returns:
        A `bilingual` segment dict with `ja_plain`, escaped `ja_html` (only `<ruby>`/`<rt>` markup),
        structured `ruby_parts` and passthrough fields, or None when invalid.
    """
    try:
        data = yaml.safe_load(body)
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    ja_raw = data.get("ja")
    ruby = data.get("ruby")
    if not isinstance(ja_raw, str) or not ja_raw.strip():
        if isinstance(ruby, str) and ruby.strip():
            ja_raw = ruby
        else:
            return None
    ja_raw = ja_raw.strip()
    inline_parts = _merge_plain_parts(_parts_from_inline(ja_raw))
    ja_plain = "".join(p["base"] for p in inline_parts)

    parts: list[dict[str, str]]
    if isinstance(ruby, str) and ruby.strip():
        parts = _parts_from_inline(ruby.strip())
        if "".join(p["base"] for p in parts) != ja_plain:
            # Ruby string disagrees with ja: keep ja text, annotate with the ruby string's readings.
            pairs = [(p["base"], p["rt"]) for p in parts if "rt" in p]
            parts = _parts_from_pairs(ja_plain, pairs)
    elif isinstance(ruby, list):
        pairs: list[tuple[str, str]] = []
        for item in ruby:
            if isinstance(item, dict) and isinstance(item.get("base"), str) and isinstance(item.get("rt"), str):
                pairs.append((item["base"], item["rt"]))
            elif isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(x, str) for x in item):
                pairs.append((item[0], item[1]))
        parts = _parts_from_pairs(ja_plain, pairs)
    elif isinstance(ruby, dict):
        parts = _parts_from_mapping(ja_plain, {str(k): str(v) for k, v in ruby.items()})
    else:
        parts = inline_parts
    parts = _merge_plain_parts(parts)

    html_chunks: list[str] = []
    for p in parts:
        base = _html_escape(p["base"])
        if "rt" in p:
            html_chunks.append(f"<ruby>{base}<rt>{_html_escape(p['rt'])}</rt></ruby>")
        else:
            html_chunks.append(base)

    zh = data.get("zh")
    segment: dict[str, Any] = {
        "type": "bilingual",
        "ja": ja_raw,
        "ja_plain": ja_plain,
        "ja_html": "".join(html_chunks),
        "ruby_parts": parts,
        "zh": zh.strip() if isinstance(zh, str) else "",
    }
    if ruby is not None and isinstance(ruby, (str, list, dict)):
        segment["ruby"] = ruby
    speaker = data.get("speaker", data.get("name"))
    if isinstance(speaker, str) and speaker.strip():
        segment["speaker"] = speaker.strip()
    voice = _media_ref(data.get("voice"))
    if voice is not None:
        segment["voice"] = voice
    tokens = data.get("tokens")
    if isinstance(tokens, list):
        clean_tokens = [t for t in tokens if isinstance(t, dict) and isinstance(t.get("surface"), str)]
        if clean_tokens:
            segment["tokens"] = clean_tokens
    return segment


def parse_card_block_body(body: str) -> dict[str, Any] | None:
    """Parse a ```card YAML body; requires a string `kind`, every other field is passed through verbatim."""
    try:
        data = yaml.safe_load(body)
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    kind = data.get("kind")
    if not isinstance(kind, str) or not CARD_KIND_RE.match(kind.strip()):
        return None
    payload = {str(k): _coerce_media_tag(v) for k, v in data.items() if k != "kind"}
    return {"type": "card", "kind": kind.strip(), "data": _jsonable(payload)}


def _jsonable(value: Any) -> Any:
    """Coerce YAML-loaded values (dates, sets, ...) into JSON-serializable structures."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _localized_text(value: Any) -> str | None:
    """Return a display string from a scalar or a `{zh, ja, en}`-style mapping."""
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        text = str(value).strip()
        return text or None
    if isinstance(value, dict):
        for key in _NAME_LANG_KEYS:
            v = value.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def card_media_refs(data: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Return top-level card fields whose value is exactly a `[file: ...]` / `[voice: ...]` tag."""
    refs: dict[str, dict[str, str]] = {}
    for key, value in data.items():
        if isinstance(value, str):
            m = _MEDIA_TAG_RE.match(value)
            if m:
                refs[key] = {"kind": m.group("kind"), "ref": m.group("path").strip()}
    return refs


def render_bilingual_as_text(segment: dict[str, Any]) -> str:
    """Plain-text fallback for a bilingual segment (`speaker：ja`, `zh`, and a voice tag when present)."""
    ja = str(segment.get("ja_plain") or "").strip()
    speaker = str(segment.get("speaker") or "").strip()
    lines = [f"{speaker}：{ja}" if speaker else ja]
    zh = str(segment.get("zh") or "").strip()
    if zh:
        lines.append(zh)
    voice = segment.get("voice")
    if isinstance(voice, dict) and voice.get("ref"):
        lines.append(f"[voice: {voice['ref']}]")
    return "\n".join(line for line in lines if line)


def render_card_as_text(segment: dict[str, Any]) -> str:
    """Plain-text fallback for a card: a bold title plus up to 12 scalar key points; media tags are kept."""
    data = segment.get("data") if isinstance(segment.get("data"), dict) else {}
    title: str | None = None
    title_key: str | None = None
    for key in _CARD_TITLE_KEYS:
        title = _localized_text(data.get(key))
        if title:
            title_key = key
            break
    lines = [f"**{title or segment.get('kind', 'card')}**"]
    media_refs = card_media_refs(data)
    for key, value in data.items():
        if len(lines) > CARD_FALLBACK_MAX_LINES:
            break
        if key == title_key or key in media_refs or key == "glossary":
            continue
        text: str | None = None
        if isinstance(value, list):
            items = [t for t in (_localized_text(v) for v in value) if t]
            if items:
                text = "、".join(items)
        else:
            text = _localized_text(value)
        if text:
            if len(text) > CARD_FALLBACK_VALUE_MAX_CHARS:
                text = text[: CARD_FALLBACK_VALUE_MAX_CHARS - 1] + "…"
            lines.append(f"- {key}: {text}")
    for ref in media_refs.values():
        lines.append(f"[{ref['kind']}: {ref['ref']}]")
    return "\n".join(lines)


def parse_rich_block(kind: str, body: str) -> dict[str, Any] | None:
    """Parse a ```bilingual / ```card block body into its segment, or None when invalid."""
    if kind == "bilingual":
        return parse_bilingual_block_body(body)
    if kind == "card":
        return parse_card_block_body(body)
    return None


def render_rich_block_as_text(segment: dict[str, Any]) -> str:
    """Plain-text fallback for any rich segment."""
    if segment.get("type") == "bilingual":
        return render_bilingual_as_text(segment)
    if segment.get("type") == "card":
        return render_card_as_text(segment)
    return ""


def flatten_rich_blocks(text: str) -> str:
    """Replace every valid ```bilingual / ```card block in `text` with its readable plain-text fallback.

    Invalid blocks are left untouched.
    """
    if not text or "```" not in text:
        return text

    def _replace(match: re.Match[str]) -> str:
        segment = parse_rich_block(match.group("kind"), match.group("body"))
        if segment is None:
            return match.group(0)
        return render_rich_block_as_text(segment)

    return RICH_BLOCK_RE.sub(_replace, text)
