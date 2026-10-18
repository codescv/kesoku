"""Restart-persistent Discord components for interactive questions and collapsible sections.

Components are `discord.ui.DynamicItem`s whose custom_id encodes the session and the rendered kesoku message:

* `kui:<session_id>:<message_id>:<idx>`   - a button for option `idx` of a single-select question
* `kuis:<session_id>:<message_id>`        - a select menu (single-select with >5 options, or multiselect)
* `kud:<session_id>:<details_key>`        - the expand/collapse toggle of a `<details>` section

The question/details spec is stored in the `ui_prompts` DB table when rendered and loaded again on click, so no
in-memory View object is needed and clicks keep working after a process restart (the item classes are registered
with `Client.add_dynamic_items` in `setup_hook`).
"""

import re
from typing import Any

import discord
from kesoku.logger import setup_logger
from kesoku.utils.ui_markup import format_multi_selection

from .ui import (
    DETAILS_COLLAPSE_LABEL,
    DETAILS_EXPAND_LABEL,
    MAX_BUTTON_LABEL_LEN,
    post_simulated_choice,
    render_details_content,
)

logger = setup_logger(__name__)

CUSTOM_ID_MAX_LEN = 100
MAX_INLINE_BUTTONS = 5
MAX_SELECT_OPTIONS = 25
MAX_SELECT_LABEL_LEN = 100
MAX_EMBED_TITLE_LEN = 256
SHORT_LABEL_THRESHOLD = 15
EMBED_COLOR_ANSWERED = discord.Color.dark_grey()

QUESTION_BUTTON_PREFIX = "kui"
QUESTION_SELECT_PREFIX = "kuis"
DETAILS_TOGGLE_PREFIX = "kud"

# Registry from Discord client to its kesoku chatbot adapter, used by DynamicItem callbacks after a restart.
_CHATBOTS: dict[int, Any] = {}


def register_chatbot(client: discord.Client, chatbot: Any) -> None:
    """Associate a Discord client with its chatbot adapter for persistent component callbacks."""
    _CHATBOTS[id(client)] = chatbot


def chatbot_for(interaction: discord.Interaction) -> Any | None:
    """Return the chatbot adapter owning the interaction's client (None if unregistered)."""
    return _CHATBOTS.get(id(interaction.client))


def question_button_id(session_id: str, message_id: str, idx: int) -> str:
    """custom_id of a question option button."""
    return f"{QUESTION_BUTTON_PREFIX}:{session_id}:{message_id}:{idx}"


def question_select_id(session_id: str, message_id: str) -> str:
    """custom_id of a question select menu."""
    return f"{QUESTION_SELECT_PREFIX}:{session_id}:{message_id}"


def details_toggle_id(session_id: str, details_key: str) -> str:
    """custom_id of a details toggle button."""
    return f"{DETAILS_TOGGLE_PREFIX}:{session_id}:{details_key}"


def ids_fit(session_id: str, message_id: str) -> bool:
    """Whether persistent custom_ids for this session/message fit Discord's limits and template syntax."""
    if ":" in session_id or ":" in message_id:
        return False
    return len(question_button_id(session_id, message_id, MAX_SELECT_OPTIONS)) <= CUSTOM_ID_MAX_LEN


def build_question_spec(question: str, choices: list[str], multi: bool, max_choices: int) -> dict[str, Any]:
    """Build the JSON spec stored in `ui_prompts` for a question."""
    options = list(choices)
    if len(options) > MAX_SELECT_OPTIONS:
        logger.warning(f"Question has {len(options)} options; Discord shows at most {MAX_SELECT_OPTIONS}")
        options = options[:MAX_SELECT_OPTIONS]
    max_choices = max(1, min(max_choices, len(options))) if multi else 1
    return {"type": "question", "prompt": question, "options": options, "multi": multi, "max": max_choices}


def uses_select(spec: dict[str, Any]) -> bool:
    """A select menu is used for multiselect questions and single-select questions with >5 options."""
    return bool(spec.get("multi")) or len(spec.get("options", [])) > MAX_INLINE_BUTTONS


def uses_short_labels(spec: dict[str, Any]) -> bool:
    """Buttons show letters (A, B, ...) and the embed lists the full options when any option is long."""
    return not uses_select(spec) and any(len(o) > SHORT_LABEL_THRESHOLD for o in spec.get("options", []))


def _truncate(text: str, limit: int) -> str:
    """Truncate text to a Discord length limit with an ellipsis."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_question_embed(spec: dict[str, Any]) -> discord.Embed:
    """Embed showing the question prompt (and full option texts when labels are shortened)."""
    answered = spec.get("answered")
    embed = discord.Embed(
        title=_truncate(f"❓ {spec.get('prompt', '')}", MAX_EMBED_TITLE_LEN),
        color=EMBED_COLOR_ANSWERED if answered else discord.Color.blurple(),
    )
    options = spec.get("options", [])
    if uses_short_labels(spec) or (uses_select(spec) and any(len(o) > MAX_SELECT_LABEL_LEN for o in options)):
        embed.description = "\n".join(f"**{chr(ord('A') + idx)}.** {opt}" for idx, opt in enumerate(options))
    if answered:
        embed.set_footer(text=_truncate(f"✓ {answered}", 2048))
    return embed


class QuestionButton(
    discord.ui.DynamicItem[discord.ui.Button],
    template=r"kui:(?P<sid>[^:]+):(?P<mid>[^:]+):(?P<idx>\d+)",
):
    """Persistent option button of a single-select question."""

    def __init__(
        self,
        session_id: str,
        message_id: str,
        idx: int,
        label: str | None = None,
        disabled: bool = False,
        chosen: bool = False,
    ) -> None:
        """Create the button (row 0)."""
        button = discord.ui.Button(
            style=discord.ButtonStyle.success if chosen else discord.ButtonStyle.primary,
            label=_truncate(label or str(idx + 1), MAX_BUTTON_LABEL_LEN),
            custom_id=question_button_id(session_id, message_id, idx),
            disabled=disabled,
            row=0,
        )
        # discord.py does not always set _row from the constructor argument; set it explicitly.
        button._row = 0
        super().__init__(button)
        self.session_id = session_id
        self.message_id = message_id
        self.idx = idx

    @classmethod
    async def from_custom_id(
        cls, interaction: discord.Interaction, item: discord.ui.Button, match: re.Match[str], /
    ) -> "QuestionButton":
        """Rebuild from a clicked component (also after a restart)."""
        return cls(match["sid"], match["mid"], int(match["idx"]), label=item.label)

    async def callback(self, interaction: discord.Interaction) -> None:
        """Answer the question with this option."""
        await handle_question_answer(interaction, self.session_id, self.message_id, [self.idx])


class QuestionSelect(
    discord.ui.DynamicItem[discord.ui.Select],
    template=r"kuis:(?P<sid>[^:]+):(?P<mid>[^:]+)",
):
    """Persistent select menu (single-select with many options, or multiselect)."""

    def __init__(
        self,
        session_id: str,
        message_id: str,
        options: list[str] | None = None,
        multi: bool = False,
        max_choices: int = 1,
        disabled: bool = False,
        chosen: list[int] | None = None,
    ) -> None:
        """Create the select menu (row 1, after any buttons on row 0)."""
        chosen_set = set(chosen or [])
        select_options = [
            discord.SelectOption(
                label=_truncate(f"{chr(ord('A') + idx)}. {opt}" if len(opt) > MAX_SELECT_LABEL_LEN else opt,
                                MAX_SELECT_LABEL_LEN),
                value=str(idx),
                default=idx in chosen_set,
            )
            for idx, opt in enumerate(options or [])
        ] or [discord.SelectOption(label="-", value="0")]
        max_values = max(1, min(max_choices, len(select_options))) if multi else 1
        select = discord.ui.Select(
            custom_id=question_select_id(session_id, message_id),
            placeholder=f"最多选择 {max_values} 项" if multi else "请选择",
            options=select_options,
            min_values=1,
            max_values=max_values,
            disabled=disabled,
            row=1,
        )
        # discord.ui.Select(row=...) does not set _row; set it explicitly (buttons row 0, select row 1).
        select._row = 1
        super().__init__(select)
        self.session_id = session_id
        self.message_id = message_id

    @classmethod
    async def from_custom_id(
        cls, interaction: discord.Interaction, item: discord.ui.Select, match: re.Match[str], /
    ) -> "QuestionSelect":
        """Rebuild from a used select menu; the real options come from the DB on callback."""
        return cls(match["sid"], match["mid"])

    async def callback(self, interaction: discord.Interaction) -> None:
        """Answer the question with the selected option indices."""
        data: Any = interaction.data or {}
        indices: list[int] = []
        for value in data.get("values") or []:
            try:
                indices.append(int(value))
            except (TypeError, ValueError):
                continue
        await handle_question_answer(interaction, self.session_id, self.message_id, indices)


def build_question_view(
    session_id: str, message_id: str, spec: dict[str, Any], chosen: list[int] | None = None
) -> discord.ui.View:
    """Build the persistent view of a question; with `spec["answered"]` set, every component is disabled.

    Args:
        session_id: Session ID encoded in custom_ids.
        message_id: Rendered kesoku message ID encoded in custom_ids.
        spec: The stored question spec.
        chosen: Indices of the chosen options (highlighted once answered).

    Returns:
        A `discord.ui.View` with timeout=None.
    """
    view = discord.ui.View(timeout=None)
    disabled = bool(spec.get("answered"))
    options: list[str] = list(spec.get("options", []))
    chosen_list = list(chosen if chosen is not None else spec.get("chosen", []))
    if uses_select(spec):
        view.add_item(
            QuestionSelect(
                session_id,
                message_id,
                options,
                multi=bool(spec.get("multi")),
                max_choices=int(spec.get("max", 1)),
                disabled=disabled,
                chosen=chosen_list,
            )
        )
        return view
    short = uses_short_labels(spec)
    for idx, opt in enumerate(options):
        label = chr(ord("A") + idx) if short else opt
        view.add_item(
            QuestionButton(session_id, message_id, idx, label=label, disabled=disabled, chosen=idx in chosen_list)
        )
    return view


async def handle_question_answer(
    interaction: discord.Interaction, session_id: str, message_id: str, indices: list[int]
) -> None:
    """Resolve the chosen option(s) from the stored spec, disable the components and post the user message.

    Single-select posts the option text; multiselect posts one `选择：A；B` message.
    """
    chatbot = chatbot_for(interaction)
    db = chatbot.gateway.db if chatbot else None
    spec = await db.get_ui_prompt(session_id, message_id) if db else None
    if chatbot is None or spec is None:
        await interaction.response.send_message("⚠️ This question is no longer available.", ephemeral=True)
        return
    if spec.get("answered"):
        await interaction.response.send_message(f"✓ Already answered: {spec['answered']}", ephemeral=True)
        return
    options: list[str] = list(spec.get("options", []))
    valid = [i for i in dict.fromkeys(indices) if 0 <= i < len(options)]
    if not valid:
        await interaction.response.send_message("⚠️ Invalid selection.", ephemeral=True)
        return
    if spec.get("multi"):
        valid = sorted(valid)[: int(spec.get("max", len(options)))]
        content = format_multi_selection([options[i] for i in valid])
    else:
        valid = valid[:1]
        content = options[valid[0]]

    # Persist the answer first so concurrent clicks see it, then disable the components.
    spec["answered"] = content
    spec["chosen"] = valid
    await db.save_ui_prompt(session_id, message_id, spec)
    await interaction.response.edit_message(
        embed=build_question_embed(spec), view=build_question_view(session_id, message_id, spec)
    )
    await post_simulated_choice(chatbot.gateway, chatbot, interaction, session_id, content)


class DetailsToggle(
    discord.ui.DynamicItem[discord.ui.Button],
    template=r"kud:(?P<sid>[^:]+):(?P<key>[^:]+)",
):
    """Persistent expand/collapse button of a `<details>` section."""

    def __init__(self, session_id: str, details_key: str, expanded: bool = False) -> None:
        """Create the toggle button."""
        button = discord.ui.Button(
            style=discord.ButtonStyle.secondary,
            label=DETAILS_COLLAPSE_LABEL if expanded else DETAILS_EXPAND_LABEL,
            custom_id=details_toggle_id(session_id, details_key),
            row=0,
        )
        button._row = 0
        super().__init__(button)
        self.session_id = session_id
        self.details_key = details_key

    @classmethod
    async def from_custom_id(
        cls, interaction: discord.Interaction, item: discord.ui.Button, match: re.Match[str], /
    ) -> "DetailsToggle":
        """Rebuild from a clicked component (also after a restart)."""
        return cls(match["sid"], match["key"], expanded=item.label == DETAILS_COLLAPSE_LABEL)

    async def callback(self, interaction: discord.Interaction) -> None:
        """Toggle the section, deriving the current state from the message content."""
        chatbot = chatbot_for(interaction)
        spec = await chatbot.gateway.db.get_ui_prompt(self.session_id, self.details_key) if chatbot else None
        if spec is None:
            await interaction.response.send_message("⚠️ This section is no longer available.", ephemeral=True)
            return
        content = interaction.message.content if interaction.message else ""
        expanded = not content.startswith("▾")
        await interaction.response.edit_message(
            content=render_details_content(spec.get("title", ""), spec.get("body", ""), expanded),
            view=build_details_view(self.session_id, self.details_key, expanded),
        )


def build_details_view(session_id: str, details_key: str, expanded: bool = False) -> discord.ui.View:
    """Persistent view with the details toggle button."""
    view = discord.ui.View(timeout=None)
    view.add_item(DetailsToggle(session_id, details_key, expanded=expanded))
    return view


PERSISTENT_ITEMS: tuple[type[discord.ui.DynamicItem[Any]], ...] = (QuestionButton, QuestionSelect, DetailsToggle)
