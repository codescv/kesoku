"""Tests for restart-persistent Discord question/details components (persistent_ui.py) and ui_prompts storage."""

import re
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import discord
import pytest

from kesoku.config import DiscordConfig, KesokuConfig
from kesoku.constants import MessageRole
from kesoku.db import DatabaseManager
from kesoku.gateway.chatbot.discord import DiscordChatbot
from kesoku.gateway.chatbot.discord import persistent_ui as pui
from kesoku.gateway.chatbot.discord.ui import DETAILS_COLLAPSE_LABEL, DETAILS_EXPAND_LABEL
from kesoku.gateway.gateway import Gateway


class _MemoryDB:
    """In-memory stand-in for the ui_prompts DB methods."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}

    async def save_ui_prompt(self, session_id: str, message_id: str, spec: dict[str, Any]) -> None:
        """Store a copy of the spec."""
        self.rows[(session_id, message_id)] = dict(spec)

    async def get_ui_prompt(self, session_id: str, message_id: str) -> dict[str, Any] | None:
        """Return a copy of the stored spec."""
        row = self.rows.get((session_id, message_id))
        return dict(row) if row is not None else None


def _chatbot(db: _MemoryDB) -> MagicMock:
    """Fake chatbot adapter registered for a fake client."""
    chatbot = MagicMock()
    chatbot.chatbot_id = "discord_test"
    chatbot.gateway.db = db
    chatbot.gateway.post = AsyncMock()
    chatbot._typing_tasks = {"chan": MagicMock()}  # typing already running: no new task needed
    return chatbot


def _interaction(client: Any, values: list[str] | None = None, content: str = "") -> MagicMock:
    """Fake component interaction."""
    interaction = MagicMock()
    interaction.client = client
    interaction.data = {"values": values} if values is not None else {}
    interaction.response.defer = AsyncMock()
    interaction.response.edit_message = AsyncMock()
    interaction.response.send_message = AsyncMock()
    interaction.followup.send = AsyncMock()
    interaction.edit_original_response = AsyncMock()
    interaction.channel_id = "chan"
    interaction.user.id = 42
    interaction.user.display_name = "User"
    sent = MagicMock()
    sent.id = 777
    sent.created_at.timestamp.return_value = 123.0
    interaction.channel.send = AsyncMock(return_value=sent)
    interaction.message.content = content
    interaction.message.edit = AsyncMock()
    return interaction


@pytest.fixture
def registered() -> tuple[object, _MemoryDB, MagicMock]:
    """A fake client registered with a fake chatbot and in-memory DB."""
    client = object()
    db = _MemoryDB()
    chatbot = _chatbot(db)
    pui.register_chatbot(client, chatbot)  # type: ignore[arg-type]
    return client, db, chatbot


# ---------------------------------------------------------------------------
# ui_prompts table
# ---------------------------------------------------------------------------


def test_ui_prompts_roundtrip_and_session_delete(tmp_path: Path) -> None:
    """Specs are stored, updated in place, and removed with their session."""
    db = DatabaseManager(str(tmp_path / "k.db"))
    db.init_tables()
    assert db.get_ui_prompt("s", "m") is None
    db.save_ui_prompt("s", "m", {"type": "question", "options": ["中文", "b"]})
    assert db.get_ui_prompt("s", "m") == {"type": "question", "options": ["中文", "b"]}
    db.save_ui_prompt("s", "m", {"type": "question", "answered": "b"})
    assert db.get_ui_prompt("s", "m") == {"type": "question", "answered": "b"}
    db.delete_session("s")
    assert db.get_ui_prompt("s", "m") is None


def test_ui_prompts_migration_on_existing_db(tmp_path: Path) -> None:
    """verify_db creates the table on a database initialised before the table existed."""
    path = str(tmp_path / "old.db")
    db = DatabaseManager(path)
    db.init_tables()
    with db.connection_provider.connection() as conn:
        conn.execute("DROP TABLE ui_prompts")
        conn.commit()
    DatabaseManager(path).verify_db()
    db.save_ui_prompt("s", "m", {"x": 1})
    assert db.get_ui_prompt("s", "m") == {"x": 1}


# ---------------------------------------------------------------------------
# View construction
# ---------------------------------------------------------------------------


def test_custom_ids_and_fit() -> None:
    """custom_ids follow the kui/kuis/kud templates and fit Discord's 100-char limit."""
    assert pui.question_button_id("s", "m", 3) == "kui:s:m:3"
    assert pui.question_select_id("s", "m") == "kuis:s:m"
    assert pui.details_toggle_id("s", "m-d0") == "kud:s:m-d0"
    uuid = "12345678-1234-1234-1234-123456789012"
    assert pui.ids_fit(uuid, uuid)
    assert not pui.ids_fit("x" * 60, uuid)
    assert not pui.ids_fit("a:b", "m")


def test_select_question_with_few_options_uses_buttons_row0() -> None:
    """<=5 single-select options render buttons on row 0."""
    spec = pui.build_question_spec("Q", ["a", "b [x]"], multi=False, max_choices=1)
    view = pui.build_question_view("s", "m", spec)
    items = [c.item for c in view.children]
    assert all(isinstance(i, discord.ui.Button) for i in items)
    assert [i.label for i in items] == ["a", "b [x]"]
    assert [i.custom_id for i in items] == ["kui:s:m:0", "kui:s:m:1"]
    assert all(i._row == 0 for i in items)


def test_select_question_with_many_options_uses_single_dropdown() -> None:
    """>5 single-select options render a dropdown with max_values=1 on row 1."""
    spec = pui.build_question_spec("Q", [f"o{i}" for i in range(7)], multi=False, max_choices=1)
    view = pui.build_question_view("s", "m", spec)
    (select,) = [c.item for c in view.children]
    assert isinstance(select, discord.ui.Select)
    assert select.custom_id == "kuis:s:m"
    assert (select.min_values, select.max_values) == (1, 1)
    assert select._row == 1
    assert [o.value for o in select.options] == [str(i) for i in range(7)]


def test_multiselect_uses_select_with_max_values() -> None:
    """Multiselect renders a Select(min_values=1, max_values=max)."""
    spec = pui.build_question_spec("Q", ["a", "b", "c"], multi=True, max_choices=2)
    assert spec == {"type": "question", "prompt": "Q", "options": ["a", "b", "c"], "multi": True, "max": 2}
    (select,) = [c.item for c in pui.build_question_view("s", "m", spec).children]
    assert (select.min_values, select.max_values) == (1, 2)


def test_spec_caps_options_at_25() -> None:
    """Discord select menus hold at most 25 options."""
    spec = pui.build_question_spec("Q", [str(i) for i in range(30)], multi=True, max_choices=30)
    assert len(spec["options"]) == 25 and spec["max"] == 25


def test_long_options_use_letter_labels_and_embed_list() -> None:
    """Long option texts use A/B labels with the full text in the embed description."""
    spec = pui.build_question_spec("Q", ["a very long option text here", "b"], multi=False, max_choices=1)
    labels = [c.item.label for c in pui.build_question_view("s", "m", spec).children]
    assert labels == ["A", "B"]
    assert pui.build_question_embed(spec).description == "**A.** a very long option text here\n**B.** b"


def test_answered_view_is_disabled_and_highlights_choice() -> None:
    """After answering, every component is disabled and the chosen button is highlighted."""
    spec = pui.build_question_spec("Q", ["a", "b"], multi=False, max_choices=1)
    spec.update(answered="b", chosen=[1])
    items = [c.item for c in pui.build_question_view("s", "m", spec).children]
    assert all(i.disabled for i in items)
    assert items[1].style == discord.ButtonStyle.success
    embed = pui.build_question_embed(spec)
    assert embed.footer.text == "✓ b"


@pytest.mark.asyncio
async def test_from_custom_id_rebuilds_items() -> None:
    """DynamicItems rebuild from custom_id alone (as after a restart)."""
    match = re.fullmatch(pui.QuestionButton.__discord_ui_compiled_template__, "kui:s1:m1:2")
    assert match is not None
    btn = await pui.QuestionButton.from_custom_id(MagicMock(), discord.ui.Button(label="x"), match)
    assert (btn.session_id, btn.message_id, btn.idx) == ("s1", "m1", 2)
    match = re.fullmatch(pui.QuestionSelect.__discord_ui_compiled_template__, "kuis:s1:m1")
    sel = await pui.QuestionSelect.from_custom_id(MagicMock(), MagicMock(), match)
    assert (sel.session_id, sel.message_id) == ("s1", "m1")
    match = re.fullmatch(pui.DetailsToggle.__discord_ui_compiled_template__, "kud:s1:m1-d0")
    tog = await pui.DetailsToggle.from_custom_id(MagicMock(), discord.ui.Button(label=DETAILS_EXPAND_LABEL), match)
    assert (tog.session_id, tog.details_key) == ("s1", "m1-d0")


# ---------------------------------------------------------------------------
# Answer handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_button_click_posts_option_and_disables(registered: tuple[object, _MemoryDB, MagicMock]) -> None:
    """A button click defers immediately, posts the option text as a user message, and disables the components."""
    client, db, chatbot = registered
    await db.save_ui_prompt("s", "m", pui.build_question_spec("Q", ["a", "b [x]"], False, 1))
    interaction = _interaction(client)
    await pui.QuestionButton("s", "m", 1, label="b [x]").callback(interaction)

    interaction.response.defer.assert_awaited_once_with()
    posted = chatbot.gateway.post.call_args.args[0]
    assert posted.content == "b [x]"
    assert posted.role == MessageRole.USER and posted.session_id == "s"
    assert posted.metadata["discord_message_id"] == "777"
    interaction.channel.send.assert_awaited_once_with("<@42> selected: **b [x]**")
    view = interaction.edit_original_response.call_args.kwargs["view"]
    assert all(c.item.disabled for c in view.children)
    assert db.rows[("s", "m")]["answered"] == "b [x]"


@pytest.mark.asyncio
async def test_multiselect_submit_posts_joined_answer(registered: tuple[object, _MemoryDB, MagicMock]) -> None:
    """A multiselect submit posts ONE message `选择：A；C` in option order."""
    client, db, chatbot = registered
    await db.save_ui_prompt("s", "m", pui.build_question_spec("Q", ["A", "B", "C"], True, 2))
    interaction = _interaction(client, values=["2", "0"])
    await pui.QuestionSelect("s", "m").callback(interaction)
    interaction.response.defer.assert_awaited_once_with()
    chatbot.gateway.post.assert_awaited_once()
    assert chatbot.gateway.post.call_args.args[0].content == "选择：A；C"
    select = interaction.edit_original_response.call_args.kwargs["view"].children[0].item
    assert select.disabled
    assert [o.default for o in select.options] == [True, False, True]


@pytest.mark.asyncio
async def test_single_dropdown_submit_posts_option(registered: tuple[object, _MemoryDB, MagicMock]) -> None:
    """A single-select dropdown posts the plain option text."""
    client, db, chatbot = registered
    await db.save_ui_prompt("s", "m", pui.build_question_spec("Q", [f"o{i}" for i in range(7)], False, 1))
    await pui.QuestionSelect("s", "m").callback(_interaction(client, values=["5"]))
    assert chatbot.gateway.post.call_args.args[0].content == "o5"


@pytest.mark.asyncio
async def test_answered_or_missing_spec_is_rejected(registered: tuple[object, _MemoryDB, MagicMock]) -> None:
    """Second answers and unknown prompts get an ephemeral notice via followup and post nothing."""
    client, db, chatbot = registered
    spec = pui.build_question_spec("Q", ["a"], False, 1)
    spec["answered"] = "a"
    await db.save_ui_prompt("s", "m", spec)
    interaction = _interaction(client)
    await pui.QuestionButton("s", "m", 0).callback(interaction)
    interaction.response.defer.assert_awaited_once_with()
    assert interaction.followup.send.call_args.kwargs == {"ephemeral": True}
    interaction = _interaction(client)
    await pui.QuestionButton("s", "unknown", 0).callback(interaction)
    interaction.response.defer.assert_awaited_once_with()
    interaction.followup.send.assert_awaited_once()
    interaction = _interaction(client)
    await db.save_ui_prompt("s", "m2", pui.build_question_spec("Q", ["a"], False, 1))
    await pui.QuestionButton("s", "m2", 9).callback(interaction)
    interaction.response.defer.assert_awaited_once_with()
    interaction.followup.send.assert_awaited_once()
    chatbot.gateway.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_details_toggle_expands_and_collapses(registered: tuple[object, _MemoryDB, MagicMock]) -> None:
    """The details toggle defers immediately, derives state from the message, and edits it in place."""
    client, db, _ = registered
    await db.save_ui_prompt("s", "m-d0", {"type": "details", "title": "T", "body": "B"})
    interaction = _interaction(client, content="▸ **T**")
    await pui.DetailsToggle("s", "m-d0").callback(interaction)
    interaction.response.defer.assert_awaited_once_with()
    kwargs = interaction.edit_original_response.call_args.kwargs
    assert kwargs["content"] == "▾ **T**\nB"
    assert kwargs["view"].children[0].item.label == DETAILS_COLLAPSE_LABEL
    interaction = _interaction(client, content="▾ **T**\nB")
    await pui.DetailsToggle("s", "m-d0").callback(interaction)
    assert interaction.edit_original_response.call_args.kwargs["content"] == "▸ **T**"


# ---------------------------------------------------------------------------
# Adapter wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_setup_hook_registers_dynamic_items(tmp_path: Any) -> None:
    """The adapter registers persistent items in setup_hook and maps its client to itself."""
    cfg = KesokuConfig()
    cfg.workspace.sessions_dir = str(tmp_path)
    cfg.discord = DiscordConfig(enabled=True, bot_token="t", chatbot_id="bot")
    gw = MagicMock(spec=Gateway)
    with patch("kesoku.gateway.chatbot.discord.adapter.get_config", return_value=cfg):
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=MagicMock(id=1)):
            bot = DiscordChatbot(chatbot_id="bot", gateway=gw)
    with patch.object(bot.bot, "add_dynamic_items") as add_items:
        await bot.bot.setup_hook()
    add_items.assert_called_once_with(pui.QuestionButton, pui.QuestionSelect, pui.DetailsToggle)
    interaction = MagicMock()
    interaction.client = bot.bot
    assert pui.chatbot_for(interaction) is bot
