"""Tests for rendering `<details>` sections and question fallbacks across chatbot adapters."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import discord
import pytest
from rich.console import Console

from kesoku.config import DiscordConfig, KesokuConfig
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.gateway.chatbot.base import Chatbot
from kesoku.gateway.chatbot.cli_bot import CLIChatbot
from kesoku.gateway.chatbot.discord import DiscordChatbot
from kesoku.gateway.chatbot.discord.ui import (
    DETAILS_COLLAPSE_LABEL,
    DETAILS_EXPAND_LABEL,
    DISCORD_MESSAGE_LIMIT,
    DetailsView,
    render_details_content,
)
from kesoku.gateway.gateway import Gateway


def _assistant_msg(content: str) -> Message:
    """Build an outgoing assistant text message."""
    return Message(
        id="m1",
        session_id="s1",
        chatbot_id="bot",
        channel_id="123",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content=content,
    )


class _RecordingChatbot(Chatbot):
    """Minimal chatbot recording the text chunks it sends."""

    def __init__(self, gateway: Any) -> None:
        super().__init__("bot", gateway)
        self.sent: list[str] = []

    async def handle_message(self, message: Message) -> None:
        """Delegate to the shared renderer."""
        await self.render_outgoing_message(message)

    async def send_text_chunks(self, channel_id: str, chunks: list[str], message: Message) -> None:
        """Record chunks instead of sending."""
        self.sent.extend(chunks)


@pytest.fixture
def gateway() -> MagicMock:
    """Mock gateway with an async DB."""
    gw = MagicMock(spec=Gateway)
    gw.db = AsyncMock()
    gw.db.get_session = AsyncMock(return_value=None)
    gw.agent = None
    return gw


@pytest.mark.asyncio
async def test_base_renders_details_as_bold_title_and_body(gateway: MagicMock) -> None:
    """Platforms without collapsing get `**TITLE**` followed by BODY."""
    bot = _RecordingChatbot(gateway)
    await bot.handle_message(_assistant_msg("Intro\n<details><summary>Why</summary>Because.</details>"))
    joined = "\n".join(bot.sent)
    assert "Intro" in joined
    assert "**Why**\n\nBecause." in joined
    assert "<details>" not in joined
    gateway.db.update_message_status.assert_called_with("m1", MessageStatus.DELIVERED)


@pytest.mark.asyncio
async def test_base_passes_multi_and_max_to_question_segment(gateway: MagicMock) -> None:
    """send_question_segment receives multi/max_choices from the ```ui block."""
    bot = _RecordingChatbot(gateway)
    bot.send_question_segment = AsyncMock()  # type: ignore[method-assign]
    content = "```ui\ntype: multiselect\nprompt: Pick\noptions: [a, b, c]\nmax: 2\n```"
    await bot.handle_message(_assistant_msg(content))
    bot.send_question_segment.assert_awaited_once()
    args, kwargs = bot.send_question_segment.call_args
    assert args[1:3] == ("Pick", ["a", "b", "c"])
    assert kwargs == {"multi": True, "max_choices": 2}


@pytest.mark.asyncio
async def test_question_as_text_fallback(gateway: MagicMock) -> None:
    """The shared text fallback renders a numbered list."""
    bot = _RecordingChatbot(gateway)
    await bot.send_question_as_text("Q?", ["a", "b"], _assistant_msg(""), multi=True, max_choices=2)
    assert "Q?" in bot.sent[0] and "1. a" in bot.sent[0] and "2. b" in bot.sent[0]


def test_render_details_content() -> None:
    """Collapsed shows the title only; expanded shows the body and respects Discord's limit."""
    assert render_details_content("T", "B", expanded=False) == "▸ **T**"
    assert render_details_content("T", "B", expanded=True) == "▾ **T**\nB"
    long = render_details_content("T", "x" * 5000, expanded=True)
    assert len(long) == DISCORD_MESSAGE_LIMIT and long.endswith("…")


@pytest.mark.asyncio
async def test_details_view_toggle() -> None:
    """Clicking the toggle expands, clicking again collapses (message edited in place)."""
    view = DetailsView("T", "Body")
    assert view.toggle_button.label == DETAILS_EXPAND_LABEL
    interaction = MagicMock()
    interaction.response.edit_message = AsyncMock()
    await view.toggle(interaction)
    interaction.response.edit_message.assert_awaited_with(content="▾ **T**\nBody", view=view)
    assert view.toggle_button.label == DETAILS_COLLAPSE_LABEL
    await view.toggle(interaction)
    interaction.response.edit_message.assert_awaited_with(content="▸ **T**", view=view)
    assert view.toggle_button.label == DETAILS_EXPAND_LABEL


@pytest.mark.asyncio
async def test_discord_sends_details_with_toggle_view(tmp_path: Any, gateway: MagicMock) -> None:
    """Discord renders <details> as `▸ TITLE` plus a toggle button view."""
    cfg = KesokuConfig()
    cfg.workspace.sessions_dir = str(tmp_path / "sessions")
    cfg.discord = DiscordConfig(enabled=True, bot_token="t", chatbot_id="bot")
    with patch("kesoku.gateway.chatbot.discord.adapter.get_config", return_value=cfg):
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=MagicMock(id=1)):
            bot = DiscordChatbot(chatbot_id="bot", gateway=gateway)
            channel = AsyncMock(spec=discord.Thread)
            bot.bot.get_channel = MagicMock(return_value=channel)
            await bot.handle_message(_assistant_msg("<details><summary>Explain</summary>Long text</details>"))
            channel.send.assert_awaited_once()
            args, kwargs = channel.send.call_args
            assert args[0] == "▸ **Explain**"
            view = kwargs["view"]
            assert view.children[0].item.custom_id == "kud:s1:m1-d0"
            assert view.children[0].item.label == DETAILS_EXPAND_LABEL
            gateway.db.save_ui_prompt.assert_awaited_once_with(
                "s1", "m1-d0", {"type": "details", "title": "Explain", "body": "Long text"}
            )


@pytest.mark.asyncio
async def test_cli_renders_ui_block_and_details_as_text(gateway: MagicMock) -> None:
    """The CLI degrades a ```ui block to a numbered list and <details> to bold title + body."""
    console = Console(record=True, width=100)
    bot = CLIChatbot(chatbot_id="cli", gateway=gateway, console=console)
    content = (
        "Hi\n```ui\ntype: multiselect\nprompt: Pick two\noptions:\n  - Alpha [x]\n  - Beta\nmax: 2\n```\n"
        "<details><summary>More</summary>Hidden body</details>"
    )
    msg = _assistant_msg(content)
    msg.timestamp = bot.start_time + 1
    await bot.handle_message(msg)
    out = console.export_text()
    assert "Pick two (select up to 2)" in out
    assert "1. Alpha [x]" in out and "2. Beta" in out
    assert "More" in out and "Hidden body" in out
    assert "```ui" not in out and "<details>" not in out
    assert bot.final_response_event.is_set()
