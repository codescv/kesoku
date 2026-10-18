"""Tests that Discord system messages never trigger an agent turn."""

import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import discord
import pytest

from kesoku.config import DiscordConfig, KesokuConfig
from kesoku.db import Session
from kesoku.gateway.chatbot.discord import DiscordChatbot
from kesoku.gateway.chatbot.discord.adapter import is_user_chat_message
from kesoku.gateway.gateway import Gateway


@pytest.fixture
def mock_config(tmp_path: Any) -> KesokuConfig:
    """Provide a Kesoku configuration with temporary paths and no allowlist."""
    cfg = KesokuConfig()
    cfg.workspace.sessions_dir = str(tmp_path / "sessions")
    cfg.workspace.db_path = str(tmp_path / "kesoku.db")
    cfg.discord = DiscordConfig(enabled=True, bot_token="test_token", chatbot_id="discord_test")
    return cfg


@pytest.fixture
def mock_gateway() -> MagicMock:
    """Provide a mock Gateway instance."""
    gw = MagicMock(spec=Gateway)
    gw.db = AsyncMock()
    gw.db.get_session = AsyncMock(return_value=None)
    gw.db.get_session_by_channel = AsyncMock(return_value=None)
    gw.db.get_channel_role = AsyncMock(return_value=None)
    gw.create_session = AsyncMock(return_value=Session(id="thread123", title="Test Session"))
    gw.post = AsyncMock()
    gw.agent = None
    return gw


def _make_message(msg_type: discord.MessageType, is_system: bool) -> MagicMock:
    """Build a mock Discord message posted inside a thread."""
    thread = AsyncMock(spec=discord.Thread)
    thread.id = 4242
    thread.name = "renamed thread"
    thread.parent = None
    thread.guild = MagicMock(spec=discord.Guild)
    thread.guild.name = "Guild"
    thread.guild.members = []
    thread.typing.return_value = MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock())
    msg = MagicMock(spec=discord.Message)
    msg.type = msg_type
    msg.is_system.return_value = is_system
    msg.author = MagicMock(spec=discord.Member, id=222, display_name="User")
    msg.author.name = "user"
    msg.mentions = []
    msg.channel = thread
    msg.content = "renamed thread"
    msg.id = 999
    msg.attachments = []
    msg.created_at = datetime.datetime.now(datetime.UTC)
    return msg


@pytest.mark.parametrize(
    "msg_type",
    [
        discord.MessageType.channel_name_change,
        discord.MessageType.pins_add,
        discord.MessageType.thread_created,
        discord.MessageType.recipient_add,
        discord.MessageType.thread_starter_message,
    ],
)
def test_is_user_chat_message_rejects_system_types(msg_type: discord.MessageType) -> None:
    """Non-chat message types are rejected."""
    assert not is_user_chat_message(_make_message(msg_type, is_system=True))


@pytest.mark.parametrize("msg_type", [discord.MessageType.default, discord.MessageType.reply])
def test_is_user_chat_message_accepts_chat_types(msg_type: discord.MessageType) -> None:
    """Default and reply messages are accepted."""
    assert is_user_chat_message(_make_message(msg_type, is_system=False))


def test_is_user_chat_message_rejects_flagged_system() -> None:
    """A default-typed message flagged as system is still rejected."""
    assert not is_user_chat_message(_make_message(discord.MessageType.default, is_system=True))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "msg_type",
    [discord.MessageType.channel_name_change, discord.MessageType.pins_add, discord.MessageType.thread_created],
)
async def test_on_message_ignores_system_messages(
    mock_config: KesokuConfig, mock_gateway: MagicMock, msg_type: discord.MessageType
) -> None:
    """System messages such as thread renames never reach the gateway."""
    with patch("kesoku.gateway.chatbot.discord.adapter.get_config", return_value=mock_config):
        client_user = MagicMock(spec=discord.ClientUser, id=1)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)
            await bot.on_message(_make_message(msg_type, is_system=True))
            mock_gateway.post.assert_not_called()


@pytest.mark.asyncio
async def test_on_message_accepts_normal_message(mock_config: KesokuConfig, mock_gateway: MagicMock) -> None:
    """A normal chat message in a thread is ingested."""
    with patch("kesoku.gateway.chatbot.discord.adapter.get_config", return_value=mock_config):
        client_user = MagicMock(spec=discord.ClientUser, id=1)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)
            await bot.on_message(_make_message(discord.MessageType.default, is_system=False))
            mock_gateway.post.assert_called_once()
