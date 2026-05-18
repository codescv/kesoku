"""Unit tests for Kesoku Discord chatbot adapter."""

import asyncio
import datetime
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch
import discord
import pytest

from kesoku.config import DiscordConfig, KesokuConfig
from kesoku.constants import ROLE_ASSISTANT, STATUS_DELIVERED, STATUS_PENDING_AGENT, TYPE_TEXT, TYPE_THOUGHT
from kesoku.db import Message, Session
from kesoku.gateway.chatbot.discord import DiscordChatbot
from kesoku.gateway.gateway import Gateway


@pytest.fixture
def mock_config() -> KesokuConfig:
    """Provide a mock Kesoku configuration."""
    cfg = KesokuConfig()
    cfg.discord = DiscordConfig(
        enabled=True, bot_token="test_token", chatbot_id="discord_test", user_allowlist=["allowed_user"]
    )
    return cfg


@pytest.fixture
def mock_gateway() -> MagicMock:
    """Provide a mock Gateway instance."""
    gw = MagicMock(spec=Gateway)
    gw.get_session = AsyncMock(return_value=None)
    gw.get_session_by_channel = AsyncMock(return_value=None)
    gw.create_session = AsyncMock(return_value=Session(id="thread123", title="Test Session"))
    gw.update_session_updated_at = AsyncMock()
    gw.post = AsyncMock()
    gw.update_message_status = AsyncMock()
    return gw


@pytest.mark.asyncio
async def test_init_missing_token() -> None:
    """Test initialization without token raises ValueError."""
    cfg = KesokuConfig()
    cfg.discord = DiscordConfig(enabled=True, bot_token=None, chatbot_id="discord")
    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=cfg):
        gw = MagicMock(spec=Gateway)
        with pytest.raises(ValueError, match="Discord bot token is required"):
            DiscordChatbot(chatbot_id="discord", gateway=gw)


@pytest.mark.asyncio
async def test_on_message_ignore_self(mock_config: KesokuConfig, mock_gateway: MagicMock) -> None:
    """Test bot ignores its own messages."""
    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=mock_config):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)
            mock_msg = MagicMock(spec=discord.Message)
            mock_msg.author = mock_client_user

            await bot.on_message(mock_msg)
            mock_gateway.post.assert_not_called()


@pytest.mark.asyncio
async def test_on_message_allowlist(mock_config: KesokuConfig, mock_gateway: MagicMock) -> None:
    """Test allowlist filtering behavior."""
    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=mock_config):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)

            # Unallowed user, not mentioning bot -> ignored
            msg_unallowed = MagicMock(spec=discord.Message)
            unallowed_author = MagicMock(spec=discord.Member, id=111, display_name="unallowed")
            unallowed_author.name = "unallowed"
            msg_unallowed.author = unallowed_author
            msg_unallowed.mentions = []
            await bot.on_message(msg_unallowed)
            mock_gateway.post.assert_not_called()

            # Unallowed user, explicitly mentioning bot -> accepted
            mock_thread = AsyncMock(spec=discord.Thread)
            mock_thread.id = 12345
            mock_thread.name = "thread_name"
            mock_thread.guild = MagicMock(spec=discord.Guild)
            mock_thread.guild.name = "GuildName"
            mock_thread.guild.members = []
            mock_thread.join = AsyncMock()

            msg_unallowed.mentions = [mock_client_user]
            msg_unallowed.channel = mock_thread
            msg_unallowed.content = "Hello bot"
            msg_unallowed.id = 777
            msg_unallowed.created_at.timestamp.return_value = 1000.0
            await bot.on_message(msg_unallowed)
            mock_gateway.post.assert_called_once()
            mock_gateway.post.reset_mock()

            # Allowed user -> accepted
            msg_allowed = MagicMock(spec=discord.Message)
            allowed_author = MagicMock(spec=discord.Member, id=222, display_name="Allowed")
            allowed_author.name = "allowed_user"
            msg_allowed.author = allowed_author
            msg_allowed.mentions = []
            msg_allowed.channel = mock_thread
            msg_allowed.content = "Hello allowed"
            msg_allowed.id = 888
            msg_allowed.created_at.timestamp.return_value = 1001.0
            await bot.on_message(msg_allowed)
            mock_gateway.post.assert_called_once()
            mock_gateway.post.reset_mock()

            # Explicitly mentioning someone else -> ignored
            other_user = MagicMock(spec=discord.User, id=333)
            msg_allowed.mentions = [other_user]
            await bot.on_message(msg_allowed)
            mock_gateway.post.assert_not_called()


@pytest.mark.asyncio
async def test_on_message_concurrent_thread_creation(mock_config: KesokuConfig, mock_gateway: MagicMock) -> None:
    """Test recovering when create_thread raises HTTPException due to peer bot creating thread first."""
    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=mock_config):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)

            msg = MagicMock(spec=discord.Message)
            allowed_author = MagicMock(spec=discord.Member, id=222, display_name="Allowed")
            allowed_author.name = "allowed_user"
            msg.author = allowed_author
            msg.mentions = []
            msg.content = "Hello world"
            msg.id = 555
            msg.thread = None
            msg.created_at.timestamp.return_value = 2000.0

            mock_channel = MagicMock(spec=discord.TextChannel)
            mock_guild = MagicMock(spec=discord.Guild)
            mock_guild.name = "Guild"
            mock_channel.guild = mock_guild
            msg.channel = mock_channel

            # create_thread fails
            msg.create_thread = AsyncMock(side_effect=discord.HTTPException(AsyncMock(), "Creation failed"))

            # peer bot created thread, so get_thread finds it on retry
            peer_thread = AsyncMock(spec=discord.Thread)
            peer_thread.id = 555
            peer_thread.name = "Peer Thread"
            peer_thread.guild = mock_guild
            peer_thread.join = AsyncMock()
            mock_guild.get_thread.side_effect = [None, peer_thread]

            await bot.on_message(msg)
            peer_thread.join.assert_called_once()
            mock_gateway.post.assert_called_once()


@pytest.mark.asyncio
async def test_handle_message_chunking(mock_config: KesokuConfig, mock_gateway: MagicMock) -> None:
    """Test outgoing message > 2000 chars is chunked by newlines."""
    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=mock_config):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)
            mock_channel = AsyncMock(spec=discord.Thread)
            bot.bot.get_channel = MagicMock(return_value=mock_channel)

            # Generate message with 3 lines, each 900 characters (total 2700 chars)
            line1 = "A" * 900 + "\n"
            line2 = "B" * 900 + "\n"
            line3 = "C" * 900 + "\n"
            long_content = line1 + line2 + line3

            msg = Message(
                id="msg123",
                session_id="thread123",
                chatbot_id="discord_test",
                channel_id="12345",
                sender="Kesoku",
                role=ROLE_ASSISTANT,
                type=TYPE_TEXT,
                content=long_content,
            )

            await bot.handle_message(msg)
            # Chunk 1 should contain line1 + line2 (1802 chars). Chunk 2 should contain line3 (901 chars).
            assert mock_channel.send.call_count == 2
            mock_gateway.update_message_status.assert_called_once_with("msg123", STATUS_DELIVERED)


@pytest.mark.asyncio
async def test_handle_message_with_files_split_and_upload(mock_config: KesokuConfig, mock_gateway: MagicMock) -> None:
    """Test that messages containing valid file blocks are split and uploaded correctly."""
    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=mock_config):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)
            mock_channel = AsyncMock(spec=discord.Thread)
            bot.bot.get_channel = MagicMock(return_value=mock_channel)

            # Setup message with text before, file, and text after
            content = "Hello [file: /tmp/test_image.png] how are you?"
            msg = Message(
                id="msg123",
                session_id="thread123",
                chatbot_id="discord_test",
                channel_id="12345",
                sender="Kesoku",
                role=ROLE_ASSISTANT,
                type=TYPE_TEXT,
                content=content,
            )

            mock_file = MagicMock(spec=discord.File)
            with patch("os.path.exists", return_value=True) as mock_exists:
                with patch("discord.File", return_value=mock_file) as mock_file_class:
                    await bot.handle_message(msg)
                    
                    # Verify path existence was checked
                    mock_exists.assert_called_once_with("/tmp/test_image.png")
                    # Verify discord.File was instantiated with path
                    mock_file_class.assert_called_once_with("/tmp/test_image.png")
                    
                    # channel.send should be called 3 times: "Hello ", file=mock_file, and " how are you?"
                    assert mock_channel.send.call_count == 3
                    mock_channel.send.assert_any_call("Hello ")
                    mock_channel.send.assert_any_call(file=mock_file)
                    mock_channel.send.assert_any_call(" how are you?")
                    
                    mock_gateway.update_message_status.assert_called_once_with("msg123", STATUS_DELIVERED)


@pytest.mark.asyncio
async def test_handle_message_with_non_existent_file(mock_config: KesokuConfig, mock_gateway: MagicMock) -> None:
    """Test that missing files trigger a user-facing warning message."""
    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=mock_config):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)
            mock_channel = AsyncMock(spec=discord.Thread)
            bot.bot.get_channel = MagicMock(return_value=mock_channel)

            content = "See this: [file: /tmp/ghost.png]"
            msg = Message(
                id="msg123",
                session_id="thread123",
                chatbot_id="discord_test",
                channel_id="12345",
                sender="Kesoku",
                role=ROLE_ASSISTANT,
                type=TYPE_TEXT,
                content=content,
            )

            with patch("os.path.exists", return_value=False) as mock_exists:
                await bot.handle_message(msg)
                mock_exists.assert_called_once_with("/tmp/ghost.png")
                
                # channel.send should be called 2 times: text segment and warning segment
                assert mock_channel.send.call_count == 2
                mock_channel.send.assert_any_call("See this: ")
                mock_channel.send.assert_any_call("⚠️ File not found: /tmp/ghost.png")


@pytest.mark.asyncio
async def test_handle_message_with_empty_whitespace_guards(mock_config: KesokuConfig, mock_gateway: MagicMock) -> None:
    """Test that empty or whitespace-only text segments are guarded and not sent."""
    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=mock_config):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)
            mock_channel = AsyncMock(spec=discord.Thread)
            bot.bot.get_channel = MagicMock(return_value=mock_channel)

            # Setup content with only whitespace surrounding a file block
            content = "   [file: /tmp/only_file.zip]    "
            msg = Message(
                id="msg123",
                session_id="thread123",
                chatbot_id="discord_test",
                channel_id="12345",
                sender="Kesoku",
                role=ROLE_ASSISTANT,
                type=TYPE_TEXT,
                content=content,
            )

            mock_file = MagicMock(spec=discord.File)
            with patch("os.path.exists", return_value=True):
                with patch("discord.File", return_value=mock_file):
                    await bot.handle_message(msg)
                    
                    # channel.send should be called exactly once (only for the file attachment)
                    assert mock_channel.send.call_count == 1
                    mock_channel.send.assert_called_once_with(file=mock_file)


@pytest.mark.asyncio
async def test_on_message_timestamp_formatting(mock_config: KesokuConfig, mock_gateway: MagicMock) -> None:
    """Test that incoming Discord messages have timestamps formatted in readable local time."""
    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=mock_config):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)

            mock_thread = AsyncMock(spec=discord.Thread)
            mock_thread.id = 12345
            mock_thread.name = "thread_name"
            mock_thread.guild = MagicMock(spec=discord.Guild)
            mock_thread.guild.name = "GuildName"
            mock_thread.guild.members = []
            mock_thread.join = AsyncMock()

            msg = MagicMock(spec=discord.Message)
            msg.author = MagicMock(spec=discord.Member, id=222, display_name="Allowed")
            msg.author.name = "allowed_user"
            msg.mentions = []
            msg.channel = mock_thread
            msg.content = "Hello test"
            msg.id = 888

            # Create a mock datetime object for created_at
            tz_utc = datetime.timezone.utc
            dt = datetime.datetime(2026, 5, 18, 15, 21, 48, tzinfo=tz_utc)
            msg.created_at = dt

            await bot.on_message(msg)

            # Verify post was called with the formatted readable local time timestamp
            mock_gateway.post.assert_called_once()
            posted_msg = mock_gateway.post.call_args[0][0]

            from kesoku.gateway.chatbot.discord import _get_local_timezone_name
            tz_name = _get_local_timezone_name()
            local_time_str = dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
            expected_content = f"`Allowed` <@222> at `{local_time_str} {tz_name}`:\nHello test"
            assert posted_msg.content == expected_content


def test_build_discord_sys_prompt_dm() -> None:
    """Test prompt construction for Direct Messages."""
    from kesoku.gateway.chatbot.discord import _build_discord_sys_prompt

    mock_dm = MagicMock(spec=discord.DMChannel)
    mock_dm.guild = None
    mock_dm.id = 98765

    mock_user = MagicMock(spec=discord.User, id=12345, display_name="TestUser")

    prompt = _build_discord_sys_prompt(mock_dm, mock_user)

    assert "You are talking to the user via discord." in prompt
    assert "Users" not in prompt
    assert "TestUser" not in prompt
    assert "Mentioning Users" not in prompt
    assert "Channel Topic" not in prompt
    assert "Response Format" in prompt


def test_build_discord_sys_prompt_thread_with_topic() -> None:
    """Test prompt construction for a thread with a parent channel topic."""
    from kesoku.gateway.chatbot.discord import _build_discord_sys_prompt

    mock_parent = MagicMock(spec=discord.TextChannel)
    mock_parent.name = "general"
    mock_parent.id = 444
    mock_parent.topic = "This is the general channel topic."

    mock_thread = MagicMock(spec=discord.Thread)
    mock_thread.name = "help-thread"
    mock_thread.id = 555
    mock_thread.parent = mock_parent
    mock_thread.guild = MagicMock(spec=discord.Guild)
    mock_thread.guild.name = "AwesomeServer"
    mock_thread.guild.members = []

    mock_user = MagicMock(spec=discord.User, id=12345, display_name="TestUser")

    prompt = _build_discord_sys_prompt(mock_thread, mock_user)

    assert "You are currently chatting in a Discord thread named \"#help-thread\" (ID: 555)" in prompt
    assert "under channel \"#general\" (ID: 444) on the server 'AwesomeServer'." in prompt
    assert "## Channel Topic\nThis is the general channel topic." in prompt
    assert "Response Format" in prompt


