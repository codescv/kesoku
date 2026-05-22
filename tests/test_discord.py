"""Unit tests for Kesoku Discord chatbot adapter."""

import asyncio
import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import discord
import pytest

from kesoku.config import DiscordChannelOverride, DiscordConfig, KesokuConfig
from kesoku.constants import ROLE_ASSISTANT, STATUS_DELIVERED, TYPE_TEXT
from kesoku.db import Message, Session
from kesoku.gateway.chatbot.discord import DiscordChatbot
from kesoku.gateway.gateway import Gateway


@pytest.fixture
def mock_config(tmp_path: Any) -> KesokuConfig:
    """Provide a mock Kesoku configuration with temporary paths.

    Args:
        tmp_path: Pytest's temporary path fixture.

    Returns:
        A mock KesokuConfig instance.
    """
    cfg = KesokuConfig()
    cfg.workspace.sessions_dir = str(tmp_path / "sessions")
    cfg.workspace.db_path = str(tmp_path / "kesoku.db")
    cfg.workspace.skills_dir = str(tmp_path / "skills")
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
async def test_init_with_env_token() -> None:
    """Test initialization with DISCORD_TOKEN environment variable fallback."""
    cfg = KesokuConfig()
    cfg.discord = DiscordConfig(enabled=True, bot_token=None, chatbot_id="discord")
    with (
        patch("kesoku.gateway.chatbot.discord.get_config", return_value=cfg),
        patch.dict("os.environ", {"DISCORD_TOKEN": "env_token_value"}),
    ):
        gw = MagicMock(spec=Gateway)
        bot = DiscordChatbot(chatbot_id="discord", gateway=gw)
        assert bot.bot_token == "env_token_value"


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
            tz_utc = datetime.UTC
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


def test_build_discord_custom_prompt_dm() -> None:
    """Test prompt construction for Direct Messages."""
    from kesoku.gateway.chatbot.discord import _build_discord_custom_prompt

    mock_dm = MagicMock(spec=discord.DMChannel)
    mock_dm.guild = None
    mock_dm.id = 98765

    mock_user = MagicMock(spec=discord.User, id=12345, display_name="TestUser")

    prompt = _build_discord_custom_prompt(mock_dm, mock_user)

    assert "You are talking to the user via discord." in prompt
    assert "Users" not in prompt
    assert "TestUser" not in prompt
    assert "Mentioning Users" not in prompt
    assert "Channel Topic" not in prompt
    assert "Response Format" in prompt


def test_build_discord_custom_prompt_thread_with_topic() -> None:
    """Test prompt construction for a thread with a parent channel topic."""
    from kesoku.gateway.chatbot.discord import _build_discord_custom_prompt

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

    prompt = _build_discord_custom_prompt(mock_thread, mock_user)

    assert 'You are currently chatting in a Discord thread named "#help-thread" (ID: 555)' in prompt
    assert "under channel \"#general\" (ID: 444) on the server 'AwesomeServer'." in prompt
    assert "## Channel Topic\nThis is the general channel topic." in prompt
    assert "Response Format" in prompt


@pytest.mark.asyncio
async def test_typing_status_lifecycle(mock_config: KesokuConfig, mock_gateway: MagicMock) -> None:
    """Test typing status is started on message receive and stopped on final response delivery."""
    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=mock_config):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)

            # Mock channel / thread
            mock_thread = AsyncMock(spec=discord.Thread)
            mock_thread.id = 12345
            mock_thread.name = "thread_name"
            mock_thread.guild = MagicMock(spec=discord.Guild)
            mock_thread.guild.name = "GuildName"
            mock_thread.guild.members = []
            mock_thread.join = AsyncMock()

            # Mock typing context manager
            mock_typing = MagicMock()
            mock_typing.__aenter__ = AsyncMock()
            mock_typing.__aexit__ = AsyncMock()
            mock_thread.typing.return_value = mock_typing

            bot.bot.get_channel = MagicMock(return_value=mock_thread)

            # Verify initially empty
            assert "12345" not in bot._typing_tasks

            # Simulate incoming message to trigger typing task
            msg = MagicMock(spec=discord.Message)
            allowed_author = MagicMock(spec=discord.Member, id=222, display_name="Allowed")
            allowed_author.name = "allowed_user"
            msg.author = allowed_author
            msg.mentions = []
            msg.channel = mock_thread
            msg.content = "Trigger typing"
            msg.id = 888
            msg.created_at = datetime.datetime.now(datetime.UTC)

            await bot.on_message(msg)

            # The typing task should be created and stored
            assert "12345" in bot._typing_tasks
            task = bot._typing_tasks["12345"]
            assert not task.done()

            # Wait a tiny bit to let the task run and enter the typing context
            await asyncio.sleep(0.01)
            mock_thread.typing.assert_called_once()
            mock_typing.__aenter__.assert_called_once()

            # Simulate receiving the final assistant response to cancel typing
            final_msg = Message(
                id="msg123",
                session_id="thread123",
                chatbot_id="discord_test",
                channel_id="12345",
                sender="Kesoku",
                role=ROLE_ASSISTANT,
                type=TYPE_TEXT,
                content="Final reply",
            )

            await bot.handle_message(final_msg)

            # Wait a tiny bit for task cancellation to propagate
            await asyncio.sleep(0.01)

            # The typing task should be popped and cancelled
            assert "12345" not in bot._typing_tasks
            assert task.cancelled() or task.done()

            # Ensure typing context was exited
            await asyncio.sleep(0.01)
            mock_typing.__aexit__.assert_called_once()


@pytest.mark.asyncio
async def test_typing_status_cleanup_on_stop(mock_config: KesokuConfig, mock_gateway: MagicMock) -> None:
    """Test that all active typing tasks are cancelled when the bot is stopped."""
    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=mock_config):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)

            # Manually inject a mock task
            mock_task = MagicMock(spec=asyncio.Task)
            bot._typing_tasks["555"] = mock_task

            # Patch bot.close to prevent actual connection closure issues in tests
            bot.bot.close = AsyncMock()
            bot.bot.is_closed = MagicMock(return_value=True)

            bot.stop()

            mock_task.cancel.assert_called_once()
            assert len(bot._typing_tasks) == 0


@pytest.mark.asyncio
async def test_handle_message_tool_display_formatting(mock_config: KesokuConfig, mock_gateway: MagicMock) -> None:
    """Test refined tool display formatting with arguments in DiscordChatbot."""
    from kesoku.constants import ROLE_TOOL, TYPE_TOOL_CALL

    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=mock_config):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)
            mock_channel = AsyncMock(spec=discord.Thread)
            bot.bot.get_channel = MagicMock(return_value=mock_channel)

            # Case 1: Tool call with zero arguments
            msg_no_args = Message(
                id="msg1",
                session_id="thread_case1",
                chatbot_id="discord_test",
                channel_id="12345",
                sender="Kesoku",
                role=ROLE_TOOL,
                type=TYPE_TOOL_CALL,
                content="Calling tool...",
                metadata={"tool_name": "my_tool", "tool_arguments": {}},
            )
            await bot.handle_message(msg_no_args)
            mock_channel.send.assert_any_call("🛠️ **my_tool** ⏳")

            # Case 2: Tool call with exactly one argument
            msg_one_arg = Message(
                id="msg2",
                session_id="thread_case2",
                chatbot_id="discord_test",
                channel_id="12345",
                sender="Kesoku",
                role=ROLE_TOOL,
                type=TYPE_TOOL_CALL,
                content="Calling tool...",
                metadata={"tool_name": "my_tool", "tool_arguments": {"query": "hello world"}},
            )
            await bot.handle_message(msg_one_arg)
            mock_channel.send.assert_any_call("🛠️ **my_tool**: `hello world` ⏳")

            # Case 3: Tool call with exactly one argument and context
            msg_context_arg = Message(
                id="msg3",
                session_id="thread_case3",
                chatbot_id="discord_test",
                channel_id="12345",
                sender="Kesoku",
                role=ROLE_TOOL,
                type=TYPE_TOOL_CALL,
                content="Calling tool...",
                metadata={"tool_name": "my_tool", "tool_arguments": {"query": "hello", "context": "ignored"}},
            )
            await bot.handle_message(msg_context_arg)
            mock_channel.send.assert_any_call("🛠️ **my_tool**: `hello` ⏳")

            # Case 4: Tool call with long single argument (truncation)
            long_arg = "A" * 100
            msg_long_arg = Message(
                id="msg4",
                session_id="thread_case4",
                chatbot_id="discord_test",
                channel_id="12345",
                sender="Kesoku",
                role=ROLE_TOOL,
                type=TYPE_TOOL_CALL,
                content="Calling tool...",
                metadata={"tool_name": "my_tool", "tool_arguments": {"query": long_arg}},
            )
            await bot.handle_message(msg_long_arg)
            expected_long = "A" * 80 + "..."
            mock_channel.send.assert_any_call(f"🛠️ **my_tool**: `{expected_long}` ⏳")

            # Case 5: Tool call with multiple arguments
            msg_multiple_args = Message(
                id="msg5",
                session_id="thread_case5",
                chatbot_id="discord_test",
                channel_id="12345",
                sender="Kesoku",
                role=ROLE_TOOL,
                type=TYPE_TOOL_CALL,
                content="Calling tool...",
                metadata={"tool_name": "my_tool", "tool_arguments": {"arg1": "val1", "arg2": "val2"}},
            )
            await bot.handle_message(msg_multiple_args)
            mock_channel.send.assert_any_call("🛠️ **my_tool**: `arg1: val1, arg2: val2` ⏳")


@pytest.mark.asyncio
async def test_handle_message_tool_result_in_place_edit(mock_config: KesokuConfig, mock_gateway: MagicMock) -> None:
    """Test that a tool result edits the original tool call message in-place on Discord."""
    from kesoku.constants import ROLE_TOOL, TYPE_TOOL_RESULT

    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=mock_config):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)
            mock_channel = AsyncMock(spec=discord.Thread)
            bot.bot.get_channel = MagicMock(return_value=mock_channel)

            # Mock the sent tool call message in cache
            mock_discord_msg = AsyncMock(spec=discord.Message)
            mock_discord_msg.content = "🛠️ **my_tool**: `edit_query` ⏳"
            bot._sent_tool_calls["parent123"] = mock_discord_msg

            # Mock parent message retrieval (to verify it is NOT called)
            mock_gateway.db = MagicMock()
            mock_gateway.db.get_messages_by_filters = MagicMock()

            result_msg = Message(
                id="result123",
                session_id="thread123",
                chatbot_id="discord_test",
                channel_id="12345",
                sender="my_tool",
                role=ROLE_TOOL,
                type=TYPE_TOOL_RESULT,
                content="Result...",
                parent_id="parent123",
            )

            await bot.handle_message(result_msg)

            # Verify that edit was called with the formatted content on the discord message object
            mock_discord_msg.edit.assert_called_once_with(content="🛠️ **my_tool**: `edit_query` ✅")
            # The cache should be cleaned up
            assert "parent123" not in bot._sent_tool_calls
            # Verify DB parent lookup was bypassed
            mock_gateway.db.get_messages_by_filters.assert_not_called()


@pytest.mark.asyncio
async def test_handle_message_with_voice_success(mock_config: KesokuConfig, mock_gateway: MagicMock) -> None:
    """Test that a voice block successfully sends a native voice message via Discord API."""
    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=mock_config):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)
            mock_channel = AsyncMock(spec=discord.Thread)
            bot.bot.get_channel = MagicMock(return_value=mock_channel)

            content = "Listen here: [voice: /tmp/voice.ogg]"
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

            with patch("os.path.exists", return_value=True) as mock_exists:
                with patch("kesoku.gateway.chatbot.discord.send_voice_message") as mock_send_voice:
                    await bot.handle_message(msg)

                    mock_exists.assert_called_once_with("/tmp/voice.ogg")
                    mock_send_voice.assert_called_once_with(mock_channel, "/tmp/voice.ogg")
                    mock_channel.send.assert_called_once_with("Listen here: ")
                    mock_gateway.update_message_status.assert_called_once_with("msg123", STATUS_DELIVERED)


@pytest.mark.asyncio
async def test_handle_message_with_voice_fallback(mock_config: KesokuConfig, mock_gateway: MagicMock) -> None:
    """Test that a voice block falls back to standard file attachment if native sending fails."""
    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=mock_config):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)
            mock_channel = AsyncMock(spec=discord.Thread)
            bot.bot.get_channel = MagicMock(return_value=mock_channel)

            content = "Listen here: [voice: /tmp/voice.ogg]"
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
                with patch("kesoku.gateway.chatbot.discord.send_voice_message", side_effect=Exception("API Error")):
                    with patch("discord.File", return_value=mock_file) as mock_file_class:
                        await bot.handle_message(msg)

                        mock_exists.assert_called_once_with("/tmp/voice.ogg")
                        mock_channel.send.assert_any_call("Listen here: ")
                        mock_channel.send.assert_any_call(file=mock_file)
                        mock_gateway.update_message_status.assert_called_once_with("msg123", STATUS_DELIVERED)


@pytest.mark.asyncio
async def test_on_message_no_auto_thread_by_channel_id(mock_gateway: MagicMock) -> None:
    """Test that if incoming message channel ID matches a channel override with auto_thread=False,
    no thread is created.
    """
    cfg = KesokuConfig()
    cfg.discord = DiscordConfig(
        enabled=True,
        bot_token="test_token",
        chatbot_id="discord_test",
        channels=[
            DiscordChannelOverride(channels=["999888"], auto_thread=False)
        ],
    )
    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=cfg):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)

            mock_text_channel = MagicMock(spec=discord.TextChannel)
            mock_text_channel.id = 999888
            mock_text_channel.name = "no-thread-channel"
            mock_text_channel.typing.return_value = MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock())

            msg = MagicMock(spec=discord.Message)
            msg.author = MagicMock(spec=discord.Member, id=222, display_name="Allowed")
            msg.author.name = "allowed_user"
            msg.mentions = []
            msg.channel = mock_text_channel
            msg.content = "Hello direct channel"
            msg.id = 888
            msg.created_at = datetime.datetime.now(datetime.UTC)

            # Mock message's thread creation just in case (to verify it is NOT called)
            msg.create_thread = AsyncMock()

            await bot.on_message(msg)

            # Verify no thread was created
            msg.create_thread.assert_not_called()

            # Verify post was called with TextChannel's ID ("999888")
            mock_gateway.post.assert_called_once()
            posted_msg = mock_gateway.post.call_args[0][0]
            assert posted_msg.channel_id == "999888"


@pytest.mark.asyncio
async def test_on_message_no_auto_thread_by_channel_name(mock_gateway: MagicMock) -> None:
    """Test that if incoming message channel name matches a channel override with auto_thread=False,
    no thread is created.
    """
    cfg = KesokuConfig()
    cfg.discord = DiscordConfig(
        enabled=True,
        bot_token="test_token",
        chatbot_id="discord_test",
        channels=[
            DiscordChannelOverride(channels=["no-thread-channel"], auto_thread=False)
        ],
    )
    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=cfg):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)

            mock_text_channel = MagicMock(spec=discord.TextChannel)
            mock_text_channel.id = 123456
            mock_text_channel.name = "no-thread-channel"
            mock_text_channel.typing.return_value = MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock())

            msg = MagicMock(spec=discord.Message)
            msg.author = MagicMock(spec=discord.Member, id=222, display_name="Allowed")
            msg.author.name = "allowed_user"
            msg.mentions = []
            msg.channel = mock_text_channel
            msg.content = "Hello direct channel"
            msg.id = 888
            msg.created_at = datetime.datetime.now(datetime.UTC)

            msg.create_thread = AsyncMock()

            await bot.on_message(msg)

            # Verify no thread was created
            msg.create_thread.assert_not_called()

            # Verify post was called with TextChannel's ID ("123456")
            mock_gateway.post.assert_called_once()
            posted_msg = mock_gateway.post.call_args[0][0]
            assert posted_msg.channel_id == "123456"


@pytest.mark.asyncio
async def test_on_message_in_existing_thread_inside_no_thread_channel(mock_gateway: MagicMock) -> None:
    """Test that if incoming message is already inside a Thread, it uses the Thread,
    even if the parent channel matches a no-thread override.
    """
    cfg = KesokuConfig()
    cfg.discord = DiscordConfig(
        enabled=True,
        bot_token="test_token",
        chatbot_id="discord_test",
        channels=[
            DiscordChannelOverride(channels=["no-thread-channel"], auto_thread=False)
        ],
    )
    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=cfg):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)

            mock_parent_channel = MagicMock(spec=discord.TextChannel)
            mock_parent_channel.id = 123456
            mock_parent_channel.name = "no-thread-channel"

            mock_thread = AsyncMock(spec=discord.Thread)
            mock_thread.id = 777777
            mock_thread.name = "some-existing-thread"
            mock_thread.parent = mock_parent_channel
            mock_thread.guild = MagicMock(spec=discord.Guild)
            mock_thread.guild.name = "GuildName"
            mock_thread.guild.members = []
            mock_thread.join = AsyncMock()
            mock_thread.typing.return_value = MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock())

            msg = MagicMock(spec=discord.Message)
            msg.author = MagicMock(spec=discord.Member, id=222, display_name="Allowed")
            msg.author.name = "allowed_user"
            msg.mentions = []
            msg.channel = mock_thread  # message is in the thread!
            msg.content = "Hello inside thread"
            msg.id = 888
            msg.created_at = datetime.datetime.now(datetime.UTC)

            await bot.on_message(msg)

            # Verify post was called with the thread's ID ("777777")
            mock_gateway.post.assert_called_once()
            posted_msg = mock_gateway.post.call_args[0][0]
            assert posted_msg.channel_id == "777777"


@pytest.mark.asyncio
async def test_handle_message_intermediate_special_messages_tracking(
    mock_config: KesokuConfig, mock_gateway: MagicMock
) -> None:
    """Test that intermediate special messages (thoughts, tool calls, system) are tracked."""
    from kesoku.constants import ROLE_ASSISTANT, ROLE_SYSTEM, ROLE_TOOL, TYPE_THOUGHT, TYPE_TOOL_CALL

    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=mock_config):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)
            mock_channel = AsyncMock(spec=discord.Thread)
            bot.bot.get_channel = MagicMock(return_value=mock_channel)

            # Mock send to return mock message objects
            header_msg = AsyncMock(spec=discord.Message)
            msg1 = AsyncMock(spec=discord.Message)
            msg2 = AsyncMock(spec=discord.Message)
            msg3 = AsyncMock(spec=discord.Message)
            mock_channel.send.side_effect = [header_msg, msg1, msg2, msg3]

            # 1. Thought message (special)
            thought_msg = Message(
                id="t1",
                session_id="thread123",
                chatbot_id="discord_test",
                channel_id="12345",
                sender="Kesoku",
                role=ROLE_ASSISTANT,
                type=TYPE_THOUGHT,
                content="Thinking hard...",
            )
            await bot.handle_message(thought_msg)
            assert "12345" in bot._intermediate_messages
            assert len(bot._intermediate_messages["12345"]) == 1
            assert bot._intermediate_messages["12345"][0] == msg1

            # 2. Tool call message (special)
            tool_call = Message(
                id="tc1",
                session_id="thread123",
                chatbot_id="discord_test",
                channel_id="12345",
                sender="Kesoku",
                role=ROLE_TOOL,
                type=TYPE_TOOL_CALL,
                content="Calling tool...",
                metadata={"tool_name": "my_tool", "tool_arguments": {}},
            )
            await bot.handle_message(tool_call)
            # Grouped in the single special message via in-place edit
            assert len(bot._intermediate_messages["12345"]) == 1

            # 3. System message (special)
            sys_msg = Message(
                id="s1",
                session_id="thread123",
                chatbot_id="discord_test",
                channel_id="12345",
                sender="System",
                role=ROLE_SYSTEM,
                type=TYPE_TEXT,
                content="System event...",
            )
            await bot.handle_message(sys_msg)
            # Grouped in the single special message via in-place edit
            assert len(bot._intermediate_messages["12345"]) == 1


@pytest.mark.asyncio
async def test_handle_message_intermediate_special_messages_deletion(
    mock_config: KesokuConfig, mock_gateway: MagicMock
) -> None:
    """Test that intermediate special messages are deleted when final assistant response is reached."""
    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=mock_config):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)
            mock_channel = AsyncMock(spec=discord.Thread)
            bot.bot.get_channel = MagicMock(return_value=mock_channel)

            # Setup mocked intermediate messages
            msg1 = AsyncMock(spec=discord.Message)
            msg2 = AsyncMock(spec=discord.Message)
            bot._intermediate_messages["12345"] = [msg1, msg2]

            # Send final assistant response
            final_msg = Message(
                id="final1",
                session_id="thread123",
                chatbot_id="discord_test",
                channel_id="12345",
                sender="Kesoku",
                role=ROLE_ASSISTANT,
                type=TYPE_TEXT,
                content="Here is the final answer.",
            )
            await bot.handle_message(final_msg)

            # Verify both intermediate messages were deleted
            msg1.delete.assert_called_once()
            msg2.delete.assert_called_once()

            # Intermediate messages list should be cleared/removed for the channel
            assert "12345" not in bot._intermediate_messages


@pytest.mark.asyncio
async def test_trigger_cronjob_auto_thread(mock_gateway: MagicMock) -> None:
    """Test trigger_cronjob automatically creates a thread in an auto-thread channel."""
    cfg = KesokuConfig()
    cfg.discord = DiscordConfig(
        enabled=True,
        bot_token="test_token",
        chatbot_id="discord_test",
        channels=[
            DiscordChannelOverride(channels=["no-thread-channel"], auto_thread=False)
        ],
    )
    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=cfg):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)

            # Set bot is_ready to True
            bot.bot.is_ready = MagicMock(return_value=True)

            # Mock the text channel
            mock_text_channel = AsyncMock(spec=discord.TextChannel)
            mock_text_channel.id = 11111
            mock_text_channel.name = "auto-thread-channel"

            # Mock the starter message
            mock_starter_msg = AsyncMock(spec=discord.Message)
            mock_text_channel.send.return_value = mock_starter_msg

            # Mock the created thread
            mock_thread = AsyncMock(spec=discord.Thread)
            mock_thread.id = 22222
            mock_thread.name = "thread-name"
            mock_thread.join = AsyncMock()
            mock_starter_msg.create_thread.return_value = mock_thread

            # Setup fetch channel and typing mocks
            bot.bot.get_channel = MagicMock(return_value=mock_text_channel)
            bot._typing_tasks = {}

            # Run trigger_cronjob
            await bot.trigger_cronjob(
                channel_id="11111",
                prompt_content="Run scheduled prompt",
                mention_user_id="55555",
            )

            # Verify starter message was sent with user mention
            mock_text_channel.send.assert_called_once_with("<@55555> Scheduled job initiated.")
            # Verify thread was created on that starter message
            mock_starter_msg.create_thread.assert_called_once()
            # Verify thread was joined
            mock_thread.join.assert_called_once()

            # Verify Gateway post was called with thread's ID ("22222")
            mock_gateway.post.assert_called_once()
            posted_msg = mock_gateway.post.call_args[0][0]
            assert posted_msg.channel_id == "22222"
            assert "Run scheduled prompt" in posted_msg.content
            assert posted_msg.metadata.get("is_cronjob") is True


@pytest.mark.asyncio
async def test_handle_message_removes_stop_button_on_final_response(
    mock_config: KesokuConfig, mock_gateway: MagicMock
) -> None:
    """Test that the stop button is removed from the header view when final response is delivered."""
    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=mock_config):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)
            mock_channel = AsyncMock(spec=discord.Thread)
            bot.bot.get_channel = MagicMock(return_value=mock_channel)

            # Setup mocked header message and view
            mock_header_msg = AsyncMock(spec=discord.Message)
            mock_header_view = MagicMock()
            mock_header_view.stop_turn = MagicMock()
            bot._header_views["thread123"] = (mock_header_msg, mock_header_view)

            final_msg = Message(
                id="msg123",
                session_id="thread123",
                chatbot_id="discord_test",
                channel_id="12345",
                sender="Kesoku",
                role=ROLE_ASSISTANT,
                type=TYPE_TEXT,
                content="Final reply",
            )

            await bot.handle_message(final_msg)

            # Asserts:
            # remove_item should be called on the header view with the stop_turn button
            mock_header_view.remove_item.assert_called_once_with(mock_header_view.stop_turn)
            # The header message should be edited to update the view
            mock_header_msg.edit.assert_called_once_with(view=mock_header_view)
            mock_gateway.update_message_status.assert_called_once_with("msg123", STATUS_DELIVERED)


@pytest.mark.asyncio
async def test_handle_message_with_question(mock_config: KesokuConfig, mock_gateway: MagicMock) -> None:
    """Test that a question block in the message triggers sending a QuestionView to the channel."""
    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=mock_config):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)
            mock_channel = AsyncMock(spec=discord.Thread)
            bot.bot.get_channel = MagicMock(return_value=mock_channel)

            content = "[question: Choose? || Yes | No]"
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

            with patch("kesoku.gateway.chatbot.discord.QuestionView") as mock_question_view_class:
                mock_view = MagicMock()
                mock_question_view_class.return_value = mock_view

                with patch("discord.Embed") as mock_embed_class:
                    mock_embed = MagicMock()
                    mock_embed_class.return_value = mock_embed

                    await bot.handle_message(msg)

                    mock_question_view_class.assert_called_once_with(
                        gateway=mock_gateway,
                        session_id="thread123",
                        chatbot=bot,
                        question="Choose?",
                        choices=["Yes", "No"],
                    )
                    mock_embed_class.assert_called_once_with(
                        title="❓ Choose?",
                        color=discord.Color.blurple(),
                    )
                    mock_channel.send.assert_called_once_with(embed=mock_embed, view=mock_view)
                    mock_gateway.update_message_status.assert_called_once_with("msg123", STATUS_DELIVERED)


@pytest.mark.asyncio
async def test_channel_override_auto_thread_matching(mock_gateway: MagicMock) -> None:
    """Test that various matching criteria (parent ID, direct name, etc.) work for Discord channel overrides."""
    cfg = KesokuConfig()
    cfg.discord = DiscordConfig(
        enabled=True,
        bot_token="test_token",
        chatbot_id="discord_test",
        channels=[
            DiscordChannelOverride(channels=["parent-channel-id"], auto_thread=False)
        ],
    )
    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=cfg):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)

            override = bot._resolve_channel_override("222", "some-thread", "parent-channel-id", "general")
            assert override is not None
            assert override.auto_thread is False


@pytest.mark.asyncio
async def test_on_message_with_attachments(mock_config: KesokuConfig, mock_gateway: MagicMock) -> None:
    """Test that incoming Discord attachments are saved and tracked in metadata."""
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
            allowed_author = MagicMock(spec=discord.Member, id=222, display_name="Allowed")
            allowed_author.name = "allowed_user"
            msg.author = allowed_author
            msg.mentions = []
            msg.channel = mock_thread
            msg.content = "Look at this image"
            msg.id = 888
            msg.created_at = datetime.datetime.now(datetime.UTC)

            # Create mock attachment
            mock_attachment = AsyncMock(spec=discord.Attachment)
            mock_attachment.id = 123456
            mock_attachment.filename = "test_image.png"
            mock_attachment.content_type = "image/png"
            mock_attachment.save = AsyncMock()
            msg.attachments = [mock_attachment]

            with patch("os.makedirs") as mock_makedirs, patch("os.path.exists", return_value=False):
                await bot.on_message(msg)

                # Verify directory creation was triggered
                mock_makedirs.assert_called_once()
                # Verify attachment save was called
                mock_attachment.save.assert_called_once()

                # Verify gateway post was called
                mock_gateway.post.assert_called_once()
                posted_msg = mock_gateway.post.call_args[0][0]

                # Verify metadata has attachments details
                assert "attachments" in posted_msg.metadata
                attachments = posted_msg.metadata["attachments"]
                assert len(attachments) == 1
                assert attachments[0]["filename"] == "test_image.png"
                assert attachments[0]["mime_type"] == "image/png"
                assert "test_image.png" in attachments[0]["path"]

                # Verify message content has references
                assert "[Attachment: test_image.png (image/png)" in posted_msg.content
