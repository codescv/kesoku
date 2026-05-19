"""Unit tests for Kesoku Discord chatbot adapter."""

import asyncio
import datetime
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import discord
import pytest

from kesoku.config import DiscordConfig, KesokuConfig
from kesoku.constants import ROLE_ASSISTANT, STATUS_DELIVERED, TYPE_TEXT
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
                session_id="thread123",
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
                session_id="thread123",
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
                session_id="thread123",
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
                session_id="thread123",
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
                session_id="thread123",
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
async def test_handle_message_tool_result_displays_parent_arguments(
    mock_config: KesokuConfig, mock_gateway: MagicMock
) -> None:
    """Test tool result formatting retrieves and displays parent arguments using message.parent_id."""
    from kesoku.constants import ROLE_TOOL, TYPE_TOOL_CALL, TYPE_TOOL_RESULT

    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=mock_config):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)
            mock_channel = AsyncMock(spec=discord.Thread)
            bot.bot.get_channel = MagicMock(return_value=mock_channel)

            # Mock parent message to return when get_messages_by_filters is called
            parent_msg = Message(
                id="parent123",
                session_id="thread123",
                chatbot_id="discord_test",
                channel_id="12345",
                sender="Kesoku",
                role=ROLE_TOOL,
                type=TYPE_TOOL_CALL,
                content="Calling tool...",
                metadata={"tool_name": "my_tool", "tool_arguments": {"query": "parent_query"}},
            )
            mock_gateway.db = MagicMock()
            mock_gateway.db.get_messages_by_filters = MagicMock(return_value=[parent_msg])

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

            # Test successful tool result display formatting
            await bot.handle_message(result_msg)
            mock_channel.send.assert_any_call("📥 **my_tool**: `parent_query` ✅")

            # Test failed tool result display formatting
            result_msg.metadata["tool_error"] = "some_error"
            await bot.handle_message(result_msg)
            mock_channel.send.assert_any_call("📥 **my_tool**: `parent_query` ❌")


@pytest.mark.asyncio
async def test_handle_message_tool_result_in_place_edit(mock_config: KesokuConfig, mock_gateway: MagicMock) -> None:
    """Test that a tool result edits the original tool call message in-place on Discord."""
    from kesoku.constants import ROLE_TOOL, TYPE_TOOL_CALL, TYPE_TOOL_RESULT

    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=mock_config):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)
            mock_channel = AsyncMock(spec=discord.Thread)
            bot.bot.get_channel = MagicMock(return_value=mock_channel)

            # Mock the sent tool call message in cache
            mock_discord_msg = AsyncMock(spec=discord.Message)
            bot._sent_tool_calls["parent123"] = mock_discord_msg

            # Mock parent message retrieval
            parent_msg = Message(
                id="parent123",
                session_id="thread123",
                chatbot_id="discord_test",
                channel_id="12345",
                sender="Kesoku",
                role=ROLE_TOOL,
                type=TYPE_TOOL_CALL,
                content="Calling...",
                metadata={"tool_name": "my_tool", "tool_arguments": {"query": "edit_query"}},
            )
            mock_gateway.db = MagicMock()
            mock_gateway.db.get_messages_by_filters = MagicMock(return_value=[parent_msg])

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
            mock_discord_msg.edit.assert_called_once_with(content="📥 **my_tool**: `edit_query` ✅")
            # The cache should be cleaned up
            assert "parent123" not in bot._sent_tool_calls


@pytest.mark.asyncio
async def test_handle_message_with_voice_success(mock_config: KesokuConfig, mock_gateway: MagicMock) -> None:
    """Test that a voice block successfully sends a native voice message via Discord API."""
    with patch("kesoku.gateway.chatbot.discord.get_config", return_value=mock_config):
        mock_client_user = MagicMock(spec=discord.ClientUser, id=999)
        with patch.object(discord.Client, "user", new_callable=PropertyMock, return_value=mock_client_user):
            bot = DiscordChatbot(chatbot_id="discord_test", gateway=mock_gateway)
            mock_channel = AsyncMock(spec=discord.Thread)
            bot.bot.get_channel = MagicMock(return_value=mock_channel)

            # Mock internal channel state/http to allow low-level call to succeed
            mock_channel._state = MagicMock()
            mock_channel._state.http = AsyncMock()
            mock_channel._state.create_message = MagicMock()

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

            original_file_class = discord.File
            with patch("os.path.exists", return_value=True) as mock_exists:
                with patch("os.unlink") as mock_unlink:
                    with patch("subprocess.run") as mock_run:
                        with patch("discord.File") as mock_file_class:
                            # Configure mock file instance to serialize properly
                            mock_file_instance = MagicMock(spec=original_file_class)
                            mock_file_instance.to_dict.return_value = {"id": 0, "filename": "voice.ogg"}
                            mock_file_class.return_value = mock_file_instance

                            # Configure mock run to return a valid duration for ffprobe call
                            mock_proc = MagicMock()
                            mock_proc.stdout = "3.5\n"
                            mock_run.return_value = mock_proc

                            await bot.handle_message(msg)

                            mock_exists.assert_any_call("/tmp/voice.ogg")
                            assert mock_run.call_count == 2
                            mock_unlink.assert_called_once()
                            # Verifies channel.send is called for text before
                            mock_channel.send.assert_any_call("Listen here: ")
                            # Verifies low-level HTTP send_message was called for the voice attachment
                            mock_channel._state.http.send_message.assert_called_once()

                            # Verify message status was updated to delivered
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

            # Force native voice message sending to raise an Exception
            bot._send_voice_message = AsyncMock(side_effect=Exception("API Error"))

            mock_file = MagicMock(spec=discord.File)
            with patch("os.path.exists", return_value=True) as mock_exists:
                with patch("discord.File", return_value=mock_file) as mock_file_class:
                    await bot.handle_message(msg)

                    mock_exists.assert_called_once_with("/tmp/voice.ogg")
                    # Should have fallen back to sending standard file attachment
                    mock_channel.send.assert_any_call("Listen here: ")
                    mock_channel.send.assert_any_call(file=mock_file)

                    # Verify message status was updated to delivered
                    mock_gateway.update_message_status.assert_called_once_with("msg123", STATUS_DELIVERED)


def test_voice_file_to_dict() -> None:
    """Test that VoiceFile correctly serializes voice message metadata."""
    import io

    from kesoku.gateway.chatbot.discord import VoiceFile

    fp = io.BytesIO(b"dummy audio content")
    voice_file = VoiceFile(
        fp,
        filename="voice.ogg",
        duration_secs=3.14,
        waveform="abcde12345",
    )

    serialized = voice_file.to_dict(0)
    assert serialized["id"] == 0
    assert serialized["filename"] == "voice.ogg"
    assert serialized["duration_secs"] == 3.14
    assert serialized["waveform"] == "abcde12345"


def test_generate_pseudo_waveform() -> None:
    """Test that pseudo waveform generator returns a valid base64 string of 256 bytes."""
    import base64

    from kesoku.gateway.chatbot.discord import _generate_pseudo_waveform

    encoded = _generate_pseudo_waveform()
    assert isinstance(encoded, str)
    # Decode it and ensure it is exactly 256 bytes
    decoded = base64.b64decode(encoded.encode("utf-8"))
    assert len(decoded) == 256
    # Verify all values are in the uint8 range (0-255)
    for val in decoded:
        assert 0 <= val <= 255


@pytest.mark.asyncio
async def test_get_audio_duration_success() -> None:
    """Test that _get_audio_duration successfully extracts float duration from ffprobe output."""
    from kesoku.gateway.chatbot.discord import _get_audio_duration

    mock_proc = MagicMock()
    mock_proc.stdout = "  45.67 \n"

    with patch("subprocess.run", return_value=mock_proc) as mock_run:
        duration = await _get_audio_duration("/tmp/test.ogg")
        assert duration == 45.67
        mock_run.assert_called_once()


@pytest.mark.asyncio
async def test_get_audio_duration_failure() -> None:
    """Test that _get_audio_duration gracefully returns 0.0 on subprocess error."""
    from kesoku.gateway.chatbot.discord import _get_audio_duration

    with patch("subprocess.run", side_effect=Exception("ffprobe failed")) as mock_run:
        duration = await _get_audio_duration("/tmp/test.ogg")
        assert duration == 0.0
        mock_run.assert_called_once()
