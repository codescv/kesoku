"""Unit tests for Kesoku Discord chatbot slash commands."""

import sys
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import discord
import pytest
from discord import app_commands

from kesoku.gateway.chatbot.discord.command import setup_discord_commands
from kesoku.utils.service import _get_kesoku_executable


@pytest.fixture
def mock_chatbot() -> MagicMock:
    """Provide a mock DiscordChatbot instance with real CommandRegistry and restart_service bound."""
    from kesoku.gateway.chatbot.base import Chatbot, CommandRegistry

    chatbot = MagicMock()
    chatbot.chatbot_id = "discord_test"
    chatbot.gateway = MagicMock()
    chatbot.bot = MagicMock()
    chatbot.bot._connection._command_tree = None
    chatbot.stop = MagicMock()

    # Bind real methods
    chatbot.restart_service = Chatbot.restart_service.__get__(chatbot, Chatbot)
    chatbot._register_default_commands = Chatbot._register_default_commands.__get__(chatbot, Chatbot)

    # Bind real CommandRegistry containing default commands
    registry = CommandRegistry()
    chatbot.commands = registry
    chatbot._register_default_commands()

    # Preserve test specific restart overwrite
    async def handle_restart(reply_func):
        await reply_func("🔄 Restarting service...")
        await chatbot.restart_service()

    registry.register("restart", "Restart the Kesoku service.", handle_restart)

    if hasattr(chatbot, "tree"):
        delattr(chatbot, "tree")
    return chatbot


def test_setup_discord_commands_tree_creation(mock_chatbot: MagicMock) -> None:
    """Test that setup_discord_commands creates the CommandTree and registers commands."""
    # Ensure tree is not initially present
    if hasattr(mock_chatbot, "tree"):
        delattr(mock_chatbot, "tree")

    setup_discord_commands(mock_chatbot)

    assert hasattr(mock_chatbot, "tree")
    assert isinstance(mock_chatbot.tree, app_commands.CommandTree)

    # Check that 'restart' command is registered
    commands = mock_chatbot.tree.get_commands()
    restart_cmd = next((cmd for cmd in commands if cmd.name == "restart"), None)
    assert restart_cmd is not None
    assert restart_cmd.description == "Restart the Kesoku service."


@pytest.mark.asyncio
async def test_restart_command_success(mock_chatbot: MagicMock) -> None:
    """Test that the /restart slash command executes Popen cleanly and triggers restart."""
    setup_discord_commands(mock_chatbot)

    commands = mock_chatbot.tree.get_commands()
    restart_cmd = next((cmd for cmd in commands if cmd.name == "restart"), None)
    assert restart_cmd is not None

    # Mock Interaction
    interaction = AsyncMock(spec=discord.Interaction)
    interaction.user = MagicMock(spec=discord.User)
    interaction.user.name = "test_user"
    interaction.user.id = 123456789
    interaction.channel_id = 987654321

    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.followup.send = AsyncMock()

    kesoku_bin = _get_kesoku_executable()

    # Case 1: Default behavior without specific service env variables (defaults to --user, no name)
    with (
        patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_create_subprocess_exec,
        patch("os.execv") as mock_execv,
        patch("asyncio.sleep", AsyncMock()),
        patch.dict("os.environ", {}, clear=True),
    ):
        await restart_cmd.callback(interaction)

        # Assert first message was sent via followup
        interaction.followup.send.assert_any_call("🔄 Restarting service...")

        # Assert create_subprocess_exec was called with correct parameters
        mock_create_subprocess_exec.assert_called_once_with(
            kesoku_bin, "service", "restart", "--user", start_new_session=True
        )

        # Assert chatbot stop was called cleanly
        mock_chatbot.stop.assert_called_once()
        mock_execv.assert_not_called()

    # Reset mocks
    interaction.followup.send.reset_mock()
    mock_chatbot.stop.reset_mock()

    # Case 2: Service is configured as a system service with an instance suffix name
    with (
        patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_create_subprocess_exec,
        patch("os.execv") as mock_execv,
        patch("asyncio.sleep", AsyncMock()),
        patch.dict(
            "os.environ", {"KESOKU_SERVICE_USER": "false", "KESOKU_SERVICE_INSTANCE_NAME": "custom-inst"}, clear=True
        ),
    ):
        await restart_cmd.callback(interaction)

        # Assert create_subprocess_exec was called with --system and --name custom-inst
        mock_create_subprocess_exec.assert_called_once_with(
            kesoku_bin, "service", "restart", "--system", "--name", "custom-inst", start_new_session=True
        )

        mock_chatbot.stop.assert_called_once()
        mock_execv.assert_not_called()


@pytest.mark.asyncio
async def test_restart_command_fallback_to_execv(mock_chatbot: MagicMock) -> None:
    """Test that if create_subprocess_exec raises an exception, /restart falls back to in-place execv."""
    setup_discord_commands(mock_chatbot)

    commands = mock_chatbot.tree.get_commands()
    restart_cmd = next((cmd for cmd in commands if cmd.name == "restart"), None)
    assert restart_cmd is not None

    # Mock Interaction
    interaction = AsyncMock(spec=discord.Interaction)
    interaction.user = MagicMock(spec=discord.User)
    interaction.user.name = "test_user"
    interaction.user.id = 123456789
    interaction.channel_id = 987654321

    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.followup.send = AsyncMock()

    # Mock asyncio.create_subprocess_exec to raise an OSError (e.g. command not found)
    with (
        patch(
            "asyncio.create_subprocess_exec", side_effect=OSError("Command not found")
        ) as mock_create_subprocess_exec,
        patch("os.execv") as mock_execv,
        patch("sys.stdout.flush") as mock_stdout_flush,
        patch("sys.stderr.flush") as mock_stderr_flush,
        patch("asyncio.sleep", AsyncMock()) as mock_sleep,
    ):
        await restart_cmd.callback(interaction)

        # Assert first message was sent via followup
        interaction.followup.send.assert_any_call("🔄 Restarting service...")

        # create_subprocess_exec was called
        mock_create_subprocess_exec.assert_called_once()

        # os.execv MUST be called as a fallback
        mock_execv.assert_called_once_with(sys.executable, [sys.executable] + sys.argv)
        assert mock_stdout_flush.call_count >= 1
        assert mock_stderr_flush.call_count >= 1
        mock_chatbot.stop.assert_called_once()


@pytest.mark.asyncio
async def test_restart_command_total_failure(mock_chatbot: MagicMock) -> None:
    """Test that if both create_subprocess_exec and execv fail, the error is reported cleanly via followup."""
    setup_discord_commands(mock_chatbot)

    commands = mock_chatbot.tree.get_commands()
    restart_cmd = next((cmd for cmd in commands if cmd.name == "restart"), None)
    assert restart_cmd is not None

    # Mock Interaction
    interaction = AsyncMock(spec=discord.Interaction)
    interaction.user = MagicMock(spec=discord.User)
    interaction.user.name = "test_user"
    interaction.user.id = 123456789
    interaction.channel_id = 987654321

    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.followup.send = AsyncMock()

    # Both create_subprocess_exec and execv raise exceptions
    with (
        patch("asyncio.create_subprocess_exec", side_effect=OSError("Command not found")),
        patch("os.execv", side_effect=OSError("Permission denied")),
        patch("sys.stdout.flush"),
        patch("sys.stderr.flush"),
        patch("asyncio.sleep", AsyncMock()) as mock_sleep,
    ):
        await restart_cmd.callback(interaction)

        # Verify initial message
        interaction.followup.send.assert_any_call("🔄 Restarting service...")

        # Verify followup error message is sent
        interaction.followup.send.assert_any_call("Failed to restart service: Command not found")


@pytest.mark.asyncio
async def test_lcm_command_success(mock_chatbot: MagicMock) -> None:
    """Test that the /lcm slash command executes successfully and replies with LCM context."""
    setup_discord_commands(mock_chatbot)

    commands = mock_chatbot.tree.get_commands()
    lcm_cmd = next((cmd for cmd in commands if cmd.name == "lcm"), None)
    assert lcm_cmd is not None

    # Mock interaction
    interaction = AsyncMock(spec=discord.Interaction)
    interaction.user = MagicMock(spec=discord.User)
    interaction.user.name = "test_user"
    interaction.user.id = 123456789
    interaction.channel_id = 987654321

    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.followup.send = AsyncMock()

    # Mock get_session_lcm_context_by_channel response
    mock_context_msg = (
        "### 📖 LCM Processed Context\n"
        "⚡ **DAG Summaries**: 2 nodes\n"
        "- **[User]**: hello\n"
        "- **[Assistant]**: hi"
    )
    mock_chatbot.get_session_lcm_context_by_channel = AsyncMock(return_value=mock_context_msg)

    await lcm_cmd.callback(interaction)

    # Assert context retrieval was called with correct channel ID
    mock_chatbot.get_session_lcm_context_by_channel.assert_called_once_with("987654321")

    # Assert the message was sent
    interaction.followup.send.assert_called_once_with(mock_context_msg)


@pytest.mark.asyncio
async def test_lcm_command_response_splitting(mock_chatbot: MagicMock) -> None:
    """Test that /lcm slash command splits response into chunks if it exceeds 2000 characters."""
    setup_discord_commands(mock_chatbot)

    commands = mock_chatbot.tree.get_commands()
    lcm_cmd = next((cmd for cmd in commands if cmd.name == "lcm"), None)
    assert lcm_cmd is not None

    # Mock interaction
    interaction = AsyncMock(spec=discord.Interaction)
    interaction.user = MagicMock(spec=discord.User)
    interaction.user.name = "test_user"
    interaction.user.id = 123456789
    interaction.channel_id = 987654321

    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.followup.send = AsyncMock()

    # Mock extremely long context response (3500 characters)
    long_msg = "A" * 3500
    mock_chatbot.get_session_lcm_context_by_channel = AsyncMock(return_value=long_msg)

    await lcm_cmd.callback(interaction)

    # Assert that send was called multiple times due to chunking
    assert interaction.followup.send.call_count > 1


@pytest.mark.asyncio
async def test_lcm_command_file_attachment(mock_chatbot: MagicMock) -> None:
    """Test that if get_session_lcm_context_by_channel returns a valid file path, /lcm sends it as an attachment."""
    setup_discord_commands(mock_chatbot)

    commands = mock_chatbot.tree.get_commands()
    lcm_cmd = next((cmd for cmd in commands if cmd.name == "lcm"), None)
    assert lcm_cmd is not None

    # Mock interaction
    interaction = AsyncMock(spec=discord.Interaction)
    interaction.user = MagicMock(spec=discord.User)
    interaction.user.name = "test_user"
    interaction.user.id = 123456789
    interaction.channel_id = 987654321

    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.followup.send = AsyncMock()

    # Create a temporary mock HTML file on disk to simulate the output
    import tempfile
    with tempfile.NamedTemporaryFile(suffix="_test.html", delete=False) as tmp:
        tmp.write(b"<html>Mock Context</html>")
        tmp_path = tmp.name

    try:
        mock_chatbot.get_session_lcm_context_by_channel = AsyncMock(return_value=tmp_path)

        await lcm_cmd.callback(interaction)

        # Assert the interaction followup was called with a File parameter
        args, kwargs = interaction.followup.send.call_args
        assert kwargs.get("file") is not None
        assert isinstance(kwargs.get("file"), discord.File)
        assert kwargs.get("file").filename == "lcm_context.html"
        assert kwargs.get("content") == "📖 Here is your beautifully formatted LCM Active Context HTML download:"
    finally:
        # Clean up
        import os
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_grep_command_success(mock_chatbot: MagicMock) -> None:
    """Test that the /grep slash command executes successfully and replies."""
    setup_discord_commands(mock_chatbot)

    commands = mock_chatbot.tree.get_commands()
    grep_cmd = next((cmd for cmd in commands if cmd.name == "grep"), None)
    assert grep_cmd is not None

    # Verify command has 'query' parameter
    assert len(grep_cmd.parameters) == 1
    assert grep_cmd.parameters[0].name == "query"

    # Mock interaction
    interaction = AsyncMock(spec=discord.Interaction)
    interaction.user = MagicMock(spec=discord.User)
    interaction.user.name = "test_user"
    interaction.user.id = 123456789
    interaction.channel_id = 987654321

    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.followup.send = AsyncMock()

    # Mock chatbot.commands.execute
    mock_chatbot.commands.execute = AsyncMock()

    await grep_cmd.callback(interaction, query="test_query")

    # Assert commands.execute was called with correct parameters
    mock_chatbot.commands.execute.assert_called_once_with(
        "grep",
        ANY,
        channel_id="987654321",
        query="test_query"
    )


