"""Unit tests for Kesoku Discord chatbot slash commands."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest
from discord import app_commands

from kesoku.gateway.chatbot.discord_command import _get_kesoku_executable, setup_discord_commands


@pytest.fixture
def mock_chatbot() -> MagicMock:
    """Provide a mock DiscordChatbot instance."""
    chatbot = MagicMock()
    chatbot.bot = MagicMock()
    chatbot.bot._connection._command_tree = None
    chatbot.stop = MagicMock()
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

    kesoku_bin = _get_kesoku_executable()

    # Mock subprocess.Popen and os.execv
    with (
        patch("subprocess.Popen") as mock_popen,
        patch("os.execv") as mock_execv,
        patch("asyncio.sleep", AsyncMock()) as mock_sleep,
    ):
        await restart_cmd.callback(interaction)

        # Assert first message was sent
        interaction.response.send_message.assert_called_once_with("🔄 Restarting service...")

        # Assert Popen was called with correct parameters
        mock_popen.assert_called_once_with(
            [kesoku_bin, "service", "restart"],
            start_new_session=True
        )

        # Assert chatbot stop was called cleanly
        mock_chatbot.stop.assert_called_once()

        # os.execv should NOT be called
        mock_execv.assert_not_called()


@pytest.mark.asyncio
async def test_restart_command_fallback_to_execv(mock_chatbot: MagicMock) -> None:
    """Test that if Popen raises an exception, /restart falls back to in-place execv."""
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

    # Mock subprocess.Popen to raise an OSError (e.g. command not found)
    with (
        patch("subprocess.Popen", side_effect=OSError("Command not found")) as mock_popen,
        patch("os.execv") as mock_execv,
        patch("sys.stdout.flush") as mock_stdout_flush,
        patch("sys.stderr.flush") as mock_stderr_flush,
        patch("asyncio.sleep", AsyncMock()) as mock_sleep,
    ):
        await restart_cmd.callback(interaction)

        # Assert first message was sent
        interaction.response.send_message.assert_called_once_with("🔄 Restarting service...")

        # Popen was called
        mock_popen.assert_called_once()

        # os.execv MUST be called as a fallback
        mock_execv.assert_called_once_with(sys.executable, [sys.executable] + sys.argv)
        assert mock_stdout_flush.call_count >= 1
        assert mock_stderr_flush.call_count >= 1
        mock_chatbot.stop.assert_called_once()


@pytest.mark.asyncio
async def test_restart_command_total_failure(mock_chatbot: MagicMock) -> None:
    """Test that if both Popen and execv fail, the error is reported cleanly via followup."""
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

    # Both Popen and execv raise exceptions
    with (
        patch("subprocess.Popen", side_effect=OSError("Command not found")),
        patch("os.execv", side_effect=OSError("Permission denied")),
        patch("sys.stdout.flush"),
        patch("sys.stderr.flush"),
        patch("asyncio.sleep", AsyncMock()) as mock_sleep,
    ):
        await restart_cmd.callback(interaction)

        # Verify initial message
        interaction.response.send_message.assert_called_once_with("🔄 Restarting service...")

        # Verify followup error message is sent
        interaction.followup.send.assert_called_once_with(
            "Failed to restart service: Command not found",
            ephemeral=True
        )
