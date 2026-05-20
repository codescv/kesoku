"""Discord slash commands for Kesoku AI Agent chatbot."""

import asyncio
import os
import sys
from typing import TYPE_CHECKING

import discord
from discord import app_commands

from kesoku.logger import setup_logger

if TYPE_CHECKING:
    from kesoku.gateway.chatbot.discord import DiscordChatbot

logger = setup_logger(__name__)


def _get_kesoku_executable() -> str:
    """Retrieve the absolute path or resolved command for the 'kesoku' executable."""
    import shutil
    executable_dir = os.path.dirname(sys.executable)
    kesoku_path = os.path.join(executable_dir, "kesoku")
    if os.path.exists(kesoku_path):
        return kesoku_path
    return shutil.which("kesoku") or "kesoku"


def setup_discord_commands(chatbot: "DiscordChatbot") -> None:
    """Set up Discord application (slash) commands on the chatbot's client.

    Args:
        chatbot: The DiscordChatbot instance.
    """
    # Manually create the CommandTree if not present on the chatbot
    if not hasattr(chatbot, "tree"):
        chatbot.tree = app_commands.CommandTree(chatbot.bot)

    @chatbot.tree.command(name="restart", description="Restart the Kesoku service.")
    async def restart_command(interaction: discord.Interaction) -> None:
        """Slash command handler to restart the entire Kesoku service.

        Args:
            interaction: The incoming Discord interaction.
        """
        logger.info(
            f"Received /restart slash command from user {interaction.user.name} "
            f"(ID: {interaction.user.id}) in channel {interaction.channel_id}"
        )

        # Inform the user that the service is restarting
        await interaction.response.send_message("🔄 Restarting service...")

        try:
            # Allow a small delay for the message to be fully sent
            await asyncio.sleep(0.5)

            # Stop the chatbot cleanly to release resources
            chatbot.stop()

            # Resolve kesoku binary path
            kesoku_bin = _get_kesoku_executable()

            # Launch the restart command in a new session to decouple from parent group termination
            import subprocess
            subprocess.Popen([  # noqa: ASYNC220
                kesoku_bin,
                "service",
                "restart",
            ], start_new_session=True)
            logger.info("Successfully launched kesoku service restart command.")
        except Exception as e:
            logger.error(f"Failed to run restart command: {e}")
            try:
                # Fallback to in-place os.execv restart
                logger.info("Falling back to in-place os.execv restart...")
                sys.stdout.flush()
                sys.stderr.flush()
                os.execv(sys.executable, [sys.executable] + sys.argv)
            except Exception as fallback_error:
                logger.error(f"In-place fallback restart failed: {fallback_error}", exc_info=True)
                await interaction.followup.send(f"Failed to restart service: {e}", ephemeral=True)
