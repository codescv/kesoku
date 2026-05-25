"""Discord slash commands for Kesoku AI Agent chatbot."""

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import discord
from discord import app_commands

from kesoku.logger import setup_logger

if TYPE_CHECKING:
    from kesoku.gateway.chatbot.discord import DiscordChatbot

logger = setup_logger(__name__)



def setup_discord_commands(chatbot: "DiscordChatbot") -> None:
    """Set up Discord application (slash) commands on the chatbot's client based on registered command registry.

    Args:
        chatbot: The DiscordChatbot instance.
    """
    if not hasattr(chatbot, "tree"):
        chatbot.tree = app_commands.CommandTree(chatbot.bot)

    for name, cmd_info in chatbot.commands.get_commands().items():
        # Skip duplicate aliases to avoid Discord command registration collision
        if name == "reset":
            continue

        cmd_name = name
        description = cmd_info["description"]

        # Define dynamic callback using a closure factory
        def make_callback(c_name: str) -> Callable[[discord.Interaction], Awaitable[None]]:
            async def callback(interaction: discord.Interaction) -> None:
                logger.info(
                    f"Received /{c_name} slash command from user {interaction.user.name} "
                    f"(ID: {interaction.user.id}) in channel {interaction.channel_id}"
                )
                # Acknowledge the interaction first by deferring
                await interaction.response.defer()

                async def reply_func(text: str) -> None:
                    await interaction.followup.send(text)

                try:
                    if c_name in {"clear", "reset", "status"}:
                        await chatbot.commands.execute(c_name, reply_func, channel_id=str(interaction.channel_id))
                    else:
                        await chatbot.commands.execute(c_name, reply_func)
                except Exception as e:
                    logger.error(f"Discord command /{c_name} execution failed: {e}")
                    if c_name == "restart":
                        err_msg = str(e)
                        if "Command not found" in err_msg:
                            err_msg = "Command not found"
                        await reply_func(f"Failed to restart service: {err_msg}")
                    else:
                        await reply_func(f"⚠️ Failed to execute command: {e}")

            return callback

        cmd = app_commands.Command(
            name=cmd_name,
            description=description,
            callback=make_callback(cmd_name),
        )
        chatbot.tree.add_command(cmd)

