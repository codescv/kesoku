"""Discord slash commands for Kesoku AI Agent chatbot."""

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from kesoku.logger import setup_logger

if TYPE_CHECKING:
    from .adapter import DiscordChatbot

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
        if name in {"reset"}:
            continue

        cmd_name = name
        description = cmd_info["description"]

        # Define dynamic callback using a closure factory
        if cmd_name == "role":

            async def role_callback(interaction: discord.Interaction, role_name: str = "") -> None:
                logger.info(
                    f"Received /role slash command with role_name='{role_name}' from user {interaction.user.name} "
                    f"(ID: {interaction.user.id}) in channel {interaction.channel_id}"
                )
                await interaction.response.defer()

                async def reply_func(text: str) -> None:
                    await interaction.followup.send(text)

                try:
                    await chatbot.commands.execute(
                        "role",
                        reply_func,
                        channel_id=str(interaction.channel_id),
                        role_name=role_name,
                    )
                except Exception as e:
                    logger.error(f"Discord command /role execution failed: {e}")
                    await reply_func(f"⚠️ Failed to execute command: {e}")

            cmd = app_commands.Command(
                name="role",
                description=description,
                callback=role_callback,
            )
        elif cmd_name == "cronjob":

            async def cronjob_callback(interaction: discord.Interaction, tag: str = "") -> None:
                logger.info(
                    f"Received /cronjob slash command with tag='{tag}' from user {interaction.user.name} "
                    f"(ID: {interaction.user.id}) in channel {interaction.channel_id}"
                )
                await interaction.response.defer()

                async def reply_func(text: str) -> None:
                    from kesoku.utils.text import split_text_into_chunks
                    chunks = split_text_into_chunks(text, 2000)
                    for chunk in chunks:
                        if chunk.strip():
                            await interaction.followup.send(chunk)

                try:
                    await chatbot.commands.execute(
                        "cronjob",
                        reply_func,
                        tag=tag,
                    )
                except Exception as e:
                    logger.error(f"Discord command /cronjob execution failed: {e}")
                    await reply_func(f"⚠️ Failed to execute command: {e}")

            cmd = app_commands.Command(
                name="cronjob",
                description=description,
                callback=cronjob_callback,
            )
        elif cmd_name in {"grep", "memory-grep", "search", "memory-search"}:

            def make_search_callback(c_name: str) -> Callable[..., Awaitable[None]]:
                async def search_callback(interaction: discord.Interaction, query: str = "") -> None:
                    logger.info(
                        f"Received /{c_name} slash command with query='{query}' from user {interaction.user.name} "
                        f"(ID: {interaction.user.id}) in channel {interaction.channel_id}"
                    )
                    await interaction.response.defer()

                    async def reply_func(text: str) -> None:
                        from kesoku.utils.text import split_text_into_chunks

                        chunks = split_text_into_chunks(text, 2000)
                        for chunk in chunks:
                            if chunk.strip():
                                await interaction.followup.send(chunk)

                    try:
                        await chatbot.commands.execute(
                            c_name,
                            reply_func,
                            channel_id=str(interaction.channel_id),
                            query=query,
                        )
                    except Exception as e:
                        logger.error(f"Discord command /{c_name} execution failed: {e}")
                        await reply_func(f"⚠️ Failed to execute command: {e}")

                return search_callback

            cmd = app_commands.Command(
                name=cmd_name,
                description=description,
                callback=make_search_callback(cmd_name),
            )
        else:

            def make_callback(c_name: str) -> Callable[[discord.Interaction], Awaitable[None]]:
                async def callback(interaction: discord.Interaction) -> None:
                    logger.info(
                        f"Received /{c_name} slash command from user {interaction.user.name} "
                        f"(ID: {interaction.user.id}) in channel {interaction.channel_id}"
                    )
                    # Acknowledge the interaction first by deferring
                    await interaction.response.defer()

                    async def reply_func(text: str, file_path: str | None = None) -> None:
                        if file_path:
                            file = discord.File(file_path, filename="active_context.html")
                            await interaction.followup.send(content=text, file=file)
                        else:
                            from kesoku.utils.text import split_text_into_chunks
                            chunks = split_text_into_chunks(text, 2000)
                            for chunk in chunks:
                                if chunk.strip():
                                    await interaction.followup.send(chunk)

                    try:
                        if c_name in {
                            "clear",
                            "reset",
                            "status",
                            "compact",
                            "context",
                            "debug",
                        }:
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
