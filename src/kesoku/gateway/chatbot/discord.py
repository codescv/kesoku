"""Discord chatbot adapter for Kesoku AI Agent framework.

Connects Discord channels and threads with Kesoku Gateway using Pub/Sub.
"""

import asyncio
import logging
import discord

from kesoku.config import get_config
from kesoku.constants import (
    DEFAULT_SYSTEM_PROMPT,
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL,
    ROLE_USER,
    STATUS_COMPLETED,
    STATUS_PENDING_AGENT,
    TYPE_TEXT,
    TYPE_THOUGHT,
    TYPE_TOOL_CALL,
)
from kesoku.db import Message
from kesoku.gateway.chatbot.base import Chatbot
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger

logger = setup_logger(__name__)


class DiscordChatbot(Chatbot):
    """Discord chatbot adapter connecting to Kesoku Gateway broker."""

    def __init__(self, chatbot_id: str, gateway: Gateway, bot_token: str | None = None) -> None:
        """Initialize the Discord chatbot adapter.

        Args:
            chatbot_id: Unique identifier for this chatbot instance.
            gateway: Kesoku Gateway instance.
            bot_token: Optional Discord bot token. Defaults to config setting.
        """
        super().__init__(chatbot_id, gateway)
        self.config = get_config()
        self.bot_token = bot_token or self.config.discord.bot_token
        if not self.bot_token:
            raise ValueError("Discord bot token is required but not configured.")

        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.guilds = True
        self.bot = discord.Client(intents=intents)
        self.bot.event(self.on_ready)
        self.bot.event(self.on_message)
        self._subscriber_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the Discord bot and Gateway listener subscriber background loop."""
        self._subscriber_task = asyncio.create_task(super().start())
        logger.info(f"Connecting Discord bot '{self.chatbot_id}'...")
        await self.bot.start(self.bot_token)

    def stop(self) -> None:
        """Stop the Discord bot and subscriber listener task."""
        if self._subscriber_task and not self._subscriber_task.done():
            self._subscriber_task.cancel()
        super().stop()
        if not self.bot.is_closed():
            asyncio.create_task(self.bot.close())

    async def on_ready(self) -> None:
        """Callback invoked when Discord bot successfully connects and logs in."""
        logger.info(f"Discord chatbot '{self.chatbot_id}' successfully logged in as {self.bot.user}")

    async def on_message(self, message: discord.Message) -> None:
        """Process incoming messages from Discord and ingest into Kesoku Gateway.

        Args:
            message: Incoming discord.Message instance.
        """
        # Ignore bot's own messages
        if message.author == self.bot.user:
            return

        # Check user_allowlist filtering
        allowlist = self.config.discord.user_allowlist
        if allowlist:
            is_allowed = (
                str(message.author.id) in allowlist
                or message.author.name in allowlist
                or message.author.display_name in allowlist
            )
            if not is_allowed:
                # Unlisted users only trigger replies if explicitly mentioning the bot
                if self.bot.user not in message.mentions:
                    return

        # Check if message explicitly mentions someone else but NOT this bot
        other_mentioned = [u for u in message.mentions if u != self.bot.user]
        if other_mentioned and self.bot.user not in message.mentions:
            return

        # Thread-based context separation
        thread: discord.Thread | None = None
        if isinstance(message.channel, discord.Thread):
            thread = message.channel
        else:
            # In regular channel; find or create a thread
            if hasattr(message.channel, "guild") and message.channel.guild:
                thread = message.channel.guild.get_thread(message.id)
            if not thread and hasattr(message, "thread") and message.thread:
                thread = message.thread

            if not thread:
                try:
                    title = message.content[:30] + ("..." if len(message.content) > 30 else "")
                    if not title.strip():
                        title = f"Chat with {message.author.display_name}"
                    thread = await message.create_thread(name=title)
                except discord.HTTPException as e:
                    logger.warning(
                        f"Failed to create thread on message {message.id} (concurrent bot creation?): {e}"
                    )
                    await asyncio.sleep(0.5)
                    if hasattr(message.channel, "guild") and message.channel.guild:
                        thread = message.channel.guild.get_thread(message.id)
                    if not thread and hasattr(message.channel, "archived_threads"):
                        async for t in message.channel.archived_threads(limit=10):
                            if t.id == message.id:
                                thread = t
                                break
            if not thread:
                logger.error(f"Could not create or retrieve thread for message {message.id}")
                return

            try:
                await thread.join()
            except Exception as je:
                logger.debug(f"Already joined or error joining thread {thread.id}: {je}")

        channel_id = str(thread.id)
        session = await self.gateway.get_session_by_channel(self.chatbot_id, channel_id)

        if not session:
            guild_name = thread.guild.name if hasattr(thread, "guild") and thread.guild else "Direct"
            member_lines = []
            if hasattr(thread, "guild") and thread.guild and hasattr(thread.guild, "members"):
                for m in thread.guild.members:
                    if not m.bot:
                        member_lines.append(f"- {m.display_name} (ID: {m.id})")
            if not member_lines:
                member_lines.append(f"- {message.author.display_name} (ID: {message.author.id})")
            members_str = "\n".join(member_lines)

            special_prompt = (
                f"\n\nYou are Kesoku, a helpful AI assistant interacting in Discord thread #{thread.name} on server '{guild_name}'.\n"
                f"Users present:\n{members_str}\n"
                "When mentioning or referring to a user, use Discord tag syntax <@USER_ID>."
            )
            sys_prompt = DEFAULT_SYSTEM_PROMPT + special_prompt
            session = await self.gateway.create_session(
                title=thread.name,
                system_prompt=sys_prompt,
                created_at=message.created_at.timestamp(),
            )
            session_id = session.id
        else:
            session_id = session.id
            await self.gateway.update_session_updated_at(session_id)

        # Ingest user message into Gateway
        msg = Message(
            session_id=session_id,
            chatbot_id=self.chatbot_id,
            channel_id=channel_id,
            sender=message.author.display_name,
            role=ROLE_USER,
            type=TYPE_TEXT,
            content=message.content,
            timestamp=message.created_at.timestamp(),
            status=STATUS_PENDING_AGENT,
            metadata={"discord_message_id": str(message.id), "discord_author_id": str(message.author.id)},
        )
        await self.gateway.post(msg)

    async def handle_message(self, message: Message) -> None:
        """Process outgoing message from Gateway and send to target Discord thread.

        Args:
            message: Outgoing Message instance.
        """
        try:
            target_id = int(message.channel_id)
        except ValueError:
            logger.error(f"Invalid Discord channel_id: {message.channel_id}")
            return

        channel = self.bot.get_channel(target_id)
        if not channel:
            try:
                channel = await self.bot.fetch_channel(target_id)  # type: ignore
            except Exception as fe:
                logger.error(f"Failed to fetch Discord channel {target_id}: {fe}")
                return

        output_text = ""
        if message.role == ROLE_ASSISTANT:
            if message.type == TYPE_THOUGHT:
                output_text = f"💭 Thought:\n{message.content}"
            else:
                output_text = message.content
        elif message.role == ROLE_TOOL:
            if message.type == TYPE_TOOL_CALL:
                output_text = f"🛠️ Tool Call ({message.sender}):\n```\n{message.content}\n```"
            else:
                output_text = f"📥 Tool Output ({message.sender}):\n```\n{message.content}\n```"
        elif message.role == ROLE_SYSTEM:
            output_text = f"⚙️ System:\n{message.content}"

        if not output_text:
            output_text = message.content

        # Newline Chunking (<= 2000 chars)
        lines = output_text.splitlines(keepends=True)
        current_chunk = ""
        for line in lines:
            if len(line) > 2000:
                if current_chunk:
                    await channel.send(current_chunk)
                    current_chunk = ""
                for i in range(0, len(line), 2000):
                    await channel.send(line[i : i + 2000])
            elif len(current_chunk) + len(line) > 2000:
                await channel.send(current_chunk)
                current_chunk = line
            else:
                current_chunk += line

        if current_chunk:
            await channel.send(current_chunk)

        await self.gateway.update_message_status(message.id, STATUS_COMPLETED)
