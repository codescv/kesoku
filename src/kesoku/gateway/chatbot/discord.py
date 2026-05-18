"""Discord chatbot adapter for Kesoku AI Agent framework.

Connects Discord channels and threads with Kesoku Gateway using Pub/Sub.
"""

import asyncio
import datetime
import logging
import os
import discord
import tzlocal

from kesoku.agent.prompt import build_sys_prompt
from kesoku.config import get_config
from kesoku.constants import (
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL,
    ROLE_USER,
    STATUS_DELIVERED,
    STATUS_PENDING_AGENT,
    TYPE_TEXT,
    TYPE_THOUGHT,
    TYPE_TOOL_CALL,
)
from kesoku.db import Message
from kesoku.gateway.chatbot.base import Chatbot, parse_message_content
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger


logger = setup_logger(__name__)


def _get_local_timezone_name() -> str:
    """Retrieve the local system timezone name (e.g., 'Asia/Shanghai')."""
    try:
        return tzlocal.get_localzone().key or "UTC"
    except Exception:
        return datetime.datetime.now().astimezone().tzname() or "UTC"


def _build_discord_sys_prompt(
    channel: discord.Thread | discord.DMChannel | discord.GroupChannel | discord.TextChannel,
    author: discord.User | discord.Member,
) -> str:
    """Build the default system prompt for a Discord thread session.

    Args:
        channel: The Discord channel or thread instance.
        author: The author of the initiating message.

    Returns:
        The built system prompt string.
    """
    is_dm = getattr(channel, "guild", None) is None

    # Build user member lines (only for group chats/servers, not DMs)
    members_section = ""
    if not is_dm:
        member_lines = []
        if hasattr(channel, "guild") and channel.guild and hasattr(channel.guild, "members"):
            for m in channel.guild.members:
                if not m.bot:
                    member_lines.append(f"- {m.display_name} (ID: {m.id})")
        if not member_lines:
            member_lines.append(f"- {author.display_name} (ID: {author.id})")
        members_str = "\n".join(member_lines)
        members_section = f"\n## Users on the server\n\n{members_str}\n"

    time_section = """\n## Time
The user message contains their user id and the **real** time that the message is sent.
The time is very important to prevent your hallucination about the world status.\n"""

    # Build location instruction and channel topic
    if is_dm:
        location_instruction = "You are talking to the user via discord."
        topic_section = ""
    else:
        guild_name = channel.guild.name if channel.guild else "Unknown Server"
        if isinstance(channel, discord.Thread):
            thread_name = channel.name
            thread_id = channel.id
            parent = channel.parent
            channel_name = parent.name if parent else "unknown-channel"
            channel_id = parent.id if parent else "unknown"
            topic = getattr(parent, "topic", None) or ""
            location_instruction = (
                f"You are currently chatting in a Discord thread named \"#{thread_name}\" (ID: {thread_id}) "
                f"under channel \"#{channel_name}\" (ID: {channel_id}) on the server '{guild_name}'."
            )
        else:
            channel_name = channel.name
            channel_id = channel.id
            topic = getattr(channel, "topic", None) or ""
            location_instruction = (
                f"You are currently chatting in a Discord channel named \"#{channel_name}\" (ID: {channel_id}) "
                f"on the server '{guild_name}'."
            )

        topic_section = f"## Channel Topic\n{topic}" if topic else ""

    mention_section = ""
    if not is_dm:
        mention_section = "\n## Mentioning Users\nWhen mentioning or referring to a user, use Discord tag syntax <@USER_ID>.\n"

    format_section = """
## Response Format
This format requirement only applies for your response in discord (not for writing files etc).
- Discord doesn't support latex syntax for math, so use plain text or emojis when you want to 
show math. e.g. use "exp(x)" instead of "$e^x$", use "∞" instead of "$\\inf$".
- Discord doesn't support level 4+ headings, so use level 3 headings at most (Start with level 1 heading).
    """

    discord_prompt = f"""
# Discord Instructions
{location_instruction}
{members_section}
{mention_section}
{time_section}
{topic_section}
{format_section}
    """
    return build_sys_prompt(discord_prompt)


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
            sys_prompt = _build_discord_sys_prompt(thread, message.author)
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
        tz_name = _get_local_timezone_name()
        discord_msg_content = (
            f"`{message.author.display_name}` <@{message.author.id}> "
            f"at `{message.created_at.astimezone().strftime('%Y-%m-%d %H:%M:%S')} {tz_name}`:\n"
            f"{message.content}"
        )
        msg = Message(
            session_id=session_id,
            chatbot_id=self.chatbot_id,
            channel_id=channel_id,
            sender=message.author.display_name,
            role=ROLE_USER,
            type=TYPE_TEXT,
            content=discord_msg_content,
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
                output_text = f"💭 {message.content}"
            else:
                output_text = message.content
        elif message.role == ROLE_TOOL:
            if message.type == TYPE_TOOL_CALL:
                output_text = f"🛠️ {message.content}"
            else:
                output_text = f"📥 {message.content}"
        elif message.role == ROLE_SYSTEM:
            output_text = f"⚙️ {message.content}"

        if not output_text:
            output_text = message.content

        # Parse output text into segments (handling [file: <path>] blocks)
        segments = parse_message_content(output_text)

        for segment in segments:
            if segment["type"] == "text":
                text_content = segment["content"]
                # Only send if the text chunk is not empty and contains non-whitespace characters
                if text_content.strip():
                    # Newline Chunking (<= 2000 chars) for Discord compatibility
                    lines = text_content.splitlines(keepends=True)
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

                    if current_chunk and current_chunk.strip():
                        await channel.send(current_chunk)
            elif segment["type"] == "file":
                file_path = segment["path"]
                if not os.path.exists(file_path):
                    logger.error(f"File not found: {file_path}")
                    await channel.send(f"⚠️ File not found: {file_path}")
                else:
                    try:
                        discord_file = discord.File(file_path)
                        await channel.send(file=discord_file)
                    except Exception as e:
                        logger.error(f"Failed to send file {file_path} to Discord: {e}", exc_info=True)
                        await channel.send(f"⚠️ Failed to send file {file_path}: {e}")

        await self.gateway.update_message_status(message.id, STATUS_DELIVERED)
