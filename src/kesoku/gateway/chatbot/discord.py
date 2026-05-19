"""Discord chatbot adapter for Kesoku AI Agent framework.

Connects Discord channels and threads with Kesoku Gateway using Pub/Sub.
"""

import asyncio
import datetime
import os
from collections import defaultdict

import discord
import tzlocal

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
from kesoku.gateway.chatbot.discord_ui import MessageHeaderView
from kesoku.gateway.chatbot.discord_voice_message import send_voice_message
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger

logger = setup_logger(__name__)


def _get_local_timezone_name() -> str:
    """Retrieve the local system timezone name (e.g., 'Asia/Shanghai')."""
    try:
        return tzlocal.get_localzone().key or "UTC"
    except Exception:
        return datetime.datetime.now().astimezone().tzname() or "UTC"


def _build_discord_custom_prompt(
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
                f'You are currently chatting in a Discord thread named "#{thread_name}" (ID: {thread_id}) '
                f"under channel \"#{channel_name}\" (ID: {channel_id}) on the server '{guild_name}'."
            )
        else:
            channel_name = channel.name
            channel_id = channel.id
            topic = getattr(channel, "topic", None) or ""
            location_instruction = (
                f'You are currently chatting in a Discord channel named "#{channel_name}" (ID: {channel_id}) '
                f"on the server '{guild_name}'."
            )

        topic_section = f"## Channel Topic\n{topic}" if topic else ""

    mention_section = ""
    if not is_dm:
        mention_section = (
            "\n## Mentioning Users\nWhen mentioning or referring to a user, use Discord tag syntax <@USER_ID>.\n"
        )

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
    return discord_prompt


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
        self.bot_token = bot_token or self.config.discord.bot_token or os.environ.get("DISCORD_TOKEN")
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
        self._sent_tool_calls: dict[str, discord.Message] = {}
        self._turns_with_header: set[str] = set()
        self._typing_tasks: dict[str, asyncio.Task[None]] = {}
        self._intermediate_messages: defaultdict[str, list[discord.Message]] = defaultdict(list)

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
        for task in self._typing_tasks.values():
            task.cancel()
        self._typing_tasks.clear()
        if not self.bot.is_closed():
            asyncio.create_task(self.bot.close())

    async def _keep_typing(
        self, channel: discord.Thread | discord.DMChannel | discord.GroupChannel | discord.TextChannel
    ) -> None:
        """Keep sending typing status to Discord channel/thread in a loop.

        Automatically times out after 10 minutes to prevent infinite typing.
        """
        try:
            async with channel.typing():
                await asyncio.sleep(600)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Failed to send typing status in channel {channel.id}: {e}")
        finally:
            self._typing_tasks.pop(str(channel.id), None)

    async def _get_tool_arguments_suffix(self, message: Message) -> str:
        """Format and retrieve the tool arguments suffix for Discord status display.

        Args:
            message: The tool call or tool result Message.

        Returns:
            Formatted suffix string (e.g., ': `arg_value`'), or empty string if none.
        """
        # Retrieve the tool arguments from metadata
        tool_args = message.metadata.get("tool_arguments")

        # Format tool arguments display string according to the rules
        arg_str = ""
        if isinstance(tool_args, dict):
            # Exclude framework/context arguments
            filtered_args = {k: v for k, v in tool_args.items() if k != "context"}
            if len(filtered_args) == 1:
                # Exactly one argument: show the argument value
                val = next(iter(filtered_args.values()))
                arg_str = str(val)
            elif len(filtered_args) > 1:
                # Multiple arguments: show comma-separated name-value pairs
                arg_str = ", ".join(f"{k}: {v}" for k, v in filtered_args.items())

        if arg_str:
            # Keep display clean by replacing newlines with spaces
            arg_str = arg_str.replace("\n", " ")
            # Truncate to maximum of 80 characters
            if len(arg_str) > 80:
                arg_str = arg_str[:80] + "..."

        return f": `{arg_str}`" if arg_str else ""

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
        target_channel: discord.Thread | discord.DMChannel | discord.GroupChannel | discord.TextChannel | None = None
        if isinstance(message.channel, discord.Thread):
            target_channel = message.channel
        else:
            # In regular channel; check if this channel is in no_auto_thread_channels
            no_thread_channels = self.config.discord.no_auto_thread_channels
            channel_name = getattr(message.channel, "name", "")
            is_no_thread = str(message.channel.id) in no_thread_channels or channel_name in no_thread_channels

            if is_no_thread:
                target_channel = message.channel
            else:
                # In regular channel; find or create a thread
                thread: discord.Thread | None = None
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
                target_channel = thread

        channel_id = str(target_channel.id)
        session = await self.gateway.get_session_by_channel(self.chatbot_id, channel_id)

        if not session:
            custom_prompt = _build_discord_custom_prompt(target_channel, message.author)
            session_title = getattr(target_channel, "name", None) or f"Chat with {message.author.display_name}"
            session = await self.gateway.create_session(
                title=session_title,
                custom_prompt=custom_prompt,
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

        # Trigger typing status for the thread/channel while agent is thinking
        if channel_id not in self._typing_tasks:
            self._typing_tasks[channel_id] = asyncio.create_task(self._keep_typing(target_channel))

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

        # Try in-place editing if it is a tool result and we have the cached message
        if message.role == ROLE_TOOL and message.type != TYPE_TOOL_CALL:
            tool_call_msg_id = message.parent_id
            if tool_call_msg_id and tool_call_msg_id in self._sent_tool_calls:
                discord_msg = self._sent_tool_calls.pop(tool_call_msg_id)
                try:
                    emoji = "❌" if message.metadata.get("tool_error") else "✅"
                    content = discord_msg.content.replace("⏳", emoji)
                    await discord_msg.edit(content=content)
                except Exception as ee:
                    logger.warning(f"Failed to edit tool call message in-place: {ee}")

            await self.gateway.update_message_status(message.id, STATUS_DELIVERED)
            return

        is_special_message = False
        output_text = ""

        if message.role == ROLE_ASSISTANT:
            if message.type == TYPE_THOUGHT:
                is_special_message = True
                first_line = message.content.split("\n")[0].strip()
                hidden_chars = len(message.content) - len(first_line)
                output_text = f"💭 {first_line} ... *(+{hidden_chars} chars)*"
            else:
                output_text = message.content
        elif message.role == ROLE_TOOL:
            # Since tool results/errors are handled and early-returned above,
            # this block only ever handles TYPE_TOOL_CALL!
            is_special_message = True
            tool_name = message.metadata.get("tool_name") or message.sender or "unknown_tool"
            arg_suffix = await self._get_tool_arguments_suffix(message)
            output_text = f"🛠️ **{tool_name}**{arg_suffix} ⏳"
        elif message.role == ROLE_SYSTEM:
            is_special_message = True
            first_line = message.content.split("\n")[0].strip()
            hidden_chars = len(message.content) - len(first_line)
            output_text = f"⚙️ *System Message:* {first_line} ... *(+{hidden_chars} chars)*"
        else:
            output_text = message.content

        if not output_text:
            output_text = message.content

        # Send the MessageHeaderView at the start of the turn if it's a special message
        if is_special_message:
            turn_id = message.parent_id or message.session_id
            if turn_id not in self._turns_with_header:
                try:
                    header_view = MessageHeaderView(self.gateway, message.session_id)
                    await channel.send(view=header_view)
                    self._turns_with_header.add(turn_id)
                except Exception as he:
                    logger.warning(f"Failed to send message header: {he}")

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
                                sent_msg = await channel.send(current_chunk)
                                if is_special_message:
                                    self._intermediate_messages[message.channel_id].append(sent_msg)
                                current_chunk = ""
                            for i in range(0, len(line), 2000):
                                sent_msg = await channel.send(line[i : i + 2000])
                                if is_special_message:
                                    self._intermediate_messages[message.channel_id].append(sent_msg)
                        elif len(current_chunk) + len(line) > 2000:
                            sent_msg = await channel.send(current_chunk)
                            if is_special_message:
                                self._intermediate_messages[message.channel_id].append(sent_msg)
                            current_chunk = line
                        else:
                            current_chunk += line

                    if current_chunk and current_chunk.strip():
                        sent_msg = await channel.send(current_chunk)
                        if is_special_message:
                            self._intermediate_messages[message.channel_id].append(sent_msg)

                        # Cache the sent message object if it's a tool call for future editing
                        if message.role == ROLE_TOOL and message.type == TYPE_TOOL_CALL:
                            self._sent_tool_calls[message.id] = sent_msg
            elif segment["type"] == "file":
                file_path = segment["path"]
                if not os.path.exists(file_path):  # noqa: ASYNC240
                    logger.error(f"File not found: {file_path}")
                    await channel.send(f"⚠️ File not found: {file_path}")
                else:
                    try:
                        discord_file = discord.File(file_path)
                        await channel.send(file=discord_file)
                    except Exception as e:
                        logger.error(f"Failed to send file {file_path} to Discord: {e}", exc_info=True)
                        await channel.send(f"⚠️ Failed to send file {file_path}: {e}")
            elif segment["type"] == "voice":
                file_path = segment["path"]
                if not os.path.exists(file_path):  # noqa: ASYNC240
                    logger.error(f"Voice file not found: {file_path}")
                    await channel.send(f"⚠️ Voice file not found: {file_path}")
                else:
                    try:
                        await send_voice_message(channel, file_path)
                    except Exception as e:
                        logger.warning(
                            f"Failed to send native voice message for {file_path}: {e}. "
                            "Falling back to standard file attachment."
                        )
                        try:
                            discord_file = discord.File(file_path)
                            await channel.send(file=discord_file)
                        except Exception as fe:
                            logger.error(f"Failed to send voice file fallback for {file_path}: {fe}", exc_info=True)
                            await channel.send(f"⚠️ Failed to send voice file {file_path}: {fe}")

        await self.gateway.update_message_status(message.id, STATUS_DELIVERED)

        # Stop typing status and clean up intermediate special messages
        # when the final assistant response is successfully delivered
        if message.role == ROLE_ASSISTANT and message.type == TYPE_TEXT:
            task = self._typing_tasks.pop(message.channel_id, None)
            if task:
                task.cancel()

            # Delete all tracked intermediate special messages for this channel
            intermediate_msgs = self._intermediate_messages.pop(message.channel_id, [])
            if intermediate_msgs:
                for msg in intermediate_msgs:
                    try:
                        await msg.delete()
                    except Exception as de:
                        logger.warning(
                            f"Failed to delete intermediate message {msg.id} in channel {message.channel_id}: {de}"
                        )

    async def trigger_cronjob(
        self,
        channel_id: str,
        prompt_content: str,
        mention_user_id: str | None = None,
    ) -> None:
        """Trigger a scheduled cronjob in the specified channel/thread.

        Args:
            channel_id: Discord channel or thread identifier.
            prompt_content: The prompt message content to run.
            mention_user_id: Optional Discord user ID to mention.
        """
        if not self.bot.is_ready():
            logger.info("Discord bot is not ready yet. Waiting for connection...")
            try:
                await self.bot.wait_until_ready()
            except Exception as e:
                logger.error(f"Failed waiting for Discord bot to be ready: {e}")
                return

        try:
            target_id = int(channel_id)
        except ValueError:
            logger.error(f"Invalid Discord channel_id for cronjob: {channel_id}")
            return

        channel = self.bot.get_channel(target_id)
        if not channel:
            try:
                channel = await self.bot.fetch_channel(target_id)  # type: ignore
            except Exception as fe:
                logger.error(f"Failed to fetch Discord channel {target_id} for cronjob: {fe}")
                return

        # Determine if this is an auto-thread channel
        is_thread = isinstance(channel, discord.Thread)
        no_thread_channels = self.config.discord.no_auto_thread_channels
        channel_name = getattr(channel, "name", "")
        is_no_thread = str(channel.id) in no_thread_channels or channel_name in no_thread_channels

        target_channel = channel
        if not is_thread and not is_no_thread and hasattr(channel, "create_thread"):
            try:
                mention_str = f"<@{mention_user_id}> " if mention_user_id else ""
                starter_text = f"{mention_str}Scheduled job initiated."
                starter_msg = await channel.send(starter_text)

                # Create thread named after the job/date
                thread_title = f"Scheduled Job - {datetime.date.today().isoformat()}"
                thread = await starter_msg.create_thread(name=thread_title)
                await thread.join()
                target_channel = thread
            except Exception as e:
                logger.error(f"Failed to auto-create thread for cronjob in channel {channel.id}: {e}", exc_info=True)
        elif target_channel == channel and mention_user_id:
            # Send mention starter to direct channel/thread if not auto-threading
            try:
                await channel.send(f"<@{mention_user_id}> Scheduled job starting.")
            except Exception as e:
                logger.warning(f"Failed to send mention in channel {channel.id}: {e}")

        target_channel_id_str = str(target_channel.id)
        session = await self.gateway.get_session_by_channel(self.chatbot_id, target_channel_id_str)

        if not session:
            custom_prompt = _build_discord_custom_prompt(target_channel, self.bot.user)
            session_title = getattr(target_channel, "name", None) or f"Scheduled Job {target_channel_id_str}"
            session = await self.gateway.create_session(
                title=session_title,
                custom_prompt=custom_prompt,
            )
            session_id = session.id
        else:
            session_id = session.id
            await self.gateway.update_session_updated_at(session_id)

        tz_name = _get_local_timezone_name()
        now_dt = datetime.datetime.now()
        now_str = now_dt.strftime('%Y-%m-%d %H:%M:%S')

        discord_msg_content = (
            f"`System` at `{now_str} {tz_name}`:\n"
            f"{prompt_content}"
        )

        msg = Message(
            session_id=session_id,
            chatbot_id=self.chatbot_id,
            channel_id=target_channel_id_str,
            sender="System",
            role=ROLE_USER,
            type=TYPE_TEXT,
            content=discord_msg_content,
            timestamp=now_dt.timestamp(),
            status=STATUS_PENDING_AGENT,
            metadata={"is_cronjob": True},
        )
        await self.gateway.post(msg)

        # Trigger typing status for the thread/channel while agent is thinking
        if target_channel_id_str not in self._typing_tasks:
            self._typing_tasks[target_channel_id_str] = asyncio.create_task(self._keep_typing(target_channel))

