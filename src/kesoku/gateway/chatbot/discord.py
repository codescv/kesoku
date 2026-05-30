"""Discord chatbot adapter for Kesoku AI Agent framework.

Connects Discord channels and threads with Kesoku Gateway using Pub/Sub.
"""

import asyncio
import datetime
import os
from collections import defaultdict
from typing import Any

import discord

from kesoku.agent.prompt import build_sys_prompt
from kesoku.async_utils import (
    async_exists,
    async_realpath,
)
from kesoku.config import DiscordChannelOverride, get_config
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.gateway.chatbot.base import Chatbot, DeliveryAbortedError, get_local_timezone_name
from kesoku.gateway.chatbot.discord_command import setup_discord_commands
from kesoku.gateway.chatbot.discord_ui import MessageHeaderView, QuestionView
from kesoku.gateway.chatbot.discord_voice_message import send_voice_message
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger

logger = setup_logger(__name__)

_get_local_timezone_name = get_local_timezone_name


DISCORD_MAX_CONTENT_LENGTH = 2000


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
## (CRITICAL!!! MUST FOLLOW) Response Format
- IMPORTANT: Only use plain text or emojis when you want to show math. e.g. use "exp(x)"
  instead of "$e^x$", use "∞" instead of "$\\inf$".
- IMPORTANT: You can only use up to level 3 headings (# for h1, ## for h2, ### for h3).
  Never use #### or beyond!
- The above formatting requirement only applies for your response, not for writing files or running commands.
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
        setup_discord_commands(self)
        self._subscriber_task: asyncio.Task[None] | None = None
        self._sent_tool_calls: dict[str, discord.Message] = {}
        self._header_views: dict[str, tuple[discord.Message, MessageHeaderView]] = {}
        self._typing_tasks: dict[str, asyncio.Task[None]] = {}
        self._intermediate_messages: defaultdict[str, list[discord.Message]] = defaultdict(list)
        self._turn_special_items: dict[str, list[dict[str, Any]]] = {}
        self._turn_special_msg: dict[str, discord.Message] = {}

    def _resolve_channel_override(
        self,
        channel_id: str,
        channel_name: str,
        parent_id: str | None = None,
        parent_name: str | None = None,
    ) -> DiscordChannelOverride | None:
        """Resolve the matching DiscordChannelOverride for the given channel identifiers.

        Args:
            channel_id: Direct channel ID.
            channel_name: Direct channel name.
            parent_id: Optional parent channel ID (if thread).
            parent_name: Optional parent channel name (if thread).

        Returns:
            The matching DiscordChannelOverride instance if found, None otherwise.
        """
        for override in self.config.discord.channels:
            identifiers = {channel_id, channel_name}
            if parent_id:
                identifiers.add(parent_id)
            if parent_name:
                identifiers.add(parent_name)
            if any(ident in override.channels for ident in identifiers if ident):
                return override
        return None

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
        if hasattr(self, "tree"):
            try:
                logger.info("Syncing Discord slash commands...")
                await self.tree.sync()
                logger.info("Discord slash commands synced successfully.")
            except Exception as e:
                logger.error(f"Failed to sync Discord slash commands: {e}", exc_info=True)

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
            channel_id_str = str(message.channel.id)
            channel_name = getattr(message.channel, "name", "") or ""

            # Check if there is an override for this channel
            override = self._resolve_channel_override(channel_id_str, channel_name)

            # Determine auto-threading behavior
            auto_thread = True  # Default is to auto-thread
            if override is not None and override.auto_thread is not None:
                auto_thread = override.auto_thread

            if not auto_thread:
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

            # Resolve current role before session creation for accurate initial system prompt
            role = "default"
            db_role = await self.gateway.get_channel_role(self.chatbot_id, channel_id)
            if db_role:
                role = db_role
            elif isinstance(target_channel, discord.Thread) and target_channel.parent:
                parent_role = await self.gateway.get_channel_role(self.chatbot_id, str(target_channel.parent.id))
                if parent_role:
                    role = parent_role

            session = await self.gateway.create_session(
                title=session_title,
                custom_prompt=custom_prompt,
                created_at=message.created_at.timestamp(),
                chatbot_id=self.chatbot_id,
                channel_id=channel_id,
                role=role,
            )
            session_id = session.id
        else:
            session_id = session.id
            await self.gateway.update_session_updated_at(session_id)

            # Rebuild and update the system prompt with the latest Discord channel topic,
            # members list, and custom instructions
            custom_prompt = _build_discord_custom_prompt(target_channel, message.author)
            new_sys_prompt = build_sys_prompt(custom_prompt=custom_prompt, session=session)
            await self.gateway.update_session_system_prompt(session_id, new_sys_prompt)

            # Clean up any active previous turn UI elements if a new turn is started (thought interruption)
            task = self._typing_tasks.pop(channel_id, None)
            if task:
                task.cancel()

            intermediate_msgs = self._intermediate_messages.pop(channel_id, [])
            if intermediate_msgs:
                for msg in intermediate_msgs:
                    try:
                        await msg.delete()
                    except Exception as de:
                        logger.warning(f"Failed to delete intermediate message {msg.id} on interruption: {de}")

            active_turn_ids = [tid for tid in self._header_views if tid == session_id or tid.startswith(session_id)]
            for tid in active_turn_ids:
                header_msg, header_view = self._header_views.pop(tid)
                try:
                    await header_msg.delete()
                except Exception as he:
                    logger.warning(f"Failed to delete header message {header_msg.id} on interruption: {he}")

            self._turn_special_items.pop(session_id, None)
            self._turn_special_msg.pop(session_id, None)

        # Save any incoming attachments to the session staging directory
        attachments_metadata = []
        if message.attachments:
            session_staging_dir = await async_realpath(
                os.path.join(self.config.workspace.sessions_dir, session.workspace_name)
            )
            os.makedirs(session_staging_dir, exist_ok=True)

            for attachment in message.attachments:
                filename = attachment.filename
                # Sanitize filename to prevent path traversal
                safe_filename = "".join(c for c in filename if c.isalnum() or c in "._-")
                if not safe_filename:
                    safe_filename = f"attachment_{attachment.id}"

                filepath = os.path.join(session_staging_dir, safe_filename)
                # Avoid file collisions
                if await async_exists(filepath):
                    base, ext = os.path.splitext(safe_filename)
                    safe_filename = f"{base}_{attachment.id}{ext}"
                    filepath = os.path.join(session_staging_dir, safe_filename)

                logger.info(f"Saving Discord attachment {filename} to {filepath}")
                await attachment.save(filepath)

                attachments_metadata.append(
                    {
                        "path": filepath,
                        "mime_type": attachment.content_type or "application/octet-stream",
                        "filename": filename,
                    }
                )

        # Ingest user message into Gateway
        tz_name = get_local_timezone_name()
        discord_msg_content = (
            f"`{message.author.display_name}` <@{message.author.id}> "
            f"at `{message.created_at.astimezone().strftime('%Y-%m-%d %H:%M:%S')} {tz_name}`:\n"
            f"{message.content}"
        )
        if attachments_metadata:
            files_str = "\n".join(
                f"[Attachment: {a['filename']} ({a['mime_type']}) saved at {a['path']}]" for a in attachments_metadata
            )
            discord_msg_content += f"\n\nAttachments:\n{files_str}"

        # Construct metadata with parent/thread identifiers
        channel_name = getattr(target_channel, "name", "") or ""
        parent_channel_id = ""
        parent_channel_name = ""
        if isinstance(target_channel, discord.Thread):
            parent = target_channel.parent
            if parent:
                parent_channel_id = str(parent.id)
                parent_channel_name = getattr(parent, "name", "") or ""

        msg_metadata = {
            "discord_message_id": str(message.id),
            "discord_author_id": str(message.author.id),
            "channel_name": channel_name,
        }
        if attachments_metadata:
            msg_metadata["attachments"] = attachments_metadata
        if parent_channel_id:
            msg_metadata["parent_channel_id"] = parent_channel_id
        if parent_channel_name:
            msg_metadata["parent_channel_name"] = parent_channel_name

        msg = Message(
            session_id=session_id,
            chatbot_id=self.chatbot_id,
            channel_id=channel_id,
            sender=message.author.display_name,
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content=discord_msg_content,
            timestamp=message.created_at.timestamp(),
            status=MessageStatus.PENDING_AGENT,
            metadata=msg_metadata,
        )
        await self.gateway.post(msg)

        # Trigger typing status for the thread/channel while agent is thinking
        if channel_id not in self._typing_tasks:
            self._typing_tasks[channel_id] = asyncio.create_task(self._keep_typing(target_channel))

    async def handle_message(self, message: Message) -> None:
        """Process outgoing message from Gateway and send to target Discord thread."""
        await self.render_outgoing_message(message)

    async def _get_discord_channel_with_abort(self, message: Message) -> Any | None:
        try:
            target_id = int(message.channel_id)
        except ValueError:
            logger.error(f"Invalid Discord channel_id: {message.channel_id}")
            return None

        channel = self.bot.get_channel(target_id)
        if not channel:
            try:
                channel = await self.bot.fetch_channel(target_id)  # type: ignore
            except (discord.NotFound, discord.Forbidden) as fe:
                logger.warning(
                    f"Message {message.id} Failed to fetch Discord channel {target_id} (deleted or forbidden): {fe}. "
                    "Aborting session and marking message as delivered to stop retrying."
                )
                await self.gateway.abort_session(message.session_id)
                await self.gateway.update_message_status(message.id, MessageStatus.DELIVERED)
                raise DeliveryAbortedError("Discord channel fetch failed (deleted or forbidden)")
            except Exception as fe:
                logger.error(f"Failed to fetch Discord channel {target_id}: {fe}")
                return None
        return channel

    def supports_intermediate_messages(self) -> bool:
        """Determine if the platform supports intermediate rendering of thoughts and tools."""
        return True

    async def handle_intermediate_message(self, message: Message) -> None:
        """Render an intermediate thought/tool/system message in the Discord UI."""
        session_id = message.session_id
        if session_id not in self._turn_special_items:
            self._turn_special_items[session_id] = []

        if message.role == MessageRole.ASSISTANT and message.type == MessageType.THOUGHT:
            first_line = message.content.split("\n")[0].strip()
            hidden_chars = len(message.content) - len(first_line)
            thought_content = f"{first_line} ... *(+{hidden_chars} chars)*"

            tc_exists = any(
                item["type"] == "thought" and item["id"] == message.id for item in self._turn_special_items[session_id]
            )
            if not tc_exists:
                self._turn_special_items[session_id].append(
                    {
                        "type": "thought",
                        "id": message.id,
                        "content": thought_content,
                    }
                )

        elif message.role == MessageRole.TOOL:
            tool_name = message.metadata.get("tool_name") or message.sender or "unknown_tool"
            arg_suffix = await self._get_tool_arguments_suffix(message)

            tc_exists = any(
                item["type"] == "tool_call" and item["id"] == message.id
                for item in self._turn_special_items[session_id]
            )
            if not tc_exists:
                self._turn_special_items[session_id].append(
                    {
                        "type": "tool_call",
                        "id": message.id,
                        "tool_name": tool_name,
                        "arg_suffix": arg_suffix,
                        "status": "⏳",
                    }
                )

        elif message.role == MessageRole.SYSTEM:
            first_line = message.content.split("\n")[0].strip()
            hidden_chars = len(message.content) - len(first_line)
            system_content = f"{first_line} ... *(+{hidden_chars} chars)*"

            tc_exists = any(
                item["type"] == "system" and item["id"] == message.id for item in self._turn_special_items[session_id]
            )
            if not tc_exists:
                self._turn_special_items[session_id].append(
                    {
                        "type": "system",
                        "id": message.id,
                        "content": system_content,
                    }
                )

        lines = []
        for item in self._turn_special_items[session_id]:
            if item["type"] == "thought":
                lines.append(f"💭 {item['content']}")
            elif item["type"] == "tool_call":
                lines.append(f"🛠️ **{item['tool_name']}**{item['arg_suffix']} {item['status']}")
            elif item["type"] == "system":
                lines.append(f"⚙️ *System Message:* {item['content']}")
        new_content = "\n".join(lines)
        if len(new_content) > DISCORD_MAX_CONTENT_LENGTH:
            new_content = new_content[: DISCORD_MAX_CONTENT_LENGTH - len(" (omitted)")] + " (omitted)"

        channel = await self._get_discord_channel_with_abort(message)
        if not channel:
            return

        # Send the MessageHeaderView at the start of the turn if it's a special message
        turn_id = message.parent_id or message.session_id
        if turn_id not in self._header_views:
            try:
                is_thread = isinstance(channel, discord.Thread)
                header_view = MessageHeaderView(
                    self.gateway,
                    message.session_id,
                    chatbot=self,
                    is_thread=is_thread,
                )
                header_msg = await channel.send(
                    content=f"🔍 **Session ID:** `{message.session_id}`",
                    view=header_view,
                )
                self._header_views[turn_id] = (header_msg, header_view)
            except Exception as he:
                logger.warning(f"Failed to send message header: {he}")

        if session_id in self._turn_special_msg:
            discord_msg = self._turn_special_msg[session_id]
            try:
                await discord_msg.edit(content=new_content)
            except Exception as ee:
                logger.warning(f"Failed to edit single special message: {ee}")
            await self.gateway.update_message_status(message.id, MessageStatus.DELIVERED)
            return

        sent_msg = await channel.send(new_content)
        self._intermediate_messages[message.channel_id].append(sent_msg)
        self._turn_special_msg[session_id] = sent_msg
        if message.role == MessageRole.TOOL and message.type == MessageType.TOOL_CALL:
            self._sent_tool_calls[message.id] = sent_msg

        await self.gateway.update_message_status(message.id, MessageStatus.DELIVERED)

    async def handle_tool_result(self, message: Message) -> None:
        """Update the status of a previously executed tool to Success or Error emoji in the special messages list."""
        tool_call_msg_id = message.parent_id
        session_id = message.session_id
        if session_id in self._turn_special_items:
            items = self._turn_special_items[session_id]
            tc_item = None
            for item in items:
                if item["type"] == "tool_call" and item["id"] == tool_call_msg_id:
                    tc_item = item
                    break

            if tc_item:
                emoji = "❌" if message.metadata.get("tool_error") else "✅"
                tc_item["status"] = emoji

                # Re-render the combined statuses of all special messages
                lines = []
                for item in items:
                    if item["type"] == "thought":
                        lines.append(f"💭 {item['content']}")
                    elif item["type"] == "tool_call":
                        lines.append(f"🛠️ **{item['tool_name']}**{item['arg_suffix']} {item['status']}")
                    elif item["type"] == "system":
                        lines.append(f"⚙️ *System Message:* {item['content']}")
                new_content = "\n".join(lines)
                if len(new_content) > DISCORD_MAX_CONTENT_LENGTH:
                    new_content = new_content[: DISCORD_MAX_CONTENT_LENGTH - len(" (omitted)")] + " (omitted)"

                discord_msg = self._turn_special_msg.get(session_id)
                if discord_msg:
                    try:
                        await discord_msg.edit(content=new_content)
                    except Exception as ee:
                        logger.warning(f"Failed to edit single special message: {ee}")

                await self.gateway.update_message_status(message.id, MessageStatus.DELIVERED)
                return

        if tool_call_msg_id and tool_call_msg_id in self._sent_tool_calls:
            discord_msg = self._sent_tool_calls.pop(tool_call_msg_id)
            try:
                emoji = "❌" if message.metadata.get("tool_error") else "✅"
                content = discord_msg.content.replace("⏳", emoji)
                if len(content) > DISCORD_MAX_CONTENT_LENGTH:
                    content = content[: DISCORD_MAX_CONTENT_LENGTH - len(" (omitted)")] + " (omitted)"
                await discord_msg.edit(content=content)
            except Exception as ee:
                logger.warning(f"Failed to edit tool call message in-place: {ee}")

        await self.gateway.update_message_status(message.id, MessageStatus.DELIVERED)

    async def send_text_chunks(self, channel_id: str, chunks: list[str], message: Message) -> None:
        """Deliver formatted text chunks sequentially to the target channel."""
        channel = await self._get_discord_channel_with_abort(message)
        if not channel:
            return

        is_special = self.is_intermediate_message(message)
        for chunk in chunks:
            if chunk.strip():
                sent_msg = await channel.send(chunk)
                if is_special:
                    self._intermediate_messages[message.channel_id].append(sent_msg)
                    self._turn_special_msg[message.session_id] = sent_msg
                    if message.role == MessageRole.TOOL and message.type == MessageType.TOOL_CALL:
                        self._sent_tool_calls[message.id] = sent_msg

    async def send_file_segment(self, channel_id: str, file_path: str, message: Message) -> None:
        """Deliver a file attachment segment to the target Discord thread."""
        channel = await self._get_discord_channel_with_abort(message)
        if not channel:
            return

        if not await async_exists(file_path):
            logger.error(f"File not found: {file_path}")
            await channel.send(f"⚠️ File not found: {file_path}")
        else:
            try:
                discord_file = discord.File(file_path)
                await channel.send(file=discord_file)
            except Exception as e:
                logger.error(f"Failed to send file {file_path} to Discord: {e}", exc_info=True)
                await channel.send(f"⚠️ Failed to send file {file_path}: {e}")

    async def send_voice_segment(self, channel_id: str, file_path: str, message: Message) -> None:
        """Deliver a voice segment to the target Discord thread (falling back to standard attachment if failed)."""
        channel = await self._get_discord_channel_with_abort(message)
        if not channel:
            return

        if not await async_exists(file_path):
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

    async def send_question_segment(self, channel_id: str, question: str, choices: list[str], message: Message) -> None:
        """Deliver a multiple choice question block as a dynamic action button view embed."""
        channel = await self._get_discord_channel_with_abort(message)
        if not channel:
            return

        try:
            question_view = QuestionView(
                gateway=self.gateway,
                session_id=message.session_id,
                chatbot=self,
                question=question,
                choices=choices,
            )
            embed = discord.Embed(
                title=f"❓ {question}",
                color=discord.Color.blurple(),
            )
            await channel.send(embed=embed, view=question_view)
        except Exception as qe:
            logger.error(f"Failed to send question view to Discord: {qe}", exc_info=True)
            await channel.send(f"⚠️ Failed to send question: {question}")

    async def on_message_delivered(self, message: Message) -> None:
        """Post-delivery lifecycle callback: clean up typing indicator, intermediate lists, and update metrics."""
        if message.role == MessageRole.ASSISTANT and message.type == MessageType.TEXT:
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

            # Clean up tool call single-message caches for the session
            self._turn_special_items.pop(message.session_id, None)
            self._turn_special_msg.pop(message.session_id, None)

            # Remove stop button from the header view for this turn
            turn_id = message.parent_id or message.session_id
            if turn_id in self._header_views:
                header_msg, header_view = self._header_views[turn_id]
                try:
                    header_view.remove_item(header_view.stop_turn)

                    header_content = f"🔍 **Session ID:** `{message.session_id}`"
                    metrics = message.metadata.get("turn_metrics")
                    if metrics:
                        session_turns = metrics.get("session_turns", 0)
                        context_tokens = metrics.get("context_tokens", 0)
                        cached_tokens = metrics.get("cached_tokens", 0)
                        context_percent = metrics.get("context_percent", 0.0)
                        turn_tool_calls = metrics.get("turn_tool_calls", 0)
                        turn_tokens = metrics.get("turn_tokens", 0)
                        turn_time = metrics.get("turn_time", 0.0)

                        context_k = f"{round(context_tokens / 1000)}K"
                        turn_k = f"{round(turn_tokens / 1000)}K"
                        cached_k = f"{round(cached_tokens / 1000)}K"

                        context_str = f"{context_k} tokens"
                        if cached_tokens > 0:
                            context_str += f" (Cached: {cached_k})"

                        header_content = (
                            f"🔍 **Session ID:** `{message.session_id}`\n"
                            f"⚡ **Session:** {session_turns} turns | "
                            f"**Context:** {context_str} ({context_percent:.1f}% of window)\n"
                            f"⏱️ **Turn:** {turn_tool_calls} tool calls | {turn_k} tokens | {turn_time:.1f}s"
                        )

                    if header_content is not None:
                        await header_msg.edit(content=header_content, view=header_view)
                    else:
                        await header_msg.edit(view=header_view)
                except Exception as ee:
                    logger.warning(f"Failed to update header view with metrics: {ee}")

    async def trigger_cronjob(
        self,
        channel_id: str,
        prompt_content: str,
        mention_user_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Trigger a scheduled cronjob in the specified channel/thread.

        Args:
            channel_id: Discord channel or thread identifier.
            prompt_content: The prompt message content to run.
            mention_user_id: Optional Discord user ID to mention.
            **kwargs: Additional optional arguments.
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
        channel_id_str = str(channel.id)
        channel_name = getattr(channel, "name", "") or ""

        auto_thread = True
        if not is_thread:
            override = self._resolve_channel_override(channel_id_str, channel_name)
            if override is not None and override.auto_thread is not None:
                auto_thread = override.auto_thread
        else:
            auto_thread = False

        target_channel = channel
        if auto_thread and hasattr(channel, "create_thread"):
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
            try:
                await channel.send(f"<@{mention_user_id}> Scheduled job starting.")
            except Exception as e:
                logger.warning(f"Failed to send mention in channel {channel.id}: {e}")

        target_channel_id_str = str(target_channel.id)
        custom_prompt = _build_discord_custom_prompt(target_channel, self.bot.user)

        # Construct metadata with parent/thread identifiers
        channel_name = getattr(target_channel, "name", "") or ""
        parent_channel_id = ""
        parent_channel_name = ""
        if isinstance(target_channel, discord.Thread):
            parent = target_channel.parent
            if parent:
                parent_channel_id = str(parent.id)
                parent_channel_name = getattr(parent, "name", "") or ""

        msg_metadata = {
            "channel_name": channel_name,
        }
        if parent_channel_id:
            msg_metadata["parent_channel_id"] = parent_channel_id
        if parent_channel_name:
            msg_metadata["parent_channel_name"] = parent_channel_name

        await self.trigger_cronjob_message(
            channel_id=target_channel_id_str,
            prompt_content=prompt_content,
            sender_name="Cronjob",
            custom_prompt=custom_prompt,
            metadata=msg_metadata,
        )

        # Trigger typing status for the thread/channel while agent is thinking
        if target_channel_id_str not in self._typing_tasks:
            self._typing_tasks[target_channel_id_str] = asyncio.create_task(self._keep_typing(target_channel))
