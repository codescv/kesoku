"""Base class for Kesoku chatbot adapters."""

import asyncio
import datetime
import os
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

import tzlocal
from pydantic import BaseModel, Field

from kesoku.agent.prompt import build_sys_prompt
from kesoku.config import get_config
from kesoku.constants import SYSTEM_START_TIME, MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger
from kesoku.utils.service import restart_service as utils_restart_service
from kesoku.utils.text import format_text, split_text_into_chunks

logger = setup_logger(__name__)


def parse_message_content(content: str) -> list[dict[str, Any]]:
    """Parse message content to extract zero or more blocks.

    Recognizes `[file: /path]`, `[voice: /path]`, or `[question: <question> | choice1 | ...]`.

    Args:
        content: Raw message text content to parse.

    Returns:
        A list of segment dictionaries. Text segments have format:
        {"type": "text", "content": "..."}, file segments have format:
        {"type": "file", "path": "..."}, voice segments have format:
        {"type": "voice", "path": "..."}, and question segments have format:
        {"type": "question", "question": "...", "choices": [...]}.
    """
    # Regex matches [file: <path>], [voice: <path>], or [question: <text>]
    # where <text> is any character except closed bracket
    pattern = re.compile(r"\[(file|voice|question):\s*([^\]]+)\s*\]")
    segments: list[dict[str, Any]] = []
    last_idx = 0

    for match in pattern.finditer(content):
        text_before = content[last_idx : match.start()]
        if text_before:
            segments.append({"type": "text", "content": text_before})

        block_type = match.group(1)
        inner_val = match.group(2).strip()

        if block_type == "question":
            if "||" in inner_val:
                q_part, choices_part = inner_val.split("||", 1)
                question_text = q_part.strip()
                choices = [c.strip() for c in choices_part.split("|") if c.strip()]
            else:
                parts = [p.strip() for p in inner_val.split("|")]
                question_text = parts[0]
                choices = parts[1:]
            segments.append(
                {
                    "type": "question",
                    "question": question_text,
                    "choices": choices,
                }
            )
        else:
            segments.append({"type": block_type, "path": inner_val})
        last_idx = match.end()

    text_after = content[last_idx:]
    if text_after:
        segments.append({"type": "text", "content": text_after})

    return segments


class DeliveryAbortedError(Exception):
    """Exception raised to immediately halt outgoing message delivery."""

    pass


def get_local_timezone_name() -> str:
    """Retrieve the local system timezone name (e.g., 'Asia/Shanghai')."""
    try:
        return tzlocal.get_localzone().key or "UTC"
    except Exception:
        return datetime.datetime.now().astimezone().tzname() or "UTC"


def _format_uptime(td: datetime.timedelta) -> str:
    """Format a timedelta into a concise string representing uptime.

    Args:
        td: The timedelta to format.

    Returns:
        A human-readable uptime string, e.g., '2d 4h 15m 3s'.
    """
    total_seconds = int(td.total_seconds())
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or days > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0 or days > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)

class InboundMessageAttachment(BaseModel):
    """Unified attachment metadata for inbound messages."""
    path: str
    mime_type: str
    filename: str


class InboundMessageDTO(BaseModel):
    """Unified Data Transfer Object for inbound messages across all platforms."""
    sender_id: str
    channel_id: str
    text: str = ""
    message_id: str = ""
    timestamp: float = Field(default_factory=time.time)
    attachments: list[InboundMessageAttachment] = Field(default_factory=list)
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
    session_title: str | None = None
    custom_prompt: str | None = None
    role: str = "default"


class CommandRegistry:
    """Unified registry for chatbot slash commands."""

    def __init__(self) -> None:
        """Initialize the CommandRegistry with an empty command mapping."""
        self._commands: dict[str, dict[str, Any]] = {}

    def register(self, name: str, description: str, handler: Callable[..., Awaitable[None]]) -> None:
        """Register a command with its description and async handler.

        Args:
            name: The command name (e.g., 'restart').
            description: A short description of the command.
            handler: An async callback function to handle the command execution.
        """
        self._commands[name] = {
            "description": description,
            "handler": handler,
        }

    def get_commands(self) -> dict[str, dict[str, Any]]:
        """Get all registered commands."""
        return self._commands

    async def execute(self, name: str, *args: Any, **kwargs: Any) -> None:
        """Execute the registered command.

        Args:
            name: The name of the command.
            *args: Variable length argument list for the handler.
            **kwargs: Arbitrary keyword arguments for the handler.
        """
        if name in self._commands:
            await self._commands[name]["handler"](*args, **kwargs)
        else:
            raise ValueError(f"Command '{name}' is not registered.")


class Chatbot(ABC):
    """Abstract base class for chatbot adapters connecting to Kesoku Gateway."""

    def __init__(self, chatbot_id: str, gateway: Gateway, session_id: str | None = None) -> None:
        """Initialize the chatbot with a unique identifier, gateway instance, and optional session ID.

        Args:
            chatbot_id: Unique identifier for this chatbot instance (e.g., 'console', 'discord_primary').
            gateway: The Kesoku Gateway instance managing routing and persistence.
            session_id: Optional specific session ID to listen to.
        """
        self.chatbot_id = chatbot_id
        self.gateway = gateway
        self.session_id = session_id
        self._listener_task: asyncio.Task[None] | None = None

        self.commands = CommandRegistry()
        self._register_default_commands()

    def _register_default_commands(self) -> None:
        """Register platform-agnostic standard commands."""

        async def handle_restart(reply_func: Callable[[str], Awaitable[None]]) -> None:
            await reply_func("🔄 Restarting service...")
            await self.restart_service()

        async def handle_clear(reply_func: Callable[[str], Awaitable[None]], channel_id: str) -> None:
            status_msg = await self.clear_session_by_channel(channel_id)
            await reply_func(status_msg)

        async def handle_status(reply_func: Callable[[str], Awaitable[None]], channel_id: str) -> None:
            status_msg = await self.get_session_status_by_channel(channel_id)
            await reply_func(status_msg)

        async def handle_compact(reply_func: Callable[[str], Awaitable[None]], channel_id: str) -> None:
            status_msg = await self.manual_compact_session_by_channel(channel_id)
            await reply_func(status_msg)

        async def handle_role(
            reply_func: Callable[[str], Awaitable[None]],
            channel_id: str,
            role_name: str = "",
        ) -> None:
            status_msg = await self.update_role_by_channel(channel_id, role_name)
            await reply_func(status_msg)

        self.commands.register("restart", "Restart the Kesoku service.", handle_restart)
        self.commands.register("clear", "Clear the active conversation session.", handle_clear)
        self.commands.register("reset", "Clear the active conversation session.", handle_clear)
        self.commands.register("status", "Get conversation and performance statistics.", handle_status)
        self.commands.register("compact", "Manually compact conversation history.", handle_compact)
        self.commands.register(
            "role",
            "Update or view the active roleplay persona for the current channel.",
            handle_role,
        )

    async def restart_service(self) -> None:
        """Restart the Kesoku service."""
        await utils_restart_service(self.chatbot_id, self.stop)

    async def execute_command_from_text(
        self,
        text: str,
        reply_func: Callable[[str], Awaitable[None]],
        channel_id: str | None = None,
    ) -> None:
        """Parse and execute a slash command from text.

        Args:
            text: Raw text containing the command (e.g., '/role helper').
            reply_func: Async callback to send response back.
            channel_id: Optional channel ID context.
        """
        parts = text.strip().split()
        if not parts:
            return

        raw_command = parts[0]
        if not raw_command.startswith("/"):
            return

        command = raw_command.lower().lstrip("/")

        try:
            if command in {"clear", "reset", "status", "compact"}:
                if not channel_id:
                    await reply_func("⚠️ Channel ID is required for this command.")
                    return
                await self.commands.execute(command, reply_func, channel_id=channel_id)
            elif command == "role":
                if not channel_id:
                    await reply_func("⚠️ Channel ID is required for this command.")
                    return
                role_name = " ".join(parts[1:]) if len(parts) > 1 else ""
                await self.commands.execute("role", reply_func, channel_id=channel_id, role_name=role_name)
            elif command == "restart":
                await self.commands.execute(command, reply_func)
            else:
                await reply_func(f"⚠️ Unrecognized command: /{command}")
        except Exception as e:
            logger.error(f"Command /{command} execution failed: {e}", exc_info=True)
            await reply_func(f"⚠️ Failed to execute command: {e}")

    async def clear_session(self, session_id: str) -> None:
        """Stop any active worker for the session, and delete the session database record and workspace."""
        logger.info(f"Chatbot '{self.chatbot_id}' clearing session '{session_id}'.")
        agent = self.gateway.agent
        if agent:
            worker = agent.workers.get(session_id)
            if worker:
                worker.stop()
                agent.workers.pop(session_id, None)
        await self.gateway.delete_session(session_id)

    async def clear_session_by_channel(self, channel_id: str) -> str:
        """Clear session associated with the channel. Returns status message."""
        session = await self.gateway.get_session_by_channel(self.chatbot_id, channel_id)
        if session:
            await self.clear_session(session.id)
            return "♻️ Session successfully cleared. The next message will initiate a new session."
        return "⚠️ No active session found for this chat."

    async def manual_compact_session_by_channel(self, channel_id: str) -> str:
        """Manually trigger conversation compaction by posting a trigger system user message."""
        session = await self.gateway.get_session_by_channel(self.chatbot_id, channel_id)
        if not session:
            return "⚠️ No active session found for this chat."

        # Create system trigger user message
        # Note: MUST use MessageRole.USER so the agent dispatcher listens to and picks it up to run the turn!
        trigger_content = (
            "[System Notification] The user has manually requested session compaction.\n"
            "Please write a comprehensive summary of the conversation history so far "
            "using the defined structured template (including the 'Key Commands & Executions' section "
            "with the most commonly used/successful shell commands under active skills), "
            "and call the 'compact_history' tool immediately to transition this channel to a clean session."
        )
        trigger_msg = Message(
            session_id=session.id,
            chatbot_id=self.chatbot_id,
            channel_id=channel_id,
            sender="System",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content=trigger_content,
            status=MessageStatus.PENDING_AGENT,
        )
        await self.gateway.post(trigger_msg)
        return "🔄 Initiating history compaction. Please wait a moment..."

    async def get_session_status_by_channel(self, channel_id: str) -> str:
        """Get session statistics for the channel."""
        session = await self.gateway.get_session_by_channel(self.chatbot_id, channel_id)
        if not session:
            return "⚠️ No active session found for this chat."

        history = await self.gateway.get_session_history(session.id, limit=100)
        metrics = None
        for msg in reversed(history):
            if msg.role == MessageRole.ASSISTANT and msg.metadata and msg.metadata.get("turn_metrics"):
                metrics = msg.metadata.get("turn_metrics")
                break

        session_turns = len([m for m in history if m.role == MessageRole.USER])
        context_tokens = metrics.get("context_tokens", 0) if metrics else 0
        cached_tokens = metrics.get("cached_tokens", 0) if metrics else 0
        turn_tool_calls = (
            metrics.get("turn_tool_calls", 0)
            if metrics
            else len([m for m in history if m.role == MessageRole.TOOL and m.type == MessageType.TOOL_CALL])
        )
        turn_tokens = metrics.get("turn_tokens", 0) if metrics else 0
        turn_time = metrics.get("turn_time", 0.0) if metrics else 0.0

        context_k = f"{round(context_tokens / 1000)}K" if context_tokens else "0K"
        turn_k = f"{round(turn_tokens / 1000)}K" if turn_tokens else "0K"
        cached_k = f"{round(cached_tokens / 1000)}K" if cached_tokens else "0K"

        context_str = f"{context_k} tokens"
        if cached_tokens > 0:
            context_str += f" (Cached: {cached_k})"

        uptime_td = datetime.datetime.now() - SYSTEM_START_TIME
        uptime_str = _format_uptime(uptime_td)
        started_str = SYSTEM_START_TIME.strftime("%Y-%m-%d %H:%M:%S")

        return (
            f"【Current Stats】\n"
            f"⏰ Uptime: {uptime_str} (started: {started_str})\n"
            f"⚡ Session: {session_turns} turns (ID: {session.id})\n"
            f"📖 Context: {context_str}\n"
            f"⏱️ Last Turn:\n"
            f"  - Tool Calls: {turn_tool_calls}\n"
            f"  - Tokens: {turn_k}\n"
            f"  - Time: {turn_time:.1f}s"
        )

    async def trigger_cronjob_message(
        self,
        channel_id: str,
        prompt_content: str,
        sender_name: str = "Cronjob",
        custom_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
        title: str | None = None,
    ) -> Message:
        """Unified helper to create a session (if not exists) and post a scheduled cronjob message to the gateway.

        Returns:
            The posted Message instance.
        """
        session = await self.gateway.get_session_by_channel(self.chatbot_id, channel_id)
        if not session:
            session_title = title or f"{self.chatbot_id.capitalize()} Scheduled Job {channel_id}"
            session = await self.gateway.create_session(
                session_id=None,
                title=session_title,
                custom_prompt=custom_prompt,
                chatbot_id=self.chatbot_id,
                channel_id=channel_id,
            )
        else:
            await self.gateway.update_session_updated_at(session.id)

        now_dt = datetime.datetime.now()
        tz_name = get_local_timezone_name()
        now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")

        msg_content = f"`{sender_name}` at `{now_str} {tz_name}`:\n{prompt_content}"

        msg_metadata = {"is_cronjob": True}
        if metadata:
            msg_metadata.update(metadata)

        msg = Message(
            session_id=session.id,
            chatbot_id=self.chatbot_id,
            channel_id=channel_id,
            sender=sender_name,
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content=msg_content,
            timestamp=now_dt.timestamp(),
            status=MessageStatus.PENDING_AGENT,
            metadata=msg_metadata,
        )
        await self.gateway.post(msg)
        return msg

    def get_session_staging_dir(self, workspace_name: str) -> str:
        """Get and ensure the session's absolute staging directory exists."""
        cfg = get_config()
        sessions_dir = cfg.workspace.sessions_dir
        staging_dir = os.path.realpath(os.path.join(sessions_dir, workspace_name))
        os.makedirs(staging_dir, exist_ok=True)
        return staging_dir

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize a filename to prevent path traversal."""
        return "".join(c for c in filename if c.isalnum() or c in "._-")

    def is_intermediate_message(self, message: Message) -> bool:
        """Check if the message is an intermediate/thought/system/tool_call message."""
        return (
            (message.role == MessageRole.ASSISTANT and message.type == MessageType.THOUGHT)
            or (message.role == MessageRole.TOOL and message.type == MessageType.TOOL_CALL)
            or (message.role == MessageRole.SYSTEM)
        )

    def supports_intermediate_messages(self) -> bool:
        """Whether the platform supports rendering thoughts and tools."""
        return False

    async def handle_intermediate_message(self, message: Message) -> None:
        """Hook to render intermediate thought/tool/system message."""
        pass

    async def handle_tool_result(self, message: Message) -> None:
        """Hook to handle tool result status updates (e.g., updating status indicator in-place)."""
        await self.gateway.update_message_status(message.id, MessageStatus.DELIVERED)

    def format_text(self, text: str) -> str:
        """Format/normalize markdown or lines before chunking.

        Cleans up headers, shifts header levels starting from level 1, clamps to
        maximum level 3, ensures blank line before headings, and collapses 3+
        consecutive newlines (outside code blocks).

        Args:
            text: The raw input markdown/text.

        Returns:
            The formatted and cleaned text.
        """
        return format_text(text)

    def get_max_text_length(self) -> int:
        """Maximum text length allowed per message chunk for the platform."""
        return 2000

    def split_text_into_chunks(self, text: str, max_length: int) -> list[str]:
        """Split text into chunks of at most max_length.

        Avoids splitting in the middle of code blocks (triple backticks). If a
        chunk would exceed max_length, it closes the code block with triple
        backticks at the end of the current chunk, and prepends the matching
        opening tag at the beginning of the next chunk.

        Args:
            text: The formatted text to split.
            max_length: The maximum characters allowed in a single chunk.

        Returns:
            A list of message chunks.
        """
        return split_text_into_chunks(text, max_length)

    async def render_outgoing_message(self, message: Message) -> None:
        """Common template method to process and render an outgoing Gateway message.

        Handles routing to intermediate hooks (for thought/tool/system) or final hooks
        (for text/file/voice/question segments), including automatic chunking and delivery status updates.
        """
        try:
            # 1. Resolve if platform wants to handle intermediate messages (thought, tool, system)
            if self.is_intermediate_message(message):
                if not self.supports_intermediate_messages():
                    # Mark as delivered and return if not supported
                    await self.gateway.update_message_status(message.id, MessageStatus.DELIVERED)
                    return

                # Handle intermediate/special message (thought, tool, system)
                await self.handle_intermediate_message(message)
                return

            # 2. Handle Tool Results (updating tool call status to checkmark or error emoji)
            if message.role == MessageRole.TOOL and message.type != MessageType.TOOL_CALL:
                await self.handle_tool_result(message)
                return

            # 3. Parse message content to extract text, file, voice, or question segments
            segments = parse_message_content(message.content)

            for segment in segments:
                if segment["type"] == "text":
                    text_content = segment["content"]
                    if text_content.strip():
                        # Clean/normalize/format text if needed per platform
                        formatted_text = self.format_text(text_content)

                        # Split into chunks matching platform limits (default 2000)
                        chunks = self.split_text_into_chunks(formatted_text, max_length=self.get_max_text_length())

                        await self.send_text_chunks(message.channel_id, chunks, message)

                elif segment["type"] == "file":
                    await self.send_file_segment(message.channel_id, segment["path"], message)

                elif segment["type"] == "voice":
                    await self.send_voice_segment(message.channel_id, segment["path"], message)

                elif segment["type"] == "question":
                    await self.send_question_segment(
                        message.channel_id, segment["question"], segment["choices"], message
                    )

            # 4. Update Gateway status to DELIVERED
            await self.gateway.update_message_status(message.id, MessageStatus.DELIVERED)

            # 5. Post-delivery lifecycle hook (e.g., stop typing indicator, update metrics, finalize card/header)
            await self.on_message_delivered(message)
        except DeliveryAbortedError:
            # Stop further execution of the template since the delivery was aborted/handled elsewhere
            pass

    async def start(self) -> None:
        """Start listening as a decentralized subscriber for model responses.

        Subscribes to gateway messages for this session_id (if set) or chatbot_id
        and routes non-user messages to handle_message.
        """
        self._listener_task = asyncio.current_task()
        filters = {}
        if self.session_id:
            filters["session_id"] = self.session_id
        else:
            filters["chatbot_id"] = self.chatbot_id

        try:
            async for msg in self.gateway.listen(
                exclude_statuses=[MessageStatus.DELIVERED], exclude_roles=[MessageRole.USER], **filters
            ):
                await self.handle_message(msg)
        except asyncio.CancelledError:
            logger.debug(f"Chatbot '{self.chatbot_id}' listener cancelled.")

    def stop(self) -> None:
        """Stop the subscriber listener task."""
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()

    @abstractmethod
    async def handle_message(self, message: Message) -> None:
        """Process an outgoing message (e.g., tool call, thought, or final assistant text).

        Args:
            message: The Message instance to handle.
        """
        pass

    async def send_text_chunks(self, channel_id: str, chunks: list[str], message: Message) -> None:
        """Send text chunks to the specified channel."""
        pass

    async def send_file_segment(self, channel_id: str, file_path: str, message: Message) -> None:
        """Send a file to the specified channel."""
        pass

    async def send_voice_segment(self, channel_id: str, file_path: str, message: Message) -> None:
        """Send a voice message/file to the specified channel."""
        pass

    async def send_question_segment(self, channel_id: str, question: str, choices: list[str], message: Message) -> None:
        """Send a multiple choice question to the specified channel."""
        pass

    async def on_message_delivered(self, message: Message) -> None:
        """Lifecycle hook triggered after a message is successfully delivered."""
        pass

    async def pre_ingest_hook(self, dto: InboundMessageDTO) -> None:
        """Hook executed before session resolution or creation.

        Adapters can override this to perform platform-specific setup (e.g. token stores, typing).
        """
        pass

    async def pre_ingest_interruption_hook(self, session: Any, dto: InboundMessageDTO) -> None:
        """Hook executed after session is resolved/created, but before posting.

        Adapters can override this to handle thought interruption (e.g. deleting old UI cards).
        """
        pass

    async def post_ingest_hook(self, session: Any, message: Message, dto: InboundMessageDTO) -> None:
        """Hook executed after the message is successfully posted to the gateway.

        Adapters can override this to perform post-ingestion actions (e.g. adding reactions).
        """
        pass

    async def process_attachments_hook(
        self, session: Any, dto: InboundMessageDTO, raw_message: Any
    ) -> list[InboundMessageAttachment]:
        """Hook to process and save attachments using the resolved session workspace.

        Adapters should override this to download/decrypt assets and save them using AttachmentManager.
        """
        return dto.attachments

    def _format_inbound_content(self, dto: InboundMessageDTO) -> str:
        """Format the inbound message content, including attachments list if present."""
        msg_content = dto.text
        if dto.attachments:
            files_str = "\n".join(
                f"[Attachment: {a.filename} ({a.mime_type}) saved at {a.path}]" for a in dto.attachments
            )
            if msg_content:
                msg_content += f"\n\nAttachments:\n{files_str}"
            else:
                msg_content = f"Attachments:\n{files_str}"
        return msg_content

    async def _resolve_or_create_session(self, dto: InboundMessageDTO) -> tuple[Any, bool]:
        """Resolve an existing session or create a new one for the channel."""
        session = await self.gateway.get_session_by_channel(self.chatbot_id, dto.channel_id)
        created = False
        if not session:
            title = dto.session_title or f"Session: {dto.text[:30]}"
            custom_prompt = dto.custom_prompt or ""

            session = await self.gateway.create_session(
                session_id=None,
                title=title,
                custom_prompt=custom_prompt,
                chatbot_id=self.chatbot_id,
                channel_id=dto.channel_id,
                role=dto.role,
                created_at=dto.timestamp,
            )
            created = True
        else:
            await self.gateway.update_session_updated_at(session.id)

            if dto.custom_prompt:
                new_sys_prompt = build_sys_prompt(custom_prompt=dto.custom_prompt, session=session)
                await self.gateway.update_session_system_prompt(session.id, new_sys_prompt)

        return session, created

    async def ingest_message(
        self,
        dto: InboundMessageDTO,
        raw_message: Any = None,
        reply_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> bool:
        """Ingest an inbound message from the platform into the Kesoku Gateway.

        Handles:
        1. Slash command interception (if reply_callback is provided).
        2. Pre-ingest hook.
        3. Resolve or create session.
        4. Interruption hook.
        5. Process attachments (via hook).
        6. Format content.
        7. Post to Gateway.
        8. Post-ingest hook.

        Args:
            dto: Unified inbound message data transfer object.
            raw_message: Optional raw platform message payload for attachment processing.
            reply_callback: Optional async callback to reply to slash commands.

        Returns:
            True if the message was intercepted as a slash command, False otherwise.
        """
        # 1. Slash command interception
        if reply_callback and dto.text.startswith("/"):
            await self.execute_command_from_text(dto.text, reply_callback, channel_id=dto.channel_id)
            return True

        # 2. Pre-ingest hook
        await self.pre_ingest_hook(dto)

        # 3. Resolve or create session
        session, _ = await self._resolve_or_create_session(dto)

        # 4. Interruption hook
        await self.pre_ingest_interruption_hook(session, dto)

        # 5. Process attachments
        dto.attachments = await self.process_attachments_hook(session, dto, raw_message)

        # 6. Format content
        msg_content = self._format_inbound_content(dto)

        # 7. Post to Gateway
        user_msg = Message(
            session_id=session.id,
            chatbot_id=self.chatbot_id,
            channel_id=dto.channel_id,
            sender=dto.sender_id,
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content=msg_content,
            timestamp=dto.timestamp,
            status=MessageStatus.PENDING_AGENT,
            metadata={
                **dto.raw_metadata,
                "attachments": [a.model_dump() for a in dto.attachments],
            },
        )
        await self.gateway.post(user_msg)

        # 8. Post-ingest hook
        await self.post_ingest_hook(session, user_msg, dto)
        return False

    async def update_role_by_channel(self, channel_id: str, role_name: str = "") -> str:
        """Update or query the active roleplay persona for the current channel. Returns status message."""
        role_name = role_name.strip()
        cfg = get_config()

        # List available roles
        roles_dir = cfg.workspace.roles_dir

        def list_roles(path: str) -> list[str]:
            if os.path.exists(path):
                try:
                    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                except Exception as e:
                    logger.warning(f"Failed to list roles directory: {e}")
            return []

        available_roles = await asyncio.to_thread(list_roles, roles_dir)
        if not available_roles:
            available_roles = ["default"]

        if not role_name:
            # Query current role
            session = await self.gateway.get_session_by_channel(self.chatbot_id, channel_id)
            session_id = session.id if session else None
            current_role = await self.gateway.get_channel_role_with_inheritance(
                self.chatbot_id,
                channel_id,
                session_id,
            )
            return (
                f"🎭 **Active Persona:** `{current_role}`\n"
                f"✨ **Available Personas:** {', '.join(f'`{r}`' for r in sorted(available_roles))}\n"
                f"💡 Use `/role {{name}}` to switch personas."
            )

        if role_name not in available_roles:
            return (
                f"⚠️ **Error:** Persona `{role_name}` not found.\n"
                f"✨ **Available Personas:** {', '.join(f'`{r}`' for r in sorted(available_roles))}"
            )

        # 1. Update in database
        await self.gateway.set_channel_role(self.chatbot_id, channel_id, role_name)

        # 2. Rebuild the active session system prompt if a session exists
        session = await self.gateway.get_session_by_channel(self.chatbot_id, channel_id)
        if session:
            new_sys_prompt = build_sys_prompt(session=session)
            await self.gateway.update_session_system_prompt(session.id, new_sys_prompt)

        return f"🎭 Persona for this channel has been successfully changed to **`{role_name}`**."
