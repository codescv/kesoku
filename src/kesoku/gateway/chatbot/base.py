"""Base class for Kesoku chatbot adapters."""

import asyncio
import datetime
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from kesoku.config import get_config
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger

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
            segments.append({
                "type": "question",
                "question": question_text,
                "choices": choices,
            })
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
        import tzlocal
        return tzlocal.get_localzone().key or "UTC"
    except Exception:
        return datetime.datetime.now().astimezone().tzname() or "UTC"


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

        self.commands.register("restart", "Restart the Kesoku service.", handle_restart)
        self.commands.register("clear", "Clear the active conversation session.", handle_clear)
        self.commands.register("reset", "Clear the active conversation session.", handle_clear)
        self.commands.register("status", "Get conversation and performance statistics.", handle_status)

    def _get_kesoku_executable(self) -> str:
        import shutil
        import sys
        executable_dir = os.path.dirname(sys.executable)
        kesoku_path = os.path.join(executable_dir, "kesoku")
        if os.path.exists(kesoku_path):
            return kesoku_path
        return shutil.which("kesoku") or "kesoku"

    async def restart_service(self) -> None:
        """Restart the Kesoku service."""
        logger.info(f"Chatbot '{self.chatbot_id}' requesting service restart.")
        self.stop()

        import subprocess
        import sys

        kesoku_bin = self._get_kesoku_executable()
        cmd = [kesoku_bin, "service", "restart"]

        service_user = os.environ.get("KESOKU_SERVICE_USER", "true") == "true"
        if service_user:
            cmd.append("--user")
        else:
            cmd.append("--system")

        instance_name = os.environ.get("KESOKU_SERVICE_INSTANCE_NAME")
        if instance_name:
            cmd.extend(["--name", instance_name])

        logger.info(f"Launching restart command: {' '.join(cmd)}")
        try:
            subprocess.Popen(cmd, start_new_session=True)  # noqa: ASYNC220
            logger.info("Successfully launched kesoku service restart command.")
        except Exception as e:
            logger.error(f"Failed to run restart command: {e}")
            # Fallback to in-place os.execv restart
            logger.info("Falling back to in-place os.execv restart...")
            try:
                sys.stdout.flush()
                sys.stderr.flush()
                os.execv(sys.executable, [sys.executable] + sys.argv)
            except Exception as fallback_error:
                logger.error(f"In-place fallback restart failed: {fallback_error}")
                raise e

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
        turn_tool_calls = (
            metrics.get("turn_tool_calls", 0)
            if metrics
            else len([m for m in history if m.role == MessageRole.TOOL and m.type == MessageType.TOOL_CALL])
        )
        turn_tokens = metrics.get("turn_tokens", 0) if metrics else 0
        turn_time = metrics.get("turn_time", 0.0) if metrics else 0.0

        context_k = f"{round(context_tokens / 1000)}K" if context_tokens else "0K"
        turn_k = f"{round(turn_tokens / 1000)}K" if turn_tokens else "0K"

        return (
            f"【Current Stats】\n"
            f"⚡ Session: {session_turns} turns\n"
            f"📖 Context: {context_k} tokens\n"
            f"⏱️ Last Turn:\n"
            f"  - Tool Calls: {turn_tool_calls}\n"
            f"  - Tokens: {turn_k}\n"
            f"  - Time: {turn_time:.1f}s"
        )

    async def trigger_cronjob_message(
        self,
        channel_id: str,
        prompt_content: str,
        sender_name: str = "System",
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
            )
        else:
            await self.gateway.update_session_updated_at(session.id)

        now_dt = datetime.datetime.now()
        tz_name = get_local_timezone_name()
        now_str = now_dt.strftime('%Y-%m-%d %H:%M:%S')

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
        """Format/normalize markdown or lines before chunking. Default returns unmodified."""
        return text

    def get_max_text_length(self) -> int:
        """Maximum text length allowed per message chunk for the platform."""
        return 2000

    def split_text_into_chunks(self, text: str, max_length: int) -> list[str]:
        """Generic helper to split text into chunks of at most max_length.

        Default uses standard line-aware chunking.
        """
        if len(text) <= max_length:
            return [text]

        chunks = []
        lines = text.splitlines(keepends=True)
        current_chunk = ""

        for line in lines:
            if len(line) > max_length:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                # Force split extremely long line
                for i in range(0, len(line), max_length):
                    chunks.append(line[i : i + max_length])
            elif len(current_chunk) + len(line) > max_length:
                chunks.append(current_chunk)
                current_chunk = line
            else:
                current_chunk += line

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

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
