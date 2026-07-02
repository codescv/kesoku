"""Base class for Kesoku chatbot adapters."""

import asyncio
import datetime
import difflib
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from kesoku.agent.history import build_history
from kesoku.agent.prompt import build_sys_prompt
from kesoku.agent.tools.registry import ToolContext
from kesoku.config import get_config
from kesoku.constants import SYSTEM_START_TIME, MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.gateway.chatbot.context_reporter import ContextHtmlReporter
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger
from kesoku.utils.async_fs import async_exists, async_realpath
from kesoku.utils.path import PathResolver
from kesoku.utils.service import restart_service as utils_restart_service
from kesoku.utils.table import parse_markdown_tables, render_table_to_image
from kesoku.utils.text import format_text, split_text_into_chunks

PATH_RESOLUTION_CONFIDENCE_THRESHOLD = 0.9
"""Similarity score threshold for auto path resolution of misspelled absolute paths."""

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
    role: str | None = None


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

        async def handle_restart(reply_func: Callable[[str], Awaitable[None]], **kwargs: Any) -> None:
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

        async def handle_context(reply_func: Callable[..., Awaitable[None]], channel_id: str) -> None:
            res = await self.get_session_active_context_by_channel(channel_id)
            if await async_exists(res):
                await reply_func("📖 Here is your beautifully formatted Active Context HTML download:", file_path=res)
            else:
                await reply_func(res)

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
        self.commands.register(
            "context",
            "View the currently active prompt context (what the LLM sees).",
            handle_context,
        )

        async def handle_grep(
            reply_func: Callable[..., Awaitable[None]],
            channel_id: str,
            query: str = "",
        ) -> None:
            if not query:
                await reply_func("⚠️ Please provide search keywords (e.g., /grep keyword).")
                return

            session = await self.gateway.db.get_session_by_channel(self.chatbot_id, channel_id)
            if not session:
                session = await self.gateway.create_session(
                    session_id=None,
                    title=f"Grep {channel_id}",
                    chatbot_id=self.chatbot_id,
                    channel_id=channel_id,
                )

            from kesoku.agent.tools.memory import memory_grep

            ctx = ToolContext(
                session_id=session.id,
                session_workspace=session.workspace_name,
                gateway=self.gateway,
            )
            try:
                res = await memory_grep(query=query, context=ctx)
                await reply_func(res)
            except Exception as e:
                logger.error(f"Failed grep execution: {e}")
                await reply_func(f"⚠️ Failed to execute grep: {e}")

        self.commands.register(
            "grep",
            "Search active memories and past messages for the current bound role.",
            handle_grep,
        )
        self.commands.register(
            "memory-grep",
            "Search active memories and past messages for the current bound role.",
            handle_grep,
        )


        async def handle_debug(
            reply_func: Callable[[str], Awaitable[None]],
            channel_id: str,
        ) -> None:
            cfg = get_config()
            cfg.agent.raw_llm_logs = not cfg.agent.raw_llm_logs
            if cfg.agent.raw_llm_logs:
                session = await self.gateway.db.get_session_by_channel(self.chatbot_id, channel_id)
                if session:
                    staging_dir = self.get_session_staging_dir(session.workspace_name)
                    await reply_func(f"🐞 Debug mode enabled.\nraw_llm_logs = True\nStaging dir: `{staging_dir}`")
                else:
                    await reply_func(
                        "🐞 Debug mode enabled.\nraw_llm_logs = True\n⚠️ No active session found to resolve staging dir."
                    )
            else:
                await reply_func("🐞 Debug mode disabled.\nraw_llm_logs = False")

        self.commands.register(
            "debug",
            "Toggle debug mode (raw LLM logs and staging directory visibility).",
            handle_debug,
        )

        async def handle_cronjob(
            reply_func: Callable[[str], Awaitable[None]],
            tag: str = "",
            **kwargs: Any,
        ) -> None:
            if tag:
                status_msg = await self.trigger_cronjobs_by_tag(tag)
            else:
                status_msg = await self.list_cronjobs()
            await reply_func(status_msg)

        self.commands.register(
            "cronjob",
            "List all cronjobs or trigger them by tag immediately.",
            handle_cronjob,
        )

    async def restart_service(self) -> None:
        """Restart the Kesoku service."""
        await utils_restart_service(self.chatbot_id, self.stop)

    async def list_cronjobs(self) -> str:
        """List all configured cronjobs."""
        cron_mgr = getattr(self.gateway, "cron_manager", None)
        if not cron_mgr:
            return "⚠️ CronManager is not initialized or active."

        try:
            jobs = cron_mgr.get_all_jobs()
            if not jobs:
                return "ℹ️ No cronjobs configured."

            lines = ["📋 **Configured Cronjobs:**"]
            for idx, job in enumerate(jobs):
                tag_str = f" | tag: `{job.get('tag')}`" if job.get("tag") else ""
                lines.append(
                    f"  {idx + 1}. **{job.get('chatbot_id')}** -> `{job.get('prompt')}`\n"
                    f"     schedule: `{job.get('schedule')}`{tag_str} | channel: `{job.get('channel_id') or 'auto'}`"
                )
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Failed to list cronjobs: {e}", exc_info=True)
            return f"⚠️ Failed to list cronjobs: {e}"

    async def trigger_cronjobs_by_tag(self, tag: str) -> str:
        """Trigger cronjobs matching the given tag immediately."""
        cron_mgr = getattr(self.gateway, "cron_manager", None)
        if not cron_mgr:
            return "⚠️ CronManager is not initialized or active."

        try:
            count = await cron_mgr.trigger_jobs_by_tag(tag)
            if count > 0:
                return f"🚀 Successfully triggered {count} cronjob(s) with tag `{tag}`."
            else:
                return f"⚠️ No cronjobs found matching tag `{tag}`."
        except Exception as e:
            logger.error(f"Failed to trigger cronjobs for tag {tag}: {e}", exc_info=True)
            return f"⚠️ Failed to trigger cronjobs: {e}"

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
            if command in {"clear", "reset", "status", "compact", "debug"}:
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
            elif command in {"grep", "memory-grep", "memory_grep"}:
                if not channel_id:
                    await reply_func("⚠️ Channel ID is required for this command.")
                    return
                query = " ".join(parts[1:]) if len(parts) > 1 else ""
                await self.commands.execute("grep", reply_func, channel_id=channel_id, query=query)

            elif command == "cronjob":
                tag = " ".join(parts[1:]) if len(parts) > 1 else ""
                await self.commands.execute("cronjob", reply_func, tag=tag)
            else:
                await reply_func(f"⚠️ Unrecognized command: /{command}")
        except Exception as e:
            logger.error(f"Command /{command} execution failed: {e}", exc_info=True)
            await reply_func(f"⚠️ Failed to execute command: {e}")

    async def clear_session_by_channel(self, channel_id: str) -> str:
        """Unbind the active session for the channel, stop its workers/jobs, and start a new session.

        Preserves the old session data in the database.
        """
        session = await self.gateway.db.get_session_by_channel(self.chatbot_id, channel_id)
        if session:
            logger.info(f"Chatbot '{self.chatbot_id}' unbinding and stopping session '{session.id}'.")

            # 1. Stop worker
            agent = self.gateway.agent
            if agent:
                await agent.stop_session_worker(session.id, immediate=True)

            # 2. Stop background jobs
            try:
                await self.gateway.context.active_jobs.stop_all_for_session(session.id)
            except Exception as e:
                logger.warning(f"Failed to clean up background jobs in clear_session_by_channel: {e}")

            # 3. Create and bind new session (overwrites active mapping)
            new_session = await self.gateway.create_session(
                session_id=None,
                title="New Session",
                chatbot_id=self.chatbot_id,
                channel_id=channel_id,
            )
            return f"♻️ New session '{new_session.id}' started. Old session '{session.id}' preserved."
        return "⚠️ No active session found for this chat."

    async def manual_compact_session_by_channel(self, channel_id: str) -> str:
        """Manually trigger context compaction on the active history of this channel."""
        session = await self.gateway.db.get_session_by_channel(self.chatbot_id, channel_id)
        if not session:
            return "⚠️ No active session found for this chat."

        # Fetch active history
        history = await build_history(self.gateway, session.id, heal_orphans=False)
        if not history:
            return "⚠️ Active session has no messages to compact."

        from kesoku.agent.compressor import HistoryCompressor
        from kesoku.agent.llm import get_llm

        compressor = HistoryCompressor(self.gateway.db)
        cfg = self.gateway.context.config
        llm = get_llm(provider=cfg.agent.llm, config=cfg)

        try:
            # Trigger compaction
            compacted = await compressor.auto_compact_session(
                session_id=session.id,
                history=history,
                llm=llm,
                config=cfg,
            )
            if compacted:
                return "🔄 Context Compaction completed successfully! Old turns have been compacted into summary nodes."
            else:
                return "ℹ️ Context compaction is not needed right now: turns do not meet threshold limits."
        except Exception as e:
            logger.error(f"Failed manual compaction for session {session.id}: {e}")
            return f"⚠️ Failed to compact history: {e}"

    async def get_session_status_by_channel(self, channel_id: str) -> str:
        """Get session statistics for the channel."""
        session = await self.gateway.db.get_session_by_channel(self.chatbot_id, channel_id)
        if not session:
            return "⚠️ No active session found for this chat."

        history = await self.gateway.db.get_session_history(session.id, limit=100)
        metrics = None
        for msg in reversed(history):
            if msg.role == MessageRole.ASSISTANT and msg.metadata and msg.metadata.get("turn_metrics"):
                metrics = msg.metadata.get("turn_metrics")
                break

        session_turns = await self.gateway.db.get_session_turns_count(session.id)
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

    async def get_session_active_context_by_channel(self, channel_id: str) -> str:
        """Get the currently active assembled prompt context (what the LLM sees) for the channel."""
        session = await self.gateway.db.get_session_by_channel(self.chatbot_id, channel_id)
        if not session:
            return "⚠️ No active session found for this chat."

        history = await build_history(self.gateway, session.id, heal_orphans=False)
        if not history:
            return "⚠️ Active session has no messages."

        from kesoku.agent.compressor import HistoryCompressor

        compressor = HistoryCompressor(self.gateway.db)
        cfg = self.gateway.context.config

        try:
            # Segment turns
            turns = compressor.segment_turns(history)
            protect_front = cfg.agent.protect_front_turns
            protect_tail = cfg.agent.protect_tail_turns

            # Assemble Protected Head
            protected_head = []
            for t in turns[:protect_front]:
                protected_head.extend(t)

            # Assemble Protected Tail
            protected_tail = []
            if len(turns) > protect_front:
                tail_start = max(protect_front, len(turns) - protect_tail)
                for t in turns[tail_start:]:
                    protected_tail.extend(t)

            # Assemble Buffer
            buffer = []
            if len(turns) > protect_front + protect_tail:
                middle_turns = turns[protect_front:-protect_tail]
                for t in middle_turns:
                    if not any(msg.summary_node_id is not None for msg in t):
                        buffer.extend(t)

            # Query summaries
            root_summaries = await self.gateway.db.get_root_summary_nodes(session.id)
            all_summaries = await self.gateway.db.get_all_summary_nodes(session.id)

            sys_msg = session.system_prompt or ""

            # Extract metrics of the last execution turn
            last_metrics = None
            for m in reversed(history):
                if m.role == MessageRole.ASSISTANT and m.metadata and m.metadata.get("turn_metrics"):
                    last_metrics = m.metadata.get("turn_metrics")
                    break

            return ContextHtmlReporter.render_to_temp_file(
                session=session,
                root_summaries=root_summaries,
                all_summaries=all_summaries,
                protected_head=protected_head,
                buffer=buffer,
                protected_tail=protected_tail,
                sys_msg=sys_msg,
                last_metrics=last_metrics,
            )
        except Exception as e:
            logger.error(f"Failed to get active context by channel: {e}")
            return f"⚠️ Failed to retrieve active context: {e}"

    async def trigger_cronjob_message(
        self,
        channel_id: str,
        prompt_content: str,
        sender_name: str = "Cronjob",
        custom_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
        title: str | None = None,
        tag: str | None = None,
        role: str | None = None,
    ) -> Message:
        """Unified helper to create a session (if not exists) and post a scheduled cronjob message to the gateway.

        Returns:
            The posted Message instance.
        """
        session = await self.gateway.db.get_session_by_channel(self.chatbot_id, channel_id)
        if not session:
            if tag:
                session_title = f"Cronjob {self.chatbot_id} {tag}"
            else:
                session_title = title or f"{self.chatbot_id.capitalize()} Scheduled Job {channel_id}"

            # Resolve role with parent inheritance support
            if not role:
                db_role = await self.gateway.db.get_channel_role(self.chatbot_id, channel_id)
                if isinstance(db_role, str):
                    role = db_role
                elif metadata and "parent_channel_id" in metadata:
                    parent_id = metadata["parent_channel_id"]
                    parent_role = await self.gateway.db.get_channel_role(self.chatbot_id, str(parent_id))
                    if isinstance(parent_role, str):
                        role = parent_role
            if not role:
                role = "default"

            session = await self.gateway.create_session(
                session_id=None,
                title=session_title,
                custom_prompt=custom_prompt,
                chatbot_id=self.chatbot_id,
                channel_id=channel_id,
                role=role,
            )
        else:
            await self.gateway.db.update_session_updated_at(session.id, time.time())

        now_dt = datetime.datetime.now()
        msg_content = prompt_content

        msg_metadata = {"is_cronjob": True}
        if tag:
            msg_metadata["tag"] = tag
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
        return PathResolver.get_session_staging_dir(workspace_name)

    async def resolve_outbound_path(self, raw_path: str, session_id: str) -> str:
        """Resolve a potentially misspelled absolute path by fuzzy matching against files in STAGING_DIR.

        If the path exists exactly as given, it is returned immediately.
        Otherwise, it searches all files recursively in the session's staging directory,
        calculating a similarity score that comprehensively considers:
        - The match score of the filename (60% weight)
        - The match score of the full absolute path (40% weight)

        If a strong match (score >= PATH_RESOLUTION_CONFIDENCE_THRESHOLD) is found, the corrected path is returned.

        Args:
            raw_path: The raw absolute file path written by the agent.
            session_id: Active session ID context.

        Returns:
            The resolved corrected absolute path, or the original path if not matched.
        """
        # Clean up raw path
        cleaned_path = raw_path.strip()
        if not cleaned_path:
            return raw_path

        # 1. Check if the exact path exists
        if os.path.isabs(cleaned_path) and await async_exists(cleaned_path):
            return await async_realpath(cleaned_path)

        # Try to resolve relative to AWD (Agent Working Directory)
        if not os.path.isabs(cleaned_path):
            candidate_path = PathResolver.resolve(cleaned_path)
            if await async_exists(candidate_path):
                return await async_realpath(candidate_path)

        # 2. Get session staging directory
        staging_dir = None
        session = await self.gateway.db.get_session(session_id)
        if session:
            staging_dir = self.get_session_staging_dir(session.workspace_name)

        # If it's a relative path, try to resolve it against staging_dir first
        if staging_dir and not os.path.isabs(cleaned_path):
            candidate_path = os.path.join(staging_dir, cleaned_path)
            if await async_exists(candidate_path):
                return await async_realpath(candidate_path)

        # 3. Fuzzy matching inside the session's staging directory
        if staging_dir and await async_exists(staging_dir):
            try:
                # List all files in staging directory with their real absolute paths
                def _list_staging_files_abs(s_dir: str) -> list[str]:
                    files = []
                    for root, _, filenames in os.walk(s_dir):
                        for f in filenames:
                            files.append(os.path.realpath(os.path.join(root, f)))
                    return files

                abs_staging_files = await asyncio.to_thread(_list_staging_files_abs, staging_dir)
                if not abs_staging_files:
                    return raw_path

                # Resolve raw_abs_path relative to staging_dir if it's relative
                if not os.path.isabs(cleaned_path):
                    raw_abs_path = os.path.join(staging_dir, cleaned_path)
                else:
                    raw_abs_path = cleaned_path
                raw_abs_path = await async_realpath(raw_abs_path)
                raw_filename = os.path.basename(raw_abs_path)

                best_candidate = None
                best_score = 0.0

                for candidate in abs_staging_files:
                    candidate_filename = os.path.basename(candidate)

                    # 60% weight for filename similarity, 40% weight for full path similarity
                    fn_ratio = difflib.SequenceMatcher(None, raw_filename, candidate_filename).ratio()
                    path_ratio = difflib.SequenceMatcher(None, raw_abs_path, candidate).ratio()

                    score = 0.6 * fn_ratio + 0.4 * path_ratio
                    if score > best_score:
                        best_score = score
                        best_candidate = candidate

                # If we found a high confidence match
                if best_candidate and best_score >= PATH_RESOLUTION_CONFIDENCE_THRESHOLD:
                    logger.warning(
                        f"Fuzzy matched misspelled path '{raw_path}' (score={best_score:.3f}) to: {best_candidate}"
                    )
                    return best_candidate
            except Exception as e:
                logger.warning(f"Failed during fuzzy path resolution: {e}")

        # Fallback to original
        return raw_path

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
        await self.gateway.db.update_message_status(message.id, MessageStatus.DELIVERED)

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
                    await self.gateway.db.update_message_status(message.id, MessageStatus.DELIVERED)
                    return

                # Handle intermediate/special message (thought, tool, system)
                await self.handle_intermediate_message(message)
                return

            # 2. Handle Tool Results (updating tool call status to checkmark or error emoji)
            if message.role == MessageRole.TOOL and message.type != MessageType.TOOL_CALL:
                await self.handle_tool_result(message)
                return

            # 3. Preprocess markdown tables: render to image files and replace with file tags
            from kesoku.utils.async_fs import async_write_binary_file

            tables = parse_markdown_tables(message.content)
            if tables:
                session = await self.gateway.db.get_session(message.session_id)
                if session:
                    staging_dir = self.get_session_staging_dir(session.workspace_name)
                    content = message.content
                    for table in reversed(tables):
                        try:
                            png_bytes = render_table_to_image(
                                headers=table.headers,
                                alignments=table.alignments,
                                rows=table.rows,
                            )
                            img_filename = f"table_{uuid.uuid4().hex[:8]}.png"
                            img_path = os.path.join(staging_dir, img_filename)
                            await async_write_binary_file(img_path, png_bytes)

                            file_tag = f"\n[file: {img_path}]\n"
                            content = content[: table.start_idx] + file_tag + content[table.end_idx :]
                        except Exception as re:
                            logger.error(f"Failed to render markdown table to image: {re}", exc_info=True)
                    message.content = content

            # 4. Parse message content to extract text, file, voice, or question segments
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
                    resolved_path = await self.resolve_outbound_path(segment["path"], message.session_id)
                    await self.send_file_segment(message.channel_id, resolved_path, message)

                elif segment["type"] == "voice":
                    resolved_path = await self.resolve_outbound_path(segment["path"], message.session_id)
                    await self.send_voice_segment(message.channel_id, resolved_path, message)

                elif segment["type"] == "question":
                    await self.send_question_segment(
                        message.channel_id, segment["question"], segment["choices"], message
                    )

            # 4. Update Gateway status to DELIVERED
            await self.gateway.db.update_message_status(message.id, MessageStatus.DELIVERED)

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
                exclude_statuses=[
                    MessageStatus.DELIVERED,
                    MessageStatus.PENDING_AGENT,
                    MessageStatus.PROCESSING,
                ],
                exclude_roles=[MessageRole.USER],
                **filters,
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
        session = await self.gateway.db.get_session_by_channel(self.chatbot_id, dto.channel_id)
        created = False

        current_role = dto.role
        if current_role is None:
            current_role = await self.gateway.db.get_channel_role_with_inheritance(
                self.chatbot_id, dto.channel_id, session.id if session else None
            )

        if session:
            if session.role_name != current_role:
                agent = self.gateway.agent
                if agent:
                    await agent.stop_session_worker(session.id, immediate=True)
                try:
                    await self.gateway.context.active_jobs.stop_all_for_session(session.id)
                except Exception as e:
                    logger.warning(f"Failed to clean up background jobs when switching session role: {e}")

                last_sess = await self.gateway.db.get_last_session_by_channel_and_role(
                    self.chatbot_id, dto.channel_id, current_role
                )
                if last_sess:
                    await self.gateway.db.set_active_session_for_channel(self.chatbot_id, dto.channel_id, last_sess.id)
                    session = last_sess
                else:
                    session = None

        if not session:
            title = dto.session_title or f"Session: {dto.text[:30]}"
            custom_prompt = dto.custom_prompt or ""

            session = await self.gateway.create_session(
                session_id=None,
                title=title,
                custom_prompt=custom_prompt,
                chatbot_id=self.chatbot_id,
                channel_id=dto.channel_id,
                role=current_role,
                created_at=dto.timestamp,
            )
            created = True
        else:
            await self.gateway.db.update_session_updated_at(session.id, time.time())

            # Always rebuild the system prompt to pick up on-disk template changes
            new_sys_prompt = build_sys_prompt(custom_prompt=dto.custom_prompt or None, session=session)
            await self.gateway.db.update_session_system_prompt(session.id, new_sys_prompt)

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
            session = await self.gateway.db.get_session_by_channel(self.chatbot_id, channel_id)
            session_id = session.id if session else None
            current_role = await self.gateway.db.get_channel_role_with_inheritance(
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
        await self.gateway.db.set_channel_role(self.chatbot_id, channel_id, role_name)

        # 2. Rebuild the active session system prompt if a session exists
        old_session = await self.gateway.db.get_session_by_channel(self.chatbot_id, channel_id)
        if old_session:
            agent = self.gateway.agent
            if agent:
                await agent.stop_session_worker(old_session.id, immediate=True)
            try:
                await self.gateway.context.active_jobs.stop_all_for_session(old_session.id)
            except Exception as e:
                logger.warning(f"Failed to clean up background jobs when switching role: {e}")

        session = await self.gateway.db.get_last_session_by_channel_and_role(self.chatbot_id, channel_id, role_name)
        if session:
            await self.gateway.db.set_active_session_for_channel(self.chatbot_id, channel_id, session.id)
            new_sys_prompt = build_sys_prompt(session=session)
            await self.gateway.db.update_session_system_prompt(session.id, new_sys_prompt)
        else:
            await self.gateway.create_session(
                session_id=None,
                title=f"{role_name.capitalize()} Session",
                chatbot_id=self.chatbot_id,
                channel_id=channel_id,
                role=role_name,
            )

        return f"🎭 Persona for this channel has been successfully changed to **`{role_name}`**."
