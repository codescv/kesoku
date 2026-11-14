"""Base class for Kesoku chatbot adapters."""

import asyncio
import copy
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

from kesoku.agent.context_dir import build_context_prefix
from kesoku.agent.history import build_history
from kesoku.agent.prompt import build_sys_prompt
from kesoku.agent.tools.registry import ToolContext
from kesoku.config import get_config, resolve_gateway_default_role
from kesoku.constants import SYSTEM_START_TIME, MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.gateway.chatbot.context_reporter import ContextHtmlReporter
from kesoku.gateway.chatbot.deferred_media import (
    DEAD_WORKER_GRACE_SEC,
    RECOVERY_LOOKBACK_SEC,
    clamp_timeout_sec,
    format_deferred_failure_message,
    is_deferred_sidecar_name,
    is_deferred_target_ready,
    is_pid_alive,
    read_json_dict,
    resolve_deferred_sync,
    safe_unlink,
    sanitize_error_reason,
    terminate_deferred_worker,
    unlink_if_job_matches,
    wait_for_deferred_file,
)
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger
from kesoku.utils.async_fs import (
    async_exists,
    async_read_text_file,
    async_realpath,
    async_write_text_file,
)
from kesoku.utils.path import PathResolver
from kesoku.utils.service import restart_service as utils_restart_service
from kesoku.utils.table import parse_markdown_tables, render_table_to_image
from kesoku.utils.text import format_text, render_spoilers, split_text_into_chunks
from kesoku.utils.ui_markup import (
    DETAILS_RE,
    RICH_BLOCK_RE,
    UI_BLOCK_RE,
    find_first_ui_block,
    flatten_rich_blocks,
    parse_rich_block,
    render_details_as_text,
    render_question_as_text,
)

PATH_RESOLUTION_CONFIDENCE_THRESHOLD = 0.9
"""Similarity score threshold for auto path resolution of misspelled absolute paths."""

logger = setup_logger(__name__)


_LEGACY_TAG_RE = re.compile(r"\[(file|voice|question):\s*([^\]]+)\s*\]")


def _legacy_tag_segment(match: re.Match[str]) -> dict[str, Any]:
    """Build a segment from a `[file:]`, `[voice:]` or legacy `[question:]` tag match."""
    block_type = match.group(1)
    inner_val = match.group(2).strip()
    if block_type != "question":
        return {"type": block_type, "path": inner_val}
    if "||" in inner_val:
        q_part, choices_part = inner_val.split("||", 1)
        question_text = q_part.strip()
        choices = [c.strip() for c in choices_part.split("|") if c.strip()]
    else:
        parts = [p.strip() for p in inner_val.split("|")]
        question_text = parts[0]
        choices = parts[1:]
    return {"type": "question", "question": question_text, "choices": choices, "multi": False, "max": 1}


def parse_message_content(content: str, rich_blocks: bool = False) -> list[dict[str, Any]]:
    """Parse message content to extract zero or more blocks.

    Recognizes:
    * the first fenced ```ui block (YAML `type: select|multiselect`, `prompt`, `options`, optional `max`);
      further ```ui blocks, and invalid ones, stay plain text;
    * `<details><summary>TITLE</summary>BODY</details>` collapsible sections;
    * `[file: /path]`, `[voice: /path]` and the legacy `[question: <question> || choice1 | ...]`;
    * fenced ```bilingual and ```card YAML blocks: returned as structured segments when `rich_blocks` is True,
      otherwise replaced by a clean readable plain-text fallback (so platforms without rich rendering never
      show raw YAML; media tags inside the fallback are still delivered as attachments).

    Args:
        content: Raw message text content to parse.
        rich_blocks: Whether the caller renders `bilingual` / `card` segments natively.

    Returns:
        A list of segment dictionaries:
        {"type": "text", "content": "..."}, {"type": "file", "path": "..."}, {"type": "voice", "path": "..."},
        {"type": "question", "question": "...", "choices": [...], "multi": bool, "max": int},
        {"type": "details", "title": "...", "body": "..."} and (rich mode only)
        {"type": "bilingual", ...} / {"type": "card", "kind": "...", "data": {...}}.
    """
    if not rich_blocks:
        content = flatten_rich_blocks(content)

    # Collect candidate spans from every syntax, then keep non-overlapping ones by start position, so a tag
    # inside a ```ui block or a <details> body is treated as part of that block.
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    rich_masks: list[tuple[int, int]] = []
    if rich_blocks and "```" in content:
        for m in RICH_BLOCK_RE.finditer(content):
            rich_segment = parse_rich_block(m.group("kind"), m.group("body"))
            if rich_segment is None:
                logger.warning(f"Invalid ```{m.group('kind')} block (bad YAML or missing fields); rendering as text")
                continue
            candidates.append((m.start(), m.end(), rich_segment))
            rich_masks.append((m.start(), m.end()))

    ui_match, ui_question = find_first_ui_block(content)
    if ui_match is not None:
        if ui_question is not None:
            candidates.append((ui_match.start(), ui_match.end(), ui_question.to_segment()))
        else:
            logger.warning("Invalid ```ui block (bad YAML or missing type/prompt/options); rendering as text")

    for m in DETAILS_RE.finditer(content):
        details = {"type": "details", "title": m.group("title").strip(), "body": m.group("body").strip()}
        candidates.append((m.start(), m.end(), details))

    for m in _LEGACY_TAG_RE.finditer(content):
        candidates.append((m.start(), m.end(), _legacy_tag_segment(m)))

    # Any ```ui block (including non-interactive ones) masks tags inside it.
    masks = [(m.start(), m.end()) for m in UI_BLOCK_RE.finditer(content)] if "```ui" in content else []
    masks.extend(rich_masks)

    segments: list[dict[str, Any]] = []
    last_idx = 0
    for start, end, segment in sorted(candidates, key=lambda c: c[0]):
        if start < last_idx:
            continue
        if any(ms <= start < me and (ms, me) != (start, end) for ms, me in masks):
            continue
        text_before = content[last_idx:start]
        if text_before:
            segments.append({"type": "text", "content": text_before})
        segments.append(segment)
        last_idx = end

    text_after = content[last_idx:]
    if text_after:
        segments.append({"type": "text", "content": text_after})

    return segments


class DeliveryAbortedError(Exception):
    """Exception raised to immediately halt outgoing message delivery."""

    pass


class ChannelUnavailableError(DeliveryAbortedError):
    """Raised when the target channel or DM is permanently unavailable (deleted or forbidden)."""

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

    supports_deferred_media: bool = False
    stream_min_interval_sec: float = 1.5
    _deferred_poll_initial_sec: float = 0.5
    _deferred_poll_max_sec: float = 2.0
    _deferred_dead_grace_sec: float = DEAD_WORKER_GRACE_SEC
    _voice_inline_wait_sec: float = 60.0
    _voice_inline_anchor_delay_sec: float = 1.0

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
        self._recovery_task: asyncio.Task[None] | None = None
        self._deferred_tasks: dict[str, set[asyncio.Task[None]]] = {}
        self._deferred_watched_keys: set[tuple[str, int]] = set()
        self._metadata_locks: dict[str, asyncio.Lock] = {}
        self._deferred_messages: dict[str, Message] = {}
        self._register_gateway_capabilities()

        self.commands = CommandRegistry()
        self._register_default_commands()

    def _register_gateway_capabilities(self) -> None:
        """Register this chatbot instance and its deferred-media capability on the Gateway."""
        register_bot_fn = getattr(self.gateway, "register_chatbot", None)
        if callable(register_bot_fn):
            register_bot_fn(self)
        register_deferred_fn = getattr(self.gateway, "register_deferred_media", None)
        if callable(register_deferred_fn):
            register_deferred_fn(self.chatbot_id, bool(self.supports_deferred_media))
        register_interval_fn = getattr(self.gateway, "register_stream_min_interval", None)
        if callable(register_interval_fn):
            register_interval_fn(self.chatbot_id, float(self.stream_min_interval_sec))

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

        async def handle_compact(
            reply_func: Callable[[str], Awaitable[None]], channel_id: str, instructions: str = ""
        ) -> None:
            status_msg = await self.manual_compact_session_by_channel(channel_id, instructions)
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
        self.commands.register(
            "compact",
            "Compact history now; optional focus, e.g. /compact keep only the API decisions",
            handle_compact,
        )
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

            from kesoku.agent.tools.memory import memory_search

            ctx = ToolContext(
                session_id=session.id,
                session_workspace=session.workspace_name,
                gateway=self.gateway,
            )
            try:
                res = await memory_search(query=query, context=ctx)
                await reply_func(res)
            except Exception as e:
                logger.error(f"Failed grep execution: {e}")
                await reply_func(f"⚠️ Failed to execute grep: {e}")

        self.commands.register(
            "grep",
            "Search role Markdown memories ($MEMORY_DIR) for the current bound role.",
            handle_grep,
        )
        self.commands.register(
            "chat-grep",
            "Search role Markdown memories ($MEMORY_DIR) for the current bound role.",
            handle_grep,
        )

        async def handle_search(
            reply_func: Callable[..., Awaitable[None]],
            channel_id: str,
            query: str = "",
        ) -> None:
            if not query:
                await reply_func("⚠️ Please provide search query (e.g., /search keyword).")
                return

            session = await self.gateway.db.get_session_by_channel(self.chatbot_id, channel_id)
            if not session:
                session = await self.gateway.create_session(
                    session_id=None,
                    title=f"Search {channel_id}",
                    chatbot_id=self.chatbot_id,
                    channel_id=channel_id,
                )

            from kesoku.agent.tools.memory import memory_search

            ctx = ToolContext(
                session_id=session.id,
                session_workspace=session.workspace_name,
                gateway=self.gateway,
            )
            try:
                res = await memory_search(query=query, context=ctx)
                await reply_func(res)
            except Exception as e:
                logger.error(f"Failed search execution: {e}")
                await reply_func(f"⚠️ Failed to execute search: {e}")

        self.commands.register(
            "search",
            "Perform semantic search on role Markdown memories ($MEMORY_DIR) for the current bound role.",
            handle_search,
        )
        self.commands.register(
            "chat-search",
            "Perform semantic search on role Markdown memories ($MEMORY_DIR) for the current bound role.",
            handle_search,
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

        async def handle_cron(
            reply_func: Callable[[str], Awaitable[None]],
            channel_id: str,
            **kwargs: Any,
        ) -> None:
            status_msg = await self.trigger_cronjobs_by_channel(channel_id)
            await reply_func(status_msg)

        self.commands.register(
            "cron",
            "Immediately trigger the cronjob(s) bound to the current channel.",
            handle_cron,
        )

        async def handle_notes(
            reply_func: Callable[[str], Awaitable[None]],
            channel_id: str,
            content: str = "",
        ) -> None:
            status_msg = await self.manage_session_notes_by_channel(channel_id, content)
            await reply_func(status_msg)

        self.commands.register(
            "notes",
            "View, update, or clear pinned session notes (/notes, /notes <text>, /notes clear).",
            handle_notes,
        )

    async def manage_session_notes_by_channel(self, channel_id: str, content: str = "") -> str:
        """View, update, or clear pinned session notes (notes.md) for the active session of a channel."""
        session = await self.gateway.db.get_session_by_channel(self.chatbot_id, channel_id)
        if not session:
            return "⚠️ No active session found for this chat."

        staging_dir = self.get_session_staging_dir(session.workspace_name)
        notes_path = os.path.join(staging_dir, "notes.md")
        cleaned = content.strip()

        if not cleaned:
            if await async_exists(notes_path):
                existing = (await async_read_text_file(notes_path)).strip()
                if existing:
                    return f"📌 **Pinned Session Notes:**\n\n{existing}"
            return "ℹ️ No pinned notes for the current session."

        if cleaned.lower() == "clear":
            if await async_exists(notes_path):
                await async_write_text_file(notes_path, "")
            return "🧹 Cleared pinned session notes."

        os.makedirs(staging_dir, exist_ok=True)
        await async_write_text_file(notes_path, cleaned)
        return f"📌 **Updated Pinned Session Notes:**\n\n{cleaned}"

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

    async def get_cron_channel_candidates(self, channel_id: str) -> set[str]:
        """Return channel IDs whose cronjobs should be considered bound to `channel_id`.

        Default is the channel itself. Adapters with threads (e.g. Discord) should
        also include the parent channel, since cronjobs are configured on the parent.
        """
        return {str(channel_id)}

    async def trigger_cronjobs_by_channel(self, channel_id: str) -> str:
        """Trigger cronjobs bound to this chatbot and the given channel immediately."""
        cron_mgr = getattr(self.gateway, "cron_manager", None)
        if not cron_mgr:
            return "⚠️ CronManager is not initialized or active."

        try:
            candidates = await self.get_cron_channel_candidates(channel_id)
            jobs = await cron_mgr.trigger_jobs_by_channel(self.chatbot_id, candidates)
            if not jobs:
                return "⚠️ No cronjobs are configured for this channel."
            lines = [f"🚀 Triggered {len(jobs)} cronjob(s) for this channel:"]
            for job in jobs:
                tag_str = f" (tag: `{job.get('tag')}`)" if job.get("tag") else ""
                lines.append(f"  • `{job.get('prompt')}`{tag_str}")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Failed to trigger cronjobs for channel {channel_id}: {e}", exc_info=True)
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
            if command == "compact":
                if not channel_id:
                    await reply_func("⚠️ Channel ID is required for this command.")
                    return
                instructions = text.strip()[len(raw_command) :].strip()
                await self.commands.execute("compact", reply_func, channel_id=channel_id, instructions=instructions)
            elif command in {"clear", "reset", "status", "debug", "cron"}:
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
            elif command == "notes":
                if not channel_id:
                    await reply_func("⚠️ Channel ID is required for this command.")
                    return
                raw_stripped = text.strip()
                content = raw_stripped[len(raw_command) :].strip()
                await self.commands.execute("notes", reply_func, channel_id=channel_id, content=content)
            elif command == "restart":
                await self.commands.execute(command, reply_func)
            elif command in {"grep", "chat-grep", "chat_grep"}:
                if not channel_id:
                    await reply_func("⚠️ Channel ID is required for this command.")
                    return
                query = " ".join(parts[1:]) if len(parts) > 1 else ""
                await self.commands.execute("grep", reply_func, channel_id=channel_id, query=query)
            elif command in {"search", "chat-search", "chat_search"}:
                if not channel_id:
                    await reply_func("⚠️ Channel ID is required for this command.")
                    return
                query = " ".join(parts[1:]) if len(parts) > 1 else ""
                await self.commands.execute("search", reply_func, channel_id=channel_id, query=query)

            elif command == "cronjob":
                tag = " ".join(parts[1:]) if len(parts) > 1 else ""
                await self.commands.execute("cronjob", reply_func, tag=tag)
            else:
                await reply_func(f"⚠️ Unrecognized command: /{command}")
        except Exception as e:
            logger.error(f"Command /{command} execution failed: {e}", exc_info=True)
            await reply_func(f"⚠️ Failed to execute command: {e}")

    async def is_dm_channel(self, channel_id: str) -> bool:
        """Check if the given channel is a DM channel (single user chat).

        Default implementation returns False. Adapters should override if they can detect.
        """
        return False

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

            # 2b. Cancel active deferred media watchers and workers for the cleared session
            try:
                await self.cancel_deferred_for_session(session.id)
            except Exception as e:
                logger.warning(f"Failed to cancel deferred media in clear_session_by_channel: {e}")

            # Resolve role and set appropriate session title
            bot_default = resolve_gateway_default_role(self.gateway, self.chatbot_id)
            role_name = await self.gateway.db.get_channel_role_with_inheritance(
                self.chatbot_id, channel_id, session.id, default=bot_default
            )
            if not isinstance(role_name, str) or not role_name:
                role_name = bot_default or "default"

            # 2c. Trigger T2 close extraction for the outgoing session before creating the new session
            consolidator = getattr(getattr(self.gateway, "context", None), "memory_consolidator", None)
            if consolidator is not None and hasattr(consolidator, "notify_close"):
                try:
                    await consolidator.notify_close(session.id, role=role_name, reason="new_session")
                except Exception as e:
                    logger.warning(f"Failed to notify memory consolidator on clear_session_by_channel: {e}")

            is_dm = await self.is_dm_channel(channel_id)
            display_role = role_name.capitalize() if role_name != "default" else "Default"

            if is_dm:
                title = f"{display_role} DM Session on {self.chatbot_id}"
            else:
                if role_name != "default":
                    title = f"{display_role} Session"
                else:
                    title = f"{self.chatbot_id} Session"

            # 3. Create and bind new session (overwrites active mapping)
            create_kwargs: dict[str, Any] = {
                "session_id": None,
                "title": title,
                "chatbot_id": self.chatbot_id,
                "channel_id": channel_id,
            }
            if role_name != "default":
                create_kwargs["role"] = role_name
            new_session = await self.gateway.create_session(**create_kwargs)
            return f"♻️ New session '{new_session.id}' started. Old session '{session.id}' preserved."
        return "⚠️ No active session found for this chat."

    async def manual_compact_session_by_channel(self, channel_id: str, instructions: str = "") -> str:
        """Force-compact the active history of this channel, optionally steered by focus instructions.

        Compacts every uncompressed turn between the protected head and tail regardless of the watermark. Refuses
        while a turn is in progress so the running turn's scaffold never changes between its LLM calls.

        Args:
            channel_id: Channel whose active session is compacted.
            instructions: Optional focus directive (e.g. "keep only the API decisions").

        Returns:
            A user-facing status line.
        """
        session = await self.gateway.db.get_session_by_channel(self.chatbot_id, channel_id)
        if not session:
            return "⚠️ No active session found for this chat."

        if await self._session_turn_in_progress(session.id):
            return "⏳ A turn is in progress; run /compact after it finishes."

        # Guardrail G2: inspect raw persisted messages (before build_history drops prompt_blocked rows)
        raw_msgs_res = (
            await self.gateway.db.get_session_messages(session.id)
            if hasattr(self.gateway.db, "get_session_messages")
            else []
        )
        raw_msgs = raw_msgs_res if isinstance(raw_msgs_res, list) else []
        latest_user = next((m for m in reversed(raw_msgs) if m.role == MessageRole.USER), None)
        latest_user_blocked = bool(
            latest_user and latest_user.metadata and latest_user.metadata.get("prompt_blocked")
        )

        # Fetch active history
        history = await build_history(self.gateway, session.id, heal_orphans=False)
        if not history:
            return "⚠️ Active session has no messages to compact."

        from kesoku.agent.compressor import HistoryCompressor
        from kesoku.agent.llm import get_llm

        compressor = HistoryCompressor(self.gateway.db)
        cfg = self.gateway.context.config
        llm = get_llm(provider=cfg.agent.context_llm or cfg.agent.llm, config=cfg, use_context_compression=True)
        staging_dir = self.get_session_staging_dir(session.workspace_name)

        stripped = instructions.strip()
        compact_all = False
        if stripped.lower() == "all":
            compact_all = True
            directive = None
        elif stripped.lower().startswith("all "):
            compact_all = True
            directive = stripped[4:].strip() or None
        else:
            directive = stripped or None

        segmented_turns = compressor.segment_turns(history)
        turns_before = sum(1 for t in segmented_turns if any(m.summary_node_id is not None for m in t))
        include_tail = latest_user_blocked or compact_all

        try:
            if include_tail:
                pf = (
                    0
                    if (compact_all or len(segmented_turns) <= cfg.agent.protect_front_turns)
                    else cfg.agent.protect_front_turns
                )
                nodes = await compressor.force_compact_session(
                    session.id,
                    history,
                    llm,
                    cfg,
                    directive=directive,
                    staging_dir=staging_dir,
                    protect_front_turns=pf,
                    protect_tail_turns=0,
                )
            else:
                nodes = await compressor.force_compact_session(
                    session.id, history, llm, cfg, directive=directive, staging_dir=staging_dir
                )
        except Exception as e:
            logger.error(f"Failed manual compaction for session {session.id}: {e}")
            return f"⚠️ Failed to compact history: {e}"
        if not nodes:
            return (
                "ℹ️ Nothing to compact: all turns are already summarised or inside the protected head/tail "
                f"({cfg.agent.protect_front_turns} + {cfg.agent.protect_tail_turns} turns)."
            )
        agent = getattr(self.gateway, "agent", None)
        worker = agent.workers.get(session.id) if agent is not None and hasattr(agent, "workers") else None
        clear_caches_fn = getattr(worker, "clear_explicit_caches", None) if worker is not None else None
        if callable(clear_caches_fn):
            try:
                await clear_caches_fn()
            except Exception as ce:
                logger.warning(f"Failed to clear explicit caches after /compact for session {session.id}: {ce}")
        turns_after = sum(
            1 for t in compressor.segment_turns(history) if any(m.summary_node_id is not None for m in t)
        )
        reply = f"🔄 Compacted {turns_after - turns_before} turn(s) into {nodes} summary node(s)"
        reply += " (including recent turns)." if include_tail else "."
        if directive:
            reply += f'\nFocus: "{directive}"'
        return reply

    async def _session_turn_in_progress(self, session_id: str) -> bool:
        """Whether the session has a message waiting for / being processed by the agent, or a busy worker."""
        for status in (MessageStatus.PENDING_AGENT, MessageStatus.PROCESSING):
            busy = await self.gateway.db.get_messages_by_filters({"session_id": session_id, "status": status})
            if busy:
                return True
        agent = getattr(self.gateway, "agent", None)
        worker = agent.workers.get(session_id) if agent is not None and hasattr(agent, "workers") else None
        return bool(worker is not None and getattr(worker, "_processing_turn", False))

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

            # Per-turn context blocks (notes.md + context/*.md) the agent injects into the latest user message
            turn_context = ""
            try:
                staging_dir = self.get_session_staging_dir(session.workspace_name)
                turn_context = await build_context_prefix(staging_dir, cfg.agent.context_dir_max_chars)
            except Exception as ce:
                logger.warning(f"Failed to build context blocks for /context report: {ce}")

            return ContextHtmlReporter.render_to_temp_file(
                session=session,
                root_summaries=root_summaries,
                all_summaries=all_summaries,
                protected_head=protected_head,
                buffer=buffer,
                protected_tail=protected_tail,
                sys_msg=sys_msg,
                last_metrics=last_metrics,
                turn_context=turn_context,
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
                role = resolve_gateway_default_role(self.gateway, self.chatbot_id)

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
                            if is_deferred_sidecar_name(f):
                                continue
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
        # Platforms without native hidden text keep spoiler content visible (tags stripped).
        return render_spoilers(format_text(text))

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

                        try:
                            await self.send_text_chunks(message.channel_id, chunks, message)
                        except ChannelUnavailableError:
                            raise
                        except Exception as text_err:
                            logger.error(
                                f"Text delivery failed for message {message.id} "
                                f"on channel {message.channel_id}: {text_err}"
                            )
                            if not (message.metadata and message.metadata.get("delivery_notice")):
                                try:
                                    await self.send_text_chunks(
                                        message.channel_id, ["⚠️ 回复发送失败，请再说一次"], message
                                    )
                                except Exception:
                                    pass
                            if isinstance(text_err, DeliveryAbortedError):
                                raise
                            raise DeliveryAbortedError(f"Text delivery failed: {text_err}") from text_err

                elif segment["type"] == "file":
                    await self._attempt_send_attachment(segment, message)

                elif segment["type"] == "voice":
                    await self._attempt_send_attachment(segment, message)

                elif segment["type"] == "question":
                    await self.send_question_segment(
                        message.channel_id,
                        segment["question"],
                        segment["choices"],
                        message,
                        multi=segment.get("multi", False),
                        max_choices=segment.get("max", 1),
                    )

                elif segment["type"] == "details":
                    await self.send_details_segment(message.channel_id, segment["title"], segment["body"], message)

            # 4. Update Gateway status to DELIVERED
            await self.gateway.db.update_message_status(message.id, MessageStatus.DELIVERED)

            # 5. Post-delivery lifecycle hook (e.g., stop typing indicator, update metrics, finalize card/header)
            await self.on_message_delivered(message)
        except DeliveryAbortedError as de:
            # Stop further execution of the template since the delivery was aborted/handled elsewhere
            await self.on_message_delivery_failed(message, de)

    async def start(self) -> None:
        """Start listening as a decentralized subscriber for model responses.

        Subscribes to gateway messages for this session_id (if set) or chatbot_id
        and routes non-user messages to handle_message.
        """
        self._listener_task = asyncio.current_task()
        self._register_gateway_capabilities()
        if self.supports_deferred_media and (self._recovery_task is None or self._recovery_task.done()):
            self._recovery_task = asyncio.create_task(self._recover_deferred_media())
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
                try:
                    await self.handle_message(msg)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    # A single undeliverable message must never take down the delivery loop:
                    # it is the only consumer for this chatbot, so losing it silently stops
                    # every subsequent reply from ever reaching the user.
                    logger.error(
                        f"Chatbot '{self.chatbot_id}' failed to handle message {msg.id} "
                        f"(session {msg.session_id}): {e}",
                        exc_info=True,
                    )
                    try:
                        await self.gateway.db.update_message_status(msg.id, MessageStatus.ERROR)
                    except Exception as se:
                        logger.error(f"Failed to mark message {msg.id} as errored: {se}")
        except asyncio.CancelledError:
            logger.debug(f"Chatbot '{self.chatbot_id}' listener cancelled.")
        except Exception as e:
            logger.critical(
                f"Chatbot '{self.chatbot_id}' listener loop terminated unexpectedly: {e}",
                exc_info=True,
            )
            raise

    def spawn_subscriber_task(self) -> asyncio.Task[None]:
        """Run the gateway subscriber loop as a background task with crash logging.

        Platform adapters occupy their main coroutine with their own client loop, so the
        subscriber runs in a separate task. Without an explicit done-callback its death is
        invisible: the task object stays referenced by the adapter, so Python never emits
        "Task exception was never retrieved" and outbound delivery stops with no log at all.

        Returns:
            The created subscriber task.
        """
        task = asyncio.create_task(Chatbot.start(self))
        task.add_done_callback(self._log_subscriber_task_exit)
        return task

    def _log_subscriber_task_exit(self, task: asyncio.Task[None]) -> None:
        """Report why the gateway subscriber task finished.

        Args:
            task: The completed subscriber task.
        """
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.critical(
                f"Chatbot '{self.chatbot_id}' subscriber task died; outbound delivery has stopped: {exc}",
                exc_info=exc,
            )
        else:
            logger.warning(f"Chatbot '{self.chatbot_id}' subscriber task exited; outbound delivery has stopped.")

    def stop(self) -> None:
        """Stop the subscriber listener task and cancel in-memory deferred watchers.

        Note:
            Service stop/restart cancels ONLY in-memory ``asyncio.Task`` instances
            without signaling detached worker PIDs or altering ``metadata`` so
            startup recovery can re-attach watchers on the next boot.
        """
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
        if self._recovery_task and not self._recovery_task.done():
            self._recovery_task.cancel()
        for task_set in list(self._deferred_tasks.values()):
            for task in list(task_set):
                if not task.done():
                    task.cancel()
        self._deferred_tasks.clear()
        self._deferred_watched_keys.clear()

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

    async def send_deferred_placeholder(
        self,
        channel_id: str,
        segment: dict[str, Any],
        message: Message,
        marker: dict[str, Any],
    ) -> Any:
        """Send an optional placeholder message for a pending deferred attachment."""
        _ = (channel_id, segment, message, marker)
        return None

    def serialize_deferred_handle(self, handle: Any) -> dict[str, Any] | None:
        """Serialize a platform placeholder handle into JSON-storable metadata."""
        _ = handle
        return None

    async def restore_deferred_handle(self, raw_handle: dict[str, Any] | None) -> Any:
        """Restore a platform placeholder handle from persisted metadata on startup."""
        _ = raw_handle
        return None

    async def deliver_deferred_attachment(
        self,
        channel_id: str,
        file_path: str,
        message: Message,
        handle: Any,
        kind: str,
    ) -> None:
        """Deliver a ready deferred attachment via the platform adapter."""
        _ = handle
        if kind == "voice":
            await self.send_voice_segment(channel_id, file_path, message)
        else:
            await self.send_file_segment(channel_id, file_path, message)

    async def fail_deferred_attachment(
        self,
        handle: Any,
        reason: str,
        message: Message,
        segment: dict[str, Any],
    ) -> None:
        """Report a deferred attachment failure to the user without triggering self-healing."""
        _ = handle
        notice = format_deferred_failure_message(segment.get("path", ""), reason)
        await self.send_text_chunks(message.channel_id, [notice], message)

    async def discard_deferred_placeholder(
        self,
        handle: Any,
        message: Message,
        segment: dict[str, Any],
    ) -> None:
        """Discard an inline placeholder or reaction anchor after successful synchronous delivery."""
        _ = (handle, message, segment)

    async def _get_staging_dir_for_session(self, session_id: str) -> str | None:
        """Resolve the session staging directory for ``session_id`` if available."""
        try:
            get_sess = getattr(self.gateway.db, "get_session", None)
            if not callable(get_sess):
                return None
            session = await get_sess(session_id)
            if session and getattr(session, "workspace_name", None):
                return self.get_session_staging_dir(session.workspace_name)
        except Exception as exc:
            logger.debug(f"Failed to resolve staging dir for session {session_id}: {exc}")
        return None

    async def _resolve_deferred(
        self,
        raw_path: str,
        session_id: str,
    ) -> tuple[str, str, dict[str, Any]] | None:
        """Inspect deferred sidecars for ``raw_path`` strictly before fuzzy path matching."""
        staging_dir = await self._get_staging_dir_for_session(session_id)
        return await asyncio.to_thread(resolve_deferred_sync, raw_path, staging_dir)

    def _get_metadata_lock(self, message_id: str) -> asyncio.Lock:
        """Return a per-message asyncio lock guarding concurrent ``message.metadata`` updates."""
        lock = self._metadata_locks.get(message_id)
        if lock is None:
            lock = asyncio.Lock()
            self._metadata_locks[message_id] = lock
        return lock

    async def _persist_message_metadata_locked(self, msg_obj: Message) -> None:
        """Persist a deep copy of ``msg_obj.metadata`` to SQLite while holding its metadata lock."""
        update_fn = getattr(self.gateway.db, "update_message_metadata", None)
        if not callable(update_fn):
            return
        try:
            await update_fn(msg_obj.id, copy.deepcopy(msg_obj.metadata))
        except Exception as exc:
            logger.warning(f"Failed to persist deferred metadata for message {msg_obj.id}: {exc}")

    async def _start_deferred_tracking(
        self,
        segment: dict[str, Any],
        message: Message,
        target_path: str,
        marker: dict[str, Any],
        *,
        placeholder_sent: bool = False,
        handle: Any = None,
    ) -> None:
        """Send placeholder (unless already sent inline), persist pending metadata, and spawn watcher task."""
        if not placeholder_sent:
            handle = None
            try:
                handle = await self.send_deferred_placeholder(message.channel_id, segment, message, marker)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    f"Failed to send deferred placeholder for '{segment.get('path')}' "
                    f"(will deliver without placeholder when ready): {exc}"
                )
                handle = None

        job_id = str(marker.get("job_id") or uuid.uuid4().hex[:12])
        kind = str(segment.get("type") or "file")
        entry: dict[str, Any] = {
            "id": job_id,
            "path": target_path,
            "raw_path": str(segment.get("path") or target_path),
            "kind": kind,
            "state": "pending",
            "marker": copy.deepcopy(marker) if isinstance(marker, dict) else {},
            "handle": self.serialize_deferred_handle(handle),
        }

        lock = self._get_metadata_lock(message.id)
        async with lock:
            canonical_msg = self._deferred_messages.setdefault(message.id, message)
            if not isinstance(canonical_msg.metadata, dict):
                canonical_msg.metadata = {}
            entries = canonical_msg.metadata.get("deferred_media")
            if not isinstance(entries, list):
                entries = []
                canonical_msg.metadata["deferred_media"] = entries
            entry_index = len(entries)
            entries.append(entry)
            await self._persist_message_metadata_locked(canonical_msg)

        self._spawn_deferred_watcher(
            message=canonical_msg,
            entry_index=entry_index,
            target_path=target_path,
            initial_marker=entry["marker"],
            handle=handle,
            segment=segment,
        )

    def _spawn_deferred_watcher(
        self,
        message: Message,
        entry_index: int,
        target_path: str,
        initial_marker: dict[str, Any],
        handle: Any,
        segment: dict[str, Any],
    ) -> None:
        """Spawn and register an ``asyncio.Task`` watching ``(message.id, entry_index)``."""
        watch_key = (message.id, entry_index)
        if watch_key in self._deferred_watched_keys:
            return
        self._deferred_watched_keys.add(watch_key)

        task = asyncio.create_task(
            self._watch_deferred_attachment(
                message=message,
                entry_index=entry_index,
                target_path=target_path,
                initial_marker=initial_marker,
                handle=handle,
                segment=segment,
            )
        )
        session_tasks = self._deferred_tasks.setdefault(message.session_id, set())
        session_tasks.add(task)

        def _cleanup_done(done_task: asyncio.Task[None]) -> None:
            self._deferred_watched_keys.discard(watch_key)
            bucket = self._deferred_tasks.get(message.session_id)
            if bucket is not None:
                bucket.discard(done_task)
                if not bucket:
                    self._deferred_tasks.pop(message.session_id, None)
            if not any(mid == message.id for mid, _ in self._deferred_watched_keys):
                msg_lock = self._metadata_locks.get(message.id)
                if msg_lock is not None and not msg_lock.locked():
                    self._metadata_locks.pop(message.id, None)
                self._deferred_messages.pop(message.id, None)

        task.add_done_callback(_cleanup_done)

    async def _set_deferred_entry_state(self, message: Message, entry_index: int, new_state: str) -> None:
        """Update ``message.metadata['deferred_media'][entry_index]['state']`` under lock and persist."""
        lock = self._get_metadata_lock(message.id)
        async with lock:
            canonical_msg = self._deferred_messages.get(message.id, message)
            if not isinstance(canonical_msg.metadata, dict):
                return
            entries = canonical_msg.metadata.get("deferred_media")
            if not isinstance(entries, list) or not (0 <= entry_index < len(entries)):
                return
            entry = entries[entry_index]
            if isinstance(entry, dict):
                entry["state"] = new_state
                await self._persist_message_metadata_locked(canonical_msg)

    async def _safe_fail_deferred(
        self,
        handle: Any,
        reason: str,
        message: Message,
        segment: dict[str, Any],
    ) -> None:
        """Invoke ``fail_deferred_attachment`` catching and logging any adapter exceptions."""
        try:
            await self.fail_deferred_attachment(handle, sanitize_error_reason(reason), message, segment)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                f"Adapter fail_deferred_attachment raised error for '{segment.get('path')}': {exc}",
                exc_info=True,
            )

    async def _safe_discard_placeholder(
        self,
        handle: Any,
        message: Message,
        segment: dict[str, Any],
    ) -> None:
        """Invoke ``discard_deferred_placeholder`` catching and logging any adapter exceptions."""
        try:
            await self.discard_deferred_placeholder(handle, message, segment)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                f"Adapter discard_deferred_placeholder raised error for '{segment.get('path')}': {exc}",
                exc_info=True,
            )

    async def _wait_voice_inline(
        self,
        segment: dict[str, Any],
        message: Message,
        raw_path: str,
        target_path: str,
        sidecar_data: dict[str, Any],
    ) -> tuple[tuple[str, str, dict[str, Any]] | None, Any, bool, str | None, bool]:
        """Wait inline for a pending voice segment (silent Phase A then anchored Phase B).

        Returns:
            Tuple of ``(rechecked_state, inline_handle, placeholder_attempted, wait_err, budget_exhausted)``.
        """
        wait_cap = max(0.0, float(self._voice_inline_wait_sec or 0.0))
        marker_timeout = clamp_timeout_sec(sidecar_data)
        try:
            created_at = float(sidecar_data.get("created_at") or 0.0)
        except (TypeError, ValueError):
            created_at = 0.0
        if created_at > 0.0:
            remaining_timeout = max(0.0, (created_at + marker_timeout) - time.time())
            effective_wait = min(wait_cap, remaining_timeout)
        else:
            effective_wait = min(wait_cap, marker_timeout)

        deadline = time.monotonic() + effective_wait
        poll_iv = min(0.25, max(0.02, float(self._deferred_poll_initial_sec)))
        dead_grace = max(0.05, float(self._deferred_dead_grace_sec))

        inline_handle: Any = None
        placeholder_attempted = False
        wait_err: str | None = None

        anchor_delay = max(0.0, float(self._voice_inline_anchor_delay_sec or 0.0))
        phase_a_cap = min(anchor_delay, effective_wait)
        if phase_a_cap > 0.0:
            _, wait_err = await wait_for_deferred_file(
                [target_path],
                poll_interval=poll_iv,
                max_wait_cap=phase_a_cap,
                grace_sec=dead_grace,
            )

        # CRITICAL: Never treat inline wait expiry `(None, "生成超时")` as failure.
        # Re-inspect sidecar state for exact 3-state dispatch: ready / pending / failed.
        rechecked = await self._resolve_deferred(raw_path, message.session_id)

        if (
            rechecked is not None
            and rechecked[0] == "pending"
            and wait_err in (None, "生成超时")
            and (deadline - time.monotonic()) > 0.05
        ):
            placeholder_attempted = True
            try:
                inline_handle = await self.send_deferred_placeholder(
                    message.channel_id,
                    segment,
                    message,
                    rechecked[2],
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    f"Failed to send inline voice placeholder for '{segment.get('path')}' "
                    f"(will continue waiting without placeholder): {exc}"
                )
                inline_handle = None

            rem = max(0.05, deadline - time.monotonic())
            _, wait_err = await wait_for_deferred_file(
                [rechecked[1]],
                poll_interval=poll_iv,
                max_wait_cap=rem,
                grace_sec=dead_grace,
            )
            rechecked = await self._resolve_deferred(raw_path, message.session_id)

        return (
            rechecked,
            inline_handle,
            placeholder_attempted,
            wait_err,
            time.monotonic() >= deadline - 0.05,
        )

    async def _watch_deferred_attachment(
        self,
        message: Message,
        entry_index: int,
        target_path: str,
        initial_marker: dict[str, Any],
        handle: Any,
        segment: dict[str, Any],
    ) -> None:
        """Poll a pending deferred attachment and deliver or fail without blocking ``Chatbot.start()``."""
        pending_path = f"{target_path}.pending.json"
        error_path = f"{target_path}.error.json"
        expected_job_id = str(initial_marker.get("job_id") or "").strip()
        kind = str(segment.get("type") or "file")
        effective_marker: dict[str, Any] = copy.deepcopy(initial_marker) if isinstance(initial_marker, dict) else {}
        timeout_sec = clamp_timeout_sec(effective_marker)
        try:
            created_at = float(effective_marker.get("created_at") or 0.0)
        except (TypeError, ValueError):
            created_at = 0.0
        if created_at <= 0.0:
            created_at = time.time()

        try:
            tracked_pid = int(effective_marker.get("pid") or 0)
        except (TypeError, ValueError):
            tracked_pid = 0

        dead_since: float | None = None
        missing_since: float | None = None
        poll_delay = max(0.01, float(self._deferred_poll_initial_sec))
        max_poll_delay = max(poll_delay, float(self._deferred_poll_max_sec))
        grace_sec = max(0.05, float(self._deferred_dead_grace_sec))

        try:
            while True:
                live_marker = await asyncio.to_thread(read_json_dict, pending_path)
                if isinstance(live_marker, dict) and live_marker:
                    live_job_id = str(live_marker.get("job_id") or "").strip()
                    if expected_job_id and live_job_id and live_job_id != expected_job_id:
                        await self._set_deferred_entry_state(message, entry_index, "failed")
                        await self._safe_fail_deferred(handle, "已被新任务取代", message, segment)
                        return
                    effective_marker = live_marker
                    timeout_sec = clamp_timeout_sec(effective_marker)
                    try:
                        live_pid = int(live_marker.get("pid") or 0)
                        if live_pid > 0:
                            tracked_pid = live_pid
                            if isinstance(initial_marker, dict):
                                initial_marker["pid"] = live_pid
                    except (TypeError, ValueError):
                        pass
                    missing_since = None

                if await asyncio.to_thread(is_deferred_target_ready, target_path, pending_path, effective_marker):
                    await asyncio.to_thread(unlink_if_job_matches, pending_path, expected_job_id)
                    await asyncio.to_thread(unlink_if_job_matches, error_path, expected_job_id)
                    final_path = await async_realpath(target_path)
                    try:
                        await self.deliver_deferred_attachment(
                            message.channel_id,
                            final_path,
                            message,
                            handle,
                            kind,
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.error(
                            f"Deferred delivery failed for '{target_path}' in session {message.session_id}: {exc}",
                            exc_info=True,
                        )
                        await self._set_deferred_entry_state(message, entry_index, "failed")
                        await self._safe_fail_deferred(handle, "附件上传失败", message, segment)
                        return

                    await self._set_deferred_entry_state(message, entry_index, "delivered")
                    return

                err_dict = await asyncio.to_thread(read_json_dict, error_path)
                if err_dict is not None:
                    err_job_id = str(err_dict.get("job_id") or "").strip()
                    if not expected_job_id or not err_job_id or err_job_id == expected_job_id:
                        await asyncio.to_thread(unlink_if_job_matches, error_path, expected_job_id)
                        await asyncio.to_thread(unlink_if_job_matches, pending_path, expected_job_id)
                        reason = sanitize_error_reason(err_dict.get("error"), default="后台生成失败")
                        await self._set_deferred_entry_state(message, entry_index, "failed")
                        await self._safe_fail_deferred(handle, reason, message, segment)
                        return

                now_mono = time.monotonic()
                if tracked_pid > 0 and not await asyncio.to_thread(is_pid_alive, tracked_pid):
                    if dead_since is None:
                        dead_since = now_mono
                    elif now_mono - dead_since >= grace_sec:
                        await asyncio.to_thread(unlink_if_job_matches, pending_path, expected_job_id)
                        await self._set_deferred_entry_state(message, entry_index, "failed")
                        await self._safe_fail_deferred(handle, "后台生成进程异常退出", message, segment)
                        return
                else:
                    dead_since = None

                if live_marker is None:
                    if missing_since is None:
                        missing_since = now_mono
                    elif now_mono - missing_since >= grace_sec:
                        recheck = await asyncio.to_thread(read_json_dict, pending_path)
                        await self._set_deferred_entry_state(message, entry_index, "failed")
                        if (
                            isinstance(recheck, dict)
                            and recheck
                            and expected_job_id
                            and str(recheck.get("job_id") or "").strip() not in ("", expected_job_id)
                        ):
                            await self._safe_fail_deferred(handle, "已被新任务取代", message, segment)
                        else:
                            await self._safe_fail_deferred(handle, "生成任务中断", message, segment)
                        return

                if time.time() > created_at + timeout_sec:
                    await terminate_deferred_worker(target_path, effective_marker)
                    await self._set_deferred_entry_state(message, entry_index, "failed")
                    await self._safe_fail_deferred(handle, "生成超时", message, segment)
                    return

                if dead_since is not None or missing_since is not None:
                    sleep_for = min(poll_delay, max(0.02, grace_sec / 4.0))
                else:
                    sleep_for = poll_delay
                    poll_delay = min(max_poll_delay, poll_delay * 1.5)
                await asyncio.sleep(sleep_for)
        except asyncio.CancelledError:
            # Service stop or explicit session cancel: do NOT touch worker PID or metadata here.
            raise
        except Exception as exc:
            logger.error(
                f"Unexpected error in deferred watcher for '{target_path}' (message {message.id}): {exc}",
                exc_info=True,
            )

    async def cancel_deferred_for_session(self, session_id: str) -> None:
        """Cancel active deferred watchers, terminate worker PIDs/units, and mark ``cancelled`` for ``session_id``."""
        tasks = list(self._deferred_tasks.pop(session_id, set()))
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        msg_by_id: dict[str, Message] = {
            mid: m for mid, m in list(self._deferred_messages.items()) if m.session_id == session_id
        }
        get_msgs_fn = getattr(self.gateway.db, "get_messages_by_filters", None)
        if callable(get_msgs_fn):
            try:
                db_msgs = await get_msgs_fn({"session_id": session_id, "role": MessageRole.ASSISTANT})
                if isinstance(db_msgs, list):
                    for dm in db_msgs:
                        msg_by_id.setdefault(dm.id, dm)
            except Exception as exc:
                logger.debug(f"Failed to load session messages in cancel_deferred_for_session({session_id}): {exc}")

        for msg in msg_by_id.values():
            lock = self._get_metadata_lock(msg.id)
            async with lock:
                meta = getattr(msg, "metadata", None)
                if not isinstance(meta, dict):
                    continue
                entries = meta.get("deferred_media")
                if not isinstance(entries, list):
                    continue
                changed = False
                for entry in entries:
                    if isinstance(entry, dict) and entry.get("state") == "pending":
                        target = str(entry.get("path") or "")
                        marker = entry.get("marker") if isinstance(entry.get("marker"), dict) else {}
                        if target:
                            await terminate_deferred_worker(target, marker)
                        entry["state"] = "cancelled"
                        changed = True
                if changed:
                    await self._persist_message_metadata_locked(msg)

    async def _recover_deferred_media(self) -> None:
        """Scan last 24h messages for pending deferred attachments and re-attach watchers on startup."""
        if not self.supports_deferred_media:
            return
        cutoff = time.time() - RECOVERY_LOOKBACK_SEC
        messages: list[Message] = []
        try:
            fast_fn = getattr(self.gateway.db, "get_pending_deferred_media_messages", None)
            if callable(fast_fn):
                res = await fast_fn(self.chatbot_id, cutoff)
                if isinstance(res, list):
                    messages = res
            else:
                get_msgs_fn = getattr(self.gateway.db, "get_messages_by_filters", None)
                if callable(get_msgs_fn):
                    res = await get_msgs_fn({"chatbot_id": self.chatbot_id, "role": MessageRole.ASSISTANT})
                    if isinstance(res, list):
                        messages = [m for m in res if getattr(m, "timestamp", 0.0) >= cutoff]
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"Failed to scan pending deferred media on startup for '{self.chatbot_id}': {exc}")
            return

        for msg in messages:
            if self.session_id and msg.session_id != self.session_id:
                continue
            meta = getattr(msg, "metadata", None)
            if not isinstance(meta, dict):
                continue
            entries = meta.get("deferred_media")
            if not isinstance(entries, list):
                continue

            canonical_msg = self._deferred_messages.setdefault(msg.id, msg)
            canonical_entries = canonical_msg.metadata.get("deferred_media", [])
            for idx, entry in enumerate(canonical_entries):
                if not isinstance(entry, dict) or entry.get("state") != "pending":
                    continue
                if (canonical_msg.id, idx) in self._deferred_watched_keys:
                    continue
                target_path = str(entry.get("path") or "").strip()
                if not target_path:
                    continue
                raw_handle = entry.get("handle") if isinstance(entry.get("handle"), dict) else None
                handle: Any = None
                if raw_handle is not None:
                    try:
                        handle = await self.restore_deferred_handle(raw_handle)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.warning(
                            f"Failed to restore deferred handle for message {canonical_msg.id} entry {idx}: {exc}"
                        )
                        handle = None
                kind = str(entry.get("kind") or "file")
                raw_path = str(entry.get("raw_path") or target_path)
                marker = entry.get("marker") if isinstance(entry.get("marker"), dict) else {}
                segment = {"type": kind, "path": raw_path}
                self._spawn_deferred_watcher(
                    message=canonical_msg,
                    entry_index=idx,
                    target_path=target_path,
                    initial_marker=marker,
                    handle=handle,
                    segment=segment,
                )

    async def _attempt_send_attachment(self, segment: dict[str, Any], message: Message) -> None:
        """Attempt to deliver a file or voice attachment segment, with automatic Self-Healing on failure.

        If sending fails on the first attempt, an error notification is injected into the session
        history to let the Agent automatically correct the filename/path. If a self-healing retry
        also fails, a user-facing error message is sent to the channel.

        Args:
            segment: Segment dictionary with 'type' and 'path'.
            message: The Message instance being processed.

        Raises:
            DeliveryAbortedError: Raised to abort delivery of remaining segments in this message.
        """
        raw_path = segment["path"]
        inline_handle: Any = None
        inline_placeholder_attempted = False
        wait_err: str | None = None
        budget_exhausted = True
        if self.supports_deferred_media:
            deferred_state = await self._resolve_deferred(raw_path, message.session_id)
            if deferred_state is not None:
                state_kind, target_path, sidecar_data = deferred_state
                inline_wait = float(self._voice_inline_wait_sec or 0.0)
                if segment.get("type") == "voice" and state_kind == "pending" and inline_wait > 0.0:
                    (
                        rechecked,
                        inline_handle,
                        inline_placeholder_attempted,
                        wait_err,
                        budget_exhausted,
                    ) = await self._wait_voice_inline(
                        segment,
                        message,
                        raw_path,
                        target_path,
                        sidecar_data,
                    )
                    if rechecked is not None:
                        state_kind, target_path, sidecar_data = rechecked
                    elif await asyncio.to_thread(os.path.isfile, target_path):
                        state_kind = "ready"
                        target_path = await async_realpath(target_path)
                    else:
                        await self._safe_fail_deferred(
                            inline_handle,
                            wait_err or "生成任务中断",
                            message,
                            segment,
                        )
                        return

                if state_kind == "failed":
                    await asyncio.to_thread(safe_unlink, f"{target_path}.error.json")
                    raw_err = sidecar_data.get("error") if isinstance(sidecar_data, dict) else None
                    reason = sanitize_error_reason(raw_err, default="后台生成失败")
                    await self._safe_fail_deferred(inline_handle, reason, message, segment)
                    return
                if state_kind == "pending":
                    if segment.get("type") == "voice" and (
                        wait_err not in (None, "生成超时") or (not budget_exhausted and wait_err is not None)
                    ):
                        expected_job_id = str((sidecar_data or {}).get("job_id") or "")
                        await asyncio.to_thread(
                            unlink_if_job_matches,
                            f"{target_path}.pending.json",
                            expected_job_id,
                        )
                        await self._safe_fail_deferred(
                            inline_handle,
                            wait_err or "生成任务中断",
                            message,
                            segment,
                        )
                        return
                    await self._start_deferred_tracking(
                        segment,
                        message,
                        target_path,
                        sidecar_data,
                        placeholder_sent=inline_placeholder_attempted,
                        handle=inline_handle,
                    )
                    return
                resolved_path = target_path
            else:
                resolved_path = await self.resolve_outbound_path(raw_path, message.session_id)
        else:
            resolved_path = await self.resolve_outbound_path(raw_path, message.session_id)

        _path_errors = (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError)
        try:
            if segment["type"] == "file":
                await self.send_file_segment(message.channel_id, resolved_path, message)
            elif segment["type"] == "voice":
                try:
                    await self.send_voice_segment(message.channel_id, resolved_path, message)
                finally:
                    if inline_handle is not None:
                        await self._safe_discard_placeholder(inline_handle, message, segment)
        except ChannelUnavailableError:
            raise
        except Exception as e:
            file_exists = bool(resolved_path) and await asyncio.to_thread(os.path.isfile, resolved_path)
            if isinstance(e, _path_errors) or not file_exists:
                logger.warning(
                    f"Attachment delivery failed for '[{segment['type']}: {raw_path}]' "
                    f"(resolved: {resolved_path}): {e}"
                )
                is_already_healing = await self._is_self_healing_retry(message)
                if is_already_healing:
                    logger.error(
                        f"Agent self-healing failed for attachment '[{segment['type']}: {raw_path}]'. "
                        f"Delivering error notice to user."
                    )
                    err_notice = f"⚠️ 文件发送失败 ({os.path.basename(raw_path)}): {e}"
                    await self.send_text_chunks(message.channel_id, [err_notice], message)
                else:
                    logger.info(f"Initiating Agent self-healing for attachment '[{segment['type']}: {raw_path}]'.")
                    await self._inject_self_healing_notification(segment, raw_path, resolved_path, message, e)
                    raise DeliveryAbortedError(f"Self-healing initiated for failed attachment: {raw_path}")
            else:
                logger.warning(
                    f"Transport failure sending existing attachment '[{segment['type']}: {raw_path}]' "
                    f"(resolved: {resolved_path}): {e}"
                )
                await self.send_text_chunks(
                    message.channel_id,
                    [f"⚠️ 附件发送失败（网络错误，已重试）: {os.path.basename(raw_path)}"],
                    message,
                )
                return

    async def _is_self_healing_retry(self, message: Message) -> bool:
        """Check if the current message is generated from an attachment self-healing retry.

        Args:
            message: The outgoing Message instance being inspected.

        Returns:
            True if the active conversational turn was triggered by an injected self-healing notification.
        """
        try:
            history = await self.gateway.db.get_session_history(message.session_id, limit=20)
            for msg in reversed(history):
                if msg.id == message.id:
                    continue
                if msg.role in (MessageRole.USER, MessageRole.SYSTEM):
                    if msg.metadata and msg.metadata.get("is_self_healing_retry"):
                        return True
                    return False
        except Exception as e:
            logger.warning(f"Failed to check self-healing retry status for session {message.session_id}: {e}")
        return False

    async def _get_staging_files_list(self, session_id: str) -> list[str]:
        """List relative paths of files in the session staging directory.

        Args:
            session_id: Active session ID context.

        Returns:
            A list of up to 20 relative file paths in the staging directory.
        """
        try:
            session = await self.gateway.db.get_session(session_id)
            if not session:
                return []
            staging_dir = self.get_session_staging_dir(session.workspace_name)
            if not await async_exists(staging_dir):
                return []

            def _list_files(s_dir: str) -> list[str]:
                now = time.time()
                excluded_dirs = {"context", "game"}
                excluded_prefixes = ("background_job_", "cmd_output_", "hist_")
                pending_entries: dict[str, float] = {}
                file_entries: dict[str, float] = {}

                for root, dirs, filenames in os.walk(s_dir):
                    dirs[:] = [
                        d
                        for d in dirs
                        if not d.startswith(".") and d not in excluded_dirs
                    ]
                    for f in filenames:
                        if f.startswith("."):
                            continue
                        full_path = os.path.join(root, f)
                        if f.endswith(".pending.json"):
                            target_full = full_path[: -len(".pending.json")]
                            target_base = os.path.basename(target_full)
                            if target_base.startswith(".") or target_base.startswith(excluded_prefixes):
                                continue
                            try:
                                p_mtime = os.path.getmtime(full_path)
                            except OSError:
                                continue
                            marker = read_json_dict(full_path) or {}
                            if now - p_mtime <= clamp_timeout_sec(marker) and not is_deferred_target_ready(
                                target_full, full_path, marker
                            ):
                                rel_target = os.path.relpath(target_full, s_dir)
                                pending_entries[rel_target] = max(pending_entries.get(rel_target, 0.0), p_mtime)
                            continue
                        if is_deferred_sidecar_name(f) or f.startswith(excluded_prefixes):
                            continue
                        try:
                            f_mtime = os.path.getmtime(full_path)
                        except OSError:
                            f_mtime = 0.0
                        rel_file = os.path.relpath(full_path, s_dir)
                        file_entries[rel_file] = f_mtime

                combined: list[tuple[float, str]] = []
                for rel_target, p_mtime in pending_entries.items():
                    combined.append((p_mtime, f"{rel_target} (生成中)"))
                for rel_file, f_mtime in file_entries.items():
                    if rel_file in pending_entries:
                        continue
                    combined.append((f_mtime, rel_file))
                combined.sort(key=lambda item: (item[0], item[1]), reverse=True)
                return [label for _, label in combined[:20]]

            return await asyncio.to_thread(_list_files, staging_dir)
        except Exception as e:
            logger.warning(f"Failed to list staging directory files for session {session_id}: {e}")
            return []

    async def _inject_self_healing_notification(
        self,
        segment: dict[str, Any],
        raw_path: str,
        resolved_path: str,
        message: Message,
        exception: Exception,
    ) -> None:
        """Inject a system error notification into the session to trigger Agent self-healing.

        Args:
            segment: The attachment segment dictionary.
            raw_path: The raw path string originally written by the Agent.
            resolved_path: The path after resolution attempts.
            message: The Message instance being processed.
            exception: The exception raised during delivery.
        """
        available_files = await self._get_staging_files_list(message.session_id)
        files_str = "\n".join(f"- {f}" for f in available_files) if available_files else "No files found."

        content = (
            f"File transmission failed for attachment tag '[{segment['type']}: {raw_path}]' "
            f"(resolved path '{resolved_path}'): {exception}\n"
            f"Please inspect your generated filenames and reply with the correct "
            f"'[{segment['type']}: <actual_filename>]' tag to deliver the file to the user. "
            f"Re-send the full reply including any ```ui block of the failed message.\n"
            f"Available files in your staging directory:\n{files_str}"
        )

        healing_msg = Message(
            session_id=message.session_id,
            chatbot_id=self.chatbot_id,
            channel_id=message.channel_id,
            sender="System",
            role=MessageRole.SYSTEM,
            type=MessageType.TEXT,
            content=content,
            timestamp=time.time(),
            status=MessageStatus.PENDING_AGENT,
            metadata={
                "is_self_healing_retry": True,
                "failed_path": raw_path,
            },
        )
        await self.gateway.post(healing_msg)
        logger.info(
            f"Injected self-healing notification message {healing_msg.id} into session {message.session_id} "
            f"for failed attachment {raw_path}."
        )

    async def send_question_segment(
        self,
        channel_id: str,
        question: str,
        choices: list[str],
        message: Message,
        multi: bool = False,
        max_choices: int = 1,
    ) -> None:
        """Send a single- or multi-select question to the specified channel."""
        pass

    async def send_details_segment(self, channel_id: str, title: str, body: str, message: Message) -> None:
        """Send a collapsible `<details>` section; the default renders it as `**TITLE**` followed by BODY."""
        await self.send_formatted_text(render_details_as_text(title, body), message)

    async def send_formatted_text(self, text: str, message: Message) -> None:
        """Format, chunk and send plain text through the adapter's `send_text_chunks`."""
        if not text.strip():
            return
        formatted_text = self.format_text(text)
        chunks = self.split_text_into_chunks(formatted_text, max_length=self.get_max_text_length())
        await self.send_text_chunks(message.channel_id, chunks, message)

    async def send_question_as_text(
        self, question: str, choices: list[str], message: Message, multi: bool = False, max_choices: int = 1
    ) -> None:
        """Fallback for platforms without interactive components: send the question as a numbered list."""
        await self.send_formatted_text(render_question_as_text(question, choices, multi, max_choices), message)

    async def on_message_delivered(self, message: Message) -> None:
        """Lifecycle hook triggered after a message is successfully delivered."""
        pass

    async def on_message_delivery_failed(self, message: Message, exception: Exception) -> None:
        """Lifecycle hook triggered after a message delivery fails."""
        await self.gateway.db.update_message_status(message.id, MessageStatus.ERROR)

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
            lines = []
            for a in dto.attachments:
                mime_lower = (a.mime_type or "").lower()
                if mime_lower.startswith(("image/", "audio/")):
                    lines.append(
                        f"[Attachment: {a.filename} ({a.mime_type}) saved at {a.path} "
                        "— image/audio attachments are already visible to you in this message; "
                        "do not call analyze_media on them]"
                    )
                else:
                    lines.append(f"[Attachment: {a.filename} ({a.mime_type}) saved at {a.path}]")
            files_str = "\n".join(lines)
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
            is_dm = await self.is_dm_channel(dto.channel_id)
            display_role = current_role.capitalize() if current_role and current_role != "default" else "Default"

            if is_dm:
                title = f"{display_role} DM Session on {self.chatbot_id}"
            else:
                title = dto.text[:30].strip() if dto.text else f"{self.chatbot_id} Session"
                if not title:
                    title = f"{self.chatbot_id} Session"
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

            # Always rebuild the system prompt to pick up on-disk template changes.
            # Pass pre-resolved role so default-role sessions skip opening a new DatabaseManager.
            new_sys_prompt = build_sys_prompt(
                custom_prompt=dto.custom_prompt or None,
                session=session,
                role=current_role or None,
            )
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

        old_session = await self.gateway.db.get_session_by_channel(self.chatbot_id, channel_id)
        old_role: str | None = None
        if old_session:
            bot_default = resolve_gateway_default_role(self.gateway, self.chatbot_id)
            old_role = await self.gateway.db.get_channel_role_with_inheritance(
                self.chatbot_id, channel_id, old_session.id, default=bot_default
            )

        # 1. Update in database
        await self.gateway.db.set_channel_role(self.chatbot_id, channel_id, role_name)

        # 2. Rebuild the active session system prompt if a session exists
        if old_session:
            agent = self.gateway.agent
            if agent:
                await agent.stop_session_worker(old_session.id, immediate=True)
            consolidator = getattr(getattr(self.gateway, "context", None), "memory_consolidator", None)
            if consolidator is not None and hasattr(consolidator, "notify_close"):
                try:
                    await consolidator.notify_close(old_session.id, role=old_role, reason="role_switch")
                except Exception as e:
                    logger.warning(f"Failed to notify memory consolidator on role switch: {e}")
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
