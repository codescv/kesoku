"""Base class for Kesoku chatbot adapters."""

import asyncio
import datetime
import difflib
import html
import json
import os
import re
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from kesoku.agent.history import build_clean_history, messages_to_openlcm_dicts
from kesoku.agent.prompt import build_sys_prompt
from kesoku.config import get_config
from kesoku.constants import SYSTEM_START_TIME, MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger
from kesoku.utils.async_fs import async_exists, async_realpath
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

        async def handle_lcm(reply_func: Callable[..., Awaitable[None]], channel_id: str) -> None:
            res = await self.get_session_lcm_context_by_channel(channel_id)
            if await async_exists(res):
                await reply_func(
                    "📖 Here is your beautifully formatted LCM Active Context HTML download:",
                    file_path=res
                )
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
            "lcm",
            "View the currently active Lossless Context Management (LCM) context.",
            handle_lcm,
        )
        self.commands.register(
            "context",
            "View the currently active Lossless Context Management (LCM) context.",
            handle_lcm,
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
        session = await self.gateway.db.get_session_by_channel(self.chatbot_id, channel_id)
        if session:
            await self.clear_session(session.id)
            return "♻️ Session successfully cleared. The next message will initiate a new session."
        return "⚠️ No active session found for this chat."

    async def manual_compact_session_by_channel(self, channel_id: str) -> str:
        """Manually trigger OpenLCM context compaction on the active history of this channel."""
        session = await self.gateway.db.get_session_by_channel(self.chatbot_id, channel_id)
        if not session:
            return "⚠️ No active session found for this chat."

        # Fetch active history
        history = await build_clean_history(self.gateway, session.id)
        if not history:
            return "⚠️ Active session has no messages to compact."

        lcm_engine = self.gateway.context.lcm_engine
        original_session_id = lcm_engine.current_session_id

        try:
            lcm_engine.bind_session(session.id)
            lcm_input = messages_to_openlcm_dicts(history)

            # Prepend system prompt
            if session.system_prompt:
                lcm_input.insert(0, {"role": "system", "content": session.system_prompt})

            # Check if there is eligible raw backlog
            eligible, reason = lcm_engine._leaf_compaction_candidate_status(lcm_input)
            if not eligible:
                return f"ℹ️ Context compaction is not needed right now: {reason}."

            # Force compress
            await lcm_engine.compress(lcm_input)
            return (
                "🔄 Lossless Context Compaction completed successfully! "
                "Old turns have been compacted into summary nodes."
            )
        except Exception as e:
            logger.error(f"Failed manual compaction for session {session.id}: {e}")
            return f"⚠️ Failed to compact history: {e}"
        finally:
            if original_session_id and original_session_id != session.id:
                lcm_engine.bind_session(original_session_id)

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

    async def get_session_lcm_context_by_channel(self, channel_id: str) -> str:
        """Get the currently active assembled LCM context (what the LLM sees) for the channel."""
        session = await self.gateway.db.get_session_by_channel(self.chatbot_id, channel_id)
        if not session:
            return "⚠️ No active session found for this chat."

        history = await build_clean_history(self.gateway, session.id)
        if not history:
            return "⚠️ Active session has no messages."

        lcm_engine = self.gateway.context.lcm_engine
        original_session_id = lcm_engine.current_session_id

        try:
            # Ensure session is bound to read summaries correctly from openlcm.db
            lcm_engine.bind_session(session.id)

            lcm_input = messages_to_openlcm_dicts(history)

            system_message = None
            remaining_messages = list(lcm_input)
            if remaining_messages and remaining_messages[0].get("role") == "system":
                system_message = remaining_messages.pop(0)

            if not system_message and session.system_prompt:
                system_message = {"role": "system", "content": session.system_prompt}

            assembled = lcm_engine._assemble_context(system_message, remaining_messages)

            sys_msg = ""
            fresh_msgs = []

            for msg in assembled:
                role = msg.get("role")
                content = msg.get("content") or ""
                if role == "system":
                    sys_msg = content
                elif role == "user" and "[Note: This conversation uses Lossless Context Management" in content:
                    continue
                elif role == "assistant" and (
                    "I have access to the full conversation history through LCM tools" in content
                ):
                    continue
                else:
                    fresh_msgs.append(msg)

            all_nodes = lcm_engine._dag.get_session_nodes(session.id)

            from openlcm.core.tokens import count_messages_tokens
            total_tokens = count_messages_tokens(assembled)

            summary_nodes_html = []
            from collections import defaultdict
            by_depth = defaultdict(list)
            for node in all_nodes:
                by_depth[node.depth].append(node)

            max_dag_depth = max(by_depth.keys()) if by_depth else -1
            for depth in range(max_dag_depth, -1, -1):
                nodes_at_depth = sorted(by_depth.get(depth, []), key=lambda nd: nd.created_at)
                depth_label = {0: "Recent", 1: "Session Arc", 2: "Durable"}.get(depth, f"Depth-{depth}")
                for node in nodes_at_depth:
                    safe_summary = html.escape(node.summary).replace("\n", "<br>")
                    savings = round(node.source_token_count / node.token_count, 1) if node.token_count > 0 else 1
                    hint_text = node.expand_hint or f"lcm_expand(node_id={node.node_id}) to retrieve details"
                    safe_hint = html.escape(hint_text)

                    node_html = f"""
                    <div class="node-card">
                        <div class="node-header">
                            <span class="badge depth-{node.depth}">{depth_label} Node {node.node_id}</span>
                            <span style="font-size:0.85rem;color:#8899a6;">
                                {node.source_token_count} tokens &rarr; {node.token_count} tokens ({savings}x savings)
                            </span>
                        </div>
                        <div style="white-space: pre-wrap; font-size:0.95rem;">{safe_summary}</div>
                        <div style="font-size:0.8rem;color:#1d9bf0;margin-top:8px;font-style:italic;">
                            Hint: {safe_hint}
                        </div>
                    </div>
                    """
                    summary_nodes_html.append(node_html)

            summary_nodes_html_str = (
                "\n".join(summary_nodes_html) if summary_nodes_html else "<p>*(No active compacted summary nodes)*</p>"
            )

            fresh_tail_html = []
            for msg in fresh_msgs:
                role = msg.get("role")
                content = msg.get("content") or ""

                bubble_class = (
                    "user"
                    if role == "user"
                    else "assistant"
                    if role == "assistant"
                    else "tool"
                    if role == "tool"
                    else "system"
                )
                if role == "user":
                    role_label = "User"
                elif role == "assistant":
                    role_label = "Assistant"
                elif role == "tool":
                    role_label = "Tool Result"
                else:
                    role_label = str(role).capitalize()

                if role == "tool":
                    tool_name = msg.get("name") or "unknown_tool"
                    safe_content = f"""
                    <div class="tool-response-block">
                        📥 <strong>Tool Output (<code>{tool_name}</code>):</strong><br>
                        <pre class="tool-response-pre">{html.escape(content)}</pre>
                    </div>
                    """
                else:
                    safe_content = html.escape(content).replace("\n", "<br>")

                # Display embedded tool calls if present inside the assistant role message
                if role == "assistant" and msg.get("tool_calls"):
                    tool_call_html_parts = []
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", {})
                        name = fn.get("name", "unknown_tool")
                        arguments = fn.get("arguments", "")
                        try:
                            if isinstance(arguments, str):
                                args_dict = json.loads(arguments)
                                arguments_pretty = json.dumps(args_dict, indent=2, ensure_ascii=False)
                            else:
                                arguments_pretty = json.dumps(arguments, indent=2, ensure_ascii=False)
                        except Exception:
                            arguments_pretty = str(arguments)

                        tool_call_html_parts.append(
                            f'<div class="tool-call-block">'
                            f'🔧 <strong>Called Tool:</strong> <code>{name}</code><br>'
                            f'<pre class="tool-args-pre">{html.escape(arguments_pretty)}</pre>'
                            f'</div>'
                        )

                    tool_calls_html = "\n".join(tool_call_html_parts)
                    if safe_content:
                        safe_content += f"<br><br>{tool_calls_html}"
                    else:
                        safe_content = tool_calls_html

                msg_html = f"""
                <div class="chat-bubble {bubble_class}">
                    <div class="bubble-content">
                        <strong>{role_label}:</strong><br>
                        {safe_content}
                    </div>
                </div>
                """
                fresh_tail_html.append(msg_html)

            fresh_tail_html_str = "\n".join(fresh_tail_html) if fresh_tail_html else "<p>*(Fresh tail is empty)*</p>"

            html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>LCM Active Context - Session {session.id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0f1419;
            color: #e1e8ed;
            margin: 0;
            padding: 20px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        header {{
            background-color: #15181c;
            border: 1px solid #2f3336;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        h1 {{
            margin-top: 0;
            font-size: 1.5rem;
            color: #1d9bf0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .stat-card {{
            background-color: #1e2732;
            border-radius: 6px;
            padding: 10px 15px;
            border-left: 4px solid #1d9bf0;
        }}
        .stat-label {{
            font-size: 0.8rem;
            color: #8899a6;
            text-transform: uppercase;
        }}
        .stat-value {{
            font-size: 1.2rem;
            font-weight: bold;
            margin-top: 5px;
        }}
        details {{
            background-color: #15181c;
            border: 1px solid #2f3336;
            border-radius: 8px;
            margin-bottom: 15px;
            padding: 15px;
        }}
        summary {{
            font-weight: bold;
            cursor: pointer;
            outline: none;
            user-select: none;
            color: #1d9bf0;
        }}
        pre {{
            background-color: #000000;
            color: #00ff00;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
            font-family: "Fira Code", Consolas, Monaco, monospace;
            font-size: 0.9rem;
            border: 1px solid #202327;
            white-space: pre-wrap;
        }}
        .node-card {{
            background-color: #1e2732;
            border: 1px solid #2f3336;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 10px;
        }}
        .node-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #2f3336;
            padding-bottom: 8px;
            margin-bottom: 10px;
        }}
        .badge {{
            background-color: #1d9bf0;
            color: #ffffff;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: bold;
        }}
        .badge.depth-0 {{ background-color: #10b981; }}
        .badge.depth-1 {{ background-color: #f59e0b; }}
        .badge.depth-2 {{ background-color: #ef4444; }}

        .chat-bubble {{
            display: flex;
            flex-direction: column;
            margin-bottom: 15px;
            max-width: 85%;
        }}
        .chat-bubble.user {{
            margin-left: auto;
            align-items: flex-end;
        }}
        .chat-bubble.assistant {{
            margin-right: auto;
            align-items: flex-start;
        }}
        .chat-bubble.tool {{
            margin-right: auto;
            align-items: flex-start;
            width: 100%;
        }}
        .bubble-content {{
            padding: 12px 16px;
            border-radius: 18px;
            font-size: 0.95rem;
        }}
        .chat-bubble.user .bubble-content {{
            background-color: #1d9bf0;
            color: #ffffff;
            border-bottom-right-radius: 4px;
        }}
        .chat-bubble.assistant .bubble-content {{
            background-color: #2f3336;
            color: #e1e8ed;
            border-bottom-left-radius: 4px;
        }}
        .chat-bubble.tool .bubble-content {{
            background-color: #14221f;
            color: #e1e8ed;
            border-bottom-left-radius: 4px;
            width: 100%;
            box-sizing: border-box;
        }}
        .tool-call-block {{
            background-color: #1e2732;
            border: 1px solid #2f3336;
            border-radius: 8px;
            padding: 12px;
            margin-top: 10px;
            font-size: 0.9rem;
            align-self: stretch;
        }}
        .tool-response-block {{
            background-color: #14221f;
            border: 1px solid #10b981;
            border-radius: 8px;
            padding: 12px;
            margin-top: 10px;
            font-size: 0.9rem;
            align-self: stretch;
        }}
        .tool-response-pre {{
            background-color: #0d1512;
            color: #34d399;
            padding: 8px;
            border-radius: 4px;
            font-family: "Fira Code", Consolas, Monaco, monospace;
            font-size: 0.85rem;
            margin: 8px 0 0 0;
            border: 1px solid #10b981;
            white-space: pre-wrap;
        }}
        .tool-args-pre {{
            background-color: #0f1419;
            color: #ffaa00;
            padding: 8px;
            border-radius: 4px;
            font-family: "Fira Code", Consolas, Monaco, monospace;
            font-size: 0.85rem;
            margin: 8px 0 0 0;
            border: 1px solid #2f3336;
            white-space: pre-wrap;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📖 Lossless Context Management (LCM) Active Context</h1>
            <p style="margin:0;color:#8899a6;">Session: <strong>{session.id}</strong></p>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Summaries</div>
                    <div class="stat-value">{len(all_nodes)} Nodes</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Fresh Tail</div>
                    <div class="stat-value">{len(fresh_msgs)} Messages</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Active Context Size</div>
                    <div class="stat-value">{total_tokens} Tokens</div>
                </div>
            </div>
        </header>

        <details>
            <summary>🛠️ System Message Instructions</summary>
            <pre>{html.escape(sys_msg)}</pre>
        </details>

        <details open>
            <summary>📦 Compacted Summary Scaffold (DAG Nodes)</summary>
            <div style="margin-top: 15px;">
                {summary_nodes_html_str}
            </div>
        </details>

        <details open>
            <summary>🧵 Active Fresh Tail (Chronological Messages)</summary>
            <div style="margin-top: 15px; display: flex; flex-direction: column;">
                {fresh_tail_html_str}
            </div>
        </details>
    </div>
</body>
</html>"""

            # Write to a temporary HTML file safely
            temp_file = tempfile.NamedTemporaryFile(
                suffix="_lcm_context.html", delete=False, mode="w", encoding="utf-8"
            )
            temp_file.write(html_template)
            temp_file.close()

            return temp_file.name
        except Exception as e:
            logger.error(f"Failed to get LCM context by channel: {e}")
            return f"⚠️ Failed to retrieve LCM context: {e}"
        finally:
            if original_session_id and original_session_id != session.id:
                lcm_engine.bind_session(original_session_id)

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
        session = await self.gateway.db.get_session_by_channel(self.chatbot_id, channel_id)
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
            await self.gateway.db.update_session_updated_at(session.id, time.time())

        now_dt = datetime.datetime.now()
        msg_content = prompt_content


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
        if await async_exists(cleaned_path):
            return await async_realpath(cleaned_path)

        # 2. Get session staging directory
        staging_dir = None
        session = await self.gateway.db.get_session(session_id)
        if session:
            staging_dir = self.get_session_staging_dir(session.workspace_name)

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

                raw_abs_path = await async_realpath(cleaned_path)
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
                            content = content[:table.start_idx] + file_tag + content[table.end_idx:]
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
        session = await self.gateway.db.get_session_by_channel(self.chatbot_id, dto.channel_id)
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
            await self.gateway.db.update_session_updated_at(session.id, time.time())

            if dto.custom_prompt:
                new_sys_prompt = build_sys_prompt(custom_prompt=dto.custom_prompt, session=session)
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
        session = await self.gateway.db.get_session_by_channel(self.chatbot_id, channel_id)
        if session:
            new_sys_prompt = build_sys_prompt(session=session)
            await self.gateway.db.update_session_system_prompt(session.id, new_sys_prompt)

        return f"🎭 Persona for this channel has been successfully changed to **`{role_name}`**."
