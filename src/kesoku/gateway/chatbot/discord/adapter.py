"""Discord chatbot adapter for Kesoku AI Agent framework.

Connects Discord channels and threads with Kesoku Gateway using Pub/Sub.
"""

import asyncio
import datetime
import inspect
import os
from collections import OrderedDict, defaultdict
from typing import Any

import aiohttp

import discord
from kesoku.config import DiscordChannelOverride, DiscordConfig, get_config, resolve_gateway_default_role
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.gateway.attachment_manager import AttachmentManager
from kesoku.gateway.chatbot.base import (
    ChannelUnavailableError,
    Chatbot,
    DeliveryAbortedError,
    InboundMessageAttachment,
    InboundMessageDTO,
    parse_message_content,
)
from kesoku.gateway.chatbot.deferred_media import (
    format_deferred_failure_message,
    sanitize_error_reason,
    sanitize_placeholder,
)
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger
from kesoku.utils.async_fs import (
    async_exists,
)
from kesoku.utils.table import parse_markdown_tables
from kesoku.utils.text import clean_latex, format_text, render_spoilers

from .command import setup_discord_commands
from .persistent_ui import (
    PERSISTENT_ITEMS,
    build_details_view,
    build_question_embed,
    build_question_spec,
    build_question_view,
    ids_fit,
    register_chatbot,
)
from .ui import DetailsView, MessageHeaderView, QuestionView, render_details_content
from .voice import send_voice_message

logger = setup_logger(__name__)


DISCORD_MAX_CONTENT_LENGTH = 2000
STREAM_PREVIEW_MAX_LENGTH = 1900

TOOL_EMOJI_MAP: dict[str, str] = {
    "run_shell_command": "💻",
    "web_search": "🔍",
    "update_file": "📝",
    "read_file": "📖",
    "analyze_media": "🖼️",
    "memory_search": "🧠",
    "use_skill": "🧩",
    "list_skills": "🧩",
    "skill_manager": "🧩",
    "manage_job": "⚙️",
    "delegate_subagent": "🤖",
}
DEFAULT_TOOL_EMOJI = "🛠️"
MAX_VISIBLE_TOOL_CALLS = 12
MAX_TRACKED_VOICE_ANCHORS = 256
ATTACHMENT_DOWNLOAD_CONCURRENCY = 6

# Discord message types that represent an actual user-authored chat message.
USER_CHAT_MESSAGE_TYPES: tuple[discord.MessageType, ...] = (
    discord.MessageType.default,
    discord.MessageType.reply,
)


def is_user_chat_message(message: discord.Message) -> bool:
    """Return True if a Discord message is a user-authored chat message.

    System messages (channel/thread renames, pins, thread creation notices,
    recipient adds, ...) must never trigger an agent turn.
    """
    if message.type not in USER_CHAT_MESSAGE_TYPES:
        return False
    return not message.is_system()


def get_tool_emoji(tool_name: str) -> str:
    """Get representative emoji for the given tool name.

    Args:
        tool_name: The tool identifier.

    Returns:
        The mapped emoji or default tool emoji.
    """
    return TOOL_EMOJI_MAP.get(tool_name, DEFAULT_TOOL_EMOJI)


def compact_path(path_str: str, max_len: int = 40) -> str:
    """Compact long file paths by keeping trailing path components.

    Args:
        path_str: File path to compact.
        max_len: Maximum length threshold before compaction.

    Returns:
        Compacted path string.
    """
    if len(path_str) <= max_len:
        return path_str

    parts = path_str.replace("\\", "/").rstrip("/").split("/")
    if len(parts) <= 1:
        return path_str[: max_len - 3] + "..."

    filename = parts[-1]
    if len(filename) >= max_len - 4:
        return "..." + filename[-(max_len - 4) :]

    compacted = filename
    for p in reversed(parts[:-1]):
        if not p:
            continue
        candidate = f"{p}/{compacted}"
        if len(candidate) + 4 <= max_len:
            compacted = candidate
        else:
            break

    return f".../{compacted}"


def format_tool_arguments(tool_name: str, tool_args: dict[str, Any] | None, max_len: int = 120) -> str:
    """Extract and format core tool arguments concisely for chat UI display.

    Args:
        tool_name: Name of the tool.
        tool_args: Raw tool arguments dictionary.
        max_len: Maximum length for the formatted arguments string.

    Returns:
        Concise, formatted argument string.
    """
    if not isinstance(tool_args, dict):
        return ""

    # Filter out context and internal framework arguments
    filtered = {k: v for k, v in tool_args.items() if k not in {"context", "session_id", "active_jobs", "gateway"}}
    if not filtered:
        return ""

    arg_str = ""

    if tool_name == "run_shell_command":
        cmd = filtered.get("command") or filtered.get("cmd") or ""
        if isinstance(cmd, str):
            arg_str = cmd.strip()

    elif tool_name in {"web_search", "memory_search"}:
        query = filtered.get("query") or filtered.get("q") or ""
        if isinstance(query, str):
            arg_str = query.strip()

    elif tool_name == "update_file":
        target = (
            filtered.get("file_name") or filtered.get("path") or filtered.get("file_path")
            or filtered.get("target_file") or ""
        )
        mode = filtered.get("mode") or filtered.get("action")
        if not mode and "file_name" in filtered:
            mode = "create" if not filtered.get("old_content") else None
        if target:
            c_path = compact_path(str(target), max_len=60)
            arg_str = f"{c_path} ({mode})" if mode else c_path

    elif tool_name == "read_file":
        target = filtered.get("file_name") or filtered.get("path") or ""
        if target:
            arg_str = compact_path(str(target), max_len=60)
            start, end = filtered.get("start_line"), filtered.get("end_line")
            if start or end:
                arg_str += f" [{start or 1}-{end or ''}]"

    elif tool_name == "analyze_media":
        media_path = filtered.get("media_path") or filtered.get("path") or ""
        prompt = filtered.get("prompt") or ""
        c_path = compact_path(str(media_path), max_len=40) if media_path else ""
        if c_path and prompt:
            arg_str = f"{c_path}: {prompt}"
        elif c_path:
            arg_str = c_path
        elif prompt:
            arg_str = str(prompt)

    elif tool_name == "use_skill":
        skill_name = filtered.get("skill_name") or filtered.get("name") or ""
        args_payload = filtered.get("args") or filtered.get("parameters")
        if skill_name and args_payload:
            arg_str = f"{skill_name} ({args_payload})"
        elif skill_name:
            arg_str = str(skill_name)

    elif tool_name == "manage_job":
        action = filtered.get("action") or ""
        job_id = filtered.get("job_id") or ""
        arg_str = f"{action} {job_id}".strip()

    elif tool_name == "delegate_subagent":
        task = filtered.get("task") or ""
        arg_str = str(task)

    elif tool_name == "skill_manager":
        action = filtered.get("action") or ""
        name = filtered.get("name") or filtered.get("skill_name") or ""
        if action and name:
            arg_str = f"{action} {name}"
        elif action:
            arg_str = str(action)
        elif name:
            arg_str = str(name)

    # Generic fallback
    if not arg_str:
        if len(filtered) == 1:
            val = next(iter(filtered.values()))
            if isinstance(val, str) and (val.startswith("/") or "/" in val):
                arg_str = compact_path(val, max_len=max_len)
            else:
                arg_str = str(val)
        else:
            pairs = []
            for k, v in filtered.items():
                if isinstance(v, str) and (v.startswith("/") or "/" in v):
                    v_str = compact_path(v, max_len=40)
                else:
                    v_str = str(v)
                pairs.append(f"{k}: {v_str}")
            arg_str = ", ".join(pairs)

    # Clean display
    arg_str = arg_str.replace("\n", " ").strip()
    if len(arg_str) > max_len:
        arg_str = arg_str[:max_len] + "..."

    return arg_str


def extract_error_snippet(content: str, max_len: int = 50) -> str:
    """Extract a concise single-line error snippet from tool failure content.

    Args:
        content: Error message or traceback content.
        max_len: Maximum length for the snippet.

    Returns:
        Cleaned, short error snippet in parentheses or empty string.
    """
    if not content:
        return ""

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return ""

    # Look for common exception or error patterns
    error_line = lines[-1]
    for line in lines:
        if any(err_kw in line for err_kw in ("Error:", "Exception:", "failed:", "exit status", "exit code", "Errno")):
            error_line = line
            break

    # Strip prefixes if present
    for prefix in ("Exception: ", "Error: ", "RuntimeError: ", "ShellCommandError: "):
        if prefix in error_line:
            error_line = error_line.split(prefix, 1)[1].strip()

    error_line = error_line.replace("\n", " ").strip()
    if len(error_line) > max_len:
        error_line = error_line[:max_len] + "..."

    return f"({error_line})" if error_line else ""


def render_turn_special_items(items: list[dict[str, Any]]) -> str:
    """Render intermediate special items (thoughts, tool calls, system messages) for Discord display.

    Applies rolling folding if the number of completed tool calls exceeds MAX_VISIBLE_TOOL_CALLS.

    Args:
        items: List of tracked special items for the turn.

    Returns:
        Formatted multi-line Discord string.
    """
    # Separate tool calls to determine if folding is needed
    tool_call_indices = [idx for idx, item in enumerate(items) if item.get("type") == "tool_call"]
    total_tools = len(tool_call_indices)

    hidden_indices = set()
    folded_count = 0
    if total_tools > MAX_VISIBLE_TOOL_CALLS:
        keep_count = MAX_VISIBLE_TOOL_CALLS - 2
        fold_candidates = [idx for idx in tool_call_indices if items[idx].get("status") in {"✅", "❌"}]
        num_to_fold = len(fold_candidates) - keep_count
        if num_to_fold > 0:
            hidden_indices = set(fold_candidates[:num_to_fold])
            folded_count = num_to_fold

    lines = []
    summary_inserted = False

    for idx, item in enumerate(items):
        item_type = item.get("type")

        if item_type == "thought":
            lines.append(f"💭 {item['content']}")
        elif item_type == "system":
            lines.append(f"⚙️ *System Message:* {item['content']}")
        elif item_type == "tool_call":
            if idx in hidden_indices:
                if not summary_inserted:
                    lines.append(f"📦 *... ({folded_count} tools completed ✅)*")
                    summary_inserted = True
                continue

            emoji = item.get("emoji") or get_tool_emoji(item.get("tool_name", ""))
            arg_str = item.get("arg_str", "")
            status = item.get("status", "⏳")
            error_snippet = item.get("error_snippet", "")

            if arg_str:
                if status == "❌" and error_snippet:
                    lines.append(f"{emoji} `{arg_str}` ❌ `{error_snippet}`")
                else:
                    lines.append(f"{emoji} `{arg_str}` {status}")
            else:
                if status == "❌" and error_snippet:
                    lines.append(f"{emoji} ❌ `{error_snippet}`")
                else:
                    lines.append(f"{emoji} {status}")

    new_content = "\n".join(lines)
    if len(new_content) > DISCORD_MAX_CONTENT_LENGTH:
        new_content = new_content[: DISCORD_MAX_CONTENT_LENGTH - len(" (omitted)")] + " (omitted)"

    return new_content


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

    # Build location instruction and channel topic
    if is_dm:
        location_instruction = "You are talking to the user via discord DM (direct message)."
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

    discord_prompt = f"""
# Discord Instructions
{location_instruction}
{members_section}
{mention_section}
{topic_section}
    """
    return discord_prompt


_TRANSIENT_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


def _is_transient_discord_error(exc: BaseException) -> bool:
    """Return True when a Discord REST/transport error is transient and safe to retry."""
    if isinstance(exc, FileNotFoundError):
        return False
    if isinstance(exc, (discord.DiscordServerError, aiohttp.ClientConnectionError, TimeoutError)):
        return True
    if isinstance(exc, (aiohttp.ClientError, ConnectionError)):
        return True
    if isinstance(exc, discord.HTTPException) and exc.status in _TRANSIENT_STATUSES:
        return True
    return False


def _is_transient_discord_upload_error(exc: Exception) -> bool:
    """Return True when a Discord attachment upload error is transient and retryable."""
    if _is_transient_discord_error(exc):
        return True
    if isinstance(exc, OSError) and not isinstance(exc, (FileNotFoundError, IsADirectoryError, PermissionError)):
        return True
    if isinstance(exc, discord.HTTPException) and exc.status >= 500:
        return True
    return False


class DiscordChatbot(Chatbot):
    """Discord chatbot adapter connecting to Kesoku Gateway broker."""

    supports_deferred_media: bool = True
    _voice_inline_grace_sec: float = 4.0

    def __init__(
        self,
        chatbot_id: str,
        gateway: Gateway,
        bot_token: str | None = None,
        discord_config: DiscordConfig | None = None,
    ) -> None:
        """Initialize the Discord chatbot adapter.

        Args:
            chatbot_id: Unique identifier for this chatbot instance.
            gateway: Kesoku Gateway instance.
            bot_token: Optional Discord bot token. Defaults to config setting.
            discord_config: Optional specific DiscordConfig. Defaults to config setting.
        """
        super().__init__(chatbot_id, gateway)
        cfg = get_config()
        if discord_config is not None:
            self.config = discord_config
        else:
            if isinstance(cfg.discord, list):
                self.config = next((c for c in cfg.discord if c.chatbot_id == chatbot_id), None)
                if not self.config:
                    self.config = DiscordConfig(chatbot_id=chatbot_id)
            else:
                self.config = cfg.discord

        self.attachment_manager = AttachmentManager()
        self.bot_token = bot_token or self.config.bot_token or os.environ.get("DISCORD_TOKEN")
        if not self.bot_token:
            raise ValueError("Discord bot token is required but not configured.")

        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.guilds = True

        proxy = (
            os.environ.get("https_proxy")
            or os.environ.get("HTTPS_PROXY")
            or os.environ.get("http_proxy")
            or os.environ.get("HTTP_PROXY")
        )
        if proxy:
            logger.info("Using proxy settings from environment variable: %s", proxy)
        self.bot = discord.Client(intents=intents, proxy=proxy)
        self.bot.event(self.on_ready)
        self.bot.event(self.on_message)
        # Restart-persistent question/details components are rebuilt from their custom_id on click.
        register_chatbot(self.bot, self)
        self.bot.setup_hook = self._setup_hook  # type: ignore[method-assign]
        setup_discord_commands(self)
        self._subscriber_task: asyncio.Task[None] | None = None
        self._sent_tool_calls: dict[str, discord.Message] = {}
        self._header_views: dict[str, tuple[discord.Message, MessageHeaderView]] = {}
        self._typing_tasks: dict[str, asyncio.Task[None]] = {}
        self._intermediate_messages: defaultdict[str, list[discord.Message]] = defaultdict(list)
        self._turn_special_items: dict[str, list[dict[str, Any]]] = {}
        self._turn_special_msg: dict[str, discord.Message] = {}
        self._details_seq: dict[str, int] = {}
        self._stream_previews: dict[str, discord.Message] = {}
        self._stream_preview_texts: dict[str, str] = {}
        self._stream_preview_channels: dict[str, str] = {}
        self._last_text_msg: OrderedDict[str, Any] = OrderedDict()
        register_stream_fn = getattr(gateway, "register_stream_handler", None)
        if callable(register_stream_fn):
            register_stream_fn(self.chatbot_id, self.handle_stream_preview)

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
        for override in self.config.channels:
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
        self._subscriber_task = self.spawn_subscriber_task()
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
        self._stream_previews.clear()
        self._stream_preview_texts.clear()
        self._stream_preview_channels.clear()
        self._last_text_msg.clear()
        if not self.bot.is_closed():
            asyncio.create_task(self.bot.close())

    def _format_stream_preview(self, text: str) -> str:
        """Format accumulated streaming text with head-truncation, code-fence auto-close, and cursor."""
        formatted = self.format_text(text)
        # Reserve up to 8 characters for ellipsis ("…"), closing fence ("\n```\n"), and cursor (" ▌").
        max_body = STREAM_PREVIEW_MAX_LENGTH - 8
        if len(formatted) > max_body:
            formatted = formatted[:max_body] + "…"
        if formatted.count("```") % 2 == 1:
            return f"{formatted}\n```\n▌"
        return f"{formatted} ▌"

    async def _cleanup_stream_preview(self, session_id: str) -> None:
        """Pop and delete any active streaming preview message for `session_id` without raising."""
        preview_msg = self._stream_previews.pop(session_id, None)
        self._stream_preview_texts.pop(session_id, None)
        self._stream_preview_channels.pop(session_id, None)
        if preview_msg is None:
            return
        try:
            await self._call_with_timeout(
                preview_msg.delete(),
                error_msg="Timeout deleting stream preview message",
            )
        except discord.NotFound:
            pass
        except Exception as e:
            logger.warning(f"Failed to delete stream preview message for session {session_id}: {e}")

    async def handle_stream_preview(
        self,
        channel_id: str,
        session_id: str,
        turn_id: str,
        text: str | None,
    ) -> None:
        """Create, edit, or delete the ephemeral streaming preview message in Discord.

        Preview messages are intentionally kept separate from `_intermediate_messages` and
        `_turn_special_msg` so post-delivery cleanup never deletes an edited-in-place final reply.
        """
        _ = turn_id
        if text is None:
            await self._cleanup_stream_preview(session_id)
            return

        agent = self.gateway.agent
        if agent:
            worker = agent.workers.get(session_id)
            if worker and not worker.queue_empty():
                return

        if not text.strip():
            return

        rendered = self._format_stream_preview(text)
        if session_id in self._stream_previews:
            if self._stream_preview_channels.get(session_id) != str(channel_id):
                await self._cleanup_stream_preview(session_id)
            else:
                if self._stream_preview_texts.get(session_id) == rendered:
                    return
                preview_msg = self._stream_previews[session_id]
                try:
                    await self._call_with_timeout(
                        preview_msg.edit(content=rendered),
                        error_msg="Timeout editing stream preview message",
                    )
                    self._stream_preview_texts[session_id] = rendered
                except discord.NotFound:
                    self._stream_previews.pop(session_id, None)
                    self._stream_preview_texts.pop(session_id, None)
                    self._stream_preview_channels.pop(session_id, None)
                except Exception as e:
                    logger.warning(f"Failed to edit stream preview in channel {channel_id}: {e}")
                return

        try:
            channel = await self._get_discord_channel(channel_id)
            if not channel:
                return
            sent_msg = await self._call_with_timeout(
                channel.send(rendered),
                error_msg="Timeout sending stream preview message",
            )
            self._stream_previews[session_id] = sent_msg
            self._stream_preview_texts[session_id] = rendered
            self._stream_preview_channels[session_id] = str(channel_id)
        except discord.NotFound:
            pass
        except Exception as e:
            logger.warning(f"Failed to send stream preview in channel {channel_id}: {e}")

    def format_text(self, text: str) -> str:
        """Clean, normalize, render `<spoiler>` as native `||...||`, and apply LaTeX filter."""
        # Render spoilers before the base formatter strips the tags.
        formatted = format_text(render_spoilers(text, "||", "||"))
        return clean_latex(formatted)

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
        tool_name = message.metadata.get("tool_name") or message.sender or "unknown_tool"
        tool_args = message.metadata.get("tool_arguments")
        arg_str = format_tool_arguments(tool_name, tool_args, max_len=120)
        return f": `{arg_str}`" if arg_str else ""

    async def _setup_hook(self) -> None:
        """Register persistent DynamicItem components so clicks on old messages work after a restart."""
        self.bot.add_dynamic_items(*PERSISTENT_ITEMS)

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

    async def pre_ingest_interruption_hook(self, session: Any, dto: InboundMessageDTO) -> None:
        """Hook executed after session is resolved/created, but before posting."""
        channel_id = dto.channel_id
        session_id = session.id

        # Clean up any active previous turn UI elements if a new turn is started (thought interruption)
        task = self._typing_tasks.pop(channel_id, None)
        if task:
            task.cancel()

        # Check if the agent is actually processing a turn for this session
        agent = self.gateway.agent
        is_interrupted = False
        if agent:
            worker = agent.workers.get(session_id)
            if worker and worker.running and worker._processing_turn:
                is_interrupted = True

        if is_interrupted:
            await self._cleanup_stream_preview(session_id)
            intermediate_msgs = self._intermediate_messages.pop(channel_id, [])
            if intermediate_msgs:
                for msg in intermediate_msgs:
                    try:
                        await msg.delete()
                    except Exception as de:
                        logger.warning(f"Failed to delete intermediate message {msg.id} on interruption: {de}")

            # Find the active turn's header view to delete
            history = await self.gateway.db.get_session_history(session_id, limit=20)
            active_user_msg = None
            for msg in reversed(history):
                if msg.role == MessageRole.USER:
                    active_user_msg = msg
                    break

            turn_id = active_user_msg.id if active_user_msg else session_id

            # Fallback: if turn_id not in _header_views, try session_id
            if turn_id not in self._header_views and session_id in self._header_views:
                turn_id = session_id

            if turn_id in self._header_views:
                header_msg, header_view = self._header_views.pop(turn_id)
                try:
                    await header_msg.delete()
                except Exception as he:
                    logger.warning(f"Failed to delete header message {header_msg.id} on interruption: {he}")

            self._turn_special_items.pop(session_id, None)
            self._turn_special_msg.pop(session_id, None)

    async def process_attachments_hook(
        self, session: Any, dto: InboundMessageDTO, raw_message: Any
    ) -> list[InboundMessageAttachment]:
        """Hook to process, download concurrently, and prewarm image attachments."""
        message = raw_message  # raw_message is the discord.Message
        if not message or not getattr(message, "attachments", None):
            return []

        sem = asyncio.Semaphore(ATTACHMENT_DOWNLOAD_CONCURRENCY)

        async def _download_one(attachment: Any) -> InboundMessageAttachment:
            async with sem:

                async def save_cb(target_path: str, att: Any = attachment) -> None:
                    await att.save(target_path)

                saved = await self.attachment_manager.save_attachment(
                    filename=attachment.filename,
                    workspace_name=session.workspace_name,
                    save_callback=save_cb,
                    collision_id=str(attachment.id),
                )
                return InboundMessageAttachment(
                    path=saved["path"],
                    mime_type=attachment.content_type or "application/octet-stream",
                    filename=attachment.filename,
                )

        attachments_metadata = list(await asyncio.gather(*(_download_one(att) for att in message.attachments)))
        saved_paths = [att.path for att in attachments_metadata if att.path]
        if saved_paths:
            from kesoku.agent.llm import prewarm_image_attachments

            await asyncio.to_thread(prewarm_image_attachments, saved_paths)
        return attachments_metadata

    async def post_ingest_hook(self, session: Any, message: Message, dto: InboundMessageDTO) -> None:
        """Hook executed after the message is successfully posted to the gateway."""
        # Trigger typing status for the thread/channel while agent is thinking
        channel_id = dto.channel_id
        if channel_id not in self._typing_tasks:
            target_channel = await self._get_discord_channel(channel_id, dto.raw_metadata)
            if target_channel:
                self._typing_tasks[channel_id] = asyncio.create_task(self._keep_typing(target_channel))

    async def on_message(self, message: discord.Message) -> None:
        """Process incoming messages from Discord and ingest into Kesoku Gateway.

        Args:
            message: Incoming discord.Message instance.
        """
        # Ignore bot's own messages
        if message.author == self.bot.user:
            return

        # Ignore Discord system messages (thread renames, pins, thread creation, member adds, ...)
        if not is_user_chat_message(message):
            return

        # Check user_allowlist filtering
        allowlist = self.config.user_allowlist
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
        elif message.guild is None:
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

        custom_prompt = _build_discord_custom_prompt(target_channel, message.author)

        # Resolve role
        role = resolve_gateway_default_role(self.gateway, self.chatbot_id)
        db_role = await self.gateway.db.get_channel_role(self.chatbot_id, channel_id)
        if isinstance(db_role, str):
            role = db_role
        elif isinstance(target_channel, discord.Thread) and target_channel.parent:
            parent_role = await self.gateway.db.get_channel_role(self.chatbot_id, str(target_channel.parent.id))
            if isinstance(parent_role, str):
                role = parent_role

        discord_msg_content = message.content

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
            "discord_message_channel_id": str(message.channel.id),
            "discord_author_id": str(message.author.id),
            "channel_name": channel_name,
            "sender_name": f"{message.author.display_name} (ID: {message.author.id})",
        }

        if parent_channel_id:
            msg_metadata["parent_channel_id"] = parent_channel_id
        if parent_channel_name:
            msg_metadata["parent_channel_name"] = parent_channel_name

        dto = InboundMessageDTO(
            sender_id=message.author.display_name,
            channel_id=channel_id,
            text=discord_msg_content,
            message_id=str(message.id),
            timestamp=message.created_at.timestamp(),
            raw_metadata=msg_metadata,
            custom_prompt=custom_prompt,
            role=role,
        )

        await self.ingest_message(dto, raw_message=message)

    def _should_discard_preview_before_render(self, message: Message) -> bool:
        """Return True if an active preview cannot be edited in place for `message`.

        Previews must be deleted and replaced by fresh messages when the target channel changed,
        when the reply contains rich segments (files, voice, questions, details, or rendered tables),
        or when there is no non-empty text segment to edit into the preview.
        """
        preview_channel = self._stream_preview_channels.get(message.session_id)
        if preview_channel is not None and preview_channel != str(message.channel_id):
            return True
        content = message.content if isinstance(message.content, str) else ""
        if parse_markdown_tables(content):
            return True
        segments = parse_message_content(content)
        if any(seg.get("type") != "text" for seg in segments):
            return True
        return not any(seg.get("type") == "text" and str(seg.get("content", "")).strip() for seg in segments)

    async def handle_message(self, message: Message) -> None:
        """Process outgoing message from Gateway and send to target Discord thread."""
        if (
            not self.is_intermediate_message(message)
            and message.role == MessageRole.ASSISTANT
            and message.type == MessageType.TEXT
            and message.session_id in self._stream_previews
            and self._should_discard_preview_before_render(message)
        ):
            await self._cleanup_stream_preview(message.session_id)
        await self.render_outgoing_message(message)
    async def _call_with_timeout(self, coro, timeout_duration=15.0, error_msg="Discord API call timed out"):
        """Wrap a coroutine in asyncio.timeout to enforce a timeout."""
        try:
            async with asyncio.timeout(timeout_duration):
                return await coro
        except TimeoutError as te:
            logger.error(f"{error_msg} (timeout={timeout_duration}s)")
            raise DeliveryAbortedError(f"{error_msg}: {te}") from te

    async def _send_with_retry(
        self,
        make_coro: Any,
        *,
        attempts: int = 3,
        base_delay: float = 1.0,
        timeout_duration: float = 15.0,
        error_msg: str = "Discord API call timed out",
    ) -> Any:
        """Execute a Discord send coroutine factory with per-attempt timeout and transient backoff retries."""
        last: Exception | None = None
        for i in range(attempts):
            try:
                async with asyncio.timeout(timeout_duration):
                    return await make_coro()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last = e
                if i == attempts - 1 or not _is_transient_discord_error(e):
                    break
                logger.warning(
                    f"Transient Discord send error ({error_msg}, attempt {i + 1}/{attempts}): {e}. Retrying..."
                )
                await asyncio.sleep(base_delay * (2**i))
        if isinstance(last, TimeoutError):
            logger.error(f"{error_msg} (timeout={timeout_duration}s, attempts={attempts})")
            raise DeliveryAbortedError(f"{error_msg}: {last}") from last
        assert last is not None
        raise last


    async def _get_discord_channel(self, channel_id_str: str, metadata: dict[str, Any] | None = None) -> Any | None:
        try:
            target_id = int(channel_id_str)
        except ValueError:
            return None

        channel = self.bot.get_channel(target_id)
        if not channel:
            try:
                channel = await self._call_with_timeout(
                    self.bot.fetch_channel(target_id),
                    error_msg=f"Timeout fetching channel {target_id}"
                )
            except (discord.NotFound, discord.Forbidden) as fe:
                raise DeliveryAbortedError(f"Discord channel fetch failed: {fe}") from fe
            except DeliveryAbortedError:
                raise
            except Exception as e:
                logger.debug(f"Failed to fetch channel {target_id} by ID: {e}")

        if not channel and metadata:
            recipient_id_str = metadata.get("discord_author_id") or metadata.get("dm_recipient_id")
            if recipient_id_str:
                try:
                    user_id = int(recipient_id_str)
                    user = self.bot.get_user(user_id)
                    if not user:
                        user = await self._call_with_timeout(
                            self.bot.fetch_user(user_id),
                            error_msg=f"Timeout fetching user {user_id}"
                        )
                    if user:
                        channel = user.dm_channel
                        if not channel:
                            channel = await self._call_with_timeout(
                                user.create_dm(),
                                error_msg=f"Timeout creating DM for user {user_id}"
                            )
                except (discord.NotFound, discord.Forbidden) as fe:
                    raise DeliveryAbortedError(f"Discord DM user fetch failed: {fe}") from fe
                except DeliveryAbortedError:
                    raise
                except Exception as de:
                    logger.warning(f"Failed to resolve DM channel for recipient {recipient_id_str}: {de}")

        return channel

    async def _get_discord_channel_with_abort(self, message: Message) -> Any | None:
        try:
            channel = await self._get_discord_channel(message.channel_id, getattr(message, "metadata", None))
            return channel
        except DeliveryAbortedError as de:
            logger.warning(
                f"Message {message.id} Failed to fetch Discord channel "
                f"{message.channel_id} (deleted or forbidden): {de}. "
                "Aborting session."
            )
            await self.gateway.abort_session(message.session_id)
            if isinstance(de, ChannelUnavailableError):
                raise
            raise ChannelUnavailableError(str(de)) from (de.__cause__ or de)

    def supports_intermediate_messages(self) -> bool:
        """Determine if the platform supports intermediate rendering of thoughts and tools."""
        return True

    async def handle_intermediate_message(self, message: Message) -> None:
        """Render an intermediate thought/tool/system message in the Discord UI."""
        if message.type == MessageType.SESSION_RENAME:
            channel = await self._get_discord_channel_with_abort(message)
            if isinstance(channel, discord.Thread):
                try:
                    await channel.edit(name=message.content)
                    logger.info(f"Renamed Discord thread {channel.id} to '{message.content}'")
                except Exception as e:
                    logger.error(f"Failed to rename Discord thread {channel.id}: {e}")
            await self.gateway.db.update_message_status(message.id, MessageStatus.DELIVERED)
            return

        session_id = message.session_id

        # Guard: If there is a pending message in the queue, an interruption has been
        # initiated. Skip rendering any further UI output for the current (aborted) turn.
        agent = self.gateway.agent
        if agent:
            worker = agent.workers.get(session_id)
            if worker and not worker.queue_empty():
                logger.info(
                    f"Skipping intermediate message rendering for session {session_id} due to pending interruption."
                )
                return

        msg_meta = getattr(message, "metadata", None)
        if (
            message.role == MessageRole.ASSISTANT
            and message.type == MessageType.THOUGHT
            and isinstance(msg_meta, dict)
            and msg_meta.get("final_response") is True
        ):
            await self.gateway.db.update_message_status(message.id, MessageStatus.DELIVERED)
            return

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
            emoji = get_tool_emoji(tool_name)
            tool_args = message.metadata.get("tool_arguments")
            arg_str = format_tool_arguments(tool_name, tool_args, max_len=120)

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
                        "emoji": emoji,
                        "arg_str": arg_str,
                        "status": "⏳",
                        "error_snippet": "",
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

        new_content = render_turn_special_items(self._turn_special_items[session_id])

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
                header_msg = await self._call_with_timeout(
                    channel.send(
                        content=f"🔍 **Session ID:** `{message.session_id}`",
                        view=header_view,
                    ),
                    error_msg="Timeout sending message header"
                )
                self._header_views[turn_id] = (header_msg, header_view)
            except DeliveryAbortedError:
                raise
            except Exception as he:
                logger.warning(f"Failed to send message header: {he}")

        if session_id in self._turn_special_msg:
            discord_msg = self._turn_special_msg[session_id]
            try:
                await self._call_with_timeout(
                    discord_msg.edit(content=new_content),
                    error_msg="Timeout editing special message"
                )
            except DeliveryAbortedError:
                raise
            except Exception as ee:
                logger.warning(f"Failed to edit single special message: {ee}")
            await self.gateway.db.update_message_status(message.id, MessageStatus.DELIVERED)
            return

        sent_msg = await self._call_with_timeout(
            channel.send(new_content),
            error_msg="Timeout sending special message"
        )
        self._intermediate_messages[message.channel_id].append(sent_msg)
        self._turn_special_msg[session_id] = sent_msg
        if message.role == MessageRole.TOOL and message.type == MessageType.TOOL_CALL:
            self._sent_tool_calls[message.id] = sent_msg

        await self.gateway.db.update_message_status(message.id, MessageStatus.DELIVERED)

    async def handle_tool_result(self, message: Message) -> None:
        """Update the status of a previously executed tool to Success or Error emoji in the special messages list."""
        tool_call_msg_id = message.parent_id
        session_id = message.session_id
        is_error = bool(message.metadata.get("tool_error"))
        error_snippet = extract_error_snippet(message.content) if is_error else ""

        if session_id in self._turn_special_items:
            items = self._turn_special_items[session_id]
            tc_item = None
            for item in items:
                if item["type"] == "tool_call" and item["id"] == tool_call_msg_id:
                    tc_item = item
                    break

            if tc_item:
                tc_item["status"] = "❌" if is_error else "✅"
                tc_item["error_snippet"] = error_snippet

                new_content = render_turn_special_items(items)

                discord_msg = self._turn_special_msg.get(session_id)
                if discord_msg:
                    try:
                        await self._call_with_timeout(
                            discord_msg.edit(content=new_content),
                            error_msg="Timeout editing special message with tool result"
                        )
                    except DeliveryAbortedError:
                        raise
                    except Exception as ee:
                        logger.warning(f"Failed to edit single special message: {ee}")

                await self.gateway.db.update_message_status(message.id, MessageStatus.DELIVERED)
                return

        if tool_call_msg_id and tool_call_msg_id in self._sent_tool_calls:
            discord_msg = self._sent_tool_calls.pop(tool_call_msg_id)
            try:
                emoji = "❌" if is_error else "✅"
                content = discord_msg.content.replace("⏳", emoji)
                if is_error and error_snippet:
                    content = f"{content} `{error_snippet}`"
                if len(content) > DISCORD_MAX_CONTENT_LENGTH:
                    content = content[: DISCORD_MAX_CONTENT_LENGTH - len(" (omitted)")] + " (omitted)"
                await self._call_with_timeout(
                    discord_msg.edit(content=content),
                    error_msg="Timeout editing tool call message with result"
                )
            except DeliveryAbortedError:
                raise
            except Exception as ee:
                logger.warning(f"Failed to edit tool call message in-place: {ee}")

        await self.gateway.db.update_message_status(message.id, MessageStatus.DELIVERED)

    def _record_last_text_msg(self, outgoing_message_id: str, discord_msg: Any) -> None:
        """Record the latest sent Discord text chunk for an outgoing assistant message (bounded LRU)."""
        if not outgoing_message_id or discord_msg is None:
            return
        self._last_text_msg[outgoing_message_id] = discord_msg
        self._last_text_msg.move_to_end(outgoing_message_id)
        while len(self._last_text_msg) > MAX_TRACKED_VOICE_ANCHORS:
            self._last_text_msg.popitem(last=False)

    def _find_live_anchor_msg(self, message_id: str | None) -> Any | None:
        """Find a live sent message object matching ``message_id`` in ``_last_text_msg`` if present."""
        if not message_id:
            return None
        for msg_obj in reversed(self._last_text_msg.values()):
            if str(getattr(msg_obj, "id", "")) == str(message_id):
                return msg_obj
        return None

    async def _resolve_voice_anchor(self, message: Message) -> tuple[str, str | None, Any]:
        """Resolve `(channel_id, anchor_message_id, live_msg)` for a `[voice:]` segment in `message`.

        Prefers the immediately preceding text chunk sent for THIS outgoing assistant `message.id`.
        When `[voice:]` appears before any text chunk in `message.id`, falls back to the triggering
        inbound user `discord_message_id` from `message.metadata` (or parent user message in DB).
        """
        live_msg = self._last_text_msg.get(message.id) if getattr(message, "id", None) else None
        if live_msg is not None:
            mid = getattr(live_msg, "id", None)
            chan = getattr(live_msg, "channel", None)
            cid = getattr(chan, "id", None) if chan is not None else getattr(live_msg, "channel_id", None)
            return (str(cid or message.channel_id), str(mid) if mid is not None else None, live_msg)

        raw_meta = getattr(message, "metadata", None)
        meta = raw_meta if isinstance(raw_meta, dict) else {}
        inbound_mid = meta.get("discord_message_id")
        inbound_cid = meta.get("discord_message_channel_id") or message.channel_id

        if not inbound_mid and getattr(message, "parent_id", None):
            get_msg_fn = getattr(getattr(self.gateway, "db", None), "get_message", None)
            if callable(get_msg_fn):
                try:
                    res = get_msg_fn(message.parent_id)
                    parent_msg = await res if inspect.isawaitable(res) else res
                    parent_meta = getattr(parent_msg, "metadata", None) if parent_msg is not None else None
                    if isinstance(parent_meta, dict) and parent_meta.get("discord_message_id"):
                        inbound_mid = parent_meta.get("discord_message_id")
                        inbound_cid = (
                            parent_meta.get("discord_message_channel_id")
                            or getattr(parent_msg, "channel_id", None)
                            or message.channel_id
                        )
                except Exception as exc:
                    logger.debug(f"Failed to look up parent discord_message_id for {message.id}: {exc}")

        return (str(inbound_cid or message.channel_id), str(inbound_mid) if inbound_mid else None, None)

    async def _best_effort_add_reaction(
        self,
        channel_id: str | None,
        message_id: str | None,
        emoji: str,
        *,
        live_msg: Any = None,
    ) -> None:
        """Best-effort add a reaction to an anchor message without raising."""
        if not channel_id or not message_id:
            return
        try:
            await self._ensure_bot_ready()
            target_live = live_msg or self._find_live_anchor_msg(message_id)
            is_default_partial = (
                getattr(self.bot.get_partial_messageable, "__func__", None)
                is discord.Client.get_partial_messageable
            )
            if (
                is_default_partial
                and target_live is not None
                and not isinstance(target_live, (discord.Message, discord.PartialMessage))
                and inspect.iscoroutinefunction(getattr(target_live, "add_reaction", None))
            ):
                async with asyncio.timeout(10.0):
                    await target_live.add_reaction(emoji)
                return

            partial = self.bot.get_partial_messageable(int(channel_id)).get_partial_message(int(message_id))
            async with asyncio.timeout(10.0):
                await partial.add_reaction(emoji)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(f"Best-effort add_reaction({emoji!r}) on {channel_id}/{message_id} failed: {exc}")

    async def _best_effort_remove_reaction(
        self,
        channel_id: str | None,
        message_id: str | None,
        emoji: str,
    ) -> None:
        """Best-effort remove the bot's own reaction from an anchor message without raising."""
        if not channel_id or not message_id:
            return
        try:
            await self._ensure_bot_ready()
            member: Any = self.bot.user if self.bot.user is not None else discord.Object(id=0)
            target_live = self._find_live_anchor_msg(message_id)
            is_default_partial = (
                getattr(self.bot.get_partial_messageable, "__func__", None)
                is discord.Client.get_partial_messageable
            )
            if (
                is_default_partial
                and target_live is not None
                and not isinstance(target_live, (discord.Message, discord.PartialMessage))
                and inspect.iscoroutinefunction(getattr(target_live, "remove_reaction", None))
            ):
                async with asyncio.timeout(10.0):
                    await target_live.remove_reaction(emoji, member)
                return

            partial = self.bot.get_partial_messageable(int(channel_id)).get_partial_message(int(message_id))
            async with asyncio.timeout(10.0):
                await partial.remove_reaction(emoji, member)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(f"Best-effort remove_reaction({emoji!r}) on {channel_id}/{message_id} failed: {exc}")

    async def _send_failure_notice_preserving_anchor(self, message: Message, text: str) -> None:
        """Send a deferred failure text notice via `send_text_chunks` without overwriting `_last_text_msg`."""
        mid = getattr(message, "id", None)
        had_prev = bool(mid and mid in self._last_text_msg)
        prev_val = self._last_text_msg.get(mid) if had_prev and mid else None
        try:
            await self.send_text_chunks(message.channel_id, [text], message)
        finally:
            if mid:
                if had_prev:
                    self._last_text_msg[mid] = prev_val
                else:
                    self._last_text_msg.pop(mid, None)

    async def send_text_chunks(self, channel_id: str, chunks: list[str], message: Message) -> None:
        """Deliver formatted text chunks sequentially to the target channel."""
        channel = await self._get_discord_channel_with_abort(message)
        if not channel:
            return

        is_special = self.is_intermediate_message(message)
        preview_msg: discord.Message | None = None
        if (
            not is_special
            and message.role == MessageRole.ASSISTANT
            and message.type == MessageType.TEXT
            and message.session_id in self._stream_previews
        ):
            # Pop BEFORE editing so `on_message_delivered` leftover cleanup never deletes the final reply.
            preview_channel = self._stream_preview_channels.pop(message.session_id, None)
            self._stream_preview_texts.pop(message.session_id, None)
            popped_preview = self._stream_previews.pop(message.session_id, None)
            if popped_preview is not None:
                if preview_channel is not None and preview_channel != str(channel_id):
                    try:
                        await self._call_with_timeout(
                            popped_preview.delete(),
                            error_msg="Timeout deleting mismatched stream preview",
                        )
                    except discord.NotFound:
                        pass
                    except Exception as e:
                        logger.warning(f"Failed to delete mismatched stream preview: {e}")
                else:
                    preview_msg = popped_preview

        last_sent_msg: Any = None
        for chunk in chunks:
            if chunk.strip():
                if preview_msg is not None:
                    target_preview = preview_msg
                    preview_msg = None
                    try:
                        await self._call_with_timeout(
                            target_preview.edit(content=chunk),
                            error_msg="Timeout replacing stream preview with final text",
                        )
                        last_sent_msg = target_preview
                        continue
                    except Exception as e:
                        logger.warning(
                            f"Failed to replace stream preview in channel {channel_id}, falling back to send: {e}"
                        )
                sent_msg = await self._send_with_retry(
                    lambda chunk_text=chunk: channel.send(chunk_text),
                    error_msg="Timeout sending text chunk",
                )
                last_sent_msg = sent_msg
                if is_special:
                    self._intermediate_messages[message.channel_id].append(sent_msg)
                    self._turn_special_msg[message.session_id] = sent_msg
                    if message.role == MessageRole.TOOL and message.type == MessageType.TOOL_CALL:
                        self._sent_tool_calls[message.id] = sent_msg

        if not is_special and last_sent_msg is not None and getattr(message, "id", None):
            self._record_last_text_msg(message.id, last_sent_msg)

    async def send_file_segment(self, channel_id: str, file_path: str, message: Message) -> None:
        """Deliver a file attachment segment to the target Discord thread."""
        channel = await self._get_discord_channel_with_abort(message)
        if not channel:
            return

        if not await async_exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        async def _send_once() -> Any:
            discord_file = discord.File(file_path)
            try:
                return await channel.send(file=discord_file)
            finally:
                try:
                    discord_file.close()
                except Exception:
                    pass

        try:
            await self._send_with_retry(
                _send_once,
                timeout_duration=60.0,
                error_msg=f"Timeout sending file {file_path}",
            )
        except DeliveryAbortedError:
            raise
        except Exception as e:
            logger.error(f"Failed to send file {file_path} to Discord: {e}", exc_info=True)
            raise

    async def send_voice_segment(
        self,
        channel_id: str,
        file_path: str,
        message: Message,
        *,
        message_reference: dict[str, Any] | None = None,
    ) -> None:
        """Deliver a voice segment to the target Discord thread (falling back to standard attachment if failed)."""
        channel = await self._get_discord_channel_with_abort(message)
        if not channel:
            return

        if not await async_exists(file_path):
            logger.error(f"Voice file not found: {file_path}")
            raise FileNotFoundError(f"Voice file not found: {file_path}")
        try:
            if message_reference:
                try:
                    await self._send_with_retry(
                        lambda: send_voice_message(channel, file_path, message_reference=message_reference),
                        timeout_duration=60.0,
                        error_msg=f"Timeout sending voice message {file_path}",
                    )
                except discord.HTTPException as http_exc:
                    err_lower = str(http_exc).lower()
                    if (
                        http_exc.status == 400
                        or "message_reference" in err_lower
                        or "reference" in err_lower
                    ):
                        logger.warning(
                            f"Discord rejected voice message_reference for {file_path} "
                            f"(status={http_exc.status}): {http_exc}. Retrying once without message_reference."
                        )
                        await self._send_with_retry(
                            lambda: send_voice_message(channel, file_path),
                            timeout_duration=60.0,
                            error_msg=f"Timeout sending voice message {file_path}",
                        )
                    else:
                        raise
            else:
                await self._send_with_retry(
                    lambda: send_voice_message(channel, file_path),
                    timeout_duration=60.0,
                    error_msg=f"Timeout sending voice message {file_path}",
                )
        except DeliveryAbortedError:
            raise
        except Exception as e:
            logger.warning(
                f"Failed to send native voice message for {file_path}: {e}. "
                "Falling back to standard file attachment."
            )

            async def _send_voice_fallback_once() -> Any:
                discord_file = discord.File(file_path)
                try:
                    return await channel.send(file=discord_file)
                finally:
                    try:
                        discord_file.close()
                    except Exception:
                        pass

            try:
                await self._send_with_retry(
                    _send_voice_fallback_once,
                    timeout_duration=60.0,
                    error_msg=f"Timeout sending voice file fallback {file_path}",
                )
            except DeliveryAbortedError:
                raise
            except Exception as fe:
                logger.error(f"Failed to send voice file fallback for {file_path}: {fe}", exc_info=True)
                raise

    async def _ensure_bot_ready(self) -> None:
        """Wait for the Discord client login/gateway session when initialized or mocked."""
        if self.bot.is_ready():
            return
        wait_fn = getattr(self.bot, "wait_until_ready", None)
        if not callable(wait_fn):
            return
        is_default_wait = getattr(wait_fn, "__func__", None) is discord.Client.wait_until_ready
        if not is_default_wait:
            await wait_fn()
            return
        ready_ev = getattr(self.bot, "_ready", discord.utils.MISSING)
        if ready_ev is discord.utils.MISSING:
            # When start() just spawned the subscriber task, bot.start() may enter __aenter__ on the next tick.
            if self._subscriber_task is not None and not self._subscriber_task.done() and not self.bot.is_closed():
                for _ in range(40):
                    if getattr(self.bot, "_ready", discord.utils.MISSING) is not discord.utils.MISSING:
                        break
                    await asyncio.sleep(0.05)
            ready_ev = getattr(self.bot, "_ready", discord.utils.MISSING)
        if ready_ev is not discord.utils.MISSING and not self.bot.is_ready():
            await wait_fn()

    async def send_deferred_placeholder(
        self,
        channel_id: str,
        segment: dict[str, Any],
        message: Message,
        marker: dict[str, Any],
    ) -> Any:
        """Send an in-order placeholder for files, or add a `🎤` reaction anchor for deferred voice."""
        _ = channel_id
        if segment.get("type") == "voice":
            anchor_cid, anchor_mid, live_msg = await self._resolve_voice_anchor(message)
            if anchor_mid:
                await self._best_effort_add_reaction(anchor_cid, anchor_mid, "🎤", live_msg=live_msg)
            return {
                "kind": "voice_anchor",
                "channel_id": str(anchor_cid),
                "message_id": str(anchor_mid) if anchor_mid else None,
                "reaction": "🎤",
            }

        channel = await self._get_discord_channel(message.channel_id, getattr(message, "metadata", None))
        if not channel:
            return None
        placeholder_text = sanitize_placeholder(marker)
        # Intentionally NOT added to _intermediate_messages or _turn_special_msg.
        return await self._send_with_retry(
            lambda: channel.send(placeholder_text),
            timeout_duration=15.0,
            error_msg="Timeout sending deferred placeholder",
        )

    def serialize_deferred_handle(self, handle: Any) -> dict[str, Any] | None:
        """Serialize a `voice_anchor` dict or `discord.Message`/`PartialMessage` into JSON metadata."""
        if handle is None:
            return None
        if isinstance(handle, dict) and handle.get("kind") == "voice_anchor":
            cid = handle.get("channel_id")
            mid = handle.get("message_id")
            return {
                "kind": "voice_anchor",
                "channel_id": str(cid) if cid else None,
                "message_id": str(mid) if mid else None,
                "reaction": str(handle.get("reaction") or "🎤"),
            }
        msg_id = getattr(handle, "id", None)
        chan = getattr(handle, "channel", None)
        chan_id = getattr(chan, "id", None) if chan is not None else getattr(handle, "channel_id", None)
        if chan_id is None or msg_id is None:
            return None
        return {"channel_id": str(chan_id), "message_id": str(msg_id)}

    async def restore_deferred_handle(self, raw_handle: dict[str, Any] | None) -> Any:
        """Restore a `voice_anchor` dict or reconstruct a `discord.PartialMessage` after bot login."""
        if not isinstance(raw_handle, dict):
            return None
        if raw_handle.get("kind") == "voice_anchor":
            await self._ensure_bot_ready()
            cid = raw_handle.get("channel_id")
            mid = raw_handle.get("message_id")
            return {
                "kind": "voice_anchor",
                "channel_id": str(cid) if cid else None,
                "message_id": str(mid) if mid else None,
                "reaction": str(raw_handle.get("reaction") or "🎤"),
            }
        cid = raw_handle.get("channel_id")
        mid = raw_handle.get("message_id")
        if not cid or not mid:
            return None
        await self._ensure_bot_ready()
        return self.bot.get_partial_messageable(int(cid)).get_partial_message(int(mid))

    async def deliver_deferred_attachment(
        self,
        channel_id: str,
        file_path: str,
        message: Message,
        handle: Any,
        kind: str,
    ) -> None:
        """Deliver a finished deferred attachment (reply-referenced voice or edited image placeholder)."""
        await self._ensure_bot_ready()
        if not await async_exists(file_path):
            raise FileNotFoundError(f"Deferred file not found: {file_path}")

        if kind == "voice":
            if isinstance(handle, dict) and handle.get("kind") == "voice_anchor":
                anchor_cid = str(handle.get("channel_id") or channel_id)
                anchor_mid = str(handle["message_id"]) if handle.get("message_id") else None
                reaction = str(handle.get("reaction") or "🎤")
                if anchor_mid:
                    ref_payload: dict[str, Any] = {
                        "channel_id": anchor_cid,
                        "message_id": anchor_mid,
                        "fail_if_not_exists": False,
                    }
                    await self.send_voice_segment(
                        channel_id,
                        file_path,
                        message,
                        message_reference=ref_payload,
                    )
                    await self._best_effort_remove_reaction(anchor_cid, anchor_mid, reaction)
                else:
                    await self.send_voice_segment(channel_id, file_path, message)
                return

            await self.send_voice_segment(channel_id, file_path, message)
            if handle is not None and callable(getattr(handle, "delete", None)):
                try:
                    async with asyncio.timeout(15.0):
                        await handle.delete()
                except discord.NotFound:
                    pass
                except Exception as de:
                    logger.debug(f"Best-effort placeholder delete after voice delivery failed: {de}")
            return

        if handle is not None:
            for attempt in range(1, 4):
                discord_file = discord.File(file_path)
                try:
                    async with asyncio.timeout(60.0):
                        # Do NOT pass view= so message components stay untouched/absent.
                        await handle.edit(content=None, attachments=[discord_file])
                    return
                except discord.NotFound:
                    logger.info(
                        f"Discord deferred placeholder was deleted before '{file_path}' finished; "
                        "falling back to send_file_segment."
                    )
                    await self.send_file_segment(channel_id, file_path, message)
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if attempt < 3 and _is_transient_discord_upload_error(exc):
                        logger.warning(
                            f"Transient Discord placeholder edit failure for '{file_path}' "
                            f"(attempt {attempt}/3): {exc}. Retrying..."
                        )
                        await asyncio.sleep(0.5 * attempt)
                        continue
                    logger.warning(
                        f"Permanent Discord placeholder edit failure for '{file_path}': {exc}. "
                        "Falling back to channel.send."
                    )
                    break
                finally:
                    try:
                        discord_file.close()
                    except Exception:
                        pass

            channel = await self._get_discord_channel_with_abort(message)
            if not channel:
                return

            async def _send_ref_once() -> Any:
                ref_file = discord.File(file_path)
                try:
                    return await channel.send(file=ref_file, reference=handle, mention_author=False)
                finally:
                    try:
                        ref_file.close()
                    except Exception:
                        pass

            try:
                await self._send_with_retry(
                    _send_ref_once,
                    timeout_duration=60.0,
                    error_msg=f"Timeout sending deferred reply-reference {file_path}",
                )
                return
            except asyncio.CancelledError:
                raise
            except Exception as ref_exc:
                logger.warning(
                    f"Discord reply-reference fallback failed for '{file_path}': {ref_exc}. "
                    "Falling back to plain send_file_segment."
                )

        await self.send_file_segment(channel_id, file_path, message)

    async def fail_deferred_attachment(
        self,
        handle: Any,
        reason: str,
        message: Message,
        segment: dict[str, Any],
    ) -> None:
        """Report deferred attachment failure (reaction swap + concise notice for voice, placeholder edit for file)."""
        await self._ensure_bot_ready()
        is_voice = segment.get("type") == "voice" or (
            isinstance(handle, dict) and handle.get("kind") == "voice_anchor"
        )
        if is_voice:
            safe_reason = sanitize_error_reason(reason)
            voice_err_text = f"⚠️ 语音生成失败（{safe_reason}）"
            anchor_cid: str | None = None
            anchor_mid: str | None = None
            if isinstance(handle, dict) and handle.get("kind") == "voice_anchor":
                anchor_cid = str(handle.get("channel_id") or message.channel_id)
                anchor_mid = str(handle["message_id"]) if handle.get("message_id") else None
                reaction = str(handle.get("reaction") or "🎤")
                if anchor_mid:
                    await self._best_effort_remove_reaction(anchor_cid, anchor_mid, reaction)
                    await self._best_effort_add_reaction(anchor_cid, anchor_mid, "⚠️")
            elif handle is None:
                anchor_cid, anchor_mid, live_msg = await self._resolve_voice_anchor(message)
                if anchor_mid:
                    await self._best_effort_add_reaction(anchor_cid, anchor_mid, "⚠️", live_msg=live_msg)
            elif callable(getattr(handle, "edit", None)):
                try:
                    async with asyncio.timeout(15.0):
                        await handle.edit(content=voice_err_text)
                    return
                except discord.NotFound:
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(f"Failed to edit legacy voice placeholder on failure: {exc}")

            is_default_send_chunks = (
                getattr(self.send_text_chunks, "__func__", None) is DiscordChatbot.send_text_chunks
            )
            if is_default_send_chunks and anchor_mid and str(anchor_cid) == str(message.channel_id):
                try:
                    channel = await self._get_discord_channel(
                        message.channel_id, getattr(message, "metadata", None)
                    )
                    if channel is not None:
                        ref_obj = discord.MessageReference(
                            message_id=int(anchor_mid),
                            channel_id=int(message.channel_id),
                            fail_if_not_exists=False,
                        )
                        await self._send_with_retry(
                            lambda: channel.send(voice_err_text, reference=ref_obj, mention_author=False),
                            timeout_duration=15.0,
                            error_msg="Timeout sending voice failure notice",
                        )
                        return
                except asyncio.CancelledError:
                    raise
                except Exception as ref_err:
                    logger.debug(f"Voice failure reference send fell back to plain text: {ref_err}")

            await self._send_failure_notice_preserving_anchor(message, voice_err_text)
            return

        err_text = format_deferred_failure_message(segment.get("path", ""), reason)
        if handle is not None:
            try:
                async with asyncio.timeout(15.0):
                    await handle.edit(content=err_text)
                return
            except discord.NotFound:
                logger.debug(
                    f"Discord deferred placeholder already deleted on failure for '{segment.get('path')}'; skipping."
                )
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    f"Failed to edit Discord placeholder on failure for '{segment.get('path')}': {exc}. "
                    "Falling back to send_text_chunks."
                )
        await self._send_failure_notice_preserving_anchor(message, err_text)

    async def send_details_segment(self, channel_id: str, title: str, body: str, message: Message) -> None:
        """Deliver a `<details>` section as `▸ TITLE` with a persistent toggle button that expands it in place."""
        channel = await self._get_discord_channel_with_abort(message)
        if not channel:
            return
        title_fmt = self.format_text(title)
        body_fmt = self.format_text(body)
        seq = self._details_seq.get(message.id, 0)
        self._details_seq[message.id] = seq + 1
        details_key = f"{message.id}-d{seq}"
        try:
            view: discord.ui.View
            if ids_fit(message.session_id, details_key):
                await self.gateway.db.save_ui_prompt(
                    message.session_id, details_key, {"type": "details", "title": title_fmt, "body": body_fmt}
                )
                view = build_details_view(message.session_id, details_key)
            else:
                view = DetailsView(title=title_fmt, body=body_fmt)
            await self._send_with_retry(
                lambda: channel.send(render_details_content(title_fmt, body_fmt, expanded=False), view=view),
                error_msg="Timeout sending details view",
            )
        except DeliveryAbortedError:
            raise
        except Exception as de:
            logger.error(f"Failed to send details view to Discord: {de}", exc_info=True)
            await super().send_details_segment(channel_id, title, body, message)

    async def send_ui_prompt(
        self,
        channel: Any,
        embed: discord.Embed,
        view: discord.ui.View,
    ) -> Any:
        """Send an interactive UI prompt embed + view with transient-error retries."""
        return await self._send_with_retry(
            lambda: channel.send(embed=embed, view=view),
            error_msg="Timeout sending question view",
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
        """Deliver a question as an embed with restart-persistent components.

        Single-select with <=10 options renders buttons; more options or multiselect render a select menu. The
        spec is stored in the `ui_prompts` table so clicks keep working after a restart.
        """
        channel = await self._get_discord_channel_with_abort(message)
        if not channel:
            return

        try:
            spec = build_question_spec(question, choices, multi, max_choices)
            view: discord.ui.View
            if ids_fit(message.session_id, message.id):
                await self.gateway.db.save_ui_prompt(message.session_id, message.id, spec)
                view = build_question_view(message.session_id, message.id, spec)
            else:
                logger.warning("Session/message id too long for persistent components; using in-memory buttons")
                view = QuestionView(
                    gateway=self.gateway,
                    session_id=message.session_id,
                    chatbot=self,
                    question=question,
                    choices=choices,
                )
            await self.send_ui_prompt(channel, build_question_embed(spec), view)
        except DeliveryAbortedError:
            raise
        except Exception as qe:
            logger.error(f"Failed to send question view to Discord: {qe}", exc_info=True)
            try:
                await self._call_with_timeout(
                    channel.send(f"⚠️ Failed to send question: {question}"),
                    error_msg="Timeout sending question failure fallback"
                )
            except Exception as fe:
                logger.error(f"Failed to send question fallback: {fe}")

    async def on_message_delivered(self, message: Message) -> None:
        """Post-delivery lifecycle callback: clean up typing indicator, intermediate lists, and update metrics."""
        self._details_seq.pop(message.id, None)
        if message.role == MessageRole.ASSISTANT and message.type == MessageType.TEXT:
            await self._cleanup_stream_preview(message.session_id)
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
                header_msg, header_view = self._header_views.pop(turn_id)
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

    async def on_message_delivery_failed(self, message: Message, exception: Exception) -> None:
        """Post-failure lifecycle callback: clean up typing indicator, intermediate lists, and update status."""
        logger.warning(f"Delivery failed for message {message.id}: {exception}")
        cause = getattr(exception, "__cause__", None)
        if isinstance(exception, (discord.NotFound, discord.Forbidden)) or isinstance(
            cause, (discord.NotFound, discord.Forbidden)
        ):
            await self.gateway.db.update_message_status(message.id, MessageStatus.DELIVERED)
        else:
            await super().on_message_delivery_failed(message, exception)

        if message.role == MessageRole.ASSISTANT and message.type == MessageType.TEXT:
            await self._cleanup_stream_preview(message.session_id)
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

            # Remove stop button from the header view for this turn and delete the header
            turn_id = message.parent_id or message.session_id
            if turn_id in self._header_views:
                header_msg, _ = self._header_views.pop(turn_id)
                try:
                    await header_msg.delete()
                except Exception as ee:
                    logger.warning(f"Failed to delete header message on failure: {ee}")

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

        channel = None
        if isinstance(channel_id, str) and channel_id.lower() == "dm":
            if not mention_user_id:
                logger.error("Cannot trigger DM cronjob without mention_user_id.")
                return
            try:
                user_id = int(mention_user_id)
                user = self.bot.get_user(user_id)
                if not user:
                    user = await self.bot.fetch_user(user_id)
                if user:
                    channel = user.dm_channel
                    if not channel:
                        channel = await user.create_dm()
                if not channel:
                    logger.error(f"Failed to resolve DM channel for user {mention_user_id}")
                    return
            except Exception as e:
                logger.error(f"Error resolving DM channel for user {mention_user_id}: {e}", exc_info=True)
                return
        else:
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
        is_dm = getattr(channel, "guild", None) is None
        channel_id_str = str(channel.id)
        channel_name = getattr(channel, "name", "") or ""

        auto_thread = True
        if is_dm:
            auto_thread = False
        elif not is_thread:
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
        if is_dm:
            msg_metadata["dm_recipient_id"] = str(mention_user_id)
        if parent_channel_id:
            msg_metadata["parent_channel_id"] = parent_channel_id
        if parent_channel_name:
            msg_metadata["parent_channel_name"] = parent_channel_name

        tag = kwargs.get("tag")
        role = kwargs.get("role")
        await self.trigger_cronjob_message(
            channel_id=target_channel_id_str,
            prompt_content=prompt_content,
            sender_name="Cronjob",
            custom_prompt=custom_prompt,
            metadata=msg_metadata,
            tag=tag,
            role=role,
        )

        # Trigger typing status for the thread/channel while agent is thinking
        if target_channel_id_str not in self._typing_tasks:
            self._typing_tasks[target_channel_id_str] = asyncio.create_task(self._keep_typing(target_channel))

    async def get_cron_channel_candidates(self, channel_id: str) -> set[str]:
        """Return the channel itself plus its parent channel if it is a thread."""
        candidates = {str(channel_id)}
        channel = await self._get_discord_channel(channel_id)
        if isinstance(channel, discord.Thread) and channel.parent_id:
            candidates.add(str(channel.parent_id))
        return candidates

    async def is_dm_channel(self, channel_id: str) -> bool:
        """Check if the given channel is a DM channel."""
        channel = await self._get_discord_channel(channel_id)
        return isinstance(channel, discord.DMChannel)
