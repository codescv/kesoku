"""Turn execution logic for the Kesoku Agent."""

import asyncio
import datetime
import inspect
import json
import os
import re
import time
import traceback
from typing import TYPE_CHECKING

import tzlocal

from kesoku.agent.compressor import SCAFFOLD_HEADER, HistoryCompressor, estimate_tokens_calibrated
from kesoku.agent.context_dir import ContextEntry, build_context_prefix
from kesoku.agent.history import (
    build_history,
    prepare_history_for_llm,
    strip_question_options,
)
from kesoku.agent.hooks import run_post_tool_use_hooks, run_stop_hooks
from kesoku.agent.llm import BaseLLM, LLMResponse
from kesoku.agent.memory_store import format_ambient_memory_block, render_context_query, search_role_memories
from kesoku.agent.repo_rules import RepoRulesManager
from kesoku.agent.state_snapshot import render_repo_rules, render_running_jobs, render_staging_artifacts
from kesoku.agent.tool_runner import ToolRunner
from kesoku.agent.tools.file import expand_path_variables
from kesoku.agent.turn_logger import TurnLogger
from kesoku.config import KesokuConfig
from kesoku.context import KesokuContext
from kesoku.utils.async_fs import async_exists, async_read_text_file, async_write_text_file
from kesoku.utils.path import PathResolver
from kesoku.utils.text import truncate_middle
from kesoku.utils.ui_markup import collapse_ui_blocks, extract_ui_options, parse_multi_selection

if TYPE_CHECKING:
    from kesoku.agent.agent import SessionWorker
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger

logger = setup_logger(__name__)
MAX_TOTAL_CROSS_SESSION_CONTEXT_LENGTH = 3000
MAX_CHATBOT_ERROR_MESSAGE_LENGTH = 500
# Number of silent re-generations attempted when the model returns no content at all
# (e.g. Vertex safety filtering yields finish_reason=SAFETY with zero parts) before
# falling back to nudging the model with an explicit system message.
EMPTY_RESPONSE_RETRIES = 2
# Max characters kept per Agent line inside auto-retrieved historical dialogue snippets.
MAX_RAG_AGENT_LINE_CHARS = 400

# Same block syntax the gateway parses: `[question: <stem> || opt1 | opt2]` (legacy: `<stem> | opt1 | opt2`).
_QUESTION_BLOCK_CAPTURE_RE = re.compile(r"\[question:\s*([^\]]+)\]")
# Short numeric selections such as "2", "2.", "选2", "選2", "第2个", "２番", "option 2".
_NUMERIC_PICK_RE = re.compile(
    r"^\s*(?:选|選|选择|選択|option|opt|#)?\s*第?\s*[0-9０-９一二三四五六七八九十]{1,2}\s*"
    r"(?:个|個|号|號|番|项|項)?\s*[.。、)）!！]?\s*$",
    re.IGNORECASE,
)


def extract_question_options(text: str | None) -> list[str]:
    """Return the options of the interactive question in text (empty list if none).

    The first ```ui block wins (it is the one rendered as interactive components); otherwise the options
    of the LAST legacy `[question: ...]` block are returned.
    """
    if not text:
        return []
    ui_options, _ = extract_ui_options(text)
    if ui_options:
        return ui_options
    matches = _QUESTION_BLOCK_CAPTURE_RE.findall(text)
    if not matches:
        return []
    inner = matches[-1].strip()
    if "||" in inner:
        _, choices_part = inner.split("||", 1)
        return [c.strip() for c in choices_part.split("|") if c.strip()]
    parts = [p.strip() for p in inner.split("|")]
    return [p for p in parts[1:] if p]


def _normalize_choice(text: str) -> str:
    """Normalize a choice/user text for comparison (whitespace collapsed, surrounding punctuation dropped)."""
    return re.sub(r"\s+", "", text or "").strip("。.!！?？")


def is_clicked_option(user_text: str | None, last_assistant_text: str | None) -> bool:
    """Detect whether the user message is a selection of the previous assistant turn's question options.

    True when the message exactly equals one of the options of the interactive question (first ```ui block, or
    the last legacy `[question: ...]` block) in the previous assistant message, when it is a multiselect answer
    (`选择：A；B`) whose parts are all options, or when it is a short numeric pick ("2", "选2", "2.").
    """
    if not user_text or not user_text.strip():
        return False
    if _NUMERIC_PICK_RE.match(user_text):
        return True
    normalized_user = _normalize_choice(user_text)
    if not normalized_user:
        return False
    options = {_normalize_choice(opt) for opt in extract_question_options(last_assistant_text)}
    if normalized_user in options:
        return True
    parts = parse_multi_selection(user_text)
    return bool(parts) and all(_normalize_choice(p) in options for p in parts)


def sanitize_rag_snippet_line(speaker: str, content: str, max_agent_chars: int = MAX_RAG_AGENT_LINE_CHARS) -> str:
    """Strip `[question: ...]` / ```ui options and truncate long Agent lines of a retrieved historical message."""
    cleaned = collapse_ui_blocks(strip_question_options(content.strip()))
    if speaker == "Agent" and len(cleaned) > max_agent_chars:
        cleaned = cleaned[:max_agent_chars] + "…[truncated]"
    return cleaned


def estimate_fixed_prompt_overhead(system_prompt: str | None, tools_list: list | None) -> int:
    """Estimate tokens of the fixed prompt overhead: system prompt plus tool declarations.

    Tool declarations are approximated from each tool's name, signature, and docstring, which is what the
    SDK turns into the function schema sent with every request.
    """
    text_parts = [system_prompt or ""]
    for tool in tools_list or []:
        name = getattr(tool, "__name__", "")
        try:
            signature = str(inspect.signature(tool))
        except (TypeError, ValueError):
            signature = ""
        text_parts.append(f"{name}{signature}\n{inspect.getdoc(tool) or ''}")
    return estimate_tokens_calibrated("\n".join(text_parts))


class TurnExecutor:
    """Orchestrates conversational turn execution, including LLM inference, thought logging, and tool calling."""

    def __init__(
        self,
        session_id: str,
        gateway: Gateway,
        tool_runner: ToolRunner,
        turn_logger: TurnLogger | None = None,
        context: KesokuContext | None = None,
    ) -> None:
        """Initialize TurnExecutor.

        Args:
            session_id: Unique conversational session identifier.
            gateway: Gateway instance.
            tool_runner: Tool runner handling actual tool execution.
            turn_logger: Optional logger to output detailed YAML logs of the turns.
            context: Optional runtime context container.
        """
        self.session_id = session_id
        self.gateway = gateway
        self.tool_runner = tool_runner
        self.turn_logger = turn_logger
        self.context = context or getattr(gateway, "context", None) or KesokuContext()
        self._turn_tools_used: set[str] = set()
        self._turn_writes = False
        self._stop_reminder_fired = False
        self._hook_retries: dict[str, int] = {}
        self._subagent_calls = 0
        self._subagent_tokens = 0
        self._repo_rules_manager: RepoRulesManager | None = None
        # Per-turn frozen state snapshot (running_jobs / staging_artifacts context blocks). It must be byte-identical
        # across all LLM calls of a turn so the implicit prefix cache keeps hitting; reset on pivot.
        self._snapshot_cache: list[ContextEntry] | None = None
        self._has_summaries: bool | None = None

    async def _get_session_turns_count(self) -> int:
        """Retrieve the count of user turns in the current session.

        Returns:
            The count of user messages.
        """
        return await self.context.db.get_session_turns_count(self.session_id)

    async def _is_bootstrap_turn(self, history: list[Message], current_msg: Message) -> bool:
        """Determine if this is a Bootstrap Turn (first turn of session or idle > 30 mins)."""
        turn_count = await self._get_session_turns_count()
        if turn_count <= 1:
            return True
        non_current_msgs = [m for m in history if m.id != current_msg.id]
        if non_current_msgs:
            last_msg_time = non_current_msgs[-1].timestamp
            # 1800 seconds = 30 minutes inactivity
            if current_msg.timestamp - last_msg_time > 1800:
                return True
        else:
            return True
        return False

    def _resolve_llm(self, current_msg: Message) -> BaseLLM:
        """Resolve the appropriate LLM instance for the current message, applying overrides.

        Args:
            current_msg: The active user message initiating the turn.

        Returns:
            A BaseLLM instance to use for this turn.
        """
        cfg = self.context.config
        discord_cfg = cfg.get_discord_config(current_msg.chatbot_id)
        if discord_cfg:
            channel_id = current_msg.channel_id
            channel_name = current_msg.metadata.get("channel_name", "")
            parent_id = current_msg.metadata.get("parent_channel_id")
            parent_name = current_msg.metadata.get("parent_channel_name")

            for override in discord_cfg.channels:
                identifiers = {channel_id, channel_name}
                if parent_id:
                    identifiers.add(parent_id)
                if parent_name:
                    identifiers.add(parent_name)
                if any(ident in override.channels for ident in identifiers if ident):
                    if override.llm or override.model_name:
                        logger.info(
                            f"Applying LLM override (provider={override.llm}, model={override.model_name}) "
                            f"for Discord channel {channel_id} ('{channel_name}')"
                        )
                        try:
                            if override.model_name:
                                return self.context.get_llm(
                                    provider=override.llm,
                                    model_name=override.model_name,
                                )
                            return self.context.get_llm(provider=override.llm)
                        except Exception as e:
                            logger.error(
                                f"Failed to get override LLM (provider='{override.llm}', "
                                f"model='{override.model_name}'): {e}"
                            )

        return self.context.get_llm()

    async def process_turn(
        self,
        current_msg: Message,
        worker: "SessionWorker",
        session_staging_dir: str,
    ) -> None:
        """Process a single conversational turn.

        Args:
            current_msg: Active user message initiating the turn.
            worker: The SessionWorker handling this conversational session.
            session_staging_dir: Path to session staging directory for outputs/logs.
        """
        chatbot_id = current_msg.chatbot_id
        channel_id = current_msg.channel_id

        cfg = self.context.config
        start_time = time.time()
        turn_tool_calls = 0
        turn_tokens = 0
        last_context_tokens = 0
        last_cached_tokens = 0

        active_cache_name = getattr(worker, "active_cache_name", None)
        if active_cache_name is not None and not isinstance(active_cache_name, str):
            active_cache_name = None
        cached_messages_len = getattr(worker, "cached_messages_len", 0)
        if not isinstance(cached_messages_len, int):
            cached_messages_len = 0

        empty_attempts = 0
        try:
            while worker.running:
                # Check-in before atomic action (Thought Interruption)
                prev_msg_id = current_msg.id
                current_msg = await worker.drain_queue_and_pivot(current_msg)
                if current_msg.id != prev_msg_id:
                    logger.info(
                        f"Pivoted from message {prev_msg_id} to {current_msg.id} during turn loop. "
                        f"Resetting turn metrics and empty-response counter."
                    )
                    self.tool_runner.tool_context.original_msg_id = current_msg.id
                    self._snapshot_cache = None
                    self._turn_tools_used.clear()
                    self._turn_writes = False
                    self._stop_reminder_fired = False
                    self._hook_retries.clear()
                    self._subagent_calls = 0
                    self._subagent_tokens = 0
                    self.tool_runner.tool_context.subagent_llm_calls = 0
                    self.tool_runner.tool_context.subagent_tokens = 0
                    empty_attempts = 0
                    turn_tool_calls = 0
                    turn_tokens = 0
                    last_context_tokens = 0
                    last_cached_tokens = 0
                    start_time = time.time()
                    if active_cache_name:
                        try:
                            # We can resolve LLM for deletion using current_msg resolved LLM or recreate it
                            temp_llm = self._resolve_llm(current_msg)
                            await temp_llm.delete_cache(active_cache_name)
                        except Exception as de:
                            logger.warning(f"Failed to delete cache on pivot: {de}")
                        active_cache_name = None
                        cached_messages_len = 0
                        worker.active_cache_name = None
                        worker.active_cache_llm = None
                        worker.cached_messages_len = 0

                # Resolve LLM dynamically for the current message
                llm = self._resolve_llm(current_msg)

                if active_cache_name and worker.active_cache_llm:
                    old_model = getattr(worker.active_cache_llm, "model_name", None)
                    new_model = getattr(llm, "model_name", None)
                    if (
                        worker.active_cache_llm.__class__.__name__ != llm.__class__.__name__
                        or old_model != new_model
                    ):
                        logger.info(
                            f"LLM changed from {worker.active_cache_llm.__class__.__name__} ({old_model}) "
                            f"to {llm.__class__.__name__} ({new_model}). Resetting obsolete context cache..."
                        )
                        try:
                            await worker.active_cache_llm.delete_cache(active_cache_name)
                        except Exception as de:
                            logger.warning(f"Failed to delete obsolete context cache on provider change: {de}")
                        active_cache_name = None
                        cached_messages_len = 0
                        worker.active_cache_name = None
                        worker.active_cache_llm = None
                        worker.cached_messages_len = 0

                # Retrieve system prompt directly from session
                session = await self.context.db.get_session(self.session_id)
                system_prompt = session.system_prompt if session else None

                tools_list = self.tool_runner.tool_registry.get_tools_list()

                # Retrieve and build the raw session history
                history = await build_history(
                    gateway=self.gateway,
                    session_id=self.session_id,
                    heal_orphans=True,
                )
                # Check and automatically compact history in-place if it exceeds threshold
                history, compacted_occurred = await self._check_and_auto_compact_history(
                    history=history,
                    system_prompt=system_prompt,
                    tools_list=tools_list,
                    llm=llm,
                    cfg=cfg,
                    current_msg=current_msg,
                    staging_dir=session_staging_dir,
                )

                # If history was compacted, the existing cache is obsolete and must be deleted.
                if compacted_occurred and active_cache_name:
                    logger.info("Session history was compacted. Deleting obsolete context cache...")
                    try:
                        await llm.delete_cache(active_cache_name)
                    except Exception as de:
                        logger.warning(f"Failed to delete obsolete cache: {de}")
                    active_cache_name = None
                    cached_messages_len = 0
                    worker.active_cache_name = None
                    worker.active_cache_llm = None
                    worker.cached_messages_len = 0

                await self._inject_context_and_trigger_consolidation(
                    history,
                    current_msg,
                    llm,
                    session_staging_dir=session_staging_dir,
                )

                # Prepare history for the LLM by stripping thoughts, attachments, and truncating historical tool outputs  # noqa: E501
                llm_history = prepare_history_for_llm(
                    history,
                    staging_dir=session_staging_dir,
                    strip_old_questions=cfg.agent.strip_historical_question_options,
                )

                # Implicit caching mode: rely on Vertex's automatic prefix caching and drop any explicit cache.
                implicit_caching = cfg.gemini.context_caching_mode == "implicit"
                if implicit_caching and active_cache_name:
                    logger.info("context_caching_mode=implicit: deleting existing explicit context cache...")
                    try:
                        await llm.delete_cache(active_cache_name)
                    except Exception as de:
                        logger.warning(f"Failed to delete explicit cache in implicit mode: {de}")
                    active_cache_name = None
                    cached_messages_len = 0
                    worker.active_cache_name = None
                    worker.active_cache_llm = None
                    worker.cached_messages_len = 0

                # Setup context cache if enabled, Gemini model, and not already created
                if (
                    cfg.gemini.context_caching
                    and not implicit_caching
                    and llm.__class__.__name__ == "GeminiLLM"
                    and active_cache_name is None
                ):
                    prefix_messages = []
                    last_user_idx = None
                    for idx, msg in enumerate(llm_history):
                        if msg.role == MessageRole.USER:
                            last_user_idx = idx
                    if last_user_idx is not None:
                        prefix_messages = llm_history[:last_user_idx]

                    if prefix_messages:
                        prefix_tokens = llm.count_tokens(
                            system_prompt=system_prompt,
                            history=prefix_messages,
                        )
                        if prefix_tokens >= cfg.gemini.context_caching_threshold:
                            logger.info(
                                f"Session prefix has {prefix_tokens} tokens. Creating explicit context cache..."
                            )
                            active_cache_name = await llm.create_cache(
                                contents=prefix_messages,
                                system_prompt=system_prompt,
                                tools=tools_list,
                                display_name=f"kesoku_{self.session_id}",
                                ttl_seconds=cfg.gemini.context_caching_ttl,
                            )
                            if active_cache_name:
                                cached_messages_len = len(prefix_messages)
                                worker.active_cache_name = active_cache_name
                                worker.active_cache_llm = llm
                                worker.cached_messages_len = cached_messages_len

                # If the prepared history length is somehow less than cached_messages_len, the cache is obsolete.
                if active_cache_name and len(llm_history) < cached_messages_len:
                    logger.info("History length is shorter than cache prefix. Deleting obsolete context cache...")
                    try:
                        await llm.delete_cache(active_cache_name)
                    except Exception as de:
                        logger.warning(f"Failed to delete obsolete cache: {de}")
                    active_cache_name = None
                    cached_messages_len = 0
                    worker.active_cache_name = None
                    worker.active_cache_llm = None
                    worker.cached_messages_len = 0

                # Partition history if using context cache
                logger.info(
                    f"Context caching debug: active_cache_name={active_cache_name}, "
                    f"llm_history_len={len(llm_history)}, cached_messages_len={cached_messages_len}"
                )
                if active_cache_name and len(llm_history) >= cached_messages_len:
                    generate_history = llm_history[cached_messages_len:]
                    generate_system_prompt = system_prompt
                    generate_tools = None
                    generate_cached_content = active_cache_name
                else:
                    generate_history = llm_history
                    generate_system_prompt = system_prompt
                    generate_tools = tools_list
                    generate_cached_content = None

                # LLM inference
                try:
                    res = await llm.generate(
                        system_prompt=generate_system_prompt,
                        history=generate_history,
                        tools=generate_tools,
                        cached_content=generate_cached_content,
                    )
                except Exception as e:
                    if active_cache_name and ("expired" in str(e) or "cache" in str(e).lower()):
                        logger.warning(
                            f"LLM generation failed, possibly due to cache expiration: {e}. "
                            "Retrying without context cache."
                        )
                        active_cache_name = None
                        cached_messages_len = 0
                        worker.active_cache_name = None
                        worker.active_cache_llm = None
                        worker.cached_messages_len = 0
                        generate_history = llm_history
                        generate_system_prompt = system_prompt
                        generate_tools = tools_list
                        generate_cached_content = None
                        res = await llm.generate(
                            system_prompt=generate_system_prompt,
                            history=generate_history,
                            tools=generate_tools,
                            cached_content=generate_cached_content,
                        )
                    else:
                        raise

                # Log the raw LLM turn using TurnLogger if enabled
                if cfg.agent.raw_llm_logs and self.turn_logger:
                    try:
                        self.turn_logger.log_llm_turn(
                            llm_provider=llm.__class__.__name__,
                            history=generate_history,
                            tools=tools_list,
                            response=res,
                            system_prompt=generate_system_prompt,
                        )
                    except Exception as le:
                        logger.error(f"Failed to log LLM turn: {le}", exc_info=True)

                # Accumulate token metrics
                last_context_tokens = res.prompt_tokens or 0
                last_cached_tokens = res.cached_tokens or 0
                if res.total_tokens:
                    turn_tokens += res.total_tokens

                if res.tool_calls:
                    turn_tool_calls += len(res.tool_calls)
                    should_continue = await self._execute_tool_calls(res, current_msg, worker, session_staging_dir)
                    if should_continue:
                        continue
                    else:
                        break
                else:
                    should_continue, empty_attempts = await self._handle_final_response(
                        res=res,
                        current_msg=current_msg,
                        last_context_tokens=last_context_tokens,
                        last_cached_tokens=last_cached_tokens,
                        turn_tool_calls=turn_tool_calls,
                        turn_tokens=turn_tokens,
                        start_time=start_time,
                        llm=llm,
                        empty_attempts=empty_attempts,
                    )
                    if should_continue:
                        continue
                    else:
                        break
        except asyncio.CancelledError:
            # Interrupted turn: save turn metrics to the initiating user message
            turn_metrics = {
                "session_turns": await self._get_session_turns_count(),
                "context_tokens": last_context_tokens,
                "cached_tokens": last_cached_tokens,
                "turn_tool_calls": turn_tool_calls,
                "turn_tokens": turn_tokens,
                "turn_time": time.time() - start_time,
                "status": "interrupted",
            }
            history = await self.context.db.get_session_history(self.session_id, limit=20)
            user_msg = None
            for msg in reversed(history):
                if msg.role == MessageRole.USER:
                    user_msg = msg
                    break
            if user_msg:
                user_msg.metadata["turn_metrics"] = turn_metrics
                await self.context.db.update_message_metadata(user_msg.id, user_msg.metadata)
            raise
        except Exception as e:
            logger.error(f"Error in session turn {self.session_id}: {e}", exc_info=True)
            tb_str = traceback.format_exc()
            error_filename = f"error_{current_msg.id}.txt"
            error_file_path = os.path.join(session_staging_dir, error_filename)
            try:
                await async_write_text_file(error_file_path, tb_str)
            except Exception as fe:
                logger.error(f"Failed to write error traceback file to {error_file_path}: {fe}")

            full_error_msg = f"⚠️ An error occurred while processing your request: {e}"
            hint = f"\n\nFull error log saved to staging directory: {error_filename}"
            max_err_len = MAX_CHATBOT_ERROR_MESSAGE_LENGTH - len(hint)
            truncated_error = truncate_middle(
                full_error_msg,
                max_err_len,
                "\n\n... [Error Truncated for Brevity] ...\n\n",
            )
            truncated_content = truncated_error + hint

            error_msg = Message(
                session_id=self.session_id,
                chatbot_id=chatbot_id,
                channel_id=channel_id,
                sender="Kesoku",
                role=MessageRole.ASSISTANT,
                type=MessageType.TEXT,
                content=truncated_content,
                status=MessageStatus.PENDING,
                parent_id=current_msg.id,
            )
            await self.gateway.post(error_msg)
            await self.context.db.update_message_status(current_msg.id, MessageStatus.ERROR)

    async def _inject_context_and_trigger_consolidation(
        self,
        history: list[Message],
        current_msg: Message,
        llm: BaseLLM,
        session_staging_dir: str | None = None,
    ) -> Message | None:
        """Inject context and user preferences, triggering background consolidation if needed.

        Modifies the history list in-place by prepending context block to the latest user message.

        Returns:
            The modified latest User Message, or None if no user message was found to inject context into.
        """
        latest_user_msg = None
        for msg in reversed(history):
            if msg.role == MessageRole.USER:
                latest_user_msg = msg
                break

        if not latest_user_msg:
            return None

        active_role = await self._resolve_active_role(current_msg)

        # Previous assistant reply (text) before the latest user message: used for clicked-option detection
        # (auto-RAG guard).
        last_assistant_text = ""
        latest_user_idx = history.index(latest_user_msg)
        for prev in reversed(history[:latest_user_idx]):
            if (
                prev.role == MessageRole.ASSISTANT
                and prev.type == MessageType.TEXT
                and prev.sender != "Notification"
                and prev.content
            ):
                last_assistant_text = prev.content
                break

        # 1. Calculate if this is a Bootstrap Turn (first turn of session or idle > 30 mins)
        is_bootstrap = await self._is_bootstrap_turn(history, current_msg)

        # 2. Read role-based preferences.md every turn
        inject_preferences = True
        instructions_prefix = ""
        if active_role:
            roles_dir = self.context.config.workspace.roles_dir
            if not os.path.isabs(roles_dir) and self.context.config.agent_working_dir:
                roles_dir = os.path.join(self.context.config.agent_working_dir, roles_dir)
            if inject_preferences:
                pref_path = os.path.join(roles_dir, active_role, "preferences.md")
                if await async_exists(pref_path):
                    try:
                        role_prefs_content = await async_read_text_file(pref_path)
                        role_prefs_content = role_prefs_content.strip()
                        if role_prefs_content:
                            instructions_prefix = f"<instructions>\n{role_prefs_content}\n</instructions>\n"
                    except Exception as e:
                        logger.warning(f"Failed to read preferences.md for role '{active_role}': {e}")

        # 2.2. Read per-turn context blocks every turn: pinned notes ($STAGING_DIR/notes.md) first, then
        # every $STAGING_DIR/context/*.md in filename order, each rendered as <context name="...">.
        context_blocks_prefix = ""
        target_staging_dir = session_staging_dir
        if not target_staging_dir and self.context and self.context.config:
            sessions_dir = self.context.config.workspace.sessions_dir
            if not os.path.isabs(sessions_dir) and self.context.config.agent_working_dir:
                sessions_dir = os.path.join(self.context.config.agent_working_dir, sessions_dir)
            target_staging_dir = os.path.join(sessions_dir, self.session_id)

        if target_staging_dir:
            try:
                context_blocks_prefix = await build_context_prefix(
                    target_staging_dir,
                    self.context.config.agent.context_dir_max_chars,
                    extra_entries=await self._get_state_snapshot_entries(target_staging_dir),
                )
            except Exception as e:
                logger.warning(f"Failed to build context blocks for session '{self.session_id}': {e}")

        # 3. Prepend Consolidated Passive Synchronization, Preferences, and Context Compression Guidelines
        # (if Bootstrap)
        full_prefix = ""
        if is_bootstrap:
            disabled_tools = set(self.context.config.agent.disabled_tools)
            hints = []
            if "memory_search" not in disabled_tools:
                hints.append(
                    "- Use `memory_search(query)` to search long-term Markdown memories in `$MEMORY_DIR` for this role."
                )
            lines = ['<background_context type="sync_guidelines">', *hints, "</background_context>"] if hints else []
            full_prefix = "\n".join(lines) + "\n" if lines else ""

        # 4. Prepend Ambient Progressive-Disclosure Memory Index (if enabled)
        augmented_context_prefix = ""
        cfg = self.context.config
        if cfg.agent.auto_context_search and active_role:
            raw_query = latest_user_msg.content or ""
            # Skip when the user merely clicked/picked an option of the previous question.
            clicked_option = is_clicked_option(raw_query, last_assistant_text)
            if clicked_option:
                logger.info("Skipping ambient memory search: user message is a clicked option of the last question")
            elif len(raw_query.strip()) >= 2:
                try:
                    context_query = render_context_query(history, latest_user_msg, max_turns=2, max_chars=2000)
                    augmented_context_prefix = await self._retrieve_auto_search_context(
                        role=active_role,
                        query_text=context_query or raw_query.strip(),
                        semantic_min=cfg.agent.auto_context_search_semantic_min,
                        limit=cfg.agent.auto_context_search_limit,
                    )
                except Exception as se:
                    logger.warning(f"Failed to retrieve ambient memory context: {se}")

        msg_idx = history.index(latest_user_msg)
        copied_msg = latest_user_msg.model_copy()
        msg_time = datetime.datetime.fromtimestamp(copied_msg.timestamp).astimezone()
        time_str = msg_time.strftime("%Y-%m-%d %H:%M:%S (%A) %Z")
        sender_name = copied_msg.metadata.get("sender_name") or copied_msg.sender
        if sender_name.lower() == "cronjob":
            sender_name = "system"
        try:
            tz_name = tzlocal.get_localzone_name()
        except Exception:
            tz_name = msg_time.tzname() or "UTC"

        copied_msg.content = (
            f"{instructions_prefix}"
            f"{context_blocks_prefix}"
            f"{full_prefix}"
            f"{augmented_context_prefix}"
            f'<current_message from="{sender_name}" time="{time_str}" timezone="{tz_name}">\n'
            f"{copied_msg.content}\n"
            "</current_message>"
        )
        history[msg_idx] = copied_msg
        latest_user_msg = copied_msg
        logger.info(
            f"Wrapped active user message {copied_msg.id} in <current_message "
            f'from="{sender_name}" time="{time_str}" timezone="{tz_name}"> '
            f"(bootstrap: {is_bootstrap})"
        )

        return latest_user_msg

    async def _resolve_active_role(self, current_msg: Message) -> str:
        """Resolve the persona role for this turn.

        The session's own `role_name` is authoritative when set to a non-default role (role switches swap
        sessions). Falls back to the channel role lookup (with parent-thread inheritance) otherwise, which
        covers thread sessions whose channel has no explicit `channel_roles` row.
        """
        try:
            session = await self.context.db.get_session(self.session_id)
            if session and session.role_name and session.role_name != "default":
                return session.role_name
        except Exception as e:
            logger.warning(f"Failed to read session role for '{self.session_id}': {e}")
        return await self.context.db.get_channel_role_with_inheritance(
            current_msg.chatbot_id,
            current_msg.channel_id,
            self.session_id,
        )

    def _resolve_memories_dir(self, role: str) -> str:
        """Resolve the memories directory path for `role` synchronously."""
        roles_dir = self.context.config.workspace.roles_dir
        if not os.path.isabs(roles_dir) and self.context.config.agent_working_dir:
            roles_dir = os.path.join(self.context.config.agent_working_dir, roles_dir)
        return os.path.abspath(os.path.join(roles_dir, role or "default", "memories"))

    async def _retrieve_auto_search_context(
        self,
        role: str,
        query_text: str,
        current_session_id: str | None = None,
        threshold: float = 0.50,
        semantic_min: float = 0.50,
        limit: int = 5,
        window_size: int = 1,
    ) -> str:
        """Search relevant Markdown memories for the active role and render a `<memory>` block."""
        _ = (current_session_id, threshold, window_size)
        sync_db = getattr(self.context.db, "_sync", getattr(self.context.db, "sync_db", self.context.db))
        memories_dir = self._resolve_memories_dir(role)

        matches = await asyncio.to_thread(
            search_role_memories,
            sync_db,
            role,
            query_text,
            limit=limit,
            min_score=semantic_min,
            memories_dir=memories_dir,
        )
        if not matches:
            return ""

        logger.info(
            f"Ambient-injected {len(matches)} memory file descriptions for role '{role}' "
            f"(query: '{query_text[:40]}...')"
        )
        return format_ambient_memory_block(matches)

    def _ensure_repo_rules_manager(
        self,
        staging_dir: str | None,
        cfg: KesokuConfig,
    ) -> RepoRulesManager | None:
        """Lazily initialize RepoRulesManager for the session and discover AWD rules on turn start."""
        if self._repo_rules_manager is None and staging_dir and cfg.agent_working_dir:
            awd = os.path.abspath(cfg.agent_working_dir)
            self._repo_rules_manager = RepoRulesManager(
                staging_dir=staging_dir,
                agent_working_dir=awd,
                max_files=cfg.agent.repo_rules_max_files,
                max_chars=cfg.agent.repo_rules_max_chars,
                rule_filenames=cfg.agent.repo_rules_files,
            )
            self._repo_rules_manager.discover([awd])
        return self._repo_rules_manager

    @staticmethod
    def _resolve_touched_path(
        raw_path: str,
        awd: str,
        staging_dir: str | None,
        tool_context: object | None = None,
    ) -> str:
        """Resolve a tool path argument into an absolute filesystem path, expanding $STAGING_DIR/$AWD."""
        cleaned = raw_path.strip().strip("\"'")
        if staging_dir:
            cleaned = cleaned.replace("${STAGING_DIR}", staging_dir).replace("$STAGING_DIR", staging_dir)
        if awd:
            cleaned = cleaned.replace("${AWD}", awd).replace("$AWD", awd)
        if "$" in cleaned:
            cleaned = expand_path_variables(cleaned, tool_context)  # type: ignore[arg-type]
        cleaned = os.path.expanduser(cleaned)
        if os.path.isabs(cleaned):
            return os.path.abspath(cleaned)
        return os.path.abspath(os.path.join(awd, cleaned))

    @staticmethod
    def _is_shell_write_command(cmd: str) -> bool:
        """Check whether a shell command string contains a write indicator (ignoring stream redirections)."""
        cmd_clean = re.sub(r"(?:\d*>&\d+|&>/dev/null|\d*>/dev/null)", "", cmd)
        if ">" in cmd_clean:
            return True
        if re.search(r"\b(?:tee|patch|mv|cp)\b", cmd_clean):
            return True
        if re.search(r"\bgit\s+apply\b", cmd_clean):
            return True
        if re.search(r"\bsed\b[^\n;|&]*\s-[a-zA-Z]*i", cmd_clean):
            return True
        return False

    async def _get_state_snapshot_entries(self, staging_dir: str | None) -> list[ContextEntry]:
        """Deterministic `running_jobs` / `repo_rules` / `staging_artifacts` context entries, computed once per turn.

        `running_jobs` comes from the session's ActiveJobsRegistry snapshot; `repo_rules` renders discovered
        AGENTS.md/GEMINI.md files; `staging_artifacts` is rendered only once the session has summary nodes (before
        that the files are visible in the verbatim history). Failures are logged and yield no entry; they never
        break the turn.
        """
        if self._snapshot_cache is not None:
            return self._snapshot_cache
        entries: list[ContextEntry] = []
        cfg = self.context.config
        try:
            active_jobs = getattr(self.context, "active_jobs", None)
            if active_jobs is not None and hasattr(active_jobs, "snapshot"):
                snapshot = active_jobs.snapshot(self.session_id)
                if inspect.isawaitable(snapshot):
                    snapshot = await snapshot
                manage_job_enabled = "manage_job" not in set(cfg.agent.disabled_tools)
                jobs_text = render_running_jobs(snapshot, manage_job_enabled=manage_job_enabled)
                if jobs_text:
                    entries.append(ContextEntry("running_jobs", jobs_text))
        except Exception as e:
            logger.warning(f"Failed to build running_jobs snapshot for session '{self.session_id}': {e}")

        try:
            mgr = self._ensure_repo_rules_manager(staging_dir, cfg)
            if mgr is not None and mgr.discovered_files:
                rules_text = await asyncio.to_thread(
                    render_repo_rules,
                    mgr.discovered_files,
                    cfg.agent.repo_rules_max_chars,
                    cfg.agent_working_dir,
                )
                if rules_text:
                    entries.append(ContextEntry("repo_rules", rules_text))
        except Exception as e:
            logger.warning(f"Failed to build repo_rules snapshot: {e}")

        try:
            has_summaries = self._has_summaries
            if has_summaries is None:
                has_summaries = bool(await self.context.db.get_root_summary_nodes(self.session_id))
            if has_summaries and staging_dir:
                artifacts_text = await asyncio.to_thread(render_staging_artifacts, staging_dir)
                if artifacts_text:
                    entries.append(ContextEntry("staging_artifacts", artifacts_text))
        except Exception as e:
            logger.warning(f"Failed to build staging_artifacts snapshot for session '{self.session_id}': {e}")
        self._snapshot_cache = entries
        return entries

    async def _check_and_auto_compact_history(
        self,
        history: list[Message],
        system_prompt: str | None,
        tools_list: list,
        llm: BaseLLM,
        cfg: KesokuConfig,
        current_msg: Message,
        staging_dir: str | None = None,
    ) -> tuple[list[Message], bool]:
        """Check context window usage and automatically compact history using custom turn-based compressor.

        Returns:
            A tuple of (active message history list, compacted_occurred bool).
        """
        if not history:
            return history, False

        compressor = HistoryCompressor(self.context.db)

        # Summarize with the dedicated context-compression model ([agent].context_llm /
        # [gemini|claude].context_model_name); fall back to the session LLM if it cannot be resolved.
        try:
            compression_llm = self.context.get_llm(use_context_compression=True)
        except Exception as e:
            logger.warning(f"Failed to resolve context compression LLM, using session LLM: {e}")
            compression_llm = llm

        # Trigger automatic compaction
        compacted_occurred = await compressor.auto_compact_session(
            session_id=self.session_id,
            history=history,
            llm=compression_llm,
            config=cfg,
            staging_dir=staging_dir,
            overhead_tokens=estimate_fixed_prompt_overhead(system_prompt, tools_list),
        )

        # Retrieve all root summary nodes from the database for this session
        root_summaries = await self.context.db.get_root_summary_nodes(self.session_id)
        self._has_summaries = bool(root_summaries)

        if not root_summaries:
            return history, compacted_occurred

        # Re-segment and assemble the scaffold context. A turn that carries a summary_node_id is represented by the
        # scaffold and never verbatim, wherever it sits (manual/forced compaction may reach into the tail).
        turns = compressor.segment_turns(history)
        protect_front = cfg.agent.protect_front_turns
        compressed = [any(msg.summary_node_id is not None for msg in t) for t in turns]
        protected_head = [m for i, t in enumerate(turns) if i < protect_front and not compressed[i] for m in t]
        uncompressed_rest = [m for i, t in enumerate(turns) if i >= protect_front and not compressed[i] for m in t]

        # Format hierarchical summaries
        scaffold_parts = [SCAFFOLD_HEADER]
        for node in root_summaries:
            depth_label = {0: "Recent", 1: "Session Arc", 2: "Durable"}.get(node.level, f"Level-{node.level}")
            scaffold_parts.append(
                f"\n[{depth_label} Summary (Level {node.level}, node {node.id})]\n"
                f"{HistoryCompressor.strip_legacy_sections(node.summary)}"
            )
        # Verbatim user messages of compressed turns (changes only on compaction, so it stays cacheable)
        recent_user_block = HistoryCompressor.build_recent_user_messages_block(
            turns,
            cfg.agent.recent_user_messages_limit,
            total_chars=cfg.agent.recent_user_messages_total_chars,
            ceiling_chars=cfg.agent.recent_user_message_ceiling_chars,
        )
        if recent_user_block:
            scaffold_parts.append("\n" + recent_user_block)
        scaffold_content = "\n".join(scaffold_parts)

        # Build assembled messages in memory
        assembled_history = []
        assembled_history.extend(protected_head)

        # Append Scaffold as a USER message with an ASSISTANT acknowledgement
        scaffold_msg = Message(
            session_id=self.session_id,
            chatbot_id=current_msg.chatbot_id,
            channel_id=current_msg.channel_id,
            sender="system",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content=scaffold_content,
            status=MessageStatus.RESPONDED,
            metadata={"is_scaffold": True},
        )
        ack_msg = Message(
            session_id=self.session_id,
            chatbot_id=current_msg.chatbot_id,
            channel_id=current_msg.channel_id,
            sender="assistant",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content="Understood. I have access to the full conversation history.",
            status=MessageStatus.RESPONDED,
            metadata={"is_scaffold_ack": True},
        )
        assembled_history.append(scaffold_msg)
        assembled_history.append(ack_msg)
        assembled_history.extend(uncompressed_rest)

        return assembled_history, compacted_occurred

    async def _execute_tool_calls(
        self,
        res: LLMResponse,
        current_msg: Message,
        worker: "SessionWorker",
        session_staging_dir: str,
    ) -> bool:
        """Execute the requested tool calls concurrently, posting thought, call, and result messages.

        Returns:
            True if process_turn should continue (loop back to LLM generation),
            False if turn execution was interrupted or transitioned to a new session (should break).
        """
        chatbot_id = current_msg.chatbot_id
        channel_id = current_msg.channel_id

        thought_text = res.thought or res.content
        if thought_text:
            thought_metadata = {}
            if res.thought_signature:
                thought_metadata["thought_signature"] = res.thought_signature
            thought_msg = Message(
                session_id=self.session_id,
                chatbot_id=chatbot_id,
                channel_id=channel_id,
                sender="Kesoku",
                role=MessageRole.ASSISTANT,
                type=MessageType.THOUGHT,
                content=thought_text,
                status=MessageStatus.RESPONDED,
                parent_id=current_msg.id,
                metadata=thought_metadata,
            )
            await self.gateway.post(thought_msg)

        tool_call_msgs = []
        for call in res.tool_calls:
            logger.info(f"Executing requested tool call: '{call.name}' with args {call.arguments}")
            call_args_json = json.dumps(call.arguments, indent=2, ensure_ascii=False)
            tool_call_msg = Message(
                session_id=self.session_id,
                chatbot_id=chatbot_id,
                channel_id=channel_id,
                sender="Kesoku",
                role=MessageRole.TOOL,
                type=MessageType.TOOL_CALL,
                content=f"Calling tool `{call.name}` with arguments:\n```json\n{call_args_json}\n```",
                status=MessageStatus.RESPONDED,
                parent_id=current_msg.id,
                metadata={
                    "tool_name": call.name,
                    "tool_arguments": call.arguments,
                    "thought_signature": call.thought_signature,
                    "tool_call_id": call.tool_call_id,
                },
            )
            await self.gateway.post(tool_call_msg)
            tool_call_msgs.append((call, tool_call_msg))

        self.tool_runner.tool_context.interrupt_check = lambda: not worker.queue_empty()
        self.tool_runner.tool_context.kesoku_context = self.context

        exec_tasks = [
            self.tool_runner.execute_tool(
                call,
                tc_msg,
                is_interrupted=lambda: not worker.queue_empty(),
            )
            for call, tc_msg in tool_call_msgs
        ]
        if not worker.queue_empty():
            logger.info("Interruption detected before launching concurrent tool execution.")
            for coro in exec_tasks:
                coro.close()
            # Mark the initiating message as interrupted to prevent it from being stuck in processing status
            await self.context.db.update_message_status(current_msg.id, MessageStatus.INTERRUPTED)
            return False

        result_msgs = await asyncio.gather(*exec_tasks)

        sub_calls = getattr(self.tool_runner.tool_context, "subagent_llm_calls", 0)
        if isinstance(sub_calls, int):
            self._subagent_calls = sub_calls
        sub_toks = getattr(self.tool_runner.tool_context, "subagent_tokens", 0)
        if isinstance(sub_toks, int):
            self._subagent_tokens = sub_toks

        # P2-2: Post-tool hooks, touch collection, and repo rules discovery
        cfg = self.context.config
        awd = PathResolver.resolve(cfg.agent_working_dir)
        mgr = self._ensure_repo_rules_manager(session_staging_dir, cfg)
        env = getattr(self.context, "env", {})

        for idx, (call_pair, rm) in enumerate(zip(tool_call_msgs, result_msgs)):
            call_obj = call_pair[0]
            if hasattr(cfg, "hooks"):
                await run_post_tool_use_hooks(
                    call_obj,
                    rm,
                    cfg.hooks,
                    awd,
                    session_staging_dir,
                    env,
                )

            paths: list[str] = []
            tool_name = call_obj.name
            self._turn_tools_used.add(tool_name)
            args = call_obj.arguments or {}
            if tool_name in ("update_file", "read_file") and args.get("file_name"):
                try:
                    paths.append(
                        self._resolve_touched_path(
                            str(args["file_name"]),
                            awd,
                            session_staging_dir,
                            self.tool_runner.tool_context,
                        )
                    )
                except Exception:
                    pass
            elif tool_name == "analyze_media" and args.get("media_path"):
                try:
                    paths.append(
                        self._resolve_touched_path(
                            str(args["media_path"]),
                            awd,
                            session_staging_dir,
                            self.tool_runner.tool_context,
                        )
                    )
                except Exception:
                    pass
            elif tool_name == "run_shell_command":
                cwd_arg = args.get("cwd")
                cmd = str(args.get("command", ""))
                if cwd_arg:
                    try:
                        paths.append(
                            self._resolve_touched_path(
                                str(cwd_arg),
                                awd,
                                session_staging_dir,
                                self.tool_runner.tool_context,
                            )
                        )
                    except Exception:
                        pass
                else:
                    m = re.match(r"^\s*cd\s+([^\s;&]+)\s*(?:&&|;)", cmd)
                    if m:
                        try:
                            paths.append(
                                self._resolve_touched_path(
                                    m.group(1),
                                    awd,
                                    session_staging_dir,
                                    self.tool_runner.tool_context,
                                )
                            )
                        except Exception:
                            pass
                    elif awd:
                        paths.append(awd)

            if paths and mgr is not None:
                new_discoveries = mgr.discover(paths)
                abs_staging = PathResolver.resolve(session_staging_dir) if session_staging_dir else None
                in_ruled_subtree = False
                for p in paths:
                    if abs_staging:
                        try:
                            if os.path.commonpath([p, abs_staging]) == abs_staging:
                                continue
                        except ValueError:
                            pass
                    for rf in mgr.discovered_files:
                        rdir = os.path.dirname(PathResolver.resolve(rf))
                        try:
                            if os.path.commonpath([p, rdir]) == rdir:
                                in_ruled_subtree = True
                                break
                        except ValueError:
                            pass
                    if in_ruled_subtree:
                        break

                if in_ruled_subtree:
                    has_tool_error = bool(rm.metadata and rm.metadata.get("tool_error"))
                    if tool_name == "update_file" and not has_tool_error:
                        self._turn_writes = True
                    elif tool_name == "run_shell_command" and self._is_shell_write_command(
                        str(args.get("command", ""))
                    ):
                        self._turn_writes = True

                if new_discoveries:
                    append_text = ""
                    for f in new_discoveries:
                        rel_path = os.path.relpath(f, awd)  # noqa: ASYNC240
                        try:
                            content_str = (await async_read_text_file(f)).strip()
                            if len(content_str) > cfg.agent.repo_rules_max_chars:
                                content_str = (
                                    content_str[: cfg.agent.repo_rules_max_chars] + "\n...[truncated]"
                                )
                        except Exception as e:
                            content_str = f"[Error reading file: {e}]"
                        append_text += f"\n\n[repo rules discovered: {rel_path}]\n{content_str}"

                    rm.content += append_text
                    if rm.metadata and "tool_result" in rm.metadata and rm.metadata["tool_result"]:
                        rm.metadata["tool_result"] += append_text

            rm.timestamp = time.time() + (idx * 0.001)
            await self.gateway.post(rm)

        # Check if history was compacted and session was transitioned
        tool_ctx = self.tool_runner.tool_context
        val = getattr(tool_ctx, "transitioned_to_session", None)
        if isinstance(val, str):
            new_session_id = val
            logger.info(
                f"Session '{self.session_id}' transitioned to '{new_session_id}'. "
                "Aborting remaining turn steps and stopping old session worker."
            )
            # Mark the initiating message as processed
            await self.context.db.update_message_status(current_msg.id, MessageStatus.PROCESSED)

            # Schedule worker stop task in the background so the current execution exits cleanly first
            if self.gateway.agent:
                asyncio.create_task(self.gateway.agent.stop_session_worker(self.session_id, immediate=True))
            return False

        return True

    async def _post_draft_before_hook(
        self,
        res: LLMResponse,
        current_msg: Message,
    ) -> None:
        """Persist the assistant's thought and draft text (status=RESPONDED) before a stop hook/reminder."""
        chatbot_id = current_msg.chatbot_id
        channel_id = current_msg.channel_id
        if res.thought:
            thought_metadata = {}
            if res.thought_signature:
                thought_metadata["thought_signature"] = res.thought_signature
            await self.gateway.post(
                Message(
                    session_id=self.session_id,
                    chatbot_id=chatbot_id,
                    channel_id=channel_id,
                    sender="Kesoku",
                    role=MessageRole.ASSISTANT,
                    type=MessageType.THOUGHT,
                    content=res.thought,
                    status=MessageStatus.RESPONDED,
                    parent_id=current_msg.id,
                    metadata=thought_metadata,
                )
            )
        if res.content and res.content.strip():
            await self.gateway.post(
                Message(
                    session_id=self.session_id,
                    chatbot_id=chatbot_id,
                    channel_id=channel_id,
                    sender="Kesoku",
                    role=MessageRole.ASSISTANT,
                    type=MessageType.TEXT,
                    content=res.content,
                    status=MessageStatus.RESPONDED,
                    parent_id=current_msg.id,
                )
            )

    async def _handle_final_response(
        self,
        res: LLMResponse,
        current_msg: Message,
        last_context_tokens: int,
        last_cached_tokens: int,
        turn_tool_calls: int,
        turn_tokens: int,
        start_time: float,
        llm: BaseLLM,
        empty_attempts: int,
    ) -> tuple[bool, int]:
        """Handle final assistant response including stop hooks, thought logging, retry logic, and metrics.

        `empty_attempts` tracks how many times the model already answered with no content this turn.

        Returns:
            tuple[should_continue, empty_attempts]:
                - should_continue=True: loop should continue (i.e. retrying/nudging the LLM).
                - should_continue=False: loop should break (i.e. successfully processed turn).
        """
        chatbot_id = current_msg.chatbot_id
        channel_id = current_msg.channel_id
        cfg = self.context.config

        if res.content and res.content.strip():
            # P2-2 Stop Hooks
            if hasattr(cfg, "hooks") and cfg.hooks.stop:
                awd = PathResolver.resolve(cfg.agent_working_dir)
                env = getattr(self.context, "env", {})
                outcomes = await run_stop_hooks(self._turn_tools_used, cfg.hooks, awd, env)
                for out in outcomes:
                    if out.exit_code != 0:
                        retries = self._hook_retries.get(out.name, 0)
                        max_retries = 2
                        for hook in cfg.hooks.stop:
                            if hook.name == out.name:
                                max_retries = hook.max_retries
                                break

                        if retries < max_retries:
                            self._hook_retries[out.name] = retries + 1
                            await self._post_draft_before_hook(res, current_msg)
                            msg = Message(
                                session_id=self.session_id,
                                chatbot_id=chatbot_id,
                                channel_id=channel_id,
                                sender="Hook",
                                role=MessageRole.SYSTEM,
                                type=MessageType.TEXT,
                                content=(
                                    f"[stop hook {out.name} exit {out.exit_code}]\n"
                                    f"{out.output}\n"
                                    "Fix the reported problems, then finish."
                                ),
                                status=MessageStatus.RESPONDED,
                                parent_id=current_msg.id,
                            )
                            await self.gateway.post(msg)
                            return True, empty_attempts
                        res.content += f"\n\n[warning: stop hook {out.name} failed after {max_retries} retries]"

            # P2-2 Built-in Stop Reminder
            if cfg.agent.repo_rules_stop_reminder and self._repo_rules_manager is not None:
                if (
                    self._repo_rules_manager.discovered_files
                    and self._turn_writes
                    and not self._stop_reminder_fired
                ):
                    self._stop_reminder_fired = True
                    await self._post_draft_before_hook(res, current_msg)
                    msg = Message(
                        session_id=self.session_id,
                        chatbot_id=chatbot_id,
                        channel_id=channel_id,
                        sender="Hook",
                        role=MessageRole.SYSTEM,
                        type=MessageType.TEXT,
                        content=(
                            'Before finishing, re-check `<context name="repo_rules">`: run any '
                            "verification/lint command it requires, add any test it mandates (e.g. "
                            "regression tests), and fix violations. If everything is already "
                            "satisfied, finish now."
                        ),
                        status=MessageStatus.RESPONDED,
                        parent_id=current_msg.id,
                    )
                    await self.gateway.post(msg)
                    return True, empty_attempts

        if res.thought:
            thought_metadata = {}
            if res.thought_signature:
                thought_metadata["thought_signature"] = res.thought_signature
            thought_msg = Message(
                session_id=self.session_id,
                chatbot_id=chatbot_id,
                channel_id=channel_id,
                sender="Kesoku",
                role=MessageRole.ASSISTANT,
                type=MessageType.THOUGHT,
                content=res.thought,
                status=MessageStatus.RESPONDED,
                parent_id=current_msg.id,
                metadata=thought_metadata,
            )
            await self.gateway.post(thought_msg)

        final_content = res.content
        if not final_content.strip():
            empty_attempts += 1
            if empty_attempts <= EMPTY_RESPONSE_RETRIES:
                # Empty responses are usually transient (e.g. the provider filtered the candidate),
                # so re-generate with the exact same history instead of polluting it with a nudge.
                logger.warning(
                    f"LLM returned empty content in session {self.session_id} "
                    f"(finish_reason={res.finish_reason}, attempt {empty_attempts}). Retrying."
                )
                return True, empty_attempts
            if empty_attempts == EMPTY_RESPONSE_RETRIES + 1:
                logger.warning(
                    f"LLM still returned empty content in session {self.session_id} "
                    f"(finish_reason={res.finish_reason}, attempt {empty_attempts}). Nudging model."
                )
                nudge_msg = Message(
                    session_id=self.session_id,
                    chatbot_id=chatbot_id,
                    channel_id=channel_id,
                    sender="System",
                    role=MessageRole.SYSTEM,
                    type=MessageType.TEXT,
                    content=(
                        "Your previous response had empty content. Please provide a final "
                        "user-facing response summarizing your results/actions."
                    ),
                    status=MessageStatus.RESPONDED,
                    parent_id=current_msg.id,
                )
                await self.gateway.post(nudge_msg)
                return True, empty_attempts
            logger.error(
                f"LLM returned empty content {empty_attempts} times in session {self.session_id} "
                f"(finish_reason={res.finish_reason}). Giving up on this turn."
            )
            final_content = (
                f"\u26a0\ufe0f The model returned an empty response after {empty_attempts} attempts "
                f"(finish_reason={res.finish_reason}). Please rephrase and try again."
            )

        limit = getattr(llm, "context_window_limit", 1048576)
        context_percent = (last_context_tokens / limit) * 100 if limit else 0.0

        final_msg = Message(
            session_id=self.session_id,
            chatbot_id=chatbot_id,
            channel_id=channel_id,
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content=final_content,
            status=MessageStatus.PENDING,
            parent_id=current_msg.id,
            metadata={
                "turn_metrics": {
                    "session_turns": await self._get_session_turns_count(),
                    "context_tokens": last_context_tokens,
                    "cached_tokens": last_cached_tokens,
                    "context_limit": limit,
                    "context_percent": context_percent,
                    "turn_tool_calls": turn_tool_calls,
                    "turn_tokens": turn_tokens,
                    "turn_time": time.time() - start_time,
                    "status": "finished",
                    "subagent_calls": self._subagent_calls,
                    "subagent_tokens": self._subagent_tokens,
                }
            },
        )
        await self.gateway.post(final_msg)
        await self.context.db.update_message_status(current_msg.id, MessageStatus.PROCESSED)
        return False, empty_attempts  # should_continue=False, counter unmodified
