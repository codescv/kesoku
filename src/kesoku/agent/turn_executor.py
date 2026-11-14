"""Turn execution logic for the Kesoku Agent."""

import asyncio
import datetime
import fnmatch
import inspect
import json
import os
import re
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import tzlocal

from kesoku.agent.compressor import SCAFFOLD_HEADER, HistoryCompressor, estimate_tokens_calibrated
from kesoku.agent.context_dir import (
    ContextEntry,
    apply_total_budget,
    collect_context_entries,
    insert_extra_entries,
    render_context_blocks,
)
from kesoku.agent.context_policy import resolve_context_policy, resolve_memory_policy
from kesoku.agent.history import (
    build_history,
    prepare_history_for_llm,
    strip_question_options,
)
from kesoku.agent.hooks import run_post_tool_use_hooks, run_stop_hooks
from kesoku.agent.llm import BaseLLM, LLMResponse, is_explicit_cache_unsupported_model
from kesoku.agent.memory_store import (
    default_pinned_cache,
    format_ambient_memory_block,
    render_context_query,
    search_role_episodes,
    search_role_memories,
)
from kesoku.agent.repo_rules import RepoRulesManager
from kesoku.agent.session_digest import (
    DigestUpdater,
    extract_bootstrap_extra_context,
    fetch_recent_digests,
    format_recent_sessions_block,
    is_excluded_from_memory,
    should_render,
)
from kesoku.agent.state_snapshot import render_repo_rules, render_running_jobs, render_staging_artifacts
from kesoku.agent.tool_runner import ToolRunner
from kesoku.agent.tools.file import expand_path_variables
from kesoku.agent.turn_logger import TurnLogger
from kesoku.config import DiscordChannelOverride, KesokuConfig, resolve_gemini_cache_ttl
from kesoku.context import KesokuContext
from kesoku.db.manager import DatabaseManager
from kesoku.utils.async_fs import async_exists, async_read_text_file, async_write_text_file
from kesoku.utils.path import PathResolver
from kesoku.utils.text import strip_injected_blocks, truncate_middle
from kesoku.utils.time_utils import humanize_gap, load_user_clock, user_local_stamp
from kesoku.utils.ui_markup import collapse_ui_blocks, extract_ui_options, parse_multi_selection

if TYPE_CHECKING:
    from kesoku.agent.agent import SessionWorker
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger

logger = setup_logger(__name__)
MAX_CHATBOT_ERROR_MESSAGE_LENGTH = 500
_REFERENTIAL_QUERY_RE = re.compile(r"刚才|之前|上次|那个|另一个|继续|下一")
# Number of silent re-generations attempted when the model returns no content at all
# (e.g. Vertex safety filtering yields finish_reason=SAFETY with zero parts) before
# falling back to nudging the model with an explicit system message.
EMPTY_RESPONSE_RETRIES = 2
# Max characters kept per Agent line inside auto-retrieved historical dialogue snippets.
MAX_RAG_AGENT_LINE_CHARS = 400
# Context block name used to inject opaque per-message client preferences (`metadata.mode_prefs`).
MODE_PREFS_CONTEXT_NAME = "app-prefs"
MODE_PREFS_MAX_BYTES = 2048
_LLM_OVERRIDE_KEYS = ("llm", "model_name", "thinking_level", "stream")
_LLM_OVERRIDE_PROVIDERS = ("gemini", "claude")
_LLM_OVERRIDE_THINKING_LEVELS = ("minimal", "low", "medium", "high", "xhigh", "max")


def filter_ignored_context_entries(entries: list[ContextEntry], globs: Any) -> list[ContextEntry]:
    """Drop context entries whose block name matches any glob in `globs` (generic; no names hard-coded).

    Args:
        entries: Context entries to render.
        globs: Glob patterns from `Message.metadata.ignore_context_globs` (non-list values are ignored).

    Returns:
        Entries whose names match none of the patterns.
    """
    if not isinstance(globs, list):
        return entries
    patterns = [g.strip() for g in globs if isinstance(g, str) and g.strip()]
    if not patterns:
        return entries
    return [e for e in entries if not any(fnmatch.fnmatchcase(e.name, p) for p in patterns)]


def render_mode_prefs_block(mode_prefs: Any) -> str:
    """Render `metadata.mode_prefs` as an opaque `<context name="app-prefs">` JSON block (empty if invalid)."""
    if not isinstance(mode_prefs, dict) or not mode_prefs:
        return ""
    try:
        payload = json.dumps(mode_prefs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return ""
    if len(payload.encode("utf-8")) > MODE_PREFS_MAX_BYTES:
        logger.warning("Ignoring oversized metadata.mode_prefs block")
        return ""
    payload = payload.replace("</context", "<\\/context")
    return f'<context name="{MODE_PREFS_CONTEXT_NAME}">\n{payload}\n</context>\n'


def validate_llm_override(raw: Any, allowed_models: list[str] | None) -> dict[str, Any] | None:
    """Validate a generic `metadata.llm_override` against whitelisted keys and `allowed_models`.

    Args:
        raw: Candidate override mapping `{llm?, model_name?, thinking_level?, stream?}`.
        allowed_models: Model names accepted for this channel; a `model_name` outside it rejects the override.

    Returns:
        A sanitized override dict, or None when invalid / empty.
    """
    if not isinstance(raw, dict):
        return None
    if any(k not in _LLM_OVERRIDE_KEYS for k in raw):
        return None
    out: dict[str, Any] = {}
    llm = raw.get("llm")
    if llm is not None:
        if not isinstance(llm, str) or llm.lower() not in _LLM_OVERRIDE_PROVIDERS:
            return None
        out["llm"] = llm.lower()
    model_name = raw.get("model_name")
    if model_name is not None:
        if not isinstance(model_name, str) or model_name not in (allowed_models or []):
            return None
        out["model_name"] = model_name
    thinking = raw.get("thinking_level")
    if thinking is not None:
        if not isinstance(thinking, str) or thinking not in _LLM_OVERRIDE_THINKING_LEVELS:
            return None
        out["thinking_level"] = thinking
    stream = raw.get("stream")
    if stream is not None:
        if not isinstance(stream, bool):
            return None
        out["stream"] = stream
    return out or None

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


def _escape_xml_attr(value: str) -> str:
    """Escape double quotes and newlines inside an XML-like header attribute value."""
    return str(value).replace("&", "&amp;").replace('"', "&quot;").replace("\n", " ").strip()


def _find_prev_human_user_timestamp(history: list[Message], current_idx: int) -> float | None:
    """Return timestamp of the last prior non-system/cronjob USER message before `current_idx`, or None."""
    for prev in reversed(history[:current_idx]):
        meta = prev.metadata or {}
        if meta.get("is_scaffold") or SCAFFOLD_HEADER in (prev.content or ""):
            break
        if prev.role != MessageRole.USER:
            continue
        raw_sender = str(prev.sender or "").strip().lower()
        raw_sender_name = str(meta.get("sender_name") or "").strip().lower()
        if raw_sender in ("cronjob", "system") or raw_sender_name in ("cronjob", "system"):
            continue
        return float(prev.timestamp)
    return None


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


EXPLICIT_CACHE_MIN_PREFIX_TOKENS = 4600
EXPLICIT_CACHE_REFRESH_MIN_SUFFIX_TOKENS = 8192
EXPLICIT_CACHE_REFRESH_RATIO = 0.25
EXPLICIT_CACHE_TTL_RENEW_BELOW_SECONDS = 600.0


@dataclass
class ExplicitCacheEntry:
    """In-memory metadata for an explicit context cache isolated by `(session_id, model)`."""

    cache_name: str
    model_name: str
    llm: BaseLLM
    cached_messages_len: int
    cached_est_tokens: int
    created_at: float
    expires_at: float


_session_cache_locks: dict[str, asyncio.Lock] = {}


def _get_session_cache_lock(session_id: str) -> asyncio.Lock:
    """Return a per-session asyncio.Lock serializing explicit cache state reads and mutations."""
    lock = _session_cache_locks.get(session_id)
    if lock is not None and not lock.locked():
        bound_loop = getattr(lock, "_loop", None)
        if bound_loop is not None and bound_loop is not asyncio.get_running_loop():
            lock = None
    if lock is None:
        lock = asyncio.Lock()
        _session_cache_locks[session_id] = lock
    return lock


def discard_session_cache_state(session_id: str) -> None:
    """Drop per-session cache lock when a session is deleted."""
    lock = _session_cache_locks.get(session_id)
    if lock is not None and not lock.locked():
        _session_cache_locks.pop(session_id, None)


def _cache_model_key(llm: BaseLLM) -> str:
    """Build a model isolation key so channel overrides never reuse incompatible CachedContents."""
    model_id = (
        getattr(llm, "model_name", None)
        or getattr(getattr(llm, "config", None), "model_name", None)
        or "default"
    )
    return f"{llm.__class__.__name__}:{model_id}"


def _llm_supports_explicit_cache(llm: Any) -> bool:
    """Return True when `llm` supports explicit context caching, preserving mock `GeminiLLM(BaseLLM)` compatibility."""
    if llm is None:
        return False
    if "supports_explicit_cache" in getattr(llm, "__dict__", {}):
        fn = getattr(llm, "supports_explicit_cache", None)
        return bool(fn()) if callable(fn) else bool(fn)
    cls_fn = getattr(type(llm), "supports_explicit_cache", None)
    if cls_fn is not None and cls_fn is not BaseLLM.supports_explicit_cache:
        fn = getattr(llm, "supports_explicit_cache", lambda: False)
        return bool(fn()) if callable(fn) else False
    if llm.__class__.__name__ == "GeminiLLM":
        llm_cfg = getattr(llm, "config", None)
        if llm_cfg is not None:
            if not getattr(llm_cfg, "context_caching", True):
                return False
            if getattr(llm_cfg, "context_caching_mode", "explicit") == "implicit":
                return False
        model_name = getattr(llm, "model_name", None) or getattr(llm_cfg, "model_name", None)
        if isinstance(model_name, str) and is_explicit_cache_unsupported_model(model_name):
            return False
        return True
    fn = getattr(llm, "supports_explicit_cache", lambda: False)
    return bool(fn()) if callable(fn) else False


def _is_cache_invalid_error(exc: BaseException) -> bool:
    """Return True if an LLM generation exception indicates an expired or missing CachedContent."""
    msg_lower = str(exc).lower()
    return (
        "404" in msg_lower
        or "not found" in msg_lower
        or "not_found" in msg_lower
        or "cachedcontent" in msg_lower
        or "cached_content" in msg_lower
        or "expired" in msg_lower
        or "cache" in msg_lower
    )


def estimate_messages_tokens_local(messages: list[Message]) -> int:
    """Estimate tokens of prepared messages locally without blocking network RPCs."""
    return sum(estimate_tokens_calibrated(m.content) for m in messages)


def estimate_prefix_tokens_local(
    prefix_messages: list[Message],
    system_prompt: str | None,
    tools_list: list[Any] | None,
) -> int:
    """Estimate total tokens of system instruction + tools + static prefix messages locally."""
    return estimate_fixed_prompt_overhead(system_prompt, tools_list) + estimate_messages_tokens_local(prefix_messages)


def _sync_worker_active_fields(worker: Any, entry: ExplicitCacheEntry | None) -> None:
    """Synchronize legacy worker attributes (`active_cache_name`, etc.) with `entry`."""
    if worker is None:
        return
    try:
        if entry is None:
            worker.active_cache_name = None
            worker.active_cache_llm = None
            worker.cached_messages_len = 0
            worker.cache_expires_at = 0.0
        else:
            worker.active_cache_name = entry.cache_name
            worker.active_cache_llm = entry.llm
            worker.cached_messages_len = entry.cached_messages_len
            worker.cache_expires_at = entry.expires_at
    except Exception:
        pass


def _ensure_worker_cache_map(
    worker: Any,
    default_llm: BaseLLM | None = None,
    default_ttl: int = 1200,
) -> dict[str, ExplicitCacheEntry]:
    """Return `worker.explicit_caches` dict, hydrating any pre-populated legacy attributes."""
    raw_map = getattr(worker, "explicit_caches", None)
    if not isinstance(raw_map, dict):
        raw_map = {}
        try:
            worker.explicit_caches = raw_map
        except Exception:
            pass

    legacy_name = getattr(worker, "active_cache_name", None)
    if isinstance(legacy_name, str) and legacy_name:
        already_tracked = any(
            isinstance(ent, ExplicitCacheEntry) and ent.cache_name == legacy_name for ent in raw_map.values()
        )
        if not already_tracked:
            legacy_llm = getattr(worker, "active_cache_llm", None)
            target_llm = legacy_llm if isinstance(legacy_llm, BaseLLM) else default_llm
            if target_llm is not None:
                c_len = getattr(worker, "cached_messages_len", 0)
                c_len_int = c_len if isinstance(c_len, int) and not isinstance(c_len, bool) else 0
                raw_exp = getattr(worker, "cache_expires_at", 0.0)
                now_ts = time.time()
                exp_ts = (
                    float(raw_exp)
                    if isinstance(raw_exp, (int, float)) and not isinstance(raw_exp, bool) and raw_exp > 0
                    else (now_ts + default_ttl)
                )
                m_name = str(getattr(target_llm, "model_name", None) or "default")
                raw_map[_cache_model_key(target_llm)] = ExplicitCacheEntry(
                    cache_name=legacy_name,
                    model_name=m_name,
                    llm=target_llm,
                    cached_messages_len=c_len_int,
                    cached_est_tokens=0,
                    created_at=now_ts,
                    expires_at=exp_ts,
                )
    return raw_map


async def _clear_worker_explicit_caches(worker: Any, fallback_llm: BaseLLM | None = None) -> None:
    """Delete all explicit caches tracked on `worker` and reset cache attributes."""
    if worker is None:
        return
    deleted: set[str] = set()
    raw_map = getattr(worker, "explicit_caches", None)
    if isinstance(raw_map, dict):
        for entry in list(raw_map.values()):
            c_name = getattr(entry, "cache_name", None)
            c_llm = getattr(entry, "llm", None) or fallback_llm
            if isinstance(c_name, str) and c_name and c_name not in deleted and c_llm is not None:
                deleted.add(c_name)
                try:
                    await c_llm.delete_cache(c_name)
                except Exception as de:
                    logger.warning(f"Failed to delete explicit context cache {c_name}: {de}")
        raw_map.clear()

    legacy_name = getattr(worker, "active_cache_name", None)
    if isinstance(legacy_name, str) and legacy_name and legacy_name not in deleted:
        legacy_llm = getattr(worker, "active_cache_llm", None)
        c_llm = legacy_llm if isinstance(legacy_llm, BaseLLM) else fallback_llm
        if c_llm is not None:
            try:
                await c_llm.delete_cache(legacy_name)
            except Exception as de:
                logger.warning(f"Failed to delete explicit context cache {legacy_name}: {de}")
    _sync_worker_active_fields(worker, None)


class _StreamPreviewPublisher:
    """Throttled, non-blocking publisher for ephemeral streaming response previews.

    Publishes the first non-empty text chunk immediately so TTFT is visible right away, then throttles
    subsequent updates to `min_interval` seconds and skips updates while a prior publish task is still
    in flight (so Discord REST latency or rate-limit backoff never stalls LLM token consumption).
    """

    def __init__(
        self,
        gateway: Gateway,
        chatbot_id: str,
        channel_id: str,
        session_id: str,
        turn_id: str,
        min_interval: float = 1.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize a per-generation stream preview publisher."""
        self.gateway = gateway
        self.chatbot_id = chatbot_id
        self.channel_id = channel_id
        self.session_id = session_id
        self.turn_id = turn_id
        self.min_interval = min_interval
        self._clock = clock
        self._last_publish_ts: float | None = None
        self._inflight_task: asyncio.Task[None] | None = None
        self._published_any: bool = False
        self._discarded: bool = False

    async def _publish(self, text: str | None) -> None:
        """Forward preview text (or None to clear) to `gateway.publish_stream` safely."""
        try:
            await self.gateway.publish_stream(
                self.chatbot_id,
                self.channel_id,
                self.session_id,
                self.turn_id,
                text,
            )
        except Exception as e:
            logger.warning(f"Stream preview publish failed for session {self.session_id}: {e}")

    async def on_chunk(self, delta: str, accumulated: str) -> None:
        """Handle a streamed text delta; schedules at most one in-flight preview update."""
        _ = delta
        if self._discarded or not accumulated:
            return
        if self._inflight_task is not None and not self._inflight_task.done():
            return
        now = self._clock()
        if self._last_publish_ts is not None and (now - self._last_publish_ts) < self.min_interval:
            return
        self._last_publish_ts = now
        self._published_any = True
        self._inflight_task = asyncio.create_task(self._publish(accumulated))
        # Yield once so fast/in-memory handlers start immediately without awaiting slow I/O.
        await asyncio.sleep(0)

    async def wait_inflight(self) -> None:
        """Await any currently running preview publish task so final delivery cannot race with it."""
        task = self._inflight_task
        self._inflight_task = None
        if task is not None and not task.done():
            try:
                await task
            except Exception as e:
                logger.warning(f"In-flight stream preview task failed for session {self.session_id}: {e}")

    async def discard(self) -> None:
        """Clear any published preview on non-final continuations (tool calls, retries, hooks, errors)."""
        self._discarded = True
        await self.wait_inflight()
        if self._published_any:
            self._published_any = False
            await self._publish(None)


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
        # Per-turn frozen cross-session memory blocks: (pinned_block, recent_sessions_block, ambient_block).
        # Reused across R1..RN (never reset on mid-turn memory writes); reset only on pivot (H1-dev).
        self._memory_snapshot: tuple[str, str, str] | None = None
        self._memory_pre_turn_ms: float = 0.0
        self._pinned_chars: int = 0
        self._recent_sessions_chars: int = 0
        self._recent_sessions_rendered: bool = False
        self._turn_context_policy: str = "full"
        # Authoritative override resolved once per turn (keyed by the turn's user message id).
        self._turn_policy_override_msg_id: str | None = None
        self._turn_policy_override: str | None = None
        self._turn_clicked_option: bool = False
        self._last_final_assistant_msg: Message | None = None
        self._turn_start_digest_task: asyncio.Task[None] | None = None
        self._has_summaries: bool | None = None
        self._compaction_pending: bool = False
        self._last_turn_prompt_blocked: bool = False
        self._bg_compaction_tasks: set[asyncio.Task[Any]] = set()
        self._turn_rounds: list[dict[str, Any]] = []
        self._turn_used_explicit_cache: bool = False
        self._turn_model_name: str = ""
        self._turn_thinking_level: str | None = None
        self._turn_stream: bool = False
        self._turn_cache_mode: str = "disabled"
        self._last_round_tool_errors: int = 0

    @staticmethod
    def _resolve_cache_mode(cfg: KesokuConfig, llm: BaseLLM) -> str:
        """Resolve effective context cache mode string for telemetry."""
        cls_name = llm.__class__.__name__
        if cls_name == "GeminiLLM":
            if not cfg.gemini.context_caching:
                return "disabled"
            if cfg.gemini.context_caching_mode == "implicit" or not _llm_supports_explicit_cache(llm):
                return "implicit"
            return str(cfg.gemini.context_caching_mode or "explicit")
        llm_cfg = getattr(llm, "config", None)
        if llm_cfg is not None and getattr(llm_cfg, "context_caching", False):
            return str(getattr(llm_cfg, "context_caching_mode", None) or "ephemeral")
        return "disabled"

    async def _build_turn_metrics(
        self,
        *,
        last_context_tokens: int,
        last_cached_tokens: int,
        turn_tool_calls: int,
        turn_tokens: int,
        start_time: float,
        status: str,
        llm: BaseLLM | None = None,
    ) -> dict[str, Any]:
        """Construct backward-compatible `turn_metrics` enriched with per-round telemetry."""
        metrics: dict[str, Any] = {
            "session_turns": await self._get_session_turns_count(),
            "context_tokens": last_context_tokens,
            "cached_tokens": last_cached_tokens,
            "turn_tool_calls": turn_tool_calls,
            "turn_tokens": turn_tokens,
            "turn_time": time.time() - start_time,
            "status": status,
            "model_name": self._turn_model_name
            or (
                str(
                    getattr(llm, "model_name", None)
                    or getattr(getattr(llm, "config", None), "model_name", None)
                    or llm.__class__.__name__
                )
                if llm is not None
                else ""
            ),
            "thinking_level": self._turn_thinking_level,
            "stream": self._turn_stream,
            "cache_mode": self._turn_cache_mode,
            "used_explicit_cache": self._turn_used_explicit_cache,
            "memory_pre_turn_ms": self._memory_pre_turn_ms,
            "pinned_chars": self._pinned_chars,
            "recent_sessions_chars": self._recent_sessions_chars,
            "recent_sessions_rendered": self._recent_sessions_rendered,
            "rounds": [dict(r) for r in self._turn_rounds],
        }
        if llm is not None:
            limit = getattr(llm, "context_window_limit", 1048576)
            context_percent = (last_context_tokens / limit) * 100 if limit else 0.0
            metrics["context_limit"] = limit
            metrics["context_percent"] = context_percent
            metrics["subagent_calls"] = self._subagent_calls
            metrics["subagent_tokens"] = self._subagent_tokens
        return metrics

    def _is_compaction_pending(self, worker: "SessionWorker") -> bool:
        """Return True if a prior turn deferred or failed post-turn compaction."""
        return self._compaction_pending or (getattr(worker, "_compaction_pending", False) is True)

    def _set_compaction_pending(self, worker: "SessionWorker", pending: bool) -> None:
        """Persist the compaction safety-net flag on both this executor and the session worker."""
        self._compaction_pending = pending
        try:
            worker._compaction_pending = pending  # type: ignore[attr-defined]
        except Exception:
            pass

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

    def _resolve_channel_override(self, current_msg: Message) -> DiscordChannelOverride | None:
        """Return the first matching DiscordChannelOverride for `current_msg`, regardless of fields set.

        Matches direct `channel_id`, `channel_name`, and thread `parent_channel_id` / `parent_channel_name`.
        """
        cfg = self.context.config
        discord_cfg = cfg.get_discord_config(current_msg.chatbot_id)
        if not discord_cfg:
            return None

        raw_meta = getattr(current_msg, "metadata", None)
        metadata = raw_meta if isinstance(raw_meta, dict) else {}
        channel_id = current_msg.channel_id
        channel_name = metadata.get("channel_name", "")
        parent_id = metadata.get("parent_channel_id")
        parent_name = metadata.get("parent_channel_name")

        identifiers = {ident for ident in (channel_id, channel_name, parent_id, parent_name) if ident}
        for override in discord_cfg.channels:
            if any(ident in override.channels for ident in identifiers):
                return override
        return None

    def _resolve_message_llm_override(self, current_msg: Message) -> dict[str, Any] | None:
        """Return the validated generic `metadata.llm_override` of `current_msg`, if any.

        Model names are checked against the `allowed_models` list of the originating channel's app config;
        channels without such a list never accept per-message model overrides.
        """
        raw_meta = getattr(current_msg, "metadata", None)
        metadata = raw_meta if isinstance(raw_meta, dict) else {}
        raw = metadata.get("llm_override")
        if raw is None:
            return None
        get_app_cfg = getattr(self.context.config, "get_app_config", None)
        app_cfg = get_app_cfg(current_msg.chatbot_id) if callable(get_app_cfg) else None
        allowed = list(getattr(app_cfg, "allowed_models", []) or []) if app_cfg is not None else []
        validated = validate_llm_override(raw, allowed)
        if validated is None:
            logger.warning(f"Ignoring invalid metadata.llm_override on message {current_msg.id}: {raw!r}")
        return validated

    def _resolve_stream_min_interval(self, chatbot_id: str) -> float:
        """Return the per-chatbot streamed preview throttle interval (default 1.5s)."""
        get_fn = getattr(self.gateway, "get_stream_min_interval", None)
        if callable(get_fn):
            try:
                value = get_fn(chatbot_id)
                if isinstance(value, (int, float)) and value >= 0:
                    return float(value)
            except Exception:
                pass
        return 1.5

    def _resolve_llm(self, current_msg: Message) -> BaseLLM:
        """Resolve the appropriate LLM instance for the current message, applying overrides.

        Precedence: validated per-message `metadata.llm_override`, then Discord channel overrides, then the
        configured default LLM.

        Args:
            current_msg: The active user message initiating the turn.

        Returns:
            A BaseLLM instance to use for this turn.
        """
        msg_override = self._resolve_message_llm_override(current_msg)
        if msg_override is not None and any(k in msg_override for k in ("llm", "model_name", "thinking_level")):
            kwargs_m: dict[str, Any] = {}
            if "llm" in msg_override:
                kwargs_m["provider"] = msg_override["llm"]
            if "model_name" in msg_override:
                kwargs_m["model_name"] = msg_override["model_name"]
            if "thinking_level" in msg_override:
                kwargs_m["thinking_level"] = msg_override["thinking_level"]
            try:
                logger.info(f"Applying per-message LLM override {msg_override} for message {current_msg.id}")
                return self.context.get_llm(**kwargs_m)
            except Exception as e:
                logger.error(f"Failed to get per-message override LLM {msg_override}: {e}")

        override = self._resolve_channel_override(current_msg)
        if override is not None and (
            override.llm is not None or override.model_name is not None or override.thinking_level is not None
        ):
            raw_meta = getattr(current_msg, "metadata", None)
            metadata = raw_meta if isinstance(raw_meta, dict) else {}
            channel_name = metadata.get("channel_name", "")
            logger.info(
                f"Applying LLM override (provider={override.llm}, model={override.model_name}, "
                f"thinking_level={override.thinking_level}) for Discord channel "
                f"{current_msg.channel_id} ('{channel_name}')"
            )
            kwargs: dict[str, Any] = {}
            if override.llm is not None:
                kwargs["provider"] = override.llm
            if override.model_name is not None:
                kwargs["model_name"] = override.model_name
            if override.thinking_level is not None:
                kwargs["thinking_level"] = override.thinking_level
            try:
                return self.context.get_llm(**kwargs)
            except Exception as e:
                logger.error(
                    f"Failed to get override LLM (provider='{override.llm}', "
                    f"model='{override.model_name}', thinking_level='{override.thinking_level}'): {e}"
                )

        return self.context.get_llm()

    def _resolve_stream(self, current_msg: Message, llm: BaseLLM) -> bool:
        """Resolve whether LLM streaming preview is enabled for `current_msg` and `llm`.

        Precedence: requires a registered Gateway stream handler for `chatbot_id`; then channel
        `override.stream` if explicitly set; otherwise provider `llm.config.stream` (default False).
        """
        has_handler_fn = getattr(self.gateway, "has_stream_handler", None)
        if callable(has_handler_fn):
            if has_handler_fn(current_msg.chatbot_id) is False:
                return False

        msg_override = self._resolve_message_llm_override(current_msg)
        if msg_override is not None and "stream" in msg_override:
            return bool(msg_override["stream"])
        override = self._resolve_channel_override(current_msg)
        if override is not None and override.stream is not None:
            return override.stream
        return bool(getattr(getattr(llm, "config", None), "stream", False))

    def _resolve_supports_deferred_media(self, chatbot_id: str) -> bool:
        """Return True when the gateway explicitly reports deferred-media support for ``chatbot_id``."""
        supports_fn = getattr(self.gateway, "supports_deferred_media", None)
        if callable(supports_fn):
            return supports_fn(chatbot_id) is True
        return False

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

        self._turn_rounds = []
        self._turn_used_explicit_cache = False
        self._turn_model_name = ""
        self._turn_thinking_level = None
        self._turn_stream = False
        self._turn_cache_mode = "disabled"
        self._last_round_tool_errors = 0
        last_round_used_explicit_cache = False
        last_round_generate_est_tokens = 0

        empty_attempts = 0
        active_publisher: _StreamPreviewPublisher | None = None
        completed_final_response = False
        self._last_turn_prompt_blocked = False
        last_raw_history: list[Message] | None = None
        last_system_prompt: str | None = None
        last_tools_list: list[Any] = []
        last_llm: BaseLLM | None = None
        self._last_final_assistant_msg = None

        def _schedule_turn_start_digest(msg: Message) -> None:
            sync_db = getattr(self.context.db, "_sync", getattr(self.context.db, "sync_db", None))
            if not isinstance(sync_db, DatabaseManager) or is_excluded_from_memory(msg):
                return

            async def _touch() -> None:
                sess_obj = await self.context.db.get_session(self.session_id)
                role_str = await self._resolve_active_role(msg)
                await DigestUpdater.touch_at_turn_start(self.context.db, sess_obj, role_str, msg)

            self._turn_start_digest_task = asyncio.create_task(_touch())

        try:
            self.tool_runner.tool_context.user_turn = await self._get_session_turns_count()
            self.tool_runner.tool_context.supports_deferred_media = self._resolve_supports_deferred_media(chatbot_id)
            _schedule_turn_start_digest(current_msg)
            while worker.running:
                # Check-in before atomic action (Thought Interruption)
                prev_msg_id = current_msg.id
                current_msg = await worker.drain_queue_and_pivot(current_msg)
                if current_msg.id != prev_msg_id:
                    logger.info(
                        f"Pivoted from message {prev_msg_id} to {current_msg.id} during turn loop. "
                        f"Resetting turn metrics and empty-response counter."
                    )
                    chatbot_id = current_msg.chatbot_id
                    channel_id = current_msg.channel_id
                    self.tool_runner.tool_context.original_msg_id = current_msg.id
                    self.tool_runner.tool_context.user_turn = await self._get_session_turns_count()
                    self.tool_runner.tool_context.supports_deferred_media = self._resolve_supports_deferred_media(
                        chatbot_id
                    )
                    self._snapshot_cache = None
                    self._memory_snapshot = None
                    self._memory_pre_turn_ms = 0.0
                    self._pinned_chars = 0
                    self._recent_sessions_chars = 0
                    self._recent_sessions_rendered = False
                    _schedule_turn_start_digest(current_msg)
                    self._turn_tools_used.clear()
                    self._turn_writes = False
                    self._stop_reminder_fired = False
                    self._hook_retries.clear()
                    self._subagent_calls = 0
                    self._subagent_tokens = 0
                    self.tool_runner.tool_context.subagent_llm_calls = 0
                    self.tool_runner.tool_context.subagent_tokens = 0
                    empty_attempts = 0
                    self._last_turn_prompt_blocked = False
                    turn_tool_calls = 0
                    turn_tokens = 0
                    last_context_tokens = 0
                    last_cached_tokens = 0
                    self._turn_rounds.clear()
                    self._turn_used_explicit_cache = False
                    self._last_round_tool_errors = 0
                    last_round_used_explicit_cache = False
                    last_round_generate_est_tokens = 0
                    start_time = time.time()

                # Resolve LLM dynamically for the current message
                llm = self._resolve_llm(current_msg)
                stream = self._resolve_stream(current_msg, llm)
                self._turn_model_name = str(
                    getattr(llm, "model_name", None)
                    or getattr(getattr(llm, "config", None), "model_name", None)
                    or llm.__class__.__name__
                )
                self._turn_thinking_level = getattr(getattr(llm, "config", None), "thinking_level", None)
                self._turn_stream = stream
                self._turn_cache_mode = self._resolve_cache_mode(cfg, llm)

                # Retrieve system prompt directly from session
                session = await self.context.db.get_session(self.session_id)
                system_prompt = session.system_prompt if session else None

                tools_list = self.tool_runner.tool_registry.get_tools_list()

                # Retrieve and build the raw session history (skip bulky compressed ASSISTANT/TOOL rows)
                raw_history = await build_history(
                    gateway=self.gateway,
                    session_id=self.session_id,
                    heal_orphans=True,
                    active_only=True,
                )

                # Safety net: only run pre-turn compaction if a previous turn skipped/failed post-turn compaction.
                if self._is_compaction_pending(worker):
                    self._set_compaction_pending(worker, False)
                    compacted_pre = await self._run_auto_compaction(
                        history=raw_history,
                        system_prompt=system_prompt,
                        tools_list=tools_list,
                        llm=llm,
                        cfg=cfg,
                        staging_dir=session_staging_dir,
                    )
                    if compacted_pre:
                        raw_history = await build_history(
                            gateway=self.gateway,
                            session_id=self.session_id,
                            heal_orphans=False,
                            active_only=True,
                        )
                        logger.info("Pre-turn fallback compaction occurred. Clearing obsolete explicit caches...")
                        async with _get_session_cache_lock(self.session_id):
                            await _clear_worker_explicit_caches(worker, fallback_llm=llm)

                last_raw_history = raw_history
                last_system_prompt = system_prompt
                last_tools_list = tools_list
                last_llm = llm

                # Assemble scaffold from existing summary nodes (no summarization LLM calls before first token)
                history = await self._assemble_scaffold_history(
                    history=raw_history,
                    cfg=cfg,
                    current_msg=current_msg,
                )

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

                # Read per-(session, model) explicit cache snapshot without blocking R1..RN on cache creation RPCs.
                implicit_caching = (not cfg.gemini.context_caching) or (cfg.gemini.context_caching_mode == "implicit")
                active_cache_name: str | None = None
                cached_messages_len: int = 0
                model_key = _cache_model_key(llm)
                ttl_cfg_secs = resolve_gemini_cache_ttl(cfg.gemini)

                if implicit_caching:
                    async with _get_session_cache_lock(self.session_id):
                        raw_existing = getattr(worker, "explicit_caches", None)
                        legacy_existing = getattr(worker, "active_cache_name", None)
                        if (isinstance(raw_existing, dict) and raw_existing) or (
                            isinstance(legacy_existing, str) and legacy_existing
                        ):
                            logger.info("Explicit caching disabled/implicit: deleting existing explicit caches...")
                            await _clear_worker_explicit_caches(worker, fallback_llm=llm)
                elif llm.__class__.__name__ == "GeminiLLM":
                    async with _get_session_cache_lock(self.session_id):
                        caches_map = _ensure_worker_cache_map(worker, default_llm=llm, default_ttl=ttl_cfg_secs)
                        entry = caches_map.get(model_key)
                        if entry is not None:
                            if entry.expires_at > 0 and time.time() >= entry.expires_at:
                                logger.info(
                                    f"Explicit cache {entry.cache_name} locally expired; skipping uncached."
                                )
                                caches_map.pop(model_key, None)
                                _sync_worker_active_fields(worker, next(iter(caches_map.values()), None))
                            elif len(llm_history) < entry.cached_messages_len:
                                logger.info(
                                    "History length is shorter than cache prefix. Deleting obsolete context cache..."
                                )
                                caches_map.pop(model_key, None)
                                _sync_worker_active_fields(worker, next(iter(caches_map.values()), None))
                                try:
                                    await entry.llm.delete_cache(entry.cache_name)
                                except Exception as de:
                                    logger.warning(f"Failed to delete obsolete cache: {de}")
                            elif len(llm_history) > entry.cached_messages_len:
                                active_cache_name = entry.cache_name
                                cached_messages_len = entry.cached_messages_len

                # Partition history if using an active explicit context cache for this model
                logger.info(
                    f"Context caching debug: model={model_key}, active_cache_name={active_cache_name}, "
                    f"llm_history_len={len(llm_history)}, cached_messages_len={cached_messages_len}"
                )
                if active_cache_name and len(llm_history) > cached_messages_len:
                    generate_history = llm_history[cached_messages_len:]
                    generate_system_prompt = system_prompt
                    generate_tools = None
                    generate_cached_content = active_cache_name
                    used_round_explicit_cache = True
                else:
                    generate_history = llm_history
                    generate_system_prompt = system_prompt
                    generate_tools = tools_list
                    generate_cached_content = None
                    used_round_explicit_cache = False

                # LLM inference (optionally streaming live preview updates to Gateway)
                publisher = (
                    _StreamPreviewPublisher(
                        gateway=self.gateway,
                        chatbot_id=chatbot_id,
                        channel_id=channel_id,
                        session_id=self.session_id,
                        turn_id=current_msg.id,
                        min_interval=self._resolve_stream_min_interval(chatbot_id),
                    )
                    if stream
                    else None
                )
                active_publisher = publisher
                llm_call_t0 = time.perf_counter()
                first_stream_chunk_ms: float | None = None

                async def _on_stream_chunk_timed(delta: str, accumulated: str) -> None:
                    nonlocal first_stream_chunk_ms
                    if first_stream_chunk_ms is None and delta:
                        first_stream_chunk_ms = round((time.perf_counter() - llm_call_t0) * 1000.0, 2)
                    if active_publisher is not None:
                        await active_publisher.on_chunk(delta, accumulated)

                gen_kwargs: dict[str, Any] = (
                    {"stream": True, "on_stream_chunk": _on_stream_chunk_timed}
                    if publisher is not None
                    else {}
                )
                try:
                    res = await llm.generate(
                        system_prompt=generate_system_prompt,
                        history=generate_history,
                        tools=generate_tools,
                        cached_content=generate_cached_content,
                        **gen_kwargs,
                    )
                except Exception as e:
                    if generate_cached_content and _is_cache_invalid_error(e):
                        logger.warning(
                            f"LLM generation with CachedContent '{generate_cached_content}' failed ({e}). "
                            "Invalidating cache and retrying same round uncached."
                        )
                        if publisher is not None:
                            await publisher.discard()
                            publisher = _StreamPreviewPublisher(
                                gateway=self.gateway,
                                chatbot_id=chatbot_id,
                                channel_id=channel_id,
                                session_id=self.session_id,
                                turn_id=current_msg.id,
                                min_interval=self._resolve_stream_min_interval(chatbot_id),
                            )
                            active_publisher = publisher
                        async with _get_session_cache_lock(self.session_id):
                            caches_map = _ensure_worker_cache_map(
                                worker, default_llm=llm, default_ttl=ttl_cfg_secs
                            )
                            popped = caches_map.pop(model_key, None)
                            if (
                                popped is not None
                                or getattr(worker, "active_cache_name", None) == generate_cached_content
                            ):
                                _sync_worker_active_fields(worker, next(iter(caches_map.values()), None))
                        active_cache_name = None
                        cached_messages_len = 0
                        used_round_explicit_cache = False
                        generate_history = llm_history
                        generate_system_prompt = system_prompt
                        generate_tools = tools_list
                        generate_cached_content = None
                        llm_call_t0 = time.perf_counter()
                        first_stream_chunk_ms = None
                        retry_kwargs: dict[str, Any] = (
                            {"stream": True, "on_stream_chunk": _on_stream_chunk_timed}
                            if publisher is not None
                            else {}
                        )
                        res = await llm.generate(
                            system_prompt=generate_system_prompt,
                            history=generate_history,
                            tools=generate_tools,
                            cached_content=generate_cached_content,
                            **retry_kwargs,
                        )
                    else:
                        raise

                round_llm_ms = round((time.perf_counter() - llm_call_t0) * 1000.0, 2)
                round_ttft_ms = (
                    round(float(res.ttft_ms), 2)
                    if res.ttft_ms is not None
                    else (first_stream_chunk_ms if first_stream_chunk_ms is not None else round_llm_ms)
                )
                if used_round_explicit_cache:
                    self._turn_used_explicit_cache = True
                last_round_used_explicit_cache = used_round_explicit_cache
                last_round_generate_est_tokens = estimate_messages_tokens_local(generate_history)

                # Log the raw LLM turn using TurnLogger if enabled
                if cfg.agent.raw_llm_logs and self.turn_logger:
                    try:
                        self.turn_logger.log_llm_turn(
                            llm_provider=llm.__class__.__name__,
                            history=generate_history,
                            tools=tools_list,
                            response=res,
                            system_prompt=generate_system_prompt,
                            extra_metrics={
                                "memory_pre_turn_ms": self._memory_pre_turn_ms,
                                "pinned_chars": self._pinned_chars,
                                "recent_sessions_chars": self._recent_sessions_chars,
                                "recent_sessions_rendered": self._recent_sessions_rendered,
                            },
                        )
                    except Exception as le:
                        logger.error(f"Failed to log LLM turn: {le}", exc_info=True)

                # Accumulate token & per-round metrics
                last_context_tokens = res.prompt_tokens or 0
                last_cached_tokens = res.cached_tokens or 0
                if res.total_tokens:
                    turn_tokens += res.total_tokens

                round_record: dict[str, Any] = {
                    "round": len(self._turn_rounds) + 1,
                    "prompt_tokens": res.prompt_tokens or 0,
                    "cached_tokens": res.cached_tokens or 0,
                    "output_tokens": res.candidates_tokens or 0,
                    "llm_ms": round_llm_ms,
                    "ttft_ms": round_ttft_ms,
                    "tool_ms": 0.0,
                    "tool_count": len(res.tool_calls),
                    "tool_errors": 0,
                }
                self._turn_rounds.append(round_record)

                if res.tool_calls:
                    if publisher is not None:
                        await publisher.discard()
                        active_publisher = None
                    turn_tool_calls += len(res.tool_calls)
                    self._last_round_tool_errors = 0
                    tool_t0 = time.perf_counter()
                    should_continue = await self._execute_tool_calls(res, current_msg, worker, session_staging_dir)
                    round_record["tool_ms"] = round((time.perf_counter() - tool_t0) * 1000.0, 2)
                    round_record["tool_errors"] = self._last_round_tool_errors
                    if should_continue:
                        continue
                    else:
                        break
                else:
                    if publisher is not None:
                        await publisher.wait_inflight()
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
                        publisher=publisher,
                        worker=worker,
                        session_staging_dir=session_staging_dir,
                    )
                    if should_continue:
                        if publisher is not None:
                            await publisher.discard()
                            active_publisher = None
                        continue
                    else:
                        active_publisher = None
                        completed_final_response = True
                        break

            # Post-turn phase: finalize session digest, FTS5 index, and extraction triggers FIRST
            # (before `queue_empty()` so queued follow-up messages never starve memory cursor/FTS updates),
            # then run auto-compaction and explicit context cache lifecycle AFTER reply delivery.
            if (
                completed_final_response
                and not self._last_turn_prompt_blocked
                and last_raw_history is not None
                and last_llm is not None
            ):
                if self._turn_start_digest_task is not None and not self._turn_start_digest_task.done():
                    try:
                        await self._turn_start_digest_task
                    except Exception:
                        pass

                sync_db_post = getattr(self.context.db, "_sync", getattr(self.context.db, "sync_db", None))
                if isinstance(sync_db_post, DatabaseManager) and not is_excluded_from_memory(current_msg):
                    try:
                        post_role = await self._resolve_active_role(current_msg)
                        turn_msgs_for_fts: list[Message] = [current_msg]
                        if self._last_final_assistant_msg is not None:
                            turn_msgs_for_fts.append(self._last_final_assistant_msg)
                        await DigestUpdater.finalize_after_turn(
                            self.context.db,
                            self.session_id,
                            post_role,
                            self._turn_context_policy,
                            turn_msgs_for_fts,
                            current_user_msg=current_msg,
                        )
                        consolidator = getattr(self.context, "memory_consolidator", None)
                        if consolidator is not None:
                            await consolidator.notify_turn(
                                session_id=self.session_id,
                                role=post_role,
                                user_msg=current_msg,
                                assistant_text=(
                                    self._last_final_assistant_msg.content
                                    if self._last_final_assistant_msg is not None
                                    else ""
                                ),
                                clicked_option=self._turn_clicked_option,
                            )
                    except Exception as mem_post_err:
                        logger.warning(
                            f"Post-turn memory digest/consolidator hook failed for session {self.session_id}: "
                            f"{mem_post_err}"
                        )

                if not worker.queue_empty():
                    logger.info(
                        f"Skipping post-turn compaction and cache management for session {self.session_id} "
                        "due to queued message; deferring to next turn."
                    )
                    self._set_compaction_pending(worker, True)
                else:
                    compacted_post = False
                    try:
                        compacted_post = await self._run_auto_compaction(
                            history=last_raw_history,
                            system_prompt=last_system_prompt,
                            tools_list=last_tools_list,
                            llm=last_llm,
                            cfg=cfg,
                            staging_dir=session_staging_dir,
                        )
                    except Exception as ce:
                        logger.warning(
                            f"Post-turn history compaction failed for session {self.session_id}: {ce}. "
                            "Will retry pre-turn on next message."
                        )
                        self._set_compaction_pending(worker, True)

                    try:
                        await self._manage_post_turn_explicit_cache(
                            worker=worker,
                            llm=last_llm,
                            cfg=cfg,
                            current_msg=current_msg,
                            system_prompt=last_system_prompt,
                            tools_list=last_tools_list,
                            session_staging_dir=session_staging_dir,
                            compacted_post=compacted_post,
                            used_explicit_cache_last_round=last_round_used_explicit_cache,
                            last_context_tokens=last_context_tokens,
                            last_cached_tokens=last_cached_tokens,
                            last_generate_est_tokens=last_round_generate_est_tokens,
                        )
                    except Exception as cache_err:
                        logger.warning(
                            f"Post-turn explicit cache management failed for session {self.session_id}: {cache_err}"
                        )
        except asyncio.CancelledError:
            if active_publisher is not None:
                try:
                    await active_publisher.discard()
                except Exception:
                    pass
                active_publisher = None
            # Interrupted turn: save turn metrics to the initiating user message
            turn_metrics = await self._build_turn_metrics(
                last_context_tokens=last_context_tokens,
                last_cached_tokens=last_cached_tokens,
                turn_tool_calls=turn_tool_calls,
                turn_tokens=turn_tokens,
                start_time=start_time,
                status="interrupted",
                llm=last_llm,
            )
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
            if active_publisher is not None:
                await active_publisher.discard()
                active_publisher = None
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

    async def _manage_post_turn_explicit_cache(
        self,
        *,
        worker: Any,
        llm: BaseLLM,
        cfg: KesokuConfig,
        current_msg: Message,
        system_prompt: str | None,
        tools_list: list[Any],
        session_staging_dir: str | None,
        compacted_post: bool,
        used_explicit_cache_last_round: bool,
        last_context_tokens: int,
        last_cached_tokens: int,
        last_generate_est_tokens: int,
    ) -> None:
        """Execute post-turn explicit context cache creation, rolling refresh, or sliding TTL renewal.

        Serialized per-session via `_get_session_cache_lock` and uses local calibrated token estimation
        so `GeminiLLM.count_tokens()` RPCs never block the asyncio event loop.
        """
        async with _get_session_cache_lock(self.session_id):
            if compacted_post:
                logger.info("Post-turn history compaction occurred. Clearing obsolete explicit context caches...")
                await _clear_worker_explicit_caches(worker, fallback_llm=llm)

            if (
                not cfg.gemini.context_caching
                or cfg.gemini.context_caching_mode == "implicit"
                or not _llm_supports_explicit_cache(llm)
            ):
                return

            post_raw = await build_history(
                gateway=self.gateway,
                session_id=self.session_id,
                heal_orphans=False,
                active_only=True,
            )
            if not post_raw:
                return

            post_assembled = await self._assemble_scaffold_history(
                history=post_raw,
                cfg=cfg,
                current_msg=current_msg,
            )

            sentinel_msg = Message(
                id="__cache_sentinel__",
                session_id=self.session_id,
                chatbot_id=current_msg.chatbot_id,
                channel_id=current_msg.channel_id,
                sender="user",
                role=MessageRole.USER,
                type=MessageType.TEXT,
                content="<current_message>\n</current_message>",
                status=MessageStatus.PENDING_AGENT,
            )
            prepared_all = prepare_history_for_llm(
                [*post_assembled, sentinel_msg],
                staging_dir=session_staging_dir,
                strip_old_questions=cfg.agent.strip_historical_question_options,
            )
            prefix_messages = (
                prepared_all[:-1]
                if prepared_all and prepared_all[-1].id == "__cache_sentinel__"
                else prepared_all
            )
            if not prefix_messages:
                return

            prefix_est = estimate_prefix_tokens_local(prefix_messages, system_prompt, tools_list)
            ttl_secs = resolve_gemini_cache_ttl(cfg.gemini)
            configured_min = int(getattr(cfg.gemini, "context_caching_threshold", 4096) or 4096)
            min_tokens = EXPLICIT_CACHE_MIN_PREFIX_TOKENS if configured_min == 4096 else configured_min

            caches_map = _ensure_worker_cache_map(worker, default_llm=llm, default_ttl=ttl_secs)
            model_key = _cache_model_key(llm)
            model_label = str(getattr(llm, "model_name", None) or "default")
            entry = caches_map.get(model_key)

            if entry is not None:
                now_check = time.time()
                if len(prefix_messages) < entry.cached_messages_len or (
                    entry.expires_at > 0 and now_check >= entry.expires_at
                ):
                    if len(prefix_messages) < entry.cached_messages_len:
                        try:
                            await entry.llm.delete_cache(entry.cache_name)
                        except Exception as de:
                            logger.debug(f"Failed deleting shorter obsolete cache {entry.cache_name}: {de}")
                    caches_map.pop(model_key, None)
                    _sync_worker_active_fields(worker, next(iter(caches_map.values()), None))
                    entry = None

            if entry is None:
                if prefix_est >= min_tokens:
                    logger.info(
                        f"Post-turn creating explicit context cache for {model_key} "
                        f"(est_tokens={prefix_est} >= {min_tokens}, ttl={ttl_secs}s)..."
                    )
                    created_name = await llm.create_cache(
                        contents=prefix_messages,
                        system_prompt=system_prompt,
                        tools=tools_list,
                        display_name=f"kesoku_{self.session_id}",
                        ttl_seconds=ttl_secs,
                    )
                    if created_name:
                        now_ts = time.time()
                        new_entry = ExplicitCacheEntry(
                            cache_name=created_name,
                            model_name=model_label,
                            llm=llm,
                            cached_messages_len=len(prefix_messages),
                            cached_est_tokens=prefix_est,
                            created_at=now_ts,
                            expires_at=now_ts + ttl_secs,
                        )
                        caches_map[model_key] = new_entry
                        _sync_worker_active_fields(worker, new_entry)
                return

            # Existing entry: decide between Rolling Refresh vs Sliding TTL Renewal.
            cached_est = (
                last_cached_tokens
                if (used_explicit_cache_last_round and last_cached_tokens > 0)
                else max(entry.cached_est_tokens, 1)
            )
            refresh_threshold = max(
                EXPLICIT_CACHE_REFRESH_MIN_SUFFIX_TOKENS,
                int(EXPLICIT_CACHE_REFRESH_RATIO * cached_est),
            )
            suffix_msgs = prefix_messages[entry.cached_messages_len :]
            suffix_text_est = estimate_messages_tokens_local(suffix_msgs)
            if used_explicit_cache_last_round and last_context_tokens > last_cached_tokens > 0:
                usage_uncached = last_context_tokens - last_cached_tokens
                ephemeral_delta = max(0, last_generate_est_tokens - suffix_text_est)
                uncached_suffix = max(suffix_text_est, usage_uncached - ephemeral_delta)
            else:
                uncached_suffix = suffix_text_est

            if uncached_suffix >= refresh_threshold and prefix_est >= min_tokens:
                logger.info(
                    f"Post-turn rolling refresh for {model_key}: uncached_suffix={uncached_suffix} >= "
                    f"threshold={refresh_threshold} (cached_est={cached_est}). Creating new cache first..."
                )
                new_name = await llm.create_cache(
                    contents=prefix_messages,
                    system_prompt=system_prompt,
                    tools=tools_list,
                    display_name=f"kesoku_{self.session_id}",
                    ttl_seconds=ttl_secs,
                )
                if new_name:
                    old_name = entry.cache_name
                    old_llm = entry.llm
                    now_ts = time.time()
                    new_entry = ExplicitCacheEntry(
                        cache_name=new_name,
                        model_name=model_label,
                        llm=llm,
                        cached_messages_len=len(prefix_messages),
                        cached_est_tokens=prefix_est,
                        created_at=now_ts,
                        expires_at=now_ts + ttl_secs,
                    )
                    # Atomically swap state BEFORE deleting old cache
                    caches_map[model_key] = new_entry
                    _sync_worker_active_fields(worker, new_entry)
                    if old_name and old_name != new_name:
                        try:
                            await old_llm.delete_cache(old_name)
                        except Exception as de:
                            logger.warning(f"Failed deleting superseded explicit cache {old_name}: {de}")
                return

            remaining_ttl = entry.expires_at - time.time()
            if used_explicit_cache_last_round and remaining_ttl < EXPLICIT_CACHE_TTL_RENEW_BELOW_SECONDS:
                logger.info(
                    f"Post-turn renewing TTL for {entry.cache_name} "
                    f"(remaining={remaining_ttl:.1f}s < {EXPLICIT_CACHE_TTL_RENEW_BELOW_SECONDS}s)..."
                )
                renewed = await llm.update_cache_ttl(entry.cache_name, ttl_seconds=ttl_secs)
                if renewed:
                    entry.expires_at = time.time() + ttl_secs
                    _sync_worker_active_fields(worker, entry)
                else:
                    caches_map.pop(model_key, None)
                    _sync_worker_active_fields(worker, next(iter(caches_map.values()), None))
                    if prefix_est >= min_tokens:
                        recreated_name = await llm.create_cache(
                            contents=prefix_messages,
                            system_prompt=system_prompt,
                            tools=tools_list,
                            display_name=f"kesoku_{self.session_id}",
                            ttl_seconds=ttl_secs,
                        )
                        if recreated_name:
                            now_ts = time.time()
                            recreated_entry = ExplicitCacheEntry(
                                cache_name=recreated_name,
                                model_name=model_label,
                                llm=llm,
                                cached_messages_len=len(prefix_messages),
                                cached_est_tokens=prefix_est,
                                created_at=now_ts,
                                expires_at=now_ts + ttl_secs,
                            )
                            caches_map[model_key] = recreated_entry
                            _sync_worker_active_fields(worker, recreated_entry)

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
        raw_context_entries: list[ContextEntry] = []
        target_staging_dir = session_staging_dir
        if not target_staging_dir and self.context and self.context.config:
            sessions_dir = self.context.config.workspace.sessions_dir
            if not os.path.isabs(sessions_dir) and self.context.config.agent_working_dir:
                sessions_dir = os.path.join(self.context.config.agent_working_dir, sessions_dir)
            target_staging_dir = os.path.join(sessions_dir, self.session_id)

        if target_staging_dir:
            try:
                raw_context_entries = await collect_context_entries(target_staging_dir, include_policy_only=True)
                renderable_entries = [e for e in raw_context_entries if e.content]
                # Generic per-message filter (e.g. a cross-channel plain-chat fallback skipping some blocks).
                # Policy resolution below still sees ALL blocks.
                renderable_entries = filter_ignored_context_entries(
                    renderable_entries, (current_msg.metadata or {}).get("ignore_context_globs")
                )
                combined_entries = insert_extra_entries(
                    renderable_entries,
                    await self._get_state_snapshot_entries(target_staging_dir),
                )
                context_blocks_prefix = render_context_blocks(
                    apply_total_budget(combined_entries, self.context.config.agent.context_dir_max_chars)
                )
            except Exception as e:
                logger.warning(f"Failed to build context blocks for session '{self.session_id}': {e}")

        context_blocks_prefix += render_mode_prefs_block((current_msg.metadata or {}).get("mode_prefs"))

        cfg = self.context.config
        static_policy = resolve_context_policy(
            [e.name for e in raw_context_entries],
            {e.name: e.memory_policy for e in raw_context_entries},
            cfg,
        )
        self._turn_context_policy = await self._apply_memory_policy_override(current_msg, static_policy)

        # 3. Prepend Consolidated Passive Synchronization, Preferences, and Context Compression Guidelines
        # (if Bootstrap)
        full_prefix = ""
        if is_bootstrap:
            disabled_tools = set(cfg.agent.disabled_tools)
            hints = []
            if "memory_search" not in disabled_tools:
                hints.append(
                    "- Use `memory_search(query)` to search long-term Markdown memories in `$MEMORY_DIR` for this role."
                )
            if cfg.agent.session_search_enabled and "session_search" not in disabled_tools:
                hints.append(
                    "- Use `session_search(query)` / `session_view(session_id)` to look up details from sibling "
                    "sessions of this role."
                )
            lines = ['<background_context type="sync_guidelines">', *hints, "</background_context>"] if hints else []
            full_prefix = "\n".join(lines) + "\n" if lines else ""

        # 4. Build or reuse per-turn frozen memory snapshot (PINNED + <recent_sessions> + <memory>)
        raw_query = latest_user_msg.content or ""
        clicked_option = is_clicked_option(raw_query, last_assistant_text)
        self._turn_clicked_option = clicked_option

        if self._memory_snapshot is None:
            mem_t0 = time.perf_counter()
            pinned_block = ""
            recent_sessions_block = ""
            augmented_context_prefix = ""
            sync_db = getattr(self.context.db, "_sync", getattr(self.context.db, "sync_db", None))
            has_real_db = isinstance(sync_db, DatabaseManager)

            if cfg.agent.memory_pinned_enabled and active_role:
                try:
                    memories_dir = self._resolve_memories_dir(active_role)
                    consolidator = getattr(self.context, "memory_consolidator", None)

                    def _on_over_budget(rel_p: str) -> None:
                        if consolidator is not None:
                            from kesoku.agent.memory_consolidator import ConsolidateFile

                            consolidator.enqueue(ConsolidateFile(role=active_role, rel_path=rel_p))

                    pinned_block = await asyncio.to_thread(
                        lambda: default_pinned_cache.get_block(
                            active_role,
                            memories_dir=memories_dir,
                            max_chars=cfg.agent.memory_pinned_max_chars,
                            pinned_files=cfg.agent.memory_pinned_files,
                            db=sync_db if has_real_db else None,
                            on_over_budget=_on_over_budget,
                        )
                    )
                except Exception as pe:
                    logger.warning(f"Failed to load <memory_pinned> for role '{active_role}': {pe}")

            digest_rows: list[dict[str, Any]] = []
            if (
                cfg.agent.recent_sessions_enabled
                and active_role
                and has_real_db
                and not is_excluded_from_memory(latest_user_msg)
            ):
                try:
                    digest_rows = await asyncio.to_thread(
                        fetch_recent_digests,
                        sync_db,
                        active_role,
                        self.session_id,
                        cfg.agent.recent_sessions_window_days,
                        cfg.agent.recent_sessions_limit,
                        float(latest_user_msg.timestamp),
                    )
                    prev_human_ts = _find_prev_human_user_timestamp(history, latest_user_idx)
                    if should_render(is_bootstrap, prev_human_ts, digest_rows):
                        recent_sessions_block = format_recent_sessions_block(
                            active_role,
                            digest_rows,
                            max_chars=cfg.agent.recent_sessions_max_chars,
                        )
                except Exception as de:
                    logger.warning(f"Failed to build <recent_sessions> for role '{active_role}': {de}")

            if cfg.agent.auto_context_search and active_role:
                if clicked_option:
                    logger.info("Skipping ambient memory search: user message is a clicked option of the last question")
                elif len(raw_query.strip()) >= 2:
                    try:
                        extra_ctx = ""
                        if is_bootstrap and digest_rows:
                            cleaned_u = strip_injected_blocks(raw_query)
                            if len(cleaned_u) < 20 or _REFERENTIAL_QUERY_RE.search(cleaned_u):
                                extra_ctx = extract_bootstrap_extra_context(digest_rows, max_sessions=2, max_chars=600)
                        context_query = render_context_query(
                            history,
                            latest_user_msg,
                            max_turns=2,
                            max_chars=2000,
                            extra_context=extra_ctx,
                        )
                        augmented_context_prefix = await self._retrieve_auto_search_context(
                            role=active_role,
                            query_text=context_query or raw_query.strip(),
                            current_session_id=self.session_id,
                            semantic_min=cfg.agent.auto_context_search_semantic_min,
                            limit=cfg.agent.auto_context_search_limit,
                        )
                    except Exception as se:
                        logger.warning(f"Failed to retrieve ambient memory context: {se}")

            self._memory_pre_turn_ms = round((time.perf_counter() - mem_t0) * 1000.0, 2)
            self._pinned_chars = len(pinned_block)
            self._recent_sessions_chars = len(recent_sessions_block)
            self._recent_sessions_rendered = bool(recent_sessions_block)
            self._memory_snapshot = (pinned_block, recent_sessions_block, augmented_context_prefix)
        else:
            pinned_block, recent_sessions_block, augmented_context_prefix = self._memory_snapshot

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

        extra_clock_attrs = ""
        raw_clock_file = (cfg.agent.user_clock_file or "").strip() if cfg and cfg.agent else ""
        if raw_clock_file:
            clock_path = raw_clock_file
            if not os.path.isabs(clock_path) and cfg.agent_working_dir:
                clock_path = os.path.join(cfg.agent_working_dir, clock_path)
            uc = load_user_clock(clock_path, now_epoch=copied_msg.timestamp)
            if uc is not None:
                local_stamp = user_local_stamp(copied_msg.timestamp, uc)
                attr_parts = [
                    f'user_local_time="{_escape_xml_attr(local_stamp)}"',
                    f'user_timezone="{_escape_xml_attr(str(uc["tz"]))}"',
                ]
                if uc.get("place"):
                    attr_parts.append(f'user_place="{_escape_xml_attr(str(uc["place"]))}"')
                prev_user_ts = _find_prev_human_user_timestamp(history, msg_idx)
                if prev_user_ts is not None:
                    gap_str = humanize_gap(max(0.0, float(copied_msg.timestamp) - prev_user_ts))
                    attr_parts.append(f'since_prev_user_msg="{gap_str}"')
                extra_clock_attrs = " " + " ".join(attr_parts)

        copied_msg.content = (
            f"{instructions_prefix}"
            f"{pinned_block}"
            f"{recent_sessions_block}"
            f"{context_blocks_prefix}"
            f"{full_prefix}"
            f"{augmented_context_prefix}"
            f'<current_message from="{sender_name}" time="{time_str}" timezone="{tz_name}"{extra_clock_attrs}>\n'
            f"{copied_msg.content}\n"
            "</current_message>"
        )
        history[msg_idx] = copied_msg
        latest_user_msg = copied_msg
        logger.info(
            f"Wrapped active user message {copied_msg.id} in <current_message "
            f'from="{sender_name}" time="{time_str}" timezone="{tz_name}"{extra_clock_attrs}> '
            f"(bootstrap: {is_bootstrap})"
        )

        return latest_user_msg

    async def _apply_memory_policy_override(self, current_msg: Message, static_policy: str) -> str:
        """Apply the registered authoritative memory policy override (resolved once per turn).

        The provider answer replaces the static policy outright (no rank merge), so a user-configured
        per-mode policy wins over static frontmatter / glob defaults in both directions.
        """
        provider = getattr(self.context, "memory_policy_override_provider", None)
        if provider is None or not callable(provider):
            return static_policy
        if self._turn_policy_override_msg_id != current_msg.id:
            try:
                self._turn_policy_override = await asyncio.to_thread(provider, self.session_id)
            except Exception as e:
                logger.warning(f"Memory policy override provider failed for session {self.session_id}: {e}")
                self._turn_policy_override = None
            self._turn_policy_override_msg_id = current_msg.id
        cached = self._turn_policy_override
        return resolve_memory_policy(self.session_id, static_policy, (lambda _sid: cached) if cached else None)

    async def _resolve_active_role(self, current_msg: Message) -> str:
        """Resolve the persona role for this turn.

        The session's own `role_name` is authoritative when set to a non-default role (role switches swap
        sessions). Falls back to the channel role lookup (with parent-thread inheritance) otherwise, which
        covers thread sessions whose channel has no explicit `channel_roles` row.
        """
        bot_default = self.context.config.get_chatbot_default_role(current_msg.chatbot_id)
        try:
            session = await self.context.db.get_session(self.session_id)
            if session and session.role_name and session.role_name not in ("default", bot_default):
                return session.role_name
        except Exception as e:
            logger.warning(f"Failed to read session role for '{self.session_id}': {e}")
        return await self.context.db.get_channel_role_with_inheritance(
            current_msg.chatbot_id,
            current_msg.channel_id,
            self.session_id,
            default=bot_default,
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
        """Search relevant Markdown memories and episode digests (pre-turn `allow_sync=False`)."""
        _ = (threshold, window_size)
        sync_db = getattr(self.context.db, "_sync", getattr(self.context.db, "sync_db", self.context.db))
        memories_dir = self._resolve_memories_dir(role)

        def _search_both() -> tuple[list[Any], list[Any]]:
            mem_hits = search_role_memories(
                sync_db,
                role,
                query_text,
                limit=limit,
                min_score=semantic_min,
                memories_dir=memories_dir,
                include_pinned=False,
                allow_sync=False,
            )
            ep_hits: list[Any] = []
            if isinstance(sync_db, DatabaseManager):
                ep_hits = search_role_episodes(
                    sync_db,
                    role,
                    query_text,
                    exclude_session_id=current_session_id or self.session_id,
                    limit=2,
                    min_score=max(0.50, semantic_min),
                )
            return mem_hits, ep_hits

        matches, episode_hits = await asyncio.to_thread(_search_both)
        if not matches and not episode_hits:
            return ""

        logger.info(
            f"Ambient-injected {len(matches)} memory file(s) and {len(episode_hits)} episode(s) for role '{role}' "
            f"(query: '{query_text[:40]}...')"
        )
        return format_ambient_memory_block(matches, episode_hits=episode_hits)

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

    async def _run_auto_compaction(
        self,
        history: list[Message],
        system_prompt: str | None,
        tools_list: list[Any],
        llm: BaseLLM,
        cfg: KesokuConfig,
        staging_dir: str | None = None,
    ) -> bool:
        """Run automatic turn-based history compaction in-place when thresholds are exceeded."""
        if not history:
            return False

        compressor = HistoryCompressor(self.context.db)

        # Summarize with the dedicated context-compression model ([agent].context_llm /
        # [gemini|claude].context_model_name); fall back to the session LLM if it cannot be resolved.
        try:
            compression_llm = self.context.get_llm(use_context_compression=True)
        except Exception as e:
            logger.warning(f"Failed to resolve context compression LLM, using session LLM: {e}")
            compression_llm = llm

        return await compressor.auto_compact_session(
            session_id=self.session_id,
            history=history,
            llm=compression_llm,
            config=cfg,
            staging_dir=staging_dir,
            overhead_tokens=estimate_fixed_prompt_overhead(system_prompt, tools_list),
        )

    async def _assemble_scaffold_history(
        self,
        history: list[Message],
        cfg: KesokuConfig,
        current_msg: Message,
    ) -> list[Message]:
        """Assemble the in-memory prompt history using existing root summary nodes from SQLite.

        Returns a new list so downstream in-place context injection never mutates `raw_history`.
        """
        if not history:
            return []

        root_summaries = await self.context.db.get_root_summary_nodes(self.session_id)
        self._has_summaries = bool(root_summaries)

        if not root_summaries:
            return list(history)

        compressor = HistoryCompressor(self.context.db)
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
        assembled_history: list[Message] = []
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

        return assembled_history

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
        """Check context window usage, compact history if needed, and return assembled prompt history.

        Returns:
            A tuple of (active message history list, compacted_occurred bool).
        """
        if not history:
            return history, False

        compacted_occurred = await self._run_auto_compaction(
            history=history,
            system_prompt=system_prompt,
            tools_list=tools_list,
            llm=llm,
            cfg=cfg,
            staging_dir=staging_dir,
        )
        assembled_history = await self._assemble_scaffold_history(
            history=history,
            cfg=cfg,
            current_msg=current_msg,
        )
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
            for call, tc_msg in tool_call_msgs:
                cancel_meta: dict[str, Any] = {
                    "tool_name": call.name,
                    "tool_error": "cancelled",
                }
                if call.tool_call_id:
                    cancel_meta["tool_call_id"] = call.tool_call_id
                await self.gateway.post(
                    Message(
                        session_id=self.session_id,
                        chatbot_id=current_msg.chatbot_id,
                        channel_id=current_msg.channel_id,
                        sender=call.name,
                        role=MessageRole.TOOL,
                        type=MessageType.TOOL_RESULT,
                        content=f"Tool `{call.name}` was cancelled before it ran because the user sent a new message.",
                        status=MessageStatus.RESPONDED,
                        parent_id=tc_msg.id,
                        metadata=cancel_meta,
                    )
                )
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

        self._last_round_tool_errors = sum(
            1 for rm in result_msgs if getattr(rm, "metadata", None) and rm.metadata.get("tool_error")
        )

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

    async def _compact_after_consecutive_block(
        self,
        *,
        llm: BaseLLM,
        cfg: KesokuConfig,
        worker: Any = None,
        staging_dir: str | None = None,
    ) -> None:
        """Best-effort background compaction (`protect_tail_turns=0`) after a second consecutive PROMPT_BLOCKED."""
        try:
            bg_history = await build_history(self.gateway, self.session_id, heal_orphans=False)
            if not bg_history:
                return
            compressor = HistoryCompressor(self.context.db)
            try:
                compression_llm = self.context.get_llm(use_context_compression=True)
            except Exception as llm_err:
                logger.warning(f"Failed to resolve context compression LLM, using session LLM: {llm_err}")
                compression_llm = llm
            turns = compressor.segment_turns(bg_history)
            if not turns:
                return
            pf = 0 if len(turns) <= cfg.agent.protect_front_turns else cfg.agent.protect_front_turns
            nodes = await compressor.force_compact_session(
                self.session_id,
                bg_history,
                compression_llm,
                cfg,
                staging_dir=staging_dir,
                protect_front_turns=pf,
                protect_tail_turns=0,
            )
            if nodes > 0 and worker is not None:
                async with _get_session_cache_lock(self.session_id):
                    await _clear_worker_explicit_caches(worker, fallback_llm=llm)
        except Exception as bg_err:
            logger.warning(
                f"Background compaction after consecutive PROMPT_BLOCKED failed in session {self.session_id}: {bg_err}"
            )

    async def _handle_prompt_blocked_response(
        self,
        *,
        res: LLMResponse,
        current_msg: Message,
        last_context_tokens: int,
        last_cached_tokens: int,
        turn_tool_calls: int,
        turn_tokens: int,
        start_time: float,
        llm: BaseLLM,
        publisher: _StreamPreviewPublisher | None,
        worker: Any,
        session_staging_dir: str | None,
    ) -> None:
        """Handle an input-blocked prompt without retries or prompt pollution."""
        finish_str = res.finish_reason or "PROMPT_BLOCKED"
        reason = finish_str.split(":", 1)[1] if ":" in finish_str else "BLOCKED"
        self._last_turn_prompt_blocked = True
        logger.warning(
            f"Prompt blocked by LLM provider in session {self.session_id} "
            f"(finish_reason={finish_str}); skipping retries and marking turn rows prompt_blocked."
        )
        if publisher is not None:
            await publisher.discard()

        raw_msgs = await self.context.db.get_session_messages(self.session_id)
        last_assistant_idx = -1
        for idx, m in enumerate(raw_msgs):
            if (
                m.role == MessageRole.ASSISTANT
                and m.type == MessageType.TEXT
                and m.parent_id != current_msg.id
            ):
                last_assistant_idx = idx

        prior_msgs = raw_msgs[: last_assistant_idx + 1] if last_assistant_idx >= 0 else []
        prev_user = next((m for m in reversed(prior_msgs) if m.role == MessageRole.USER), None)
        prev_was_blocked = bool(prev_user and prev_user.metadata and prev_user.metadata.get("prompt_blocked"))

        turn_msgs_by_id: dict[str, Message] = {}
        for m in raw_msgs[last_assistant_idx + 1 :]:
            if m.role in (MessageRole.USER, MessageRole.TOOL, MessageRole.SYSTEM) or (
                m.role == MessageRole.ASSISTANT and m.sender != "Notification"
            ):
                turn_msgs_by_id[m.id] = m
        turn_msgs_by_id.setdefault(current_msg.id, current_msg)

        for tm in turn_msgs_by_id.values():
            updated_meta = dict(tm.metadata or {})
            updated_meta["prompt_blocked"] = reason
            await self.context.db.update_message_metadata(tm.id, updated_meta)

        blocked_metrics = await self._build_turn_metrics(
            last_context_tokens=last_context_tokens,
            last_cached_tokens=last_cached_tokens,
            turn_tool_calls=turn_tool_calls,
            turn_tokens=turn_tokens,
            start_time=start_time,
            status="prompt_blocked",
            llm=llm,
        )
        notice_content = (
            f"⚠️ 模型提供方拒绝了本次请求的输入内容（{reason}），这是对最近几轮上下文的判定，重试不会改变结果。"
            "请换个说法；若仍被拒绝，发送 `/compact`（会连同最近几轮一起压缩）或 `/clear` 开新会话。"
        )
        final_msg = Message(
            session_id=self.session_id,
            chatbot_id=current_msg.chatbot_id,
            channel_id=current_msg.channel_id,
            sender="Notification",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            status=MessageStatus.PENDING,
            parent_id=current_msg.id,
            content=notice_content,
            metadata={
                "delivery_notice": True,
                "prompt_blocked": reason,
                "turn_metrics": blocked_metrics,
            },
        )
        await self.gateway.post(final_msg)
        await self.context.db.update_message_status(current_msg.id, MessageStatus.PROCESSED)

        if prev_was_blocked:
            bg_task = asyncio.create_task(
                self._compact_after_consecutive_block(
                    llm=llm,
                    cfg=self.context.config,
                    worker=worker,
                    staging_dir=session_staging_dir,
                )
            )
            self._bg_compaction_tasks.add(bg_task)
            bg_task.add_done_callback(self._bg_compaction_tasks.discard)

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
        publisher: _StreamPreviewPublisher | None = None,
        worker: Any = None,
        session_staging_dir: str | None = None,
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

        if (res.finish_reason or "").startswith("PROMPT_BLOCKED"):
            await self._handle_prompt_blocked_response(
                res=res,
                current_msg=current_msg,
                last_context_tokens=last_context_tokens,
                last_cached_tokens=last_cached_tokens,
                turn_tool_calls=turn_tool_calls,
                turn_tokens=turn_tokens,
                start_time=start_time,
                llm=llm,
                publisher=publisher,
                worker=worker,
                session_staging_dir=session_staging_dir,
            )
            return False, empty_attempts

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
                            if publisher is not None:
                                await publisher.discard()
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
                    if publisher is not None:
                        await publisher.discard()
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
            thought_metadata: dict[str, Any] = {"final_response": True}
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
                if publisher is not None:
                    await publisher.discard()
                return True, empty_attempts
            if empty_attempts == EMPTY_RESPONSE_RETRIES + 1:
                logger.warning(
                    f"LLM still returned empty content in session {self.session_id} "
                    f"(finish_reason={res.finish_reason}, attempt {empty_attempts}). Nudging model."
                )
                if publisher is not None:
                    await publisher.discard()
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

        turn_metrics = await self._build_turn_metrics(
            last_context_tokens=last_context_tokens,
            last_cached_tokens=last_cached_tokens,
            turn_tool_calls=turn_tool_calls,
            turn_tokens=turn_tokens,
            start_time=start_time,
            status="finished",
            llm=llm,
        )
        final_meta: dict[str, Any] = {"turn_metrics": turn_metrics}
        cur_meta = getattr(current_msg, "metadata", None)
        if isinstance(cur_meta, dict):
            if cur_meta.get("discord_message_id"):
                final_meta["discord_message_id"] = cur_meta["discord_message_id"]
            if cur_meta.get("discord_message_channel_id"):
                final_meta["discord_message_channel_id"] = cur_meta["discord_message_channel_id"]

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
            metadata=final_meta,
        )
        self._last_final_assistant_msg = final_msg
        await self.gateway.post(final_msg)
        await self.context.db.update_message_status(current_msg.id, MessageStatus.PROCESSED)
        return False, empty_attempts  # should_continue=False, counter unmodified
