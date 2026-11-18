"""Headless session drivers: BenchChatbot, BenchSession, PlaytestChatbot, and PlaytestSession."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import tomllib
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import kesoku.agent.llm as kesoku_llm
import kesoku.config as kesoku_config
import kesoku.utils.embedding as kesoku_embedding
from kesoku.agent.agent import Agent, SessionWorker
from kesoku.agent.compressor import HistoryCompressor
from kesoku.agent.history import build_history
from kesoku.agent.llm import BaseLLM
from kesoku.agent.turn_executor import estimate_fixed_prompt_overhead
from kesoku.config import KesokuConfig, load_config
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager, Message, Session
from kesoku.gateway.chatbot.base import Chatbot, parse_message_content
from kesoku.gateway.chatbot.deferred_media import is_pid_alive, read_json_dict
from kesoku.gateway.gateway import Gateway
from kesoku.testing.fixture import render_config_toml, sanitized_env
from kesoku.testing.grading import extract_tool_events, message_to_dict
from kesoku.testing.metering import Caps, CostLedger, MeteredLLM

CHATBOT_ID = "bench"
DEFAULT_BENCH_ROLE = "eval_engineer"
TERMINAL_STATUSES = {MessageStatus.PROCESSED, MessageStatus.ERROR, MessageStatus.INTERRUPTED}
BUSY_STATUSES = {MessageStatus.PENDING_AGENT, MessageStatus.PROCESSING}
EMBEDDING_DIM = 384

LLMFactory = Callable[[KesokuConfig], BaseLLM]


class _AwaitableDict(dict[str, Any]):
    """Dictionary return value that works synchronously or with `await`."""

    def __await__(self) -> Generator[Any, None, dict[str, Any]]:
        async def _identity() -> dict[str, Any]:
            return dict(self)

        return _identity().__await__()


class _AwaitableList(list[dict[str, Any]]):
    """List return value that works synchronously or with `await`."""

    def __await__(self) -> Generator[Any, None, list[dict[str, Any]]]:
        async def _identity() -> list[dict[str, Any]]:
            return list(self)

        return _identity().__await__()


class BenchChatbot(Chatbot):
    """Headless adapter: records every outbound message and marks final texts delivered."""

    def __init__(self, chatbot_id: str, gateway: Gateway, session_id: str | None = None) -> None:
        """Initialize the adapter with an empty transcript."""
        super().__init__(chatbot_id, gateway, session_id=session_id)
        self.transcript: list[Message] = []

    async def handle_message(self, message: Message) -> None:
        """Record the message; acknowledge assistant texts so nothing stays PENDING."""
        self.transcript.append(message)
        if message.role == MessageRole.ASSISTANT and message.type == MessageType.TEXT:
            await self.gateway.db.update_message_status(message.id, MessageStatus.DELIVERED)

    async def restart_service(self) -> None:
        """Never restart anything from a test or benchmark run."""
        return None


class PlaytestChatbot(BenchChatbot):
    """Headless playtest adapter with optional two-phase deferred media pickup."""

    _deferred_poll_initial_sec: float = 0.04
    _deferred_poll_max_sec: float = 0.2
    _deferred_dead_grace_sec: float = 0.4
    _voice_inline_wait_sec: float = 0.0

    def __init__(
        self,
        chatbot_id: str,
        gateway: Gateway,
        session_id: str | None = None,
        *,
        supports_deferred_media: bool = False,
    ) -> None:
        """Initialize the playtest adapter and register deferred-media capability."""
        self.supports_deferred_media = supports_deferred_media
        self.delivered_files: list[str] = []
        self.delivered_voices: list[str] = []
        self.deferred_failures: list[tuple[str, str]] = []
        super().__init__(chatbot_id, gateway, session_id=session_id)

    async def send_file_segment(self, channel_id: str, file_path: str, message: Message) -> None:
        """Record a delivered file segment."""
        _ = (channel_id, message)
        self.delivered_files.append(file_path)

    async def send_voice_segment(self, channel_id: str, file_path: str, message: Message) -> None:
        """Record a delivered voice segment."""
        _ = (channel_id, message)
        self.delivered_voices.append(file_path)

    async def fail_deferred_attachment(
        self,
        handle: Any,
        reason: str,
        message: Message,
        segment: dict[str, Any],
    ) -> None:
        """Record a deferred attachment failure."""
        _ = (handle, message)
        self.deferred_failures.append((str(segment.get("path") or ""), reason))

    async def handle_message(self, message: Message) -> None:
        """Record outbound messages, track deferred `[file:]`/`[voice:]` attachments, and mark delivered."""
        self.transcript.append(message)
        if message.role != MessageRole.ASSISTANT or message.type != MessageType.TEXT:
            return
        if self.supports_deferred_media and message.content:
            for seg in parse_message_content(message.content):
                if seg.get("type") not in ("file", "voice"):
                    continue
                raw_path = str(seg.get("path") or "").strip()
                if not raw_path:
                    continue
                deferred_state = await self._resolve_deferred(raw_path, message.session_id)
                if deferred_state is not None:
                    state_kind, target_path, sidecar_data = deferred_state
                    if state_kind == "pending":
                        await self._start_deferred_tracking(seg, message, target_path, sidecar_data)
                    elif state_kind == "ready":
                        if seg["type"] == "voice":
                            self.delivered_voices.append(target_path)
                        else:
                            self.delivered_files.append(target_path)
                    elif state_kind == "failed":
                        err_reason = str((sidecar_data or {}).get("error") or "deferred_error")
                        self.deferred_failures.append((raw_path, err_reason))
                else:
                    try:
                        resolved = await self.resolve_outbound_path(raw_path, message.session_id)
                    except Exception:  # noqa: BLE001
                        resolved = raw_path
                    if resolved and await asyncio.to_thread(os.path.isfile, resolved):
                        if seg["type"] == "voice":
                            self.delivered_voices.append(resolved)
                        else:
                            self.delivered_files.append(resolved)
        await self.gateway.db.update_message_status(message.id, MessageStatus.DELIVERED)


@dataclass
class TurnOutcome:
    """Result of driving one message (user turn / auto-reply) to idle."""

    msg_id: str
    status: str  # processed | error | interrupted | timeout
    start_ts: float
    end_ts: float
    final_text: str = ""
    update_file_calls: int = 0


@dataclass
class _Patches:
    """Originals of process-global monkeypatches, restored on `stop()`."""

    saved: list[tuple[Any, str, Any]] = field(default_factory=list)

    def set(self, obj: Any, name: str, value: Any) -> None:
        """Patch `obj.name`, remembering the original."""
        self.saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def restore(self) -> None:
        """Undo all patches in reverse order."""
        while self.saved:
            obj, name, value = self.saved.pop()
            setattr(obj, name, value)


async def _noop_rename(self: SessionWorker, staging_dir: str | None = None) -> None:
    """Replacement for `SessionWorker._auto_rename_session` (avoids extra LLM calls in headless runs)."""
    return None


def _zero_embedding(text: str) -> list[float]:
    """Return zero vector stub embedding."""
    _ = text
    return [0.0] * EMBEDDING_DIM


def _zero_embeddings(texts: list[str]) -> list[list[float]]:
    """Return zero vector batch stub embeddings."""
    return [[0.0] * EMBEDDING_DIM for _ in texts]


class BenchSession:
    """Drives one kesoku session in-process: post turns, wait for idle, force compaction, run commands."""

    def __init__(
        self,
        task_id: str,
        config_path: Path,
        ledger: CostLedger,
        *,
        role: str = DEFAULT_BENCH_ROLE,
        llm_factory: LLMFactory | None = None,
        stub_embeddings: bool = True,
        llm_call_timeout_s: float = 300.0,
        idle_stable_s: float = 1.0,
        poll_interval_s: float = 0.2,
    ) -> None:
        """Configure the session (nothing starts until `start()`).

        Args:
            task_id: Task id (session title / channel id).
            config_path: Rendered `ws/config.toml`.
            ledger: Shared cost ledger.
            role: Role directory name under `roles_dir`.
            llm_factory: Builds the inner LLM from the loaded config (default: kesoku `get_llm('gemini')`).
            stub_embeddings: Replace fastembed with zero vectors.
            llm_call_timeout_s: Per-call timeout for the metered proxy.
            idle_stable_s: Idle condition must hold this long.
            poll_interval_s: Poll period for idle detection.
        """
        self.task_id = task_id
        self.config_path = Path(config_path)
        self.ledger = ledger
        self.role = role
        self.llm_factory = llm_factory
        self.stub_embeddings = stub_embeddings
        self.llm_call_timeout_s = llm_call_timeout_s
        self.idle_stable_s = idle_stable_s
        self.poll_interval_s = poll_interval_s
        self.channel_id = f"bench-{task_id}"
        self._patches = _Patches()
        self.cfg: KesokuConfig | None = None
        self.llm: MeteredLLM | None = None
        self.context: KesokuContext | None = None
        self.gateway: Gateway | None = None
        self.session: Session | None = None
        self.bot: BenchChatbot | None = None
        self.agent: Agent | None = None
        self._bot_task: asyncio.Task[None] | None = None
        self._agent_task: asyncio.Task[None] | None = None
        self.staging_dir: Path | None = None
        self.command_replies: list[dict[str, Any]] = []

    async def __aenter__(self) -> BenchSession:
        """Start the session when entering an async context manager."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Stop the session when leaving an async context manager."""
        _ = (exc_type, exc, tb)
        await self.stop()

    def _prepare_config(self) -> KesokuConfig:
        """Load and return the `KesokuConfig` for this session."""
        return load_config(str(self.config_path))

    def _build_chatbot(self, gateway: Gateway, session_id: str) -> BenchChatbot:
        """Instantiate the headless chatbot adapter for this session."""
        return BenchChatbot(CHATBOT_ID, gateway, session_id=session_id)

    def _default_provider(self, cfg: KesokuConfig) -> str:
        """Return the LLM provider name to instantiate when `llm_factory` is not supplied."""
        return cfg.agent.llm or "gemini"

    async def start(self) -> None:
        """Load config, patch globals, wire Gateway/Agent/Chatbot, and create the session."""
        self._patches.set(kesoku_config, "_global_config", kesoku_config._global_config)
        cfg = self._prepare_config()
        self.cfg = cfg
        DatabaseManager(cfg.workspace.db_path).init_tables()
        real_get_llm = kesoku_llm.get_llm
        provider = self._default_provider(cfg)

        def _build_inner() -> BaseLLM:
            return self.llm_factory(cfg) if self.llm_factory else real_get_llm(provider, cfg)

        self.llm = MeteredLLM(
            _build_inner(),
            self.ledger,
            call_timeout_s=self.llm_call_timeout_s,
            inner_factory=_build_inner,
        )
        metered = self.llm

        def _metered_get_llm(*args: Any, **kwargs: Any) -> BaseLLM:
            _ = (args, kwargs)
            return metered

        # `/compact` builds its own LLM via `kesoku.agent.llm.get_llm`: route it through the metered proxy.
        self._patches.set(kesoku_llm, "get_llm", _metered_get_llm)
        self._patches.set(SessionWorker, "_auto_rename_session", _noop_rename)
        if self.stub_embeddings:
            self._patches.set(kesoku_embedding, "get_embedding", _zero_embedding)
            self._patches.set(kesoku_embedding, "get_embeddings", _zero_embeddings)

        self.context = KesokuContext(config=cfg, llm=metered)
        self.gateway = Gateway(context=self.context)
        self.session = await self.gateway.create_session(
            title=self.task_id,
            chatbot_id=CHATBOT_ID,
            channel_id=self.channel_id,
            role=self.role,
        )
        self.staging_dir = (Path(cfg.workspace.sessions_dir) / self.session.workspace_name).resolve()
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.bot = self._build_chatbot(self.gateway, self.session.id)
        self._bot_task = asyncio.create_task(self.bot.start())
        self.agent = Agent(self.gateway, context=self.context)
        self._agent_task = asyncio.create_task(self.agent.start())
        await asyncio.sleep(0)

    async def stop(self) -> None:
        """Stop the agent, kill the session's background jobs, stop the adapter, and undo patches."""
        try:
            if self.agent:
                self.agent.stop()
            if self._agent_task:
                await asyncio.gather(self._agent_task, return_exceptions=True)
            if self.context and self.session:
                await self.context.active_jobs.stop_all_for_session(self.session.id)
            if self.bot:
                self.bot.stop()
            if self._bot_task:
                await asyncio.gather(self._bot_task, return_exceptions=True)
        finally:
            self._patches.restore()

    @property
    def session_id(self) -> str:
        """Return active session id."""
        assert self.session is not None
        return self.session.id

    def _require(self) -> tuple[Gateway, KesokuContext, KesokuConfig]:
        """Return started components or raise `RuntimeError`."""
        if not (self.gateway and self.context and self.cfg and self.session):
            raise RuntimeError("BenchSession not started")
        return self.gateway, self.context, self.cfg

    async def send_user_message(self, text: str, timeout_s: float) -> TurnOutcome:
        """Post a user message (`PENDING_AGENT`) and wait until the session is idle.

        Args:
            text: Message content.
            timeout_s: Turn timeout; on expiry the worker is stopped immediately.

        Returns:
            The `TurnOutcome`.
        """
        gateway, _, _ = self._require()
        start = time.time()
        msg = Message(
            session_id=self.session_id,
            chatbot_id=CHATBOT_ID,
            channel_id=self.channel_id,
            sender="User",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content=text,
            status=MessageStatus.PENDING_AGENT,
        )
        await gateway.post(msg)
        status = await self.wait_for_idle(msg.id, timeout_s)
        end = time.time()
        final_text, update_calls = await self._turn_summary(start)
        return TurnOutcome(
            msg_id=msg.id,
            status=status,
            start_ts=start,
            end_ts=end,
            final_text=final_text,
            update_file_calls=update_calls,
        )

    async def send_command(self, text: str) -> str:
        """Execute a slash command through the chatbot adapter (e.g. `/compact ...`) and return its reply."""
        self._require()
        assert self.bot is not None
        replies: list[str] = []

        async def _reply(message: str, *args: Any, **kwargs: Any) -> None:
            _ = (args, kwargs)
            replies.append(message)

        await self.bot.execute_command_from_text(text, reply_func=_reply, channel_id=self.channel_id)
        reply = "\n".join(replies)
        self.command_replies.append({"text": text, "reply": reply, "ts": time.time()})
        return reply

    async def overhead_tokens(self) -> int:
        """Estimate the fixed prompt overhead (system prompt + tool schemas) like `TurnExecutor` does."""
        gateway, context, _ = self._require()
        session = await gateway.db.get_session(self.session_id)
        return estimate_fixed_prompt_overhead(
            session.system_prompt if session else "",
            context.tool_registry.get_tools_list(),
        )

    async def force_compaction(self, overhead_tokens: int | None = None, protect_tail_turns: int = 1) -> int:
        """Run deterministic compaction between turns through the production watermark path."""
        gateway, context, cfg = self._require()
        if overhead_tokens is None:
            overhead_tokens = await self.overhead_tokens()
        cfg_tmp = cfg.model_copy(deep=True)
        cfg_tmp.agent.context_high_watermark = 1
        cfg_tmp.agent.context_low_watermark = 0
        cfg_tmp.agent.protect_tail_min_tokens = 0
        cfg_tmp.agent.watermark_chunk_tokens = 1
        cfg_tmp.agent.protect_tail_turns = protect_tail_turns
        history = await build_history(gateway, self.session_id, heal_orphans=False)
        before = await self.summary_node_count()
        await HistoryCompressor(gateway.db).auto_compact_session(
            session_id=self.session_id,
            history=history,
            llm=context.get_llm(use_context_compression=True),
            config=cfg_tmp,
            staging_dir=str(self.staging_dir),
            overhead_tokens=overhead_tokens,
        )
        return await self.summary_node_count() - before

    async def summary_node_count(self) -> int:
        """Return number of summary nodes (all levels) of the session."""
        gateway, _, _ = self._require()
        return len(await gateway.db.get_all_summary_nodes(self.session_id))

    async def root_summary_node_count(self) -> int:
        """Return number of root summary nodes of the session."""
        gateway, _, _ = self._require()
        return len(await gateway.db.get_root_summary_nodes(self.session_id))

    def _worker_busy(self) -> bool:
        """Return True if the session worker is processing or has queued messages."""
        if not self.agent:
            return False
        worker = self.agent.workers.get(self.session_id)
        if worker is None or not worker.running:
            return False
        return bool(worker._processing_turn) or not worker.queue_empty()

    async def _session_busy(self) -> bool:
        """Return True if any session message is waiting for or being processed by the agent."""
        gateway, _, _ = self._require()
        msgs = await gateway.db.get_messages_by_filters({"session_id": self.session_id})
        return any(m.status in BUSY_STATUSES for m in msgs)

    async def wait_for_idle(self, msg_id: str, timeout_s: float) -> str:
        """Wait until the initiating message is terminal and the session is idle for `idle_stable_s`."""
        gateway, _, _ = self._require()
        deadline = time.monotonic() + timeout_s
        idle_since: float | None = None
        status = "timeout"
        while time.monotonic() < deadline:
            msg = await gateway.db.get_message(msg_id)
            terminal = msg is not None and msg.status in TERMINAL_STATUSES
            idle = terminal and not self._worker_busy() and not await self._session_busy()
            if idle:
                idle_since = idle_since or time.monotonic()
                if time.monotonic() - idle_since >= self.idle_stable_s:
                    return str(msg.status) if msg else "error"
            else:
                idle_since = None
            await asyncio.sleep(self.poll_interval_s)
        if self.agent:
            await self.agent.stop_session_worker(self.session_id, immediate=True)
        msgs = await gateway.db.get_messages_by_filters({"session_id": self.session_id})
        for m in msgs:
            if m.status in BUSY_STATUSES:
                await gateway.db.update_message_status(m.id, MessageStatus.INTERRUPTED)
        return status

    async def settle(self, max_wait_s: float, timeout_s: float) -> bool:
        """Wait up to `max_wait_s` for running background jobs to finish and wake the agent."""
        gateway, _, _ = self._require()
        if not self.active_jobs_for_session():
            return False
        woke = False
        deadline = time.monotonic() + max_wait_s
        while time.monotonic() < deadline:
            msgs = await gateway.db.get_messages_by_filters({"session_id": self.session_id})
            busy = [m for m in msgs if m.status in BUSY_STATUSES]
            if busy:
                woke = True
                await self.wait_for_idle(busy[-1].id, timeout_s)
            elif not self.active_jobs_for_session():
                break
            await asyncio.sleep(self.poll_interval_s)
        return woke

    def active_jobs_for_session(self) -> list[dict[str, Any]]:
        """Return snapshot of the session's jobs in `ActiveJobsRegistry`: `job_id, pid, pgid, alive`."""
        if not (self.context and self.session):
            return []
        jobs = []
        for job_id, job in list(self.context.active_jobs._jobs.items()):
            if job.get("session_id") != self.session.id:
                continue
            proc = job.get("process")
            pid = getattr(proc, "pid", None)
            pgid = None
            if pid:
                try:
                    pgid = os.getpgid(pid)
                except OSError:
                    pgid = None
            jobs.append(
                {
                    "job_id": job_id,
                    "pid": pid,
                    "pgid": pgid,
                    "alive": getattr(proc, "returncode", 0) is None,
                    "log_filepath": job.get("log_filepath"),
                    "creation_time": job.get("creation_time"),
                }
            )
        return jobs

    async def _turn_summary(self, since_ts: float) -> tuple[str, int]:
        """Return final assistant text and number of `update_file` calls since `since_ts`."""
        gateway, _, _ = self._require()
        msgs = await gateway.db.get_messages_by_filters({"session_id": self.session_id})
        final_text = ""
        update_calls = 0
        for m in msgs:
            if m.timestamp < since_ts:
                continue
            if m.role == MessageRole.ASSISTANT and m.type == MessageType.TEXT:
                final_text = m.content
            if m.type == MessageType.TOOL_CALL and (m.metadata or {}).get("tool_name") == "update_file":
                update_calls += 1
        return final_text, update_calls

    async def all_messages(self) -> list[Message]:
        """Return all session messages in timestamp order."""
        gateway, _, _ = self._require()
        return await gateway.db.get_messages_by_filters({"session_id": self.session_id})

    async def all_summary_nodes(self) -> list[dict[str, Any]]:
        """Return all summary nodes of the session as plain dicts."""
        gateway, _, _ = self._require()
        nodes = await gateway.db.get_all_summary_nodes(self.session_id)
        return [n.model_dump() for n in nodes]

    def llm_logs(self) -> list[Path]:
        """Return `llm-turn-*.log.yaml` files of the session in index order."""
        if not self.staging_dir or not self.staging_dir.is_dir():
            return []

        def _idx(p: Path) -> int:
            try:
                return int(p.name.split("-")[2].split(".")[0])
            except (IndexError, ValueError):
                return 0

        return sorted(self.staging_dir.glob("llm-turn-*.log.yaml"), key=_idx)


def _read_optional_json_dict(candidates: list[Path]) -> dict[str, Any]:
    """Read the first existing JSON object file from `candidates`, or return an empty dict."""
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return {}


def _read_media_log(staging_dir: Path) -> list[dict[str, Any]]:
    """Read JSONL media records from `$STAGING_DIR/.playtest_media.jsonl` if present."""
    log_path = staging_dir / ".playtest_media.jsonl"
    if not log_path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except ValueError:
            continue
        if isinstance(item, dict):
            entries.append(item)
    return entries


class PlaytestSession(BenchSession):
    """Persona-free headless playtest session with isolated shell env, `run_setup_gm`, and `snapshot()`."""

    def __init__(
        self,
        workspace: Path | str,
        role: str = "default",
        model: str = "gemini-3.8-flash",
        thinking_level: str = "low",
        env: dict[str, str] | None = None,
        memory_extraction_enabled: bool = False,
        auto_context_search: bool = False,
        *,
        config_path: Path | str | None = None,
        task_id: str = "playtest",
        ledger: CostLedger | None = None,
        llm_factory: LLMFactory | None = None,
        stub_embeddings: bool = True,
        llm_call_timeout_s: float = 180.0,
        idle_stable_s: float = 0.25,
        poll_interval_s: float = 0.1,
        setup_gm: list[list[str]] | None = None,
    ) -> None:
        """Configure a headless playtest session rooted at `workspace`.

        Args:
            workspace: Isolated workspace directory (AWD).
            role: Role name under `<workspace>/roles/` (defaults to `"default"`).
            model: Model identifier (`gemini-*` or `claude-*`).
            thinking_level: Reasoning/thinking effort level.
            env: Per-session subprocess environment variables injected into `cfg.shell.env` without
                mutating process-global `os.environ`.
            memory_extraction_enabled: Enable or disable background cross-session memory extraction.
            auto_context_search: Enable or disable ambient `<memory>` search injection.
            config_path: Explicit `config.toml` path (defaults to `<workspace>/config.toml`).
            task_id: Session title and channel suffix.
            ledger: Optional `CostLedger` (defaults to a fresh ledger with generous playtest caps).
            llm_factory: Optional factory overriding real LLM construction (e.g. `MockLLM` in unit tests).
            stub_embeddings: Replace embedding calls with zero vectors.
            llm_call_timeout_s: Per-LLM-call timeout in seconds.
            idle_stable_s: Seconds the session must stay idle before ending a turn.
            poll_interval_s: Poll period while waiting for turn completion.
            setup_gm: Optional initial `gm.py` command argv list executed automatically at the end of `start()`.
        """
        self.workspace = Path(workspace).resolve()
        resolved_cfg_path = Path(config_path).resolve() if config_path is not None else (self.workspace / "config.toml")
        effective_ledger = ledger or CostLedger(caps=Caps(max_tool_iters=25, max_task_cost_usd=5.0))
        super().__init__(
            task_id=task_id,
            config_path=resolved_cfg_path,
            ledger=effective_ledger,
            role=role,
            llm_factory=llm_factory,
            stub_embeddings=stub_embeddings,
            llm_call_timeout_s=llm_call_timeout_s,
            idle_stable_s=idle_stable_s,
            poll_interval_s=poll_interval_s,
        )
        self.model = model
        self.thinking_level = thinking_level
        self.env: dict[str, str] = {str(k): str(v) for k, v in (env or {}).items()}
        self.memory_extraction_enabled = memory_extraction_enabled
        self.auto_context_search = auto_context_search
        self._initial_setup_gm = list(setup_gm) if setup_gm else []
        self.setup_gm_outputs: list[dict[str, Any]] = []
        self._turn_counter = 0
        self._turn_windows: list[tuple[int, float]] = []
        self._turn_outcomes: dict[int, TurnOutcome] = {}

    def _prepare_config(self) -> KesokuConfig:
        """Ensure workspace layout + `config.toml`, then apply model, memory, and `shell.env` overrides."""
        self.workspace.mkdir(parents=True, exist_ok=True)
        for sub in ("sessions", "roles", "skills"):
            target = self.workspace / sub
            if not target.exists() and not target.is_symlink():
                target.mkdir(parents=True, exist_ok=True)
        if not self.config_path.is_file():
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(
                render_config_toml("default", model_name=self.model, thinking_level=self.thinking_level),
                encoding="utf-8",
            )

        # Load TOML without mutating process-global `os.environ` across parallel workers.
        with open(self.config_path, "rb") as fh:
            raw_data = tomllib.load(fh)
        file_env = raw_data.pop("env", None) or {}
        cfg = KesokuConfig.model_validate(raw_data)
        cfg.resolve_paths(str(self.config_path))
        cfg.agent_working_dir = str(self.workspace)
        cfg.agent.memory_extraction_enabled = self.memory_extraction_enabled
        cfg.agent.auto_context_search = self.auto_context_search

        if self.model.startswith("claude"):
            cfg.agent.llm = "claude"
            cfg.claude.model_name = self.model
            cfg.claude.context_model_name = self.model
            claude_tl = "low" if self.thinking_level == "minimal" else self.thinking_level
            if claude_tl in ("low", "medium", "high", "xhigh", "max"):
                cfg.claude.thinking_level = claude_tl  # type: ignore[assignment]
        else:
            cfg.agent.llm = "gemini"
            cfg.gemini.model_name = self.model
            cfg.gemini.context_model_name = self.model
            if self.thinking_level in ("minimal", "low", "medium", "high", "xhigh"):
                cfg.gemini.thinking_level = self.thinking_level  # type: ignore[assignment]

        # Inject per-session env into cfg.shell.env without mutating global os.environ.
        merged_env: dict[str, str] = {}
        for k, v in file_env.items():
            merged_env[str(k)] = str(v).lower() if isinstance(v, bool) else str(v)
        for k, v in (cfg.shell.env or {}).items():
            merged_env[str(k)] = str(v)
        merged_env.update(self.env)
        cfg.shell.env = merged_env
        kesoku_config._global_config = cfg
        return cfg

    def _build_chatbot(self, gateway: Gateway, session_id: str) -> BenchChatbot:
        """Build `PlaytestChatbot` with deferred media enabled when `KESOKU_DEFERRED_MEDIA=1`."""
        supports_deferred = bool(self.cfg and self.cfg.shell.env.get("KESOKU_DEFERRED_MEDIA") == "1")
        return PlaytestChatbot(
            CHATBOT_ID,
            gateway,
            session_id=session_id,
            supports_deferred_media=supports_deferred,
        )

    async def start(self) -> None:
        """Start the playtest session and run any initial `setup_gm` commands."""
        await super().start()
        if self._initial_setup_gm:
            self.setup_gm_outputs = list(self.run_setup_gm(self._initial_setup_gm))

    def _deferred_busy(self) -> bool:
        """Return True while deferred-media watchers or live detached workers are still finishing."""
        if isinstance(self.bot, PlaytestChatbot):
            for task_set in list(self.bot._deferred_tasks.values()):
                if any(not t.done() for t in list(task_set)):
                    return True
        if not self.staging_dir or not self.staging_dir.is_dir():
            return False
        now = time.time()
        for pending in self.staging_dir.rglob("*.pending.json"):
            marker = read_json_dict(str(pending))
            if not isinstance(marker, dict):
                continue
            try:
                created_at = float(marker.get("created_at") or 0.0)
            except (TypeError, ValueError):
                created_at = 0.0
            try:
                pid = int(marker.get("pid") or 0)
            except (TypeError, ValueError):
                pid = 0
            # Wait briefly for a just-spawned Phase-2 worker (pid=0 before backfill, or live pid <= 5s).
            if pid > 0 and is_pid_alive(pid) and (created_at <= 0.0 or now - created_at < 5.0):
                return True
            if pid <= 0 and created_at > 0.0 and (now - created_at) < 1.5:
                return True
        return False

    def _worker_busy(self) -> bool:
        """Return True if the session worker or deferred media workers are still busy."""
        return super()._worker_busy() or self._deferred_busy()

    async def send_user_message(self, text: str, timeout_s: float = 90.0) -> TurnOutcome:
        """Post a user message (`PENDING_AGENT`), advance the playtest turn counter, and wait for idle."""
        self._require()
        self._turn_counter += 1
        turn_idx = self._turn_counter
        self.ledger.begin_turn(turn_idx)
        self._turn_windows.append((turn_idx, time.time()))
        outcome = await super().send_user_message(text, timeout_s)
        self._turn_outcomes[turn_idx] = outcome
        return outcome

    def run_setup_gm(self, commands: list[list[str]]) -> list[dict[str, Any]]:
        """Execute `uv run skills/gm/gm.py <argv>` commands directly in the session staging environment.

        Runs with `USER_TURN=0`, `STAGING_DIR=<session_staging_dir>`, and `AWD=<workspace>` merged on top
        of a sanitized environment plus `cfg.shell.env`, without invoking the LLM. Can be called
        synchronously (`out = session.run_setup_gm(...)`) or awaited (`out = await session.run_setup_gm(...)`).

        Args:
            commands: List of CLI argument vectors passed to `skills/gm/gm.py` (e.g. `[["game", "start", "trpg"]]`).

        Returns:
            Parsed JSON stdout dictionaries for each executed command.

        Raises:
            RuntimeError: If the session has not started, a command exits non-zero, or stdout is not JSON.
        """
        _, _, cfg = self._require()
        assert self.staging_dir is not None

        gm_env = sanitized_env()
        gm_env.update(cfg.shell.env)
        gm_env["AWD"] = str(self.workspace)
        gm_env["STAGING_DIR"] = str(self.staging_dir)
        gm_env["USER_TURN"] = "0"

        outputs: list[dict[str, Any]] = []
        for raw_argv in commands:
            argv = [str(tok) for tok in raw_argv]
            if argv and argv[0].endswith("gm.py"):
                argv = argv[1:]
            cmd = ["uv", "run", "skills/gm/gm.py", *argv]
            proc = subprocess.run(
                cmd,
                cwd=self.workspace,
                env=gm_env,
                capture_output=True,
                text=True,
                timeout=60.0,
                check=False,
            )
            if proc.returncode != 0:
                detail = (proc.stdout or proc.stderr or "").strip()[:1000]
                raise RuntimeError(f"setup_gm command {argv!r} failed with exit code {proc.returncode}: {detail}")
            stdout = (proc.stdout or "").strip()
            if not stdout:
                outputs.append({})
                continue
            try:
                parsed = json.loads(stdout)
            except ValueError as exc:
                raise RuntimeError(f"setup_gm command {argv!r} emitted non-JSON stdout: {stdout[:500]}") from exc
            if not isinstance(parsed, dict):
                parsed = {"result": parsed}
            outputs.append(parsed)
        return _AwaitableList(outputs)

    def snapshot(self, turn: int | None = None) -> dict[str, Any]:
        """Capture session state, tool events, staging files, game JSONs, media log, and prompt overhead.

        When `turn` is None, `turn` and `final_text` reflect the latest completed user turn while
        `tool_events` and `media_log` contain all session records (each stamped with `turn_idx` / `turn`).
        When `turn` is an explicit integer, `final_text`, `tool_events`, and `media_log` are scoped to
        that turn. Can be called synchronously (`session.snapshot()`) or awaited (`await session.snapshot()`).

        Args:
            turn: Optional 1-based user turn number to filter `final_text`, `tool_events`, and `media_log`.

        Returns:
            Dictionary with keys `{turn, final_text, tool_events, staging_dir, staging_files,
            scene_json, party_json, ledger_json, image_pending, media_log, system_prompt_tokens}`.
        """
        _, context, _ = self._require()
        assert self.staging_dir is not None

        target_turn = self._turn_counter if turn is None else turn
        raw_msgs = context.sync_db.get_messages_by_filters({"session_id": self.session_id})
        msg_dicts = [message_to_dict(m, self._turn_windows) for m in raw_msgs]
        all_tool_events = extract_tool_events(msg_dicts)
        all_media_log = _read_media_log(self.staging_dir)

        if target_turn in self._turn_outcomes:
            final_text = self._turn_outcomes[target_turn].final_text
        else:
            final_text = ""
            for m in msg_dicts:
                if (
                    m["turn_idx"] == target_turn
                    and m["role"] == str(MessageRole.ASSISTANT)
                    and m["type"] == str(MessageType.TEXT)
                ):
                    final_text = str(m["content"] or "")

        if turn is not None:
            tool_events = [ev for ev in all_tool_events if ev["turn_idx"] == turn]
            media_log = [entry for entry in all_media_log if entry.get("turn") == turn]
        else:
            tool_events = all_tool_events
            media_log = all_media_log

        staging_files = sorted(
            p.relative_to(self.staging_dir).as_posix()
            for p in self.staging_dir.rglob("*")
            if p.is_file()
        )
        scene_json = _read_optional_json_dict(
            [self.staging_dir / "game" / "scene.json", self.staging_dir / "scene.json"]
        )
        party_json = _read_optional_json_dict(
            [self.staging_dir / "game" / "party.json", self.staging_dir / "party.json"]
        )
        ledger_json = _read_optional_json_dict(
            [self.staging_dir / "game" / "ledger.json", self.staging_dir / "ledger.json"]
        )
        image_pending = bool(scene_json.get("image_pending", False))

        sess = context.sync_db.get_session(self.session_id)
        system_prompt_tokens = estimate_fixed_prompt_overhead(
            sess.system_prompt if sess else "",
            context.tool_registry.get_tools_list(),
        )

        return _AwaitableDict(
            {
                "turn": target_turn,
                "final_text": final_text,
                "tool_events": tool_events,
                "staging_dir": self.staging_dir,
                "staging_files": staging_files,
                "scene_json": scene_json,
                "party_json": party_json,
                "ledger_json": ledger_json,
                "image_pending": image_pending,
                "media_log": media_log,
                "system_prompt_tokens": system_prompt_tokens,
            }
        )
