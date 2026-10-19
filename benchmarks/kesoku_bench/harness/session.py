"""Headless session driver: BenchChatbot adapter and BenchSession (Gateway + Agent wiring)."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import kesoku.agent.llm as kesoku_llm
import kesoku.utils.embedding as kesoku_embedding
from benchmarks.kesoku_bench.harness.fixture import ROLE_NAME
from benchmarks.kesoku_bench.harness.metering import CostLedger, MeteredLLM
from kesoku.agent.agent import Agent, SessionWorker
from kesoku.agent.compressor import HistoryCompressor
from kesoku.agent.history import build_history
from kesoku.agent.llm import BaseLLM
from kesoku.agent.turn_executor import estimate_fixed_prompt_overhead
from kesoku.config import KesokuConfig, load_config
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager, Message, Session
from kesoku.gateway.chatbot.base import Chatbot
from kesoku.gateway.gateway import Gateway

CHATBOT_ID = "bench"
TERMINAL_STATUSES = {MessageStatus.PROCESSED, MessageStatus.ERROR, MessageStatus.INTERRUPTED}
BUSY_STATUSES = {MessageStatus.PENDING_AGENT, MessageStatus.PROCESSING}
EMBEDDING_DIM = 384

LLMFactory = Callable[[KesokuConfig], BaseLLM]


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
        """Never restart anything from a benchmark run."""
        return None


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
    """Originals of process-global monkeypatches, restored on stop()."""

    saved: list[tuple[Any, str, Any]] = field(default_factory=list)

    def set(self, obj: Any, name: str, value: Any) -> None:
        """Patch obj.name, remembering the original."""
        self.saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def restore(self) -> None:
        """Undo all patches in reverse order."""
        while self.saved:
            obj, name, value = self.saved.pop()
            setattr(obj, name, value)


async def _noop_rename(self: SessionWorker) -> None:
    """Replacement for SessionWorker._auto_rename_session (one extra LLM call per session)."""
    return None


def _zero_embedding(text: str) -> list[float]:
    """Stub embedding (F10)."""
    return [0.0] * EMBEDDING_DIM


def _zero_embeddings(texts: list[str]) -> list[list[float]]:
    """Stub batch embedding (F10)."""
    return [[0.0] * EMBEDDING_DIM for _ in texts]


class BenchSession:
    """Drives one kesoku session in-process: post turns, wait for idle, force compaction, run commands."""

    def __init__(
        self,
        task_id: str,
        config_path: Path,
        ledger: CostLedger,
        *,
        llm_factory: LLMFactory | None = None,
        stub_embeddings: bool = True,
        llm_call_timeout_s: float = 300.0,
        idle_stable_s: float = 1.0,
        poll_interval_s: float = 0.2,
    ) -> None:
        """Configure the session (nothing starts until start()).

        Args:
            task_id: Task id (session title / channel id).
            config_path: Rendered ws/config.toml.
            ledger: Shared cost ledger.
            llm_factory: Builds the inner LLM from the loaded config (default: kesoku get_llm('gemini')).
            stub_embeddings: Replace fastembed with zero vectors.
            llm_call_timeout_s: Per-call timeout for the metered proxy.
            idle_stable_s: Idle condition must hold this long.
            poll_interval_s: Poll period for idle detection.
        """
        self.task_id = task_id
        self.config_path = Path(config_path)
        self.ledger = ledger
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

    # --- lifecycle -------------------------------------------------------------------------------
    async def start(self) -> None:
        """Load config (global AWD = ws), patch globals, wire Gateway/Agent/BenchChatbot, create the session."""
        cfg = load_config(str(self.config_path))
        self.cfg = cfg
        DatabaseManager(cfg.workspace.db_path).init_tables()
        real_get_llm = kesoku_llm.get_llm  # captured before the patch below

        def _build_inner() -> BaseLLM:
            return self.llm_factory(cfg) if self.llm_factory else real_get_llm("gemini", cfg)

        self.llm = MeteredLLM(_build_inner(), self.ledger, call_timeout_s=self.llm_call_timeout_s,
                              inner_factory=_build_inner)
        metered = self.llm

        def _metered_get_llm(*args: Any, **kwargs: Any) -> BaseLLM:
            return metered

        # /compact builds its own LLM via kesoku.agent.llm.get_llm (F4): route it through the proxy too.
        self._patches.set(kesoku_llm, "get_llm", _metered_get_llm)
        self._patches.set(SessionWorker, "_auto_rename_session", _noop_rename)
        if self.stub_embeddings:
            self._patches.set(kesoku_embedding, "get_embedding", _zero_embedding)
            self._patches.set(kesoku_embedding, "get_embeddings", _zero_embeddings)

        self.context = KesokuContext(config=cfg, llm=metered)
        self.gateway = Gateway(context=self.context)
        self.session = await self.gateway.create_session(
            title=self.task_id, chatbot_id=CHATBOT_ID, channel_id=self.channel_id, role=ROLE_NAME,
        )
        self.staging_dir = (Path(cfg.workspace.sessions_dir) / self.session.workspace_name).resolve()
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.bot = BenchChatbot(CHATBOT_ID, self.gateway, session_id=self.session.id)
        self._bot_task = asyncio.create_task(self.bot.start())
        self.agent = Agent(self.gateway, context=self.context)
        self._agent_task = asyncio.create_task(self.agent.start())
        await asyncio.sleep(0)  # let the listeners subscribe

    async def stop(self) -> None:
        """Stop the agent, kill the session's background jobs, stop the adapter, undo patches."""
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

    # --- properties ------------------------------------------------------------------------------
    @property
    def session_id(self) -> str:
        """Active session id."""
        assert self.session is not None
        return self.session.id

    def _require(self) -> tuple[Gateway, KesokuContext, KesokuConfig]:
        """Return started components or raise."""
        if not (self.gateway and self.context and self.cfg and self.session):
            raise RuntimeError("BenchSession not started")
        return self.gateway, self.context, self.cfg

    # --- driving ---------------------------------------------------------------------------------
    async def send_user_message(self, text: str, timeout_s: float) -> TurnOutcome:
        """Post a user message (PENDING_AGENT) and wait until the session is idle.

        Args:
            text: Message content.
            timeout_s: Turn timeout; on expiry the worker is stopped immediately.

        Returns:
            The TurnOutcome.
        """
        gateway, _, _ = self._require()
        start = time.time()
        msg = Message(
            session_id=self.session_id, chatbot_id=CHATBOT_ID, channel_id=self.channel_id, sender="User",
            role=MessageRole.USER, type=MessageType.TEXT, content=text, status=MessageStatus.PENDING_AGENT,
        )
        await gateway.post(msg)
        status = await self.wait_for_idle(msg.id, timeout_s)
        end = time.time()
        final_text, update_calls = await self._turn_summary(start)
        return TurnOutcome(msg_id=msg.id, status=status, start_ts=start, end_ts=end, final_text=final_text,
                           update_file_calls=update_calls)

    async def send_command(self, text: str) -> str:
        """Execute a slash command through the chatbot adapter (e.g. `/compact ...`) and return its reply."""
        self._require()
        assert self.bot is not None
        replies: list[str] = []

        async def _reply(message: str, *args: Any, **kwargs: Any) -> None:
            replies.append(message)

        await self.bot.execute_command_from_text(text, reply_func=_reply, channel_id=self.channel_id)
        reply = "\n".join(replies)
        self.command_replies.append({"text": text, "reply": reply, "ts": time.time()})
        return reply

    async def overhead_tokens(self) -> int:
        """Estimate the fixed prompt overhead (system prompt + tool schemas) like TurnExecutor does."""
        gateway, context, _ = self._require()
        session = await gateway.db.get_session(self.session_id)
        return estimate_fixed_prompt_overhead(session.system_prompt if session else "",
                                              context.tool_registry.get_tools_list())

    async def force_compaction(self, overhead_tokens: int | None = None, protect_tail_turns: int = 1) -> int:
        """Deterministic compaction point between turns (spec §2.4) through the production watermark path.

        A throw-away config with ~0 thresholds makes `_compact_watermark` summarise exactly the turns outside
        protect_front / protect_tail. Returns the number of new root summary nodes created.
        """
        gateway, context, cfg = self._require()
        if overhead_tokens is None:
            overhead_tokens = await self.overhead_tokens()
        cfg_tmp = cfg.model_copy(deep=True)
        # Attribute assignment skips _validate_watermarks (pydantic validate_assignment is off).
        cfg_tmp.agent.compression_strategy = "watermark"
        cfg_tmp.agent.context_high_watermark = 1
        cfg_tmp.agent.context_low_watermark = 0
        cfg_tmp.agent.protect_tail_min_tokens = 0
        cfg_tmp.agent.watermark_chunk_tokens = 1
        cfg_tmp.agent.protect_tail_turns = protect_tail_turns
        history = await build_history(gateway, self.session_id, heal_orphans=False)
        before = await self.summary_node_count()
        await HistoryCompressor(gateway.db).auto_compact_session(
            session_id=self.session_id, history=history, llm=context.get_llm(use_context_compression=True),
            config=cfg_tmp, staging_dir=str(self.staging_dir), overhead_tokens=overhead_tokens,
        )
        return await self.summary_node_count() - before

    async def summary_node_count(self) -> int:
        """Number of summary nodes (all levels) of the session."""
        gateway, _, _ = self._require()
        return len(await gateway.db.get_all_summary_nodes(self.session_id))

    async def root_summary_node_count(self) -> int:
        """Number of root summary nodes of the session."""
        gateway, _, _ = self._require()
        return len(await gateway.db.get_root_summary_nodes(self.session_id))

    # --- idle detection --------------------------------------------------------------------------
    def _worker_busy(self) -> bool:
        """Whether the session worker is processing or has queued messages."""
        if not self.agent:
            return False
        worker = self.agent.workers.get(self.session_id)
        if worker is None or not worker.running:
            return False
        return bool(worker._processing_turn) or not worker.queue_empty()

    async def _session_busy(self) -> bool:
        """Whether any session message is waiting for or being processed by the agent."""
        gateway, _, _ = self._require()
        msgs = await gateway.db.get_messages_by_filters({"session_id": self.session_id})
        return any(m.status in BUSY_STATUSES for m in msgs)

    async def wait_for_idle(self, msg_id: str, timeout_s: float) -> str:
        """Wait until the initiating message is terminal and the session is idle for `idle_stable_s`.

        Background-job wake-ups (SYSTEM messages, PENDING_AGENT) keep the session busy, so they extend the
        turn automatically.

        Returns:
            processed | error | interrupted | timeout.
        """
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
        # Timeout: stop the worker and release anything stuck in PROCESSING.
        if self.agent:
            await self.agent.stop_session_worker(self.session_id, immediate=True)
        msgs = await gateway.db.get_messages_by_filters({"session_id": self.session_id})
        for m in msgs:
            if m.status in BUSY_STATUSES:
                await gateway.db.update_message_status(m.id, MessageStatus.INTERRUPTED)
        return status

    async def settle(self, max_wait_s: float, timeout_s: float) -> bool:
        """After the last turn: wait up to max_wait_s for running background jobs to finish and wake the agent.

        Returns:
            True if a late wake-up turn was processed.
        """
        gateway, context, _ = self._require()
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
        """Snapshot of the session's jobs in ActiveJobsRegistry: job_id, pid, pgid, alive."""
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
            jobs.append({
                "job_id": job_id, "pid": pid, "pgid": pgid,
                "alive": getattr(proc, "returncode", 0) is None,
                "log_filepath": job.get("log_filepath"), "creation_time": job.get("creation_time"),
            })
        return jobs

    async def _turn_summary(self, since_ts: float) -> tuple[str, int]:
        """Final assistant text and number of update_file calls since a timestamp."""
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

    # --- snapshots for grading -------------------------------------------------------------------
    async def all_messages(self) -> list[Message]:
        """All session messages in timestamp order."""
        gateway, _, _ = self._require()
        return await gateway.db.get_messages_by_filters({"session_id": self.session_id})

    async def all_summary_nodes(self) -> list[dict[str, Any]]:
        """All summary nodes of the session as dicts."""
        gateway, _, _ = self._require()
        nodes = await gateway.db.get_all_summary_nodes(self.session_id)
        return [n.model_dump() for n in nodes]

    def llm_logs(self) -> list[Path]:
        """llm-turn-*.log.yaml files of the session, in index order."""
        if not self.staging_dir or not self.staging_dir.is_dir():
            return []

        def _idx(p: Path) -> int:
            try:
                return int(p.name.split("-")[2].split(".")[0])
            except (IndexError, ValueError):
                return 0

        return sorted(self.staging_dir.glob("llm-turn-*.log.yaml"), key=_idx)
