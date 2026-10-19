"""P1 tests for TurnExecutor: scaffold never repeats compressed turns verbatim (§5) and per-turn frozen state
snapshots (`running_jobs` / `staging_artifacts` context blocks, §3)."""

import asyncio
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from kesoku.agent.llm import BaseLLM, LLMResponse, ToolCallRequest
from kesoku.agent.tools.shell import ActiveJobsRegistry
from kesoku.agent.turn_executor import TurnExecutor
from kesoku.agent.turn_logger import TurnLogger
from kesoku.config import KesokuConfig, WorkspaceConfig
from kesoku.constants import MessageRole
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager, Message, SummaryNode
from kesoku.gateway.gateway import Gateway


class _CaptureLLM(BaseLLM):
    """Returns scripted responses and records every history it was called with."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.histories: list[list[Message]] = []

    async def generate(self, prompt: str | None = None, system_prompt: str | None = None,
                       history: list[Message] | None = None, tools: list[Any] | None = None,
                       **kwargs: Any) -> LLMResponse:
        self.histories.append([m.model_copy() for m in history or []])
        return self.responses.pop(0) if self.responses else LLMResponse(content="done", total_tokens=1)


def _worker(n_iterations: int) -> MagicMock:
    worker = MagicMock()
    type(worker).running = PropertyMock(side_effect=[True] * n_iterations + [False])

    async def pivot(m: Message) -> Message:
        return m

    worker.drain_queue_and_pivot.side_effect = pivot
    worker.queue_empty.return_value = True
    return worker


async def _session(tmp_path: Path, n_completed: int) -> tuple[Gateway, KesokuConfig, list[Message], Message]:
    db = str(tmp_path / "p1.db")
    DatabaseManager(db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=db, sessions_dir=str(tmp_path / "sessions")))
    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.create_session("sess_p1", title="p1")
    base = time.time() - 1000
    msgs: list[Message] = []
    for i in range(1, n_completed + 1):
        u = Message(session_id="sess_p1", chatbot_id="cli", channel_id="ch", sender="u", role=MessageRole.USER,
                    content=f"Turn {i} user", status="processed", timestamp=base + i * 10)
        a = Message(session_id="sess_p1", chatbot_id="cli", channel_id="ch", sender="Kesoku",
                    role=MessageRole.ASSISTANT, content=f"Turn {i} reply", status="delivered",
                    timestamp=base + i * 10 + 1)
        for m in (u, a):
            await gw.post(m)
            msgs.append(m)
    active = Message(session_id="sess_p1", chatbot_id="cli", channel_id="ch", sender="u", role=MessageRole.USER,
                     content="Active turn", status="pending_agent", timestamp=time.time())
    await gw.post(active)
    return gw, cfg, msgs, active


@pytest.mark.asyncio
async def test_compressed_turn_in_tail_is_not_repeated_verbatim(tmp_path: Path) -> None:
    """§5: T4 sits inside the protected tail but carries a summary_node_id -> only the scaffold represents it."""
    gw, cfg, msgs, active = await _session(tmp_path, n_completed=4)
    cfg.agent.protect_front_turns = 1
    cfg.agent.protect_tail_turns = 3  # T3, T4, active turn are "tail"
    t4 = [m for m in msgs if m.content.startswith("Turn 4")]
    node = SummaryNode(id="n-t4", session_id="sess_p1", level=0, summary="Turn 4 summary", start_timestamp=0,
                       end_timestamp=1, token_count=3, source_token_count=30)
    await gw.db.insert_summary_node(node)
    await gw.db.update_messages_summary_node([m.id for m in t4], "n-t4")

    llm = _CaptureLLM([LLMResponse(content="ok", total_tokens=1)])
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    executor = TurnExecutor("sess_p1", gw, tool_runner, MagicMock(spec=TurnLogger),
                            context=KesokuContext(config=cfg, llm=llm))
    with patch("kesoku.context.get_config", return_value=cfg), \
            patch("kesoku.agent.compressor.HistoryCompressor.auto_compact_session", return_value=False):
        await executor.process_turn(current_msg=active, worker=_worker(1),
                                    session_staging_dir=str(tmp_path / "sessions" / "sess_p1"))

    contents = [m.content for m in llm.histories[0]]
    assert not any("Turn 4 reply" in c for c in contents)
    assert sum("Turn 4 summary" in c for c in contents) == 1
    order = [next(i for i, c in enumerate(contents) if marker in c)
             for marker in ("Turn 1 user", "Turn 4 summary", "Turn 2 user", "Turn 3 reply", "Active turn")]
    assert order == sorted(order)


@pytest.mark.asyncio
async def test_state_snapshot_blocks_frozen_within_a_turn(tmp_path: Path) -> None:
    """running_jobs / staging_artifacts are computed once per turn: identical across LLM calls of the turn."""
    gw, cfg, msgs, active = await _session(tmp_path, n_completed=1)
    staging = tmp_path / "sessions" / "sess_p1"
    staging.mkdir(parents=True)
    calls = {"jobs": 0, "artifacts": 0}

    def fake_jobs(snapshot: Any, *args: Any, **kwargs: Any) -> str:
        calls["jobs"] += 1
        return f"JOBS-SNAPSHOT-{calls['jobs']}"

    def fake_artifacts(staging_dir: str, *args: Any, **kwargs: Any) -> str:
        calls["artifacts"] += 1
        return f"ARTIFACTS-{calls['artifacts']}"

    llm = _CaptureLLM([
        LLMResponse(content="", tool_calls=[ToolCallRequest(name="noop", arguments={})], total_tokens=1),
        LLMResponse(content="final", total_tokens=1),
    ])
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []

    async def exec_tool(call: Any, tc_msg: Message, is_interrupted: Any = None) -> Message:
        return Message(session_id="sess_p1", chatbot_id="cli", channel_id="ch", sender="noop",
                       role=MessageRole.TOOL, type="tool_result", content="ok", status="responded",
                       parent_id=tc_msg.id, metadata={"tool_name": "noop", "tool_result": "ok"})

    tool_runner.execute_tool.side_effect = exec_tool
    tool_runner.tool_context = MagicMock(transitioned_to_session=None)
    context = KesokuContext(config=cfg, llm=llm)
    executor = TurnExecutor("sess_p1", gw, tool_runner, MagicMock(spec=TurnLogger), context=context)
    executor._has_summaries = True  # pretend the session was compacted before
    with patch("kesoku.context.get_config", return_value=cfg), \
            patch("kesoku.agent.compressor.HistoryCompressor.auto_compact_session", return_value=False), \
            patch("kesoku.agent.turn_executor.render_running_jobs", side_effect=fake_jobs), \
            patch("kesoku.agent.turn_executor.render_staging_artifacts", side_effect=fake_artifacts), \
            patch.object(TurnExecutor, "_check_and_auto_compact_history", side_effect=_identity):
        await executor.process_turn(current_msg=active, worker=_worker(2), session_staging_dir=str(staging))

    assert len(llm.histories) == 2
    latest = [next(m.content for m in reversed(h) if m.role == MessageRole.USER) for h in llm.histories]
    assert latest[0] == latest[1]
    assert '<context name="running_jobs">' in latest[0] and "JOBS-SNAPSHOT-1" in latest[0]
    assert '<context name="staging_artifacts">' in latest[0] and "ARTIFACTS-1" in latest[0]
    assert calls == {"jobs": 1, "artifacts": 1}


async def _identity(history: list[Message], **kwargs: Any) -> tuple[list[Message], bool]:
    return history, False


@pytest.mark.asyncio
async def test_snapshot_entries_gating_and_failures(tmp_path: Path) -> None:
    """No artifacts block before any compaction; renderer failures never break the turn."""
    gw, cfg, msgs, active = await _session(tmp_path, n_completed=1)
    context = KesokuContext(config=cfg, llm=_CaptureLLM([]))
    executor = TurnExecutor("sess_p1", gw, MagicMock(), None, context=context)
    with patch("kesoku.agent.turn_executor.render_running_jobs", return_value=None), \
            patch("kesoku.agent.turn_executor.render_staging_artifacts", return_value="ART"):
        entries = await executor._get_state_snapshot_entries(str(tmp_path))
    assert entries == []  # no summary nodes yet -> artifacts gated, no jobs -> no block
    executor._snapshot_cache = None
    with patch("kesoku.agent.turn_executor.render_running_jobs", side_effect=RuntimeError("boom")), \
            patch("kesoku.agent.turn_executor.render_staging_artifacts", side_effect=RuntimeError("boom")):
        executor._has_summaries = True
        assert await executor._get_state_snapshot_entries(str(tmp_path)) == []


@pytest.mark.asyncio
async def test_running_jobs_block_from_real_registry(tmp_path: Path) -> None:
    """End-to-end with the real ActiveJobsRegistry: a running job is rendered with its pgid stop command."""
    gw, cfg, msgs, active = await _session(tmp_path, n_completed=1)
    registry = ActiveJobsRegistry()
    proc = await asyncio.create_subprocess_shell("sleep 30", preexec_fn=os.setsid)
    try:
        await registry.register("job_test_1", "sess_p1", proc, str(tmp_path / "out.log"),
                                command="sleep 30", stderr_filepath=str(tmp_path / "err.log"))
        context = KesokuContext(config=cfg, llm=_CaptureLLM([]))
        context.active_jobs = registry
        executor = TurnExecutor("sess_p1", gw, MagicMock(), None, context=context)
        entries = await executor._get_state_snapshot_entries(str(tmp_path))
        jobs = next(e for e in entries if e.name == "running_jobs").content
        assert "job_test_1" in jobs and "sleep 30" in jobs
        assert f"kill -TERM -- -{os.getpgid(proc.pid)}" in jobs
    finally:
        await registry.stop_all_for_session("sess_p1")
