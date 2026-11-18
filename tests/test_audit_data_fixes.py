"""Regression tests for the DB / memory / tools / housekeeping audit fixes.

Covers NEW-1 (per-thread connection reuse), P2-7 (ensure_ascii, history projection, auto_vacuum, thought
signature cleanup), P2-8 (session/role index), P2-9 (web_search client/model/timeout), P2-10 + NEW-3 (session
deletion vs. memory extraction), P3-13 (tracked background tasks), P3-15 (file permissions), NEW-2 (per-tool
duration_ms) and P2-12 (background job logs under `.kesoku_logs/`).
"""

import asyncio
import gc
import json
import os
import sqlite3
import stat
import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kesoku.agent.llm import BaseLLM, LLMResponse
from kesoku.agent.memory_consolidator import EpisodeExtract, MemoryConsolidator, ReindexMemories
from kesoku.agent.tool_runner import ToolRunner
from kesoku.agent.tools import ToolContext, ToolRegistry
from kesoku.agent.tools.search import WEB_SEARCH_TIMEOUT_SECONDS, WebSearchTool
from kesoku.config import GeminiConfig, KesokuConfig, WorkspaceConfig, init_config, save_config
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.context import KesokuContext
from kesoku.db import AsyncDatabaseManager, DatabaseManager, Message, Session
from kesoku.db.connection import ConnectionProvider
from kesoku.utils import tasks as tasks_mod
from kesoku.utils.permissions import secure_file_permissions


def _mode(path: str) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def db(tmp_path) -> DatabaseManager:
    manager = DatabaseManager(str(tmp_path / "audit.db"))
    manager.init_tables()
    yield manager
    manager.close()


def _msg(sid: str, mid: str, ts: float, *, role: str = MessageRole.USER, meta: dict[str, Any] | None = None) -> Message:
    return Message(
        id=mid,
        session_id=sid,
        chatbot_id="discord",
        channel_id="ch",
        sender="u",
        role=role,
        type=MessageType.TEXT,
        content="内容",
        timestamp=ts,
        status=MessageStatus.PROCESSED,
        metadata=meta or {},
    )


# --------------------------------------------------------------------------------------------- NEW-1


def test_connection_reused_per_thread_and_isolated_across_threads_and_paths(tmp_path) -> None:
    provider = ConnectionProvider(str(tmp_path / "a.db"))
    other = ConnectionProvider(str(tmp_path / "b.db"))
    with provider.connection() as c1:
        pass
    with provider.connection() as c2:
        assert c2 is c1  # reused within the thread
        assert c2.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert c2.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert c2.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    with other.connection() as c3:
        assert c3 is not c1  # never shared across db paths

    seen: list[sqlite3.Connection] = []

    def worker() -> None:
        with provider.connection() as c:
            seen.append(c)

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen and seen[0] is not c1
    # A finished thread's cached connection is released (closed), not leaked in the provider registry.
    del seen[:]
    gc.collect()
    assert len(provider._cached) == 1
    provider.close_all()
    other.close_all()


def test_connection_discards_uncommitted_work_and_restores_state(tmp_path) -> None:
    provider = ConnectionProvider(str(tmp_path / "t.db"))
    with provider.connection() as conn:
        with conn:
            conn.execute("CREATE TABLE t (x INTEGER)")
    with provider.connection() as conn:
        conn.execute("INSERT INTO t VALUES (1)")  # implicit transaction, never committed
        assert conn.in_transaction
    with provider.connection() as conn:
        assert not conn.in_transaction
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0
        conn.isolation_level = None
        conn.row_factory = None
    with provider.connection() as conn:
        assert conn.isolation_level == ""
        assert conn.row_factory is sqlite3.Row
    provider.close_all()


def test_nested_connection_gets_private_connection(tmp_path) -> None:
    provider = ConnectionProvider(str(tmp_path / "n.db"))
    with provider.connection() as outer:
        with outer:
            outer.execute("CREATE TABLE t (x INTEGER)")
        outer.execute("INSERT INTO t VALUES (1)")  # outer transaction open
        with provider.connection() as inner:
            assert inner is not outer
            # The inner block cannot see (or commit) the outer's uncommitted row.
            assert inner.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0
        assert outer.in_transaction
        outer.commit()
    with provider.connection() as again:
        assert again is outer
        assert again.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1
    provider.close_all()


def test_close_all_invalidates_cached_connections(tmp_path) -> None:
    provider = ConnectionProvider(str(tmp_path / "c.db"))
    with provider.connection() as c1:
        pass
    provider.close_all()
    with pytest.raises(sqlite3.ProgrammingError):
        c1.execute("SELECT 1")
    with provider.connection() as c2:
        assert c2 is not c1
        assert c2.execute("SELECT 1").fetchone()[0] == 1
    provider.close_all()


def test_close_all_does_not_close_connection_in_use_by_another_thread(tmp_path) -> None:
    provider = ConnectionProvider(str(tmp_path / "busy.db"))
    with provider.connection() as conn:
        with conn:
            conn.execute("CREATE TABLE t (x INTEGER)")
            conn.execute("INSERT INTO t VALUES (1)")

    inside = threading.Event()
    closed = threading.Event()
    result: dict[str, Any] = {}

    def thread_a() -> None:
        try:
            with provider.connection() as c:
                result["conn"] = c
                inside.set()
                assert closed.wait(5)
                result["count"] = c.execute("SELECT COUNT(*) FROM t").fetchone()[0]  # still usable
        except Exception as e:  # pragma: no cover - surfaced by the assertion below
            result["error"] = e

    a = threading.Thread(target=thread_a)
    a.start()
    assert inside.wait(5)
    b = threading.Thread(target=provider.close_all)  # thread B
    b.start()
    b.join()
    closed.set()
    a.join()
    assert "error" not in result, result.get("error")
    assert result["count"] == 1
    with pytest.raises(sqlite3.ProgrammingError):  # discarded by its owner on exit (stale generation)
        result["conn"].execute("SELECT 1")
    provider.close_all()


def test_init_tables_overwrite_with_cached_connection(tmp_path) -> None:
    manager = DatabaseManager(str(tmp_path / "ow.db"))
    manager.init_tables()
    manager.create_session(Session(id="s1", title="t", created_at=1.0, updated_at=1.0))
    assert manager.get_session("s1") is not None
    manager.init_tables(overwrite=True)
    assert manager.get_session("s1") is None  # fresh file, not the stale cached handle
    manager.close()


# --------------------------------------------------------------------------------------------- P2-7 / P2-8


def test_metadata_written_as_utf8_not_ascii_escapes(db: DatabaseManager) -> None:
    db.create_session(Session(id="s", title="t", created_at=1.0, updated_at=1.0))
    m = _msg("s", "m1", 1.0, meta={"note": "日本語"})
    db.save_message(m)
    db.update_message_metadata("m1", {"note": "更新", "tool_result": "结果"})
    with db.connection_provider.connection() as conn:
        raw = conn.execute("SELECT metadata FROM messages WHERE id='m1'").fetchone()[0]
    assert "更新" in raw and "\\u" not in raw
    assert json.loads(raw)["tool_result"] == "结果"  # tool_result is kept (not deduplicated)


def test_get_session_history_skips_embedding_column(db: DatabaseManager) -> None:
    db.create_session(Session(id="s", title="t", created_at=1.0, updated_at=1.0))
    db.save_message(_msg("s", "m1", 1.0))
    with db.connection_provider.connection() as conn:
        with conn:
            conn.execute("UPDATE messages SET embedding = ? WHERE id='m1'", (b"\x00" * 16,))
    hist = db.get_session_history("s", limit=0)
    assert [m.id for m in hist] == ["m1"] and hist[0].embedding is None
    assert db.get_message("m1").embedding == b"\x00" * 16  # single-row reads still return it


def test_new_db_uses_incremental_auto_vacuum_existing_db_untouched(tmp_path) -> None:
    fresh = DatabaseManager(str(tmp_path / "fresh.db"))
    fresh.init_tables()
    with fresh.connection_provider.connection() as conn:
        assert conn.execute("PRAGMA auto_vacuum").fetchone()[0] == 2  # INCREMENTAL
    fresh.close()

    legacy_path = str(tmp_path / "legacy.db")
    with sqlite3.connect(legacy_path) as raw:
        raw.execute("CREATE TABLE placeholder (x INTEGER)")
    legacy = DatabaseManager(legacy_path)
    legacy.init_tables()
    with legacy.connection_provider.connection() as conn:
        assert conn.execute("PRAGMA auto_vacuum").fetchone()[0] == 0
    legacy.close()


def test_session_role_index_created_for_new_and_migrated_dbs(db: DatabaseManager) -> None:
    with db.connection_provider.connection() as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        assert "idx_messages_session_role" in names
        with conn:
            conn.execute("DROP INDEX idx_messages_session_role")
    db.verify_db()  # migration path re-creates it
    with db.connection_provider.connection() as conn:
        plan = " ".join(
            str(tuple(r))
            for r in conn.execute(
                "EXPLAIN QUERY PLAN SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = 'user'", ("s",)
            )
        )
    assert "idx_messages_session_role" in plan


async def test_clear_thought_signatures_before(db: DatabaseManager) -> None:
    db.create_session(Session(id="s", title="t", created_at=1.0, updated_at=1.0))
    db.create_session(Session(id="other", title="t", created_at=1.0, updated_at=1.0))
    db.save_message(_msg("s", "old", 1.0, role=MessageRole.ASSISTANT, meta={"thought_signature": "ab", "k": 1}))
    db.save_message(_msg("s", "new", 5.0, role=MessageRole.ASSISTANT, meta={"thought_signature": "cd"}))
    db.save_message(_msg("s", "plain", 2.0, meta={"k": 2}))
    db.save_message(_msg("other", "o1", 1.0, role=MessageRole.ASSISTANT, meta={"thought_signature": "ef"}))

    assert db.clear_thought_signatures_before("s", 3.0) == 1
    assert db.get_message("old").metadata == {"k": 1}
    assert db.get_message("new").metadata == {"thought_signature": "cd"}
    assert db.get_message("o1").metadata == {"thought_signature": "ef"}

    adb = AsyncDatabaseManager(db)
    assert await adb.clear_thought_signatures_before("s", 10.0) == 1
    assert db.get_message("new").metadata == {}
    assert await adb.clear_thought_signatures_before("s", 10.0) == 0


def _trace_statements(db: DatabaseManager) -> list[str]:
    """Record SQL run on this thread's cached connection (reused by the next DB call on this thread)."""
    stmts: list[str] = []
    with db.connection_provider.connection() as conn:
        conn.set_trace_callback(stmts.append)
    return stmts


def _stop_trace(db: DatabaseManager) -> None:
    with db.connection_provider.connection() as conn:
        conn.set_trace_callback(None)


def test_clear_thought_signatures_hwm_skips_already_cleaned_range(db: DatabaseManager) -> None:
    db.create_session(Session(id="s", title="t", created_at=1.0, updated_at=1.0))
    db.save_message(_msg("s", "a", 1.0, role=MessageRole.ASSISTANT, meta={"thought_signature": "aa"}))
    assert db.clear_thought_signatures_before("s", 10.0) == 1
    # A row older than the high-water mark is not rescanned (the range below 10.0 is already clean).
    db.save_message(_msg("s", "late_old", 2.0, role=MessageRole.ASSISTANT, meta={"thought_signature": "bb"}))
    db.save_message(_msg("s", "b", 15.0, role=MessageRole.ASSISTANT, meta={"thought_signature": "cc"}))
    stmts = _trace_statements(db)
    try:
        assert db.clear_thought_signatures_before("s", 20.0) == 1
        assert db.clear_thought_signatures_before("s", 20.0) == 0  # hwm == before_ts: no query at all
    finally:
        _stop_trace(db)
    selects = [q for q in stmts if q.lstrip().upper().startswith("SELECT")]
    assert len(selects) == 1 and "timestamp > 10.0" in selects[0]
    assert db.get_message("late_old").metadata == {"thought_signature": "bb"}
    assert db.get_message("b").metadata == {}
    # Deleting the session forgets its high-water mark.
    db.delete_session("s")
    assert "s" not in db._thought_sig_hwm


def test_clear_thought_signatures_chunks_and_skips_write_when_clean(db: DatabaseManager, monkeypatch) -> None:
    import kesoku.db.manager as manager_mod

    monkeypatch.setattr(manager_mod, "_THOUGHT_SIG_CHUNK", 2)
    db.create_session(Session(id="s", title="t", created_at=1.0, updated_at=1.0))
    for i in range(5):
        db.save_message(_msg("s", f"m{i}", 1.0 + i, role=MessageRole.ASSISTANT, meta={"thought_signature": "x"}))
    # LIKE false positive: the key only appears inside a nested value; must not loop forever or be modified.
    db.save_message(_msg("s", "fp", 3.5, meta={"tool_result": '{"thought_signature": 1}'}))
    stmts = _trace_statements(db)
    try:
        assert db.clear_thought_signatures_before("s", 100.0) == 5
    finally:
        _stop_trace(db)
    updates = [q for q in stmts if q.lstrip().upper().startswith("UPDATE")]
    assert len(updates) == 3  # 5 matches + 1 false positive in chunks of 2
    assert all(db.get_message(f"m{i}").metadata == {} for i in range(5))
    assert db.get_message("fp").metadata == {"tool_result": '{"thought_signature": 1}'}

    # Clean session, fresh process state: read-only scan, never a write transaction.
    db._thought_sig_hwm.clear()
    stmts = _trace_statements(db)
    try:
        assert db.clear_thought_signatures_before("s", 100.0) == 0
    finally:
        _stop_trace(db)
    assert not any(q.lstrip().upper().startswith(("UPDATE", "BEGIN")) for q in stmts)


# --------------------------------------------------------------------------------------------- P3-15


def test_db_and_config_files_are_owner_only(tmp_path) -> None:
    db_path = str(tmp_path / "perm.db")
    manager = DatabaseManager(db_path)
    manager.init_tables()
    assert _mode(db_path) == 0o600
    os.chmod(db_path, 0o644)
    manager.verify_db()  # startup tightens pre-existing world-readable DBs
    assert _mode(db_path) == 0o600
    manager.close()

    cfg_path = str(tmp_path / "config.toml")
    init_config(cfg_path)
    assert _mode(cfg_path) == 0o600
    os.chmod(cfg_path, 0o644)
    save_config(KesokuConfig(workspace=WorkspaceConfig(db_path=db_path)), cfg_path)
    assert _mode(cfg_path) == 0o600

    assert secure_file_permissions(str(tmp_path / "missing")) is False


# --------------------------------------------------------------------------------------------- P3-13


async def test_spawn_background_keeps_strong_ref_and_logs_failures(caplog) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def ok() -> int:
        started.set()
        await release.wait()
        return 7

    async def boom() -> None:
        raise ValueError("kaput")

    before = tasks_mod.background_task_count()
    t = tasks_mod.spawn_background(ok(), name="ok-task")
    await started.wait()
    assert tasks_mod.background_task_count() == before + 1
    release.set()
    assert await t == 7
    await asyncio.sleep(0)
    assert tasks_mod.background_task_count() == before

    with caplog.at_level("ERROR", logger="kesoku.utils.tasks"):
        bad = tasks_mod.spawn_background(boom(), name="boom-task")
        await asyncio.gather(bad, return_exceptions=True)
        await asyncio.sleep(0)
    assert any("boom-task" in r.getMessage() and "kaput" in r.getMessage() for r in caplog.records)

    registry: set[asyncio.Task[Any]] = set()
    cancelled = tasks_mod.spawn_background(asyncio.sleep(10), registry=registry)
    assert cancelled in registry
    cancelled.cancel()
    await asyncio.gather(cancelled, return_exceptions=True)
    await asyncio.sleep(0)
    assert registry == set()


# --------------------------------------------------------------------------------------------- NEW-2


async def test_tool_runner_records_per_tool_duration_ms() -> None:
    reg = ToolRegistry()

    async def slow_tool(x: int) -> str:
        """Slow tool."""
        await asyncio.sleep(0.05)
        return f"v{x}"

    def failing_tool() -> str:
        """Failing tool."""
        raise RuntimeError("nope")

    reg.register(slow_tool)
    reg.register(failing_tool)
    runner = ToolRunner(reg, ToolContext(session_id="s", session_workspace="w"))
    tc_msg = _msg("s", "tc", 1.0, role=MessageRole.ASSISTANT)

    ok = await runner.execute_tool(SimpleNamespace(name="slow_tool", arguments={"x": 1}), tc_msg)
    assert ok.metadata["tool_result"] == "v1"
    assert isinstance(ok.metadata["duration_ms"], int) and ok.metadata["duration_ms"] >= 40

    err = await runner.execute_tool(SimpleNamespace(name="failing_tool", arguments={}), tc_msg)
    assert err.metadata["tool_error"] == "nope"
    assert isinstance(err.metadata["duration_ms"], int)

    missing = await runner.execute_tool(SimpleNamespace(name="slow_tool", arguments={}), tc_msg)
    assert "duration_ms" not in missing.metadata  # never invoked


# --------------------------------------------------------------------------------------------- P2-9


def _search_response(text: str = "answer") -> MagicMock:
    res = MagicMock()
    res.text = text
    res.candidates = []
    return res


async def test_web_search_caches_client_and_uses_context_model_with_low_thinking(monkeypatch) -> None:
    import kesoku.agent.tools.search as search_mod

    cfg = KesokuConfig()
    cfg.gemini = GeminiConfig(model_name="main-model", context_model_name="flash-model", thinking_level="high")
    monkeypatch.setattr(search_mod, "get_config", lambda: cfg)
    created: list[MagicMock] = []

    def fake_create() -> MagicMock:
        client = MagicMock()
        client.aio.models.generate_content = AsyncMock(return_value=_search_response())
        created.append(client)
        return client

    tool = WebSearchTool()
    monkeypatch.setattr(tool, "_create_client", fake_create)
    assert await tool.web_search("q1") == "answer"
    assert await tool.web_search("q2") == "answer"
    assert len(created) == 1  # client cached

    kwargs = created[0].aio.models.generate_content.await_args.kwargs
    assert kwargs["model"] == "flash-model"
    thinking = kwargs["config"].thinking_config
    assert thinking is not None and "LOW" in str(thinking.thinking_level).upper()

    cfg.gemini = GeminiConfig(model_name="main-model", context_model_name=None, thinking_level=None)
    await tool.web_search("q3")
    kwargs = created[0].aio.models.generate_content.await_args.kwargs
    assert kwargs["model"] == "main-model"
    assert kwargs["config"].thinking_config is None


async def test_web_search_retries_without_thinking_when_rejected() -> None:
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(
        side_effect=[Exception("400 INVALID_ARGUMENT: thinking_level is not supported"), _search_response("ok")]
    )
    tool = WebSearchTool(client=client)
    assert await tool.web_search("q") == "ok"
    second = client.aio.models.generate_content.await_args_list[1].kwargs["config"]
    assert second.thinking_config is None


async def test_web_search_times_out(monkeypatch) -> None:
    import kesoku.agent.tools.search as search_mod

    async def hang(**_: Any) -> Any:
        await asyncio.sleep(10)

    client = MagicMock()
    client.aio.models.generate_content = hang
    monkeypatch.setattr(search_mod, "WEB_SEARCH_TIMEOUT_SECONDS", 0.05)
    out = await WebSearchTool(client=client).web_search("slow")
    assert out.startswith("Web search timed out")
    assert WEB_SEARCH_TIMEOUT_SECONDS == 30.0


# --------------------------------------------------------------------------------------------- P2-10 / NEW-3


class _NeverLLM(BaseLLM):
    calls = 0

    async def generate(self, **kwargs: Any) -> LLMResponse:
        type(self).calls += 1
        return LLMResponse(content='{"episode_summary": "", "open_threads": [], "ops": []}')


def _consolidator(tmp_path) -> tuple[MemoryConsolidator, DatabaseManager]:
    db_path = str(tmp_path / "mem.db")
    sync_db = DatabaseManager(db_path)
    sync_db.init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=db_path, roles_dir=str(tmp_path / "roles")))
    cfg.agent_working_dir = str(tmp_path)
    ctx = KesokuContext(config=cfg, llm=_NeverLLM(), db=sync_db)
    return MemoryConsolidator(ctx, prewarm_embeddings=False, enable_timers=False), sync_db


async def test_run_episode_extract_skips_llm_when_session_deleted(tmp_path) -> None:
    consolidator, sync_db = _consolidator(tmp_path)
    # Messages exist but the session row is gone (deleted mid-flight).
    sync_db.save_message(_msg("gone", "u1", 1.0, meta={"context_policy": "full"}))
    a = _msg("gone", "a1", 1.5, role=MessageRole.ASSISTANT)
    a.parent_id = "u1"
    sync_db.save_message(a)
    _NeverLLM.calls = 0
    out = await consolidator.run_episode_extract(EpisodeExtract(role="default", session_id="gone", force=True))
    assert out.status == "session_deleted"
    assert _NeverLLM.calls == 0
    await consolidator.stop(wait_seconds=0.1)


async def test_run_episode_extract_discards_results_if_session_deleted_mid_llm_call(tmp_path) -> None:
    consolidator, sync_db = _consolidator(tmp_path)
    sync_db.create_session(Session(id="s_mid", title="t", created_at=1.0, updated_at=1.0))
    sync_db.save_message(_msg("s_mid", "u1", 1.0, meta={"context_policy": "full"}))
    a = _msg("s_mid", "a1", 1.5, role=MessageRole.ASSISTANT)
    a.parent_id = "u1"
    a.content = "好的，我记住了"
    sync_db.save_message(a)

    async def deleting_llm(prompt: str) -> tuple[dict[str, Any], int, int]:
        sync_db.delete_session("s_mid")  # deleted while the LLM call is in flight
        return {"episode_summary": "摘要", "open_threads": [], "ops": []}, 10, 5

    consolidator._generate_extraction = deleting_llm  # type: ignore[method-assign]
    out = await consolidator.run_episode_extract(EpisodeExtract(role="default", session_id="s_mid", force=True))
    assert out.status == "session_deleted"
    assert sync_db.get_session_digest("s_mid") is None
    await consolidator.stop(wait_seconds=0.1)


async def test_cancel_session_drops_timers_queued_jobs_and_debounce_state(tmp_path) -> None:
    consolidator, _ = _consolidator(tmp_path)
    blocker = asyncio.Event()

    async def blocked_execute(job: Any) -> None:
        await blocker.wait()

    consolidator._execute_job = blocked_execute  # type: ignore[method-assign]
    # First job is picked up by the consumer and blocks; the rest stay queued.
    consolidator.enqueue(ReindexMemories(role="r"))
    await asyncio.sleep(0)
    consolidator.enqueue(EpisodeExtract(role="r", session_id="victim", reason="signal"))
    consolidator.enqueue(EpisodeExtract(role="r", session_id="victim", reason="worker_stop", bypass_budget=True))
    consolidator.enqueue(EpisodeExtract(role="r", session_id="keeper"))
    consolidator._debounce_last_ts["victim"] = time.monotonic()
    consolidator._debounce_timers["victim"] = asyncio.create_task(asyncio.sleep(10))
    timer = consolidator._debounce_timers["victim"]

    assert consolidator.cancel_session("victim") == 2
    await asyncio.sleep(0)
    assert timer.cancelled()
    assert "victim" not in consolidator._debounce_last_ts
    assert "victim" not in consolidator._debounce_timers
    queue = consolidator._role_queues["r"]
    remaining = list(queue._queue)  # type: ignore[attr-defined]
    assert [getattr(j, "session_id", None) for j in remaining] == ["keeper"]
    assert not any(k[-1] == "victim" for k in consolidator._queued_keys)
    # Re-enqueue for the same session works again (keys were released).
    assert consolidator.enqueue(EpisodeExtract(role="r", session_id="victim"))

    blocker.set()
    await consolidator.drain(wait_seconds=2.0)
    assert queue._unfinished_tasks == 0  # type: ignore[attr-defined]
    await consolidator.stop(wait_seconds=0.1)


async def test_debounce_last_ts_is_pruned_once_stale(tmp_path) -> None:
    consolidator, sync_db = _consolidator(tmp_path)
    consolidator.context.config.agent.memory_extraction_debounce_seconds = 60
    consolidator.enqueue = lambda job: True  # type: ignore[method-assign]
    sync_db.create_session(Session(id="live", title="t", created_at=1.0, updated_at=1.0))
    consolidator._debounce_last_ts["stale"] = time.monotonic() - 3600
    u = _msg("live", "u1", 1.0)
    u.content = "这是一条足够长的有意义的用户消息，用来触发信号抽取"
    await consolidator.notify_turn(session_id="live", role="default", user_msg=u)
    assert "stale" not in consolidator._debounce_last_ts
    assert "live" in consolidator._debounce_last_ts  # active debounce window preserved
    await consolidator.stop(wait_seconds=0.1)


async def test_stop_session_worker_skip_memory_close() -> None:
    from kesoku.agent.agent import Agent

    agent = Agent.__new__(Agent)
    agent.workers = {}
    agent.context = SimpleNamespace(active_jobs=SimpleNamespace(stop_all_for_session=AsyncMock()))
    agent.consolidator = SimpleNamespace(notify_close=AsyncMock())

    await agent.stop_session_worker("s1", immediate=True, skip_memory_close=True)
    agent.consolidator.notify_close.assert_not_awaited()
    await agent.stop_session_worker("s1", immediate=True)
    agent.consolidator.notify_close.assert_awaited_once_with("s1", reason="worker_stop")


async def test_gateway_delete_session_skips_memory_close_and_cancels_jobs(tmp_path) -> None:
    from kesoku.gateway.gateway import Gateway

    db_path = str(tmp_path / "gw.db")
    sync_db = DatabaseManager(db_path)
    sync_db.init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=db_path, sessions_dir=str(tmp_path / "sessions")))
    gw = Gateway(context=KesokuContext(config=cfg, db=sync_db))
    sess = await gw.create_session("del_me", title="x")
    gw.agent = MagicMock()
    gw.agent.stop_session_worker = AsyncMock()
    consolidator = MagicMock()
    gw.context.memory_consolidator = consolidator

    await gw.delete_session(sess.id)
    gw.agent.stop_session_worker.assert_awaited_once_with(sess.id, immediate=True, skip_memory_close=True)
    consolidator.cancel_session.assert_called_once_with(sess.id)
    assert sync_db.get_session(sess.id) is None
