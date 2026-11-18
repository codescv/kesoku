"""Tests for agent-core fixes from the 2026-11 code & log audit (P1-2, P1-3, P1-4, P2-8, P2-12, P3-16, NEW-3)."""

import asyncio
import dataclasses
import gc
import logging
import os
import threading
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from google.genai import types

from kesoku.agent import turn_executor as te
from kesoku.agent.history import prepare_history_for_llm
from kesoku.agent.llm import (
    LLMResponse,
    _drain_blocking_stream,
    _is_explicit_cache_unsupported_error,
    summarize_request_config,
)
from kesoku.agent.turn_executor import TurnExecutor, _TurnState
from kesoku.agent.turn_logger import INTERNAL_LOGS_DIRNAME, TurnLogger
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message

SECRET_PERSONA = "# Active Persona: top-secret persona prompt " + "x" * 500


# ---------------------------------------------------------------- P1-4 explicit-cache error classification


@pytest.mark.parametrize(
    ("err", "expected"),
    [
        ("400 INVALID_ARGUMENT: Model 'gemini-4-hydrogen' does not support explicit context caching.", True),
        ("Explicit context caching is not supported for this model", True),
        ("429 RESOURCE_EXHAUSTED: quota exceeded for explicit context caching storage", False),
        ("400 INVALID_ARGUMENT: explicit context caching requires at least 4096 tokens", False),
        ("503 UNAVAILABLE", False),
    ],
)
def test_explicit_cache_unsupported_matcher_is_narrow(err: str, expected: bool) -> None:
    assert _is_explicit_cache_unsupported_error(RuntimeError(err)) is expected


# ---------------------------------------------------------------- P1-3 request config logging


def test_summarize_request_config_hides_prompt_text() -> None:
    cfg = types.GenerateContentConfig(
        system_instruction=SECRET_PERSONA,
        thinking_config=types.ThinkingConfig(thinking_budget=128),
        cached_content="projects/p/locations/global/cachedContents/123",
    )
    summary = summarize_request_config(cfg)
    assert "top-secret" not in summary
    assert f"sys_len={len(SECRET_PERSONA)}" in summary
    assert "sys_sha1=" in summary
    assert "cached_content=123" in summary
    assert len(summary) < 200


def test_gemini_request_config_logged_at_info_as_summary_only() -> None:
    from kesoku.agent.llm import GeminiLLM
    from kesoku.config import GeminiConfig

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock()
    with patch("google.genai.Client", return_value=mock_client):
        llm = GeminiLLM(GeminiConfig(model_name="gemini-3.8-flash"))
    cfg = types.GenerateContentConfig(system_instruction=SECRET_PERSONA)
    with patch("kesoku.agent.llm.logger.info") as info, patch("kesoku.agent.llm.logger.debug") as debug, patch(
        "kesoku.agent.llm.logger.isEnabledFor", return_value=False
    ):
        llm._log_request_config({"contents": [], "config": cfg}, streaming=True)
    logged = " ".join(str(c.args[0]) for c in info.call_args_list)
    assert "top-secret" not in logged
    assert "streaming request" in logged and "sys_len=" in logged
    debug.assert_not_called()


# ---------------------------------------------------------------- NEW-3 abandoned stream pump


@pytest.mark.asyncio
async def test_drain_blocking_stream_timeout_tracks_and_consumes_pump_task() -> None:
    from kesoku.agent import llm as llm_mod

    release = threading.Event()

    def stalled_stream() -> Any:
        yield "first"
        release.wait(2.0)
        raise RuntimeError("socket died after abandonment")

    seen: list[Any] = []

    async def on_item(item: Any) -> None:
        seen.append(item)

    loop = asyncio.get_running_loop()
    unhandled: list[dict[str, Any]] = []
    old_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, ctx: unhandled.append(ctx))
    try:
        with pytest.raises(TimeoutError):
            await _drain_blocking_stream(stalled_stream, on_item, idle_timeout=0.05)
        assert seen == ["first"]
        assert len(llm_mod._ORPHANED_PUMP_TASKS) >= 1
        release.set()
        for _ in range(100):
            if not llm_mod._ORPHANED_PUMP_TASKS:
                break
            await asyncio.sleep(0.02)
        assert not llm_mod._ORPHANED_PUMP_TASKS
        gc.collect()
        await asyncio.sleep(0)
        assert not [c for c in unhandled if "never retrieved" in str(c.get("message", ""))]
    finally:
        release.set()
        loop.set_exception_handler(old_handler)


# ---------------------------------------------------------------- P1-2 turn logger


def _msg(content: str, **kw: Any) -> Message:
    base: dict[str, Any] = {
        "session_id": "s1",
        "chatbot_id": "cli",
        "channel_id": "ch",
        "sender": "u",
        "role": MessageRole.USER,
        "content": content,
    }
    base.update(kw)
    return Message(**base)


@pytest.mark.asyncio
async def test_alog_llm_turn_writes_off_loop_with_isolated_snapshot(tmp_path: Any) -> None:
    staging = str(tmp_path / "sess")
    tl = TurnLogger("s1", staging)
    msg = _msg("hello", metadata={"k": ["v1"]})
    res = LLMResponse(content="ok", raw_response="HUGE-SDK-REPR", prompt_tokens=3, total_tokens=4)

    loop_tid = threading.get_ident()
    write_tids: list[int] = []
    orig_write = TurnLogger._write_log

    def spy_write(self: TurnLogger, data: dict[str, Any], idx: int) -> str:
        write_tids.append(threading.get_ident())
        # Mutate the live message while the background write is pending: the snapshot must not change.
        msg.metadata["k"].append("MUTATED")
        return orig_write(self, data, idx)

    with patch.object(TurnLogger, "_write_log", spy_write):
        path = await tl.alog_llm_turn("TestLLM", [msg], [], res)

    assert write_tids and write_tids[0] != loop_tid
    assert os.path.dirname(path) == os.path.join(staging, INTERNAL_LOGS_DIRNAME)
    data = yaml.safe_load(open(path, encoding="utf-8"))
    assert data["history"][0]["metadata"] == {"k": ["v1"]}
    assert "raw_response" not in data["response"]
    assert data["response"]["metrics"]["prompt_tokens"] == 3


def test_turn_logger_index_shared_across_instances_and_honours_legacy_root(tmp_path: Any) -> None:
    staging = tmp_path / "sess2"
    staging.mkdir()
    (staging / "llm-turn-7.log.yaml").write_text("{}")
    a = TurnLogger("s2", str(staging))
    b = TurnLogger("s2", str(staging))  # e.g. a sub-agent logger for the same session dir
    res = LLMResponse(content="x")
    p1 = a.log_llm_turn("A", [], [], res)
    p2 = b.log_llm_turn("B", [], [], res)
    p3 = a.log_llm_turn("A", [], [], res)
    assert [os.path.basename(p) for p in (p1, p2, p3)] == [
        "llm-turn-8.log.yaml",
        "llm-turn-9.log.yaml",
        "llm-turn-10.log.yaml",
    ]
    # Index scan only happens once per directory: later writes never list the directory.
    with patch("kesoku.agent.turn_logger.os.scandir", side_effect=AssertionError("rescanned")):
        a.log_llm_turn("A", [], [], res)


@pytest.mark.asyncio
async def test_alog_first_index_scan_runs_off_loop_and_turn_index_is_set(tmp_path: Any) -> None:
    staging = tmp_path / "sess3"
    staging.mkdir()
    (staging / "llm-turn-41.log.yaml").write_text("{}")
    loop_tid = threading.get_ident()
    scan_tids: list[int] = []
    from kesoku.agent import turn_logger as tl_mod

    orig_scan = tl_mod._scan_max_llm_turn_idx

    def spy_scan(directory: str) -> int:
        scan_tids.append(threading.get_ident())
        return orig_scan(directory)

    with patch.object(tl_mod, "_scan_max_llm_turn_idx", spy_scan):
        path = await TurnLogger("s3", str(staging)).alog_llm_turn("T", [], [], LLMResponse(content="x"))
    assert scan_tids and all(t != loop_tid for t in scan_tids)
    assert os.path.basename(path) == "llm-turn-42.log.yaml"
    assert yaml.safe_load(open(path, encoding="utf-8"))["metadata"]["turn_index"] == 42


# ---------------------------------------------------------------- P2-12 historical tool archives


def _tool_turn(result: str, msg_id: str = "r1") -> list[Message]:
    return [
        _msg("q1", id="u1", timestamp=1.0, status=MessageStatus.RESPONDED),
        _msg(
            "call",
            id="c1",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_CALL,
            sender="Kesoku",
            timestamp=2.0,
            metadata={"tool_name": "run_shell_command", "tool_arguments": {"command": "ls"}},
        ),
        _msg(
            result,
            id=msg_id,
            role=MessageRole.TOOL,
            type=MessageType.TOOL_RESULT,
            sender="run_shell_command",
            parent_id="c1",
            timestamp=3.0,
            metadata={"tool_name": "run_shell_command", "tool_result": result},
        ),
        _msg("q2", id="u2", timestamp=4.0, status=MessageStatus.RESPONDED),
    ]


def test_new_hist_archives_go_to_internal_logs_dir(tmp_path: Any) -> None:
    prepared = prepare_history_for_llm(_tool_turn("y" * 3000), staging_dir=str(tmp_path))
    path = prepared[2].metadata["output_filepath"]
    assert path == str(tmp_path / ".kesoku_logs" / "hist_tool_run_shell_command_r1.txt")
    assert not list(tmp_path.glob("hist_*"))


def test_legacy_root_hist_archive_keeps_its_path(tmp_path: Any) -> None:
    legacy = tmp_path / "hist_tool_run_shell_command_r1.txt"
    legacy.write_text("y" * 3000)
    prepared = prepare_history_for_llm(_tool_turn("y" * 3000), staging_dir=str(tmp_path))
    assert prepared[2].metadata["output_filepath"] == str(legacy)
    assert not (tmp_path / ".kesoku_logs").exists()


def test_existing_archive_wins_over_saved_to_reference(tmp_path: Any) -> None:
    """Once archived, the referenced path never flips (prefix determinism), even if a `saved to:` file appears."""
    saved = tmp_path / "cmd_output_1.txt"
    result = f"Full output saved to session workspace file: `{saved}`.\n" + "z" * 3000
    first = prepare_history_for_llm(_tool_turn(result), staging_dir=str(tmp_path))
    archived = first[2].metadata["output_filepath"]
    assert archived.endswith(".kesoku_logs/hist_tool_run_shell_command_r1.txt")
    saved.write_text("z" * 3000)
    second = prepare_history_for_llm(_tool_turn(result), staging_dir=str(tmp_path))
    assert second[2].metadata["output_filepath"] == archived
    assert first[2].content == second[2].content


# ---------------------------------------------------------------- P3-16 / P2-8 TurnExecutor state


def _executor() -> TurnExecutor:
    ctx = MagicMock()
    ctx.db.get_session_turns_count = AsyncMock(return_value=5)
    return TurnExecutor("s-state", MagicMock(), MagicMock(), None, context=ctx)


def test_reset_turn_state_restores_every_declared_field() -> None:
    ex = _executor()
    for f in dataclasses.fields(_TurnState):
        setattr(ex, f.name, object())
    ex._reset_turn_state()
    fresh = _TurnState()
    for f in dataclasses.fields(_TurnState):
        assert getattr(ex, f.name) == getattr(fresh, f.name), f.name
    # Mutable defaults are fresh objects per reset (never shared across turns/executors).
    other = _executor()
    assert ex._turn_rounds is not other._turn_rounds
    assert ex._hook_retries is not other._hook_retries


@pytest.mark.asyncio
async def test_session_turn_count_queried_once_per_turn() -> None:
    ex = _executor()
    assert await ex._get_session_turns_count() == 5
    assert await ex._get_session_turns_count() == 5
    assert ex.context.db.get_session_turns_count.await_count == 1
    ex._reset_turn_state()  # pivot / new turn
    await ex._get_session_turns_count()
    assert ex.context.db.get_session_turns_count.await_count == 2


@pytest.mark.asyncio
async def test_thought_signature_prune_scheduled_for_real_db_only() -> None:
    ex = _executor()
    ex.context.db.clear_thought_signatures_before = AsyncMock(return_value=3)
    msg = _msg("hi", timestamp=time.time())
    ex._schedule_thought_signature_prune(msg)
    await asyncio.sleep(0.01)
    ex.context.db.clear_thought_signatures_before.assert_awaited_once_with("s-state", msg.timestamp)

    ex2 = _executor()
    ex2.context.db.clear_thought_signatures_before = MagicMock()  # not awaitable -> skipped silently
    ex2._schedule_thought_signature_prune(msg)
    ex2.context.db.clear_thought_signatures_before.assert_not_called()


@pytest.mark.asyncio
async def test_session_cache_locks_are_pruned_beyond_cap() -> None:
    saved = dict(te._session_cache_locks)
    te._session_cache_locks.clear()
    try:
        with patch.object(te, "_SESSION_CACHE_LOCKS_MAX", 4):
            busy = te._get_session_cache_lock("busy")
            await busy.acquire()
            for i in range(10):
                te._get_session_cache_lock(f"s{i}")
            assert len(te._session_cache_locks) <= 5 + 1
            assert te._session_cache_locks.get("busy") is busy  # held locks are never evicted
            busy.release()
    finally:
        te._session_cache_locks.clear()
        te._session_cache_locks.update(saved)


def test_logging_module_untouched() -> None:
    """Sanity: helper imports did not reconfigure the root logger."""
    assert isinstance(logging.getLogger(), logging.Logger)
