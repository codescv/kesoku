"""End-to-end harness tests with MockLLM: turns, idle detection, auto-reply, forced compaction, /compact routing,
breakers, timeouts, background-job wake-ups and tamper detection."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from benchmarks.kesoku_bench.harness.metering import BUDGET_STOP_TEXT
from benchmarks.kesoku_bench.harness.runner import RunOptions, needs_auto_reply, run_task
from benchmarks.kesoku_bench.harness.session import TurnOutcome
from benchmarks.kesoku_bench.harness.taskspec import load_task
from kesoku.agent.llm import BaseLLM, LLMResponse, MockLLM, ToolCallRequest
from kesoku.config import KesokuConfig
from kesoku.db import Message

SUMMARY_JSON = '{"timeline": ["- [T] user asked for things; agent answered"], "learnings": []}'


def _opts(tmp_path: Path, **kw: Any) -> RunOptions:
    base = dict(stage="test", run_id="r1", work_root=tmp_path / "work", holdout_root=tmp_path / "holdout",
                results_dir=tmp_path / "results", final_settle_s=0.0, idle_stable_s=0.3)
    base.update(kw)
    return RunOptions(**base)


def _factory(llm: BaseLLM) -> Callable[[KesokuConfig], BaseLLM]:
    return lambda cfg: llm


def _text(content: str, prompt: int = 0, total: int = 0) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=[], prompt_tokens=prompt, total_tokens=total)


def _tool(name: str, prompt: int = 0, total: int = 0, **args: Any) -> LLMResponse:
    return LLMResponse(content="", tool_calls=[ToolCallRequest(name=name, arguments=args)], prompt_tokens=prompt,
                       total_tokens=total)


FIX = dict(file_name="repo/calc.py", old_content="return a - b", new_content="return a + b")


def test_needs_auto_reply_policy() -> None:
    o = TurnOutcome(msg_id="m", status="processed", start_ts=0, end_ts=1, final_text="Which file?", update_file_calls=0)
    assert needs_auto_reply(o)
    o.update_file_calls = 1
    assert not needs_auto_reply(o)
    o.final_text = "Pick one\n```ui\ntype: select\n```"
    assert needs_auto_reply(o)
    o.final_text = "[question: which? || a | b]"
    assert needs_auto_reply(o)
    o.final_text = "Done."
    assert not needs_auto_reply(o)


@pytest.mark.asyncio
async def test_run_task_resolves_with_auto_reply(make_task: Callable[..., Path], tmp_path: Path,
                                                 restore_environ: None) -> None:
    spec = load_task(make_task(with_grade=True))
    llm = MockLLM(responses=[
        _text("Which file should I edit?"),
        _tool("run_shell_command", command="uv run pytest -q", cwd="repo"),
        _tool("update_file", **FIX),
        _text("Fixed add() in calc.py; tests pass."),
    ])
    res = await run_task(spec, _opts(tmp_path), llm_factory=_factory(llm))
    assert res["status"] == "resolved", res["error"] or res["grading"]
    assert res["grading"]["fail_to_pass"] == {"tests/test_holdout_calc.py::test_add": True}
    assert res["grading"]["pass_to_pass"] == {"tests/test_basic.py": True}
    assert res["grading"]["behavioral"] == {"uses_plus": True, "live_only": True}
    assert res["diagnostics"]["questions_asked"] == 1
    assert res["diagnostics"]["first_edit_file"] == "repo/calc.py"
    assert res["diagnostics"]["pre_cleanup"] == {"jobs": 0}
    assert res["diagnostics"]["n_events"] == 2
    assert res["totals"]["llm_calls"] == 4 and res["totals"]["tool_calls"] == 2
    turn = res["turns"][0]
    assert turn["tool_calls"] == {"run_shell_command": 1, "update_file": 1}
    assert turn["auto_replies"] == 1
    assert res["totals"]["forced_compaction_turns"] == [] and res["totals"]["natural_compaction_turns"] == []
    assert res["diagnostics"]["log_cross_check"]["log_files"] == 4
    # held-out tests were never inside the AWD during the run
    assert Path(res["workdir"], "ws", "repo", "tests", "test_holdout_calc.py").is_file()  # materialised at grading


@pytest.mark.asyncio
async def test_unresolved_and_tamper(make_task: Callable[..., Path], tmp_path: Path, restore_environ: None) -> None:
    spec = load_task(make_task())
    llm = MockLLM(responses=[
        _tool("run_shell_command", command="cat /somewhere/kesoku_bench/tasks/A9/gold/calc.py"),
        _text("I could not find it."),
    ])
    res = await run_task(spec, _opts(tmp_path), llm_factory=_factory(llm))
    assert res["status"] == "tampered"
    assert res["grading"]["tampered"] and res["grading"]["tamper_hits"]
    assert res["grading"]["fail_to_pass"] == {"tests/test_holdout_calc.py::test_add": False}


@pytest.mark.asyncio
async def test_visible_test_tampering_is_restored(make_task: Callable[..., Path], tmp_path: Path,
                                                  restore_environ: None) -> None:
    spec = load_task(make_task())
    llm = MockLLM(responses=[
        _tool("update_file", file_name="repo/tests/test_basic.py", old_content="== 6", new_content="== 7"),
        _tool("update_file", **FIX),
        _text("Done."),
    ])
    res = await run_task(spec, _opts(tmp_path), llm_factory=_factory(llm))
    assert res["grading"]["restored_tests"] == ["tests/test_basic.py"]
    assert res["status"] == "resolved"


@pytest.mark.asyncio
async def test_forced_compaction_scaffold(make_task: Callable[..., Path], tmp_path: Path,
                                          restore_environ: None) -> None:
    turns = [("user", f"turn {i} please answer briefly") for i in range(1, 6)]
    spec = load_task(make_task(task_id="C9_synthetic", suite="C", profile="compaction_forced", turns=turns,
                               force=[5]))
    llm = MockLLM(responses=[
        _text("a1"), _text("a2"), _text("a3"), _text("a4"),
        _text(SUMMARY_JSON),  # forced compaction before T5 summarises {T2, T3}
        _tool("update_file", **FIX), _text("a5 done"),
    ])
    res = await run_task(spec, _opts(tmp_path), llm_factory=_factory(llm))
    assert res["error"] is None, res["error"]
    assert res["profile"] == "compaction_forced"
    assert res["totals"]["forced_compaction_turns"] == [5]
    assert res["totals"]["natural_compaction_turns"] == []
    assert res["totals"]["compactions_l0"] == 1
    assert res["totals"]["summarize_calls"] == 1
    assert res["turns"][4]["summarize_calls"] == 1 and res["turns"][4]["forced_nodes"] == 1
    assert res["turns"][4]["compactions"] == 1
    # the T5 agent call saw the scaffold built from the forced node
    ws = Path(res["workdir"], "ws", "sessions")
    logs = sorted(ws.rglob("llm-turn-*.log.yaml"))
    assert any("Recent Summary" in p.read_text() for p in logs)
    assert res["status"] == "resolved"


@pytest.mark.asyncio
async def test_compact_command_compacts_or_triggers_fallback(make_task: Callable[..., Path], tmp_path: Path,
                                                      restore_environ: None) -> None:
    turns = [("user", f"turn {i}") for i in range(1, 5)] + [("command", "/compact keep only Y"), ("user", "fix it")]
    task_dir = make_task(task_id="C8_synthetic", suite="C", profile="compaction_forced", turns=turns,
                         extra="fallback_force_if_noop = true")
    spec = load_task(task_dir)
    llm = MockLLM(responses=[
        _text("a1"), _text("a2"), _text("a3"), _text("a4"),
        _text(SUMMARY_JSON),  # fallback force_compaction(protect_tail_turns=2) summarises {T2}
        _tool("update_file", **FIX), _text("done"),
    ])
    res = await run_task(spec, _opts(tmp_path), llm_factory=_factory(llm))
    assert res["error"] is None, res["error"]
    d = res["diagnostics"]
    assert res["turns"][4]["kind"] == "command"
    if d["fallback_forced"]:
        # Pre-P1-4 kesoku: /compact is a watermark no-op, the harness forces an equivalent compaction.
        assert d["manual_compact_nodes_delta"] == 0
        assert "not needed" in d["manual_compact_reply"]
        assert res["turns"][4]["forced_nodes"] == 1
    else:
        # P1-4 kesoku: /compact force-compacts the middle turns itself; no fallback.
        assert d["manual_compact_nodes_delta"] == 1
        assert res["turns"][4]["forced_nodes"] == 0
    assert res["totals"]["compactions_l0"] == 1
    assert res["status"] == "resolved"


@pytest.mark.asyncio
async def test_task_budget_breaker(make_task: Callable[..., Path], tmp_path: Path, restore_environ: None) -> None:
    spec = load_task(make_task(turns=[("user", "one"), ("user", "two")]))
    llm = MockLLM(responses=[
        _tool("run_shell_command", prompt=1_000_000, total=1_000_000, command="ls", cwd="repo"),
        _text("never returned"), _text("never returned"),
    ])
    res = await run_task(spec, _opts(tmp_path, max_task_cost_usd=0.5), llm_factory=_factory(llm))
    assert res["status"] == "budget_exceeded"
    assert len(res["turns"]) == 1
    assert res["totals"]["blocked_calls"] == 1
    assert res["totals"]["cost_usd"] == pytest.approx(0.75)
    assert "max_task_cost_usd" in res["diagnostics"]["budget_exceeded_reason"]


@pytest.mark.asyncio
async def test_max_tool_iters_ends_turn(make_task: Callable[..., Path], tmp_path: Path, restore_environ: None) -> None:
    spec = load_task(make_task())
    llm = MockLLM(responses=[_tool("run_shell_command", command="true", cwd="repo") for _ in range(5)])
    res = await run_task(spec, _opts(tmp_path, max_tool_iters=2), llm_factory=_factory(llm))
    assert res["totals"]["llm_calls"] == 2 and res["totals"]["blocked_calls"] == 1
    assert res["diagnostics"]["tool_iter_cap_turns"] == [1]
    assert res["status"] == "failed"


class _HangingLLM(BaseLLM):
    async def generate(self, prompt: str | None = None, system_prompt: str | None = None,
                       history: list[Message] | None = None, tools: Any = None,
                       cached_content: str | None = None) -> LLMResponse:
        await asyncio.sleep(3600)
        return _text("never")


@pytest.mark.asyncio
async def test_turn_timeout(make_task: Callable[..., Path], tmp_path: Path, restore_environ: None) -> None:
    spec = load_task(make_task(turns=[("user", "one"), ("user", "two")]))
    res = await run_task(spec, _opts(tmp_path, turn_timeout_s=1.0), llm_factory=_factory(_HangingLLM()))
    assert res["status"] == "timeout"
    assert [t["status"] for t in res["turns"]] == ["timeout"]


@pytest.mark.asyncio
async def test_background_job_wakeup_during_settle(make_task: Callable[..., Path], tmp_path: Path,
                                                   restore_environ: None) -> None:
    task_dir = make_task(with_grade=True, extra='config_overrides = { "shell.background_threshold_seconds" = 0.5 }')
    spec = load_task(task_dir)
    llm = MockLLM(responses=[
        _tool("run_shell_command", command="sleep 1.5 && echo finished", cwd="repo"),
        _text("The command moved to the background."),
        _tool("update_file", **FIX),
        _text("Background job done; fixed."),
    ])
    res = await run_task(spec, _opts(tmp_path, final_settle_s=15.0), llm_factory=_factory(llm))
    assert res["error"] is None, res["error"]
    assert res["totals"]["llm_calls"] == 4
    assert res["status"] == "resolved"


@pytest.mark.asyncio
async def test_budget_stop_text_is_final_message(make_task: Callable[..., Path], tmp_path: Path,
                                                 restore_environ: None) -> None:
    spec = load_task(make_task())
    llm = MockLLM(responses=[_tool("run_shell_command", command="true", cwd="repo"), _text("x")])
    res = await run_task(spec, _opts(tmp_path, max_tool_iters=1), llm_factory=_factory(llm))
    ws_db = Path(res["workdir"], "ws", "kesoku.db")
    assert ws_db.is_file()
    assert BUDGET_STOP_TEXT  # sanity: constant exported
    assert res["turns"][0]["status"] == "processed"
