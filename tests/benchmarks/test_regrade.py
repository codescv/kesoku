"""Tests for $0 re-grading of saved TaskResults (grader fixed after a live run)."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from benchmarks.kesoku_bench.harness.regrade import collect_result_files, regrade_result
from benchmarks.kesoku_bench.harness.runner import RunOptions, run_task
from benchmarks.kesoku_bench.harness.taskspec import load_task
from kesoku.agent.llm import LLMResponse, MockLLM, ToolCallRequest


def _opts(tmp_path: Path) -> RunOptions:
    return RunOptions(stage="test", run_id="r1", work_root=tmp_path / "work", holdout_root=tmp_path / "holdout",
                      results_dir=tmp_path / "results", final_settle_s=0.0, idle_stable_s=0.3)


@pytest.mark.asyncio
@pytest.mark.parametrize("drop_timestamps", [False, True])
async def test_regrade_recomputes_behavioral(make_task: Callable[..., Path], tmp_path: Path,
                                             restore_environ: None, drop_timestamps: bool) -> None:
    task_dir = make_task(with_grade=True, turns=[("user", "Look around."), ("user", "Fix add() in calc.py.")])
    spec = load_task(task_dir)
    llm = MockLLM(responses=[
        LLMResponse(content="Looked."),
        LLMResponse(content="", tool_calls=[ToolCallRequest(name="update_file", arguments={
            "file_name": "repo/calc.py", "old_content": "return a - b", "new_content": "return a + b"})]),
        LLMResponse(content="Fixed."),
    ])
    res = await run_task(spec, _opts(tmp_path), llm_factory=lambda cfg: llm)
    assert res["status"] == "resolved"
    # Simulate a buggy grader having produced a failing required check.
    res["grading"]["behavioral"] = {"uses_plus": False, "live_only": True}
    res["resolved"], res["status"] = False, "failed"
    if drop_timestamps:  # results written before begin_ts was recorded: windows rebuilt from the DB
        for t in res["turns"]:
            t.pop("begin_ts", None)
    path = tmp_path / "results" / "A9_synthetic.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(res))

    out = regrade_result(path, tasks_root=task_dir.parent)
    assert out["grading"]["behavioral"] == {"uses_plus": True, "live_only": True}
    assert out["status"] == "resolved" and out["resolved"] is True
    assert out["regraded"]["previous"]["status"] == "failed"
    assert out["diagnostics"]["n_events"] == 1
    assert json.loads(path.read_text())["status"] == "resolved"
    assert json.loads(path.with_name(path.name + ".orig").read_text())["status"] == "failed"


def test_collect_result_files(tmp_path: Path) -> None:
    run = tmp_path / "baseline" / "r1"
    run.mkdir(parents=True)
    for name in ("C3_multi.json", "A1_x.json", "scorecard.json", "C3_multi.json.orig"):
        (run / name).write_text("{}")
    assert [p.name for p in collect_result_files([tmp_path])] == ["A1_x.json", "C3_multi.json"]
    assert [p.name for p in collect_result_files([run], ["C3"])] == ["C3_multi.json"]
