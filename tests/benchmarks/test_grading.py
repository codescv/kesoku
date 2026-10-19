"""Tests for junit parsing, id evaluation, tool-event extraction, tamper audit and grader integration."""

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.kesoku_bench.harness import grading as gr
from benchmarks.kesoku_bench.harness.fixture import materialize_grading_tests, overlay_gold
from benchmarks.kesoku_bench.harness.taskspec import load_task

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="1" failures="1" skipped="1" tests="6" time="1.25">
<testcase classname="tests.test_a" name="test_ok" file="tests/test_a.py" line="1" time="0.01"/>
<testcase classname="tests.test_a" name="test_bad" file="tests/test_a.py" line="5" time="0.01">
  <failure message="assert 1 == 2">trace</failure></testcase>
<testcase classname="tests.test_b.TestX" name="test_m[1]" file="tests/test_b.py" line="3" time="0.01"/>
<testcase classname="tests.test_b.TestX" name="test_m[2]" file="tests/test_b.py" line="3" time="0.01"/>
<testcase classname="tests.integration.test_live" name="test_live" file="tests/integration/test_live.py"
  line="1" time="0.0"><skipped message="needs MOCKRPC_URL"/></testcase>
<testcase classname="" name="tests.test_broken" time="0.0"><error message="collection failure">ImportError</error>
</testcase>
</testsuite></testsuites>
"""


@pytest.fixture
def junit_file(tmp_path: Path) -> Path:
    path = tmp_path / "junit.xml"
    path.write_text(JUNIT)
    return path


def test_parse_junit_nodeids(junit_file: Path) -> None:
    cases, tests_time = gr.parse_junit(junit_file)
    ids = {c.nodeid: c.outcome for c in cases}
    assert ids["tests/test_a.py::test_ok"] == "passed"
    assert ids["tests/test_a.py::test_bad"] == "failed"
    assert ids["tests/test_b.py::TestX::test_m[1]"] == "passed"
    assert ids["tests/integration/test_live.py::test_live"] == "skipped"
    assert tests_time == pytest.approx(1.25)


def test_parse_missing_or_corrupt(tmp_path: Path) -> None:
    assert gr.parse_junit(tmp_path / "nope.xml") == ([], 0.0)
    bad = tmp_path / "bad.xml"
    bad.write_text("<not-closed")
    assert gr.parse_junit(bad) == ([], 0.0)


def test_evaluate_semantics(junit_file: Path) -> None:
    cases, t = gr.parse_junit(junit_file)
    run = gr.PytestRun(returncode=1, cases=cases, duration_s=1.0, tests_time_s=t, tail="")
    res = run.evaluate([
        "tests/test_a.py::test_ok", "tests/test_a.py::test_bad", "tests/test_a.py",
        "tests/test_b.py::TestX::test_m", "tests/test_b.py::TestX", "tests/test_broken.py",
        "tests/test_missing.py::test_x", "tests/integration/test_live.py",
    ])
    assert res["tests/test_a.py::test_ok"] is True
    assert res["tests/test_a.py::test_bad"] is False
    assert res["tests/test_a.py"] is False  # contains a failure
    assert res["tests/test_b.py::TestX::test_m"] is True  # parametrised cases
    assert res["tests/test_b.py::TestX"] is True
    assert res["tests/test_broken.py"] is False  # collection error -> no matching case
    assert res["tests/test_missing.py::test_x"] is False
    assert res["tests/integration/test_live.py"] is False  # skipped only: not a pass for F2P
    p2p = run.evaluate(["tests/integration/test_live.py"], require_pass=False)
    assert p2p["tests/integration/test_live.py"] is True  # skips are fine for P2P
    assert run.summary() == {"passed": 3, "failed": 1, "skipped": 1, "error": 1}


def _msg(mid: str, mtype: str, ts: float, meta: dict, parent: str | None = None, turn: int = 1) -> dict:
    return {"id": mid, "role": "tool", "type": mtype, "content": "", "metadata": meta, "timestamp": ts,
            "status": "responded", "parent_id": parent, "sender": "x", "summary_node_id": None, "turn_idx": turn}


def test_extract_tool_events_and_tamper_audit() -> None:
    msgs = [
        _msg("c1", "tool_call", 1.0, {"tool_name": "run_shell_command", "tool_arguments": {"command": "ls"}}),
        _msg("r1", "tool_result", 1.1, {"tool_name": "run_shell_command", "tool_result": "a b"}, parent="c1"),
        _msg("c2", "tool_call", 2.0, {"tool_name": "update_file",
                                      "tool_arguments": {"file_name": "repo/x.py", "old_content": "q"}}, turn=2),
        _msg("r2", "tool_result", 2.1, {"tool_name": "update_file",
                                        "tool_result": "Error: old_content not found in file 'x'."}, parent="c2"),
        _msg("c3", "tool_call", 3.0, {"tool_name": "run_shell_command",
                                      "tool_arguments": {"command": "cat /x/kesoku_bench/tasks/A1/gold/f.py"}}),
        _msg("r3", "tool_result", 3.1, {"tool_name": "run_shell_command", "tool_error": "boom"}, parent="c3"),
        _msg("c4", "tool_call", 4.0, {"tool_name": "list_skills", "tool_arguments": {}}),
    ]
    events = gr.extract_tool_events(msgs)
    assert [e["name"] for e in events] == ["run_shell_command", "update_file", "run_shell_command", "list_skills"]
    assert events[0]["result"] == "a b" and not events[0]["is_error"]
    assert events[1]["is_error"] and events[1]["turn_idx"] == 2
    assert events[2]["is_error"] and events[2]["result"] == "boom"
    assert events[3]["has_result"] is False
    hits = gr.tamper_audit(events)
    assert len(hits) == 1 and "kesoku_bench" in hits[0]


def test_turn_index_for() -> None:
    windows = [(1, 10.0), (2, 20.0), (3, 30.0)]
    assert gr.turn_index_for(5.0, windows) == 0
    assert gr.turn_index_for(10.0, windows) == 1
    assert gr.turn_index_for(25.0, windows) == 2
    assert gr.turn_index_for(99.0, windows) == 3


def test_normalize_outcome_variants() -> None:
    o = gr.normalize_outcome({"checks": {"a": True}, "diagnostics": {"d": 1}})
    assert o.checks == {"a": True} and o.diagnostics == {"d": 1}
    o = gr.normalize_outcome({"a": True, "b": None, "n": 3})
    assert o.checks == {"a": True, "b": None} and o.diagnostics == {"n": 3}
    o = gr.normalize_outcome(SimpleNamespace(checks={"x": False}, diagnostics={}))
    assert o.checks == {"x": False}
    o = gr.normalize_outcome(gr.GradeOutcome(checks={"y": True}))
    assert o.checks == {"y": True}
    assert gr.normalize_outcome(None).checks == {}
    with pytest.raises(TypeError):
        gr.normalize_outcome(42)


def test_required_checks_pass() -> None:
    o = gr.GradeOutcome(checks={"a": True, "b": None, "c": False})
    assert gr.required_checks_pass(o, ["a"])
    assert not gr.required_checks_pass(o, ["a", "b"])
    assert gr.required_checks_pass(o, ["a", "b"], gold_mode=True)
    assert not gr.required_checks_pass(o, ["c"], gold_mode=True)
    assert not gr.required_checks_pass(o, ["missing"])


def test_grade_module_load_run_and_crash(tmp_path: Path) -> None:
    good = tmp_path / "t1" / "grade.py"
    good.parent.mkdir()
    good.write_text("def grade(ctx, gold_mode=False):\n    return {'ok': ctx.read('f.txt') == 'hi', 'gm': gold_mode}\n"
                    "def pre_cleanup(ctx):\n    return {'port': ctx.extras['port']}\n")
    (tmp_path / "f.txt").write_text("hi")
    ctx = gr.GradeContext(task_id="t1", task_dir=good.parent, repo_dir=tmp_path, gold_mode=True,
                          extras={"port": 99})
    mod = gr.load_grade_module(good)
    out = gr.run_behavioral(mod, ctx)
    assert out.checks == {"ok": True, "gm": True}
    assert gr.run_pre_cleanup(mod, ctx) == {"port": 99}
    assert gr.run_pre_cleanup(None, ctx) == {}
    bad = tmp_path / "t2" / "grade.py"
    bad.parent.mkdir()
    bad.write_text("def grade(ctx, gold_mode=False):\n    raise RuntimeError('nope')\n")
    out = gr.run_behavioral(gr.load_grade_module(bad), ctx)
    assert out.checks == {"grader_error": False} and "nope" in out.diagnostics["grader_exception"]


def test_context_helpers(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    ctx = gr.GradeContext(task_id="t", task_dir=tmp_path, repo_dir=tmp_path)
    assert ctx.read("a.txt") == "x"
    assert ctx.read("missing.txt") is None
    assert ctx.run(["ls"]).returncode == 0


def test_run_pytest_real_repo(make_task: Callable[..., Path], tmp_path: Path) -> None:
    """End-to-end: F2P fails before gold, passes after; collection errors don't hide other ids."""
    spec = load_task(make_task())
    repo = tmp_path / "repo"
    shutil.copytree(spec.repo_dir, repo)
    subprocess.run(["uv", "sync", "--quiet"], cwd=repo, check=True, capture_output=True)
    materialize_grading_tests(spec, repo)
    (repo / "tests" / "test_holdout_broken.py").write_text("import does_not_exist\n")
    ids = [*spec.grading.fail_to_pass, *spec.grading.pass_to_pass, "tests/test_holdout_broken.py"]
    run = gr.run_pytest(repo, ids, tmp_path / "j1.xml")
    res = run.evaluate(ids)
    assert res == {"tests/test_holdout_calc.py::test_add": False, "tests/test_basic.py": True,
                   "tests/test_holdout_broken.py": False}
    overlay_gold(spec, repo)
    run2 = gr.run_pytest(repo, spec.grading.fail_to_pass, tmp_path / "j2.xml")
    assert run2.evaluate(spec.grading.fail_to_pass) == {"tests/test_holdout_calc.py::test_add": True}
    assert run2.returncode == 0
