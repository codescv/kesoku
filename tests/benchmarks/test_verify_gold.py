"""verify-gold tests: synthetic positive/negative fixtures plus the tester's gate over every shipped task."""

from collections.abc import Callable
from pathlib import Path

import pytest

from benchmarks.kesoku_bench.harness.taskspec import discover_task_dirs, load_task
from benchmarks.kesoku_bench.harness.verify_gold import format_reports, reports_to_json, verify_task

REAL_TASK_DIRS = discover_task_dirs()


def test_verify_gold_synthetic_ok(make_task: Callable[..., Path]) -> None:
    report = verify_task(load_task(make_task(with_grade=True)))
    assert report.ok, report.errors
    assert report.steps["before_gold"]["fail_to_pass"] == {"tests/test_holdout_calc.py::test_add": False}
    assert report.steps["after_gold"]["fail_to_pass"] == {"tests/test_holdout_calc.py::test_add": True}
    assert report.steps["behavioral_gold"]["checks"] == {"uses_plus": True, "live_only": None}
    assert "OK" in format_reports([report])
    assert reports_to_json([report])[0]["ok"] is True


def test_verify_gold_detects_f2p_passing_before_gold(make_task: Callable[..., Path]) -> None:
    report = verify_task(load_task(make_task(bug="a + b")))
    assert not report.ok
    assert any("already pass before gold" in e for e in report.errors)


def test_verify_gold_detects_broken_gold(make_task: Callable[..., Path]) -> None:
    report = verify_task(load_task(make_task(gold_ok=False)))
    assert not report.ok
    assert any("not passing after gold" in e for e in report.errors)


def test_verify_gold_detects_failing_visible_tests(make_task: Callable[..., Path]) -> None:
    task_dir = make_task()
    (task_dir / "repo" / "tests" / "test_basic.py").write_text("def test_mul():\n    assert False\n")
    report = verify_task(load_task(task_dir))
    assert not report.ok
    assert any("visible tests do not pass" in e for e in report.errors)


@pytest.mark.slow
@pytest.mark.parametrize("task_dir", REAL_TASK_DIRS, ids=[p.name for p in REAL_TASK_DIRS])
def test_verify_gold_real_task(task_dir: Path) -> None:
    """The tester's gate: every shipped task passes the $0 gold check."""
    report = verify_task(load_task(task_dir))
    assert report.ok, format_reports([report])
