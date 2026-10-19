"""Tests for task.toml loading/validation and for the structure of every shipped task directory."""

from collections.abc import Callable
from pathlib import Path

import pytest

from benchmarks.kesoku_bench.harness.taskspec import (
    TASKS_DIR,
    discover_task_dirs,
    load_task,
    select_tasks,
)

REAL_TASK_DIRS = discover_task_dirs()


def test_load_synthetic_task(make_task: Callable[..., Path]) -> None:
    task_dir = make_task(turns=[("user", "a"), ("user", "b"), ("user", "c"), ("user", "d")], force=[4])
    spec = load_task(task_dir)
    assert spec.id == "A9_synthetic"
    assert spec.suite == "A"
    assert spec.repo_dir == task_dir / "repo"
    assert spec.task.force_compact_before_turn == [4]
    assert spec.is_multi_turn
    assert spec.grade_path is None


def test_id_must_match_directory(make_task: Callable[..., Path]) -> None:
    task_dir = make_task()
    renamed = task_dir.parent / "A8_other"
    task_dir.rename(renamed)
    with pytest.raises(ValueError, match="does not match"):
        load_task(renamed)


def test_force_point_out_of_range(make_task: Callable[..., Path]) -> None:
    with pytest.raises(ValueError, match="force_compact_before_turn"):
        load_task(make_task(turns=[("user", "a"), ("user", "b"), ("user", "c")], force=[3]))


def test_fallback_requires_command_turn(make_task: Callable[..., Path]) -> None:
    task_dir = make_task(extra="fallback_force_if_noop = true")
    with pytest.raises(ValueError, match="fallback_force_if_noop"):
        load_task(task_dir)


def test_select_tasks_by_prefix_and_suite(make_task: Callable[..., Path]) -> None:
    a = make_task(task_id="A9_synthetic")
    make_task(task_id="B9_synthetic", suite="B")
    root = a.parent
    assert [t.id for t in select_tasks(["A9"], tasks_root=root)] == ["A9_synthetic"]
    assert [t.id for t in select_tasks(suites=["b"], tasks_root=root)] == ["B9_synthetic"]
    assert len(select_tasks(tasks_root=root)) == 2
    with pytest.raises(ValueError, match="unknown task"):
        select_tasks(["Z1"], tasks_root=root)


def test_missing_task_toml(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_task(tmp_path)


@pytest.mark.parametrize("task_dir", REAL_TASK_DIRS, ids=[p.name for p in REAL_TASK_DIRS])
def test_real_task_structure(task_dir: Path) -> None:
    """Every shipped task loads and has repo/, holdout/tests, gold/, and consistent grading ids."""
    spec = load_task(task_dir)
    assert spec.task_dir.parent == TASKS_DIR
    assert (task_dir / "repo").is_dir()
    assert (task_dir / "repo" / "pyproject.toml").is_file()
    assert (task_dir / "repo" / ".gitignore").is_file()
    assert (task_dir / "holdout" / "tests").is_dir()
    assert (task_dir / "gold").is_dir()
    assert len(spec.turns) >= 1
    for node in spec.grading.fail_to_pass:
        rel = node.split("::")[0]
        assert rel.startswith("tests/"), node
        assert (task_dir / "holdout" / rel).is_file(), f"F2P file not in holdout/: {node}"
        assert not (task_dir / "repo" / rel).exists(), f"held-out test shipped in repo/: {node}"
    for node in spec.grading.pass_to_pass:
        rel = node.split("::")[0]
        assert (task_dir / "repo" / rel).exists() or (task_dir / "holdout" / rel).exists(), node
    if spec.grading.behavioral:
        assert spec.grade_path is not None and spec.grade_path.is_file()
    if spec.suite == "C":
        assert spec.task.config_profile == "compaction_forced"
        assert spec.task.force_compact_before_turn
    if spec.task.critical_turn and spec.task.force_compact_before_turn:
        assert spec.task.critical_turn <= min(spec.task.force_compact_before_turn) - 2
