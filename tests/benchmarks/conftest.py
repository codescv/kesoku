"""Shared fixtures for kesoku-bench harness tests: a tiny synthetic task built in tmp_path."""

import os
import textwrap
from collections.abc import Callable, Generator
from pathlib import Path

import pytest

TASK_TOML = '''
[task]
id = "{task_id}"
suite = "{suite}"
title = "synthetic calc task"
config_profile = "{profile}"
force_compact_before_turn = {force}
turn_timeout_s = 60
{extra}

{turns}

[grading]
fail_to_pass = ["tests/test_holdout_calc.py::test_add"]
pass_to_pass = ["tests/test_basic.py"]
{behavioral}
'''

GRADE_PY = '''
def grade(ctx, gold_mode=False):
    src = ctx.read("calc.py") or ""
    checks = {"uses_plus": "a + b" in src}
    if gold_mode:
        checks["live_only"] = None
    else:
        checks["live_only"] = bool(ctx.turns)
    return {"checks": checks, "diagnostics": {"n_events": len(ctx.tool_events)}}


def pre_cleanup(ctx):
    return {"jobs": len(ctx.extras.get("active_jobs_for_session", []))}
'''


def write_task(
    root: Path,
    task_id: str = "A9_synthetic",
    suite: str = "A",
    profile: str = "default",
    turns: list[tuple[str, str]] | None = None,
    force: list[int] | None = None,
    with_grade: bool = False,
    gold_ok: bool = True,
    bug: str = "a - b",
    extra: str = "",
) -> Path:
    """Create a synthetic task directory and return it."""
    task_dir = root / task_id
    (task_dir / "repo" / "tests").mkdir(parents=True)
    (task_dir / "holdout" / "tests").mkdir(parents=True)
    (task_dir / "gold").mkdir()
    turns = turns or [("user", "Fix add() in calc.py.")]
    turns_toml = "\n".join(f'[[turns]]\nkind = "{k}"\ntext = """{t}"""\n' for k, t in turns)
    behavioral = 'behavioral = "grade.py"\nrequired_behavioral = ["uses_plus"]' if with_grade else ""
    (task_dir / "task.toml").write_text(
        TASK_TOML.format(task_id=task_id, suite=suite, profile=profile, force=force or [], turns=turns_toml,
                         behavioral=behavioral, extra=extra),
        encoding="utf-8",
    )
    repo = task_dir / "repo"
    (repo / "pyproject.toml").write_text(textwrap.dedent('''
        [project]
        name = "calc"
        version = "0.1.0"
        requires-python = ">=3.12"
        dependencies = []

        [dependency-groups]
        dev = ["pytest>=8"]

        [tool.pytest.ini_options]
        pythonpath = ["."]
    '''), encoding="utf-8")
    (repo / ".gitignore").write_text(".venv/\n__pycache__/\n", encoding="utf-8")
    (repo / "README.md").write_text("calc\n", encoding="utf-8")
    (repo / "calc.py").write_text(f"def add(a, b):\n    return {bug}\n\n\ndef mul(a, b):\n    return a * b\n",
                                  encoding="utf-8")
    (repo / "tests" / "test_basic.py").write_text(
        "from calc import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 6\n", encoding="utf-8")
    (task_dir / "holdout" / "tests" / "test_holdout_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n", encoding="utf-8")
    fixed = "a + b" if gold_ok else "a - b"
    (task_dir / "gold" / "calc.py").write_text(
        f"def add(a, b):\n    return {fixed}\n\n\ndef mul(a, b):\n    return a * b\n", encoding="utf-8")
    if with_grade:
        (task_dir / "grade.py").write_text(GRADE_PY, encoding="utf-8")
    return task_dir


@pytest.fixture
def make_task(tmp_path: Path) -> Callable[..., Path]:
    """Factory building synthetic tasks under tmp_path/tasks."""
    root = tmp_path / "tasks"
    root.mkdir(exist_ok=True)

    def _make(**kwargs: object) -> Path:
        return write_task(root, **kwargs)  # type: ignore[arg-type]

    return _make


@pytest.fixture
def restore_environ() -> Generator[None, None, None]:
    """Snapshot and restore os.environ (run_task sanitizes it in place)."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)
