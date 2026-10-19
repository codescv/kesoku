"""Task specification schema (`task.toml`) and task discovery for kesoku-bench."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

BENCH_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = BENCH_ROOT / "tasks"
COMMON_DIR = BENCH_ROOT / "common"
RESULTS_DIR = BENCH_ROOT / "results"

ConfigProfile = Literal["default", "compaction_forced", "compaction_small"]


class TurnSpec(BaseModel):
    """One scripted turn: a user message or a slash command."""

    model_config = ConfigDict(extra="allow")

    kind: Literal["user", "command"] = "user"
    text: str = Field(..., min_length=1)


class GradingSpec(BaseModel):
    """Grading lists: held-out FAIL_TO_PASS ids, shipped PASS_TO_PASS ids, optional behavioural grader."""

    model_config = ConfigDict(extra="allow")

    fail_to_pass: list[str] = Field(..., min_length=1)
    pass_to_pass: list[str] = Field(default_factory=list)
    behavioral: str | None = None
    required_behavioral: list[str] = Field(default_factory=list)
    pytest_timeout_s: float = 300.0


class TaskMeta(BaseModel):
    """The `[task]` table."""

    model_config = ConfigDict(extra="allow")

    id: str
    suite: Literal["A", "B", "C", "D"]
    title: str = ""
    config_profile: ConfigProfile = "default"
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    force_compact_before_turn: list[int] = Field(default_factory=list)
    critical_turn: int | None = None
    turn_timeout_s: float = 900.0
    continue_on_timeout: bool = False
    fallback_force_if_noop: bool = False


class TaskSpec(BaseModel):
    """A fully loaded and validated task definition."""

    model_config = ConfigDict(extra="allow")

    task: TaskMeta
    turns: list[TurnSpec] = Field(..., min_length=1)
    grading: GradingSpec
    _task_dir: Path | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_turn_refs(self) -> TaskSpec:
        """Check that compaction points and the critical turn refer to existing turns."""
        n = len(self.turns)
        for t in self.task.force_compact_before_turn:
            # force_compaction() before turn t sees t-1 completed turns and needs > protect_front(1) + tail(1).
            if not 4 <= t <= n:
                raise ValueError(f"force_compact_before_turn entry {t} must be within [4, {n}]")
        if self.task.critical_turn is not None and not 1 <= self.task.critical_turn <= n:
            raise ValueError(f"critical_turn {self.task.critical_turn} out of range 1..{n}")
        if self.task.fallback_force_if_noop and not any(t.kind == "command" for t in self.turns):
            raise ValueError("fallback_force_if_noop requires at least one kind='command' turn")
        if self.grading.behavioral is None and self.grading.required_behavioral:
            raise ValueError("required_behavioral needs grading.behavioral")
        return self

    # --- convenience accessors -------------------------------------------------------------------
    @property
    def id(self) -> str:
        """Task identifier."""
        return self.task.id

    @property
    def suite(self) -> str:
        """Suite letter (A-D)."""
        return self.task.suite

    @property
    def task_dir(self) -> Path:
        """Directory the task was loaded from."""
        if self._task_dir is None:
            raise ValueError("TaskSpec was not loaded from a directory")
        return self._task_dir

    @property
    def repo_dir(self) -> Path:
        """Initial target repository."""
        return self.task_dir / "repo"

    @property
    def holdout_dir(self) -> Path:
        """Held-out grading material (tests + optional ref)."""
        return self.task_dir / "holdout"

    @property
    def gold_dir(self) -> Path:
        """Reference solution overlay."""
        return self.task_dir / "gold"

    @property
    def skills_dir(self) -> Path:
        """Task-specific skills (Suite D)."""
        return self.task_dir / "skills"

    @property
    def grade_path(self) -> Path | None:
        """Path to the behavioural grader module, if any."""
        if not self.grading.behavioral:
            return None
        return self.task_dir / self.grading.behavioral

    @property
    def is_multi_turn(self) -> bool:
        """Whether the task has more than one scripted turn."""
        return len(self.turns) > 1


def load_task(task_dir: Path | str) -> TaskSpec:
    """Load and validate `<task_dir>/task.toml`.

    Args:
        task_dir: Task directory containing `task.toml`.

    Returns:
        The validated TaskSpec.

    Raises:
        FileNotFoundError: If task.toml is missing.
        ValueError: If the spec is invalid or its id does not match the directory name.
    """
    task_dir = Path(task_dir).resolve()
    path = task_dir / "task.toml"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    with open(path, "rb") as f:
        data = tomllib.load(f)
    spec = TaskSpec.model_validate(data)
    if spec.task.id != task_dir.name:
        raise ValueError(f"task id {spec.task.id!r} does not match directory name {task_dir.name!r}")
    spec._task_dir = task_dir
    return spec


def discover_task_dirs(tasks_root: Path | None = None) -> list[Path]:
    """List task directories (those containing a task.toml), sorted by name."""
    root = tasks_root or TASKS_DIR
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "task.toml").is_file())


def discover_tasks(tasks_root: Path | None = None) -> list[TaskSpec]:
    """Load every task under the tasks root."""
    return [load_task(p) for p in discover_task_dirs(tasks_root)]


def select_tasks(
    task_ids: list[str] | None = None,
    suites: list[str] | None = None,
    tasks_root: Path | None = None,
) -> list[TaskSpec]:
    """Select tasks by id (exact or prefix such as `A1`) and/or suite letter.

    Args:
        task_ids: Optional ids or id prefixes; an unknown id raises.
        suites: Optional suite letters.
        tasks_root: Optional override of the tasks directory.

    Returns:
        Selected task specs in directory order.
    """
    tasks = discover_tasks(tasks_root)
    if task_ids:
        selected: list[TaskSpec] = []
        for wanted in task_ids:
            matches = [t for t in tasks if t.id == wanted] or [t for t in tasks if t.id.split("_")[0] == wanted]
            if not matches:
                raise ValueError(f"unknown task {wanted!r}")
            selected.extend(m for m in matches if m not in selected)
        tasks = selected
    if suites:
        wanted_suites = {s.upper() for s in suites}
        tasks = [t for t in tasks if t.suite in wanted_suites]
    return tasks
