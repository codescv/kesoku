"""Fixture materialisation: per-task workspace (AWD), target repo, held-out tree, rendered config.toml."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.kesoku_bench.harness.taskspec import COMMON_DIR, TaskSpec
from kesoku.testing.fixture import (
    copy_tree as _copy_tree,
)
from kesoku.testing.fixture import (
    default_holdout_root,
    free_port,
    git_init,
    sanitize_process_env,
    sanitized_env,
    substitute,
    uv_sync,
)
from kesoku.testing.fixture import (
    render_config_dict as _base_render_config_dict,
)
from kesoku.testing.fixture import (
    render_config_toml as _base_render_config_toml,
)

CONFIG_TEMPLATE = COMMON_DIR / "config.template.toml"
DISTRACTOR_SKILLS_DIR = COMMON_DIR / "distractor_skills"
ROLE_NAME = "eval_engineer"
DEFAULT_WORK_ROOT = Path("/tmp/kesoku-bench")

__all__ = [
    "CONFIG_TEMPLATE",
    "DEFAULT_WORK_ROOT",
    "DISTRACTOR_SKILLS_DIR",
    "ROLE_NAME",
    "Fixture",
    "copy_skills",
    "default_holdout_root",
    "free_port",
    "git_init",
    "materialize",
    "materialize_grading_tests",
    "overlay_gold",
    "render_config_dict",
    "render_config_toml",
    "restore_visible_tests",
    "sanitize_process_env",
    "sanitized_env",
    "substitute",
    "uv_sync",
]


def render_config_dict(
    profile: str,
    overrides: dict[str, Any] | None = None,
    *,
    thinking_level: str | None = None,
    model_name: str | None = None,
    template_path: Path = CONFIG_TEMPLATE,
) -> dict[str, Any]:
    """Build the config dict for a benchmark profile."""
    return _base_render_config_dict(
        profile,
        overrides,
        thinking_level=thinking_level,
        model_name=model_name,
        template_path=template_path,
    )


def render_config_toml(profile: str, overrides: dict[str, Any] | None = None, **kwargs: Any) -> str:
    """Render the config for a benchmark profile as TOML text."""
    kwargs.setdefault("template_path", CONFIG_TEMPLATE)
    return _base_render_config_toml(profile, overrides, **kwargs)


@dataclass
class Fixture:
    """Paths of one materialised task instance."""

    task_id: str
    work_dir: Path
    ws_dir: Path
    repo_dir: Path
    holdout_dir: Path
    config_path: Path
    port: int
    profile: str
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def pristine_tests_dir(self) -> Path:
        """Pristine copy of the shipped visible tests (outside the AWD)."""
        return self.holdout_dir / "visible_tests_pristine"

    @property
    def grading_dir(self) -> Path:
        """Where held-out tests are staged at grading time."""
        return self.holdout_dir / "grading"


def copy_skills(spec: TaskSpec, skills_dst: Path, distractors_dir: Path = DISTRACTOR_SKILLS_DIR) -> list[str]:
    """Copy task skills + shared distractor skills (Suite D only) into the workspace skills dir."""
    skills_dst.mkdir(parents=True, exist_ok=True)
    if spec.suite != "D":
        return []
    installed: list[str] = []
    for src_root in (spec.skills_dir, distractors_dir):
        if not src_root.is_dir():
            continue
        for skill in sorted(p for p in src_root.iterdir() if p.is_dir()):
            _copy_tree(skill, skills_dst / skill.name)
            installed.append(skill.name)
    return installed


def materialize(
    spec: TaskSpec,
    work_root: Path,
    run_id: str,
    holdout_root: Path,
    *,
    profile: str | None = None,
    extra_overrides: dict[str, Any] | None = None,
    thinking_level: str | None = None,
    model_name: str | None = None,
    prewarm: bool = True,
    port: int | None = None,
) -> Fixture:
    """Materialise `<work_root>/<run_id>/<task_id>/ws` and the held-out tree for one task."""
    work_root = Path(work_root).resolve()
    holdout_root = Path(holdout_root).resolve()
    if holdout_root == work_root or work_root in holdout_root.parents:
        raise ValueError("holdout_root must not live under work_root")
    work_dir = work_root / run_id / spec.id
    if work_dir.exists():
        shutil.rmtree(work_dir)
    ws_dir = work_dir / "ws"
    repo_dir = ws_dir / "repo"
    ws_dir.mkdir(parents=True)
    for sub in ("sessions", "skills", "roles"):
        (ws_dir / sub).mkdir()

    _copy_tree(COMMON_DIR / "roles" / ROLE_NAME, ws_dir / "roles" / ROLE_NAME)
    _copy_tree(spec.repo_dir, repo_dir)
    sync_rc: int | None = None
    sync_err = ""
    if prewarm:
        res = uv_sync(repo_dir)
        sync_rc, sync_err = res.returncode, res.stderr[-2000:]
    git_init(repo_dir)

    skills = copy_skills(spec, ws_dir / "skills")

    chosen_profile = profile or spec.task.config_profile
    overrides = {**spec.task.config_overrides, **(extra_overrides or {})}
    config_path = ws_dir / "config.toml"
    config_path.write_text(
        render_config_toml(chosen_profile, overrides, thinking_level=thinking_level, model_name=model_name),
        encoding="utf-8",
    )

    holdout_dir = holdout_root / run_id / spec.id
    if holdout_dir.exists():
        shutil.rmtree(holdout_dir)
    holdout_dir.mkdir(parents=True)
    fixture = Fixture(
        task_id=spec.id,
        work_dir=work_dir,
        ws_dir=ws_dir,
        repo_dir=repo_dir,
        holdout_dir=holdout_dir,
        config_path=config_path,
        port=port or free_port(),
        profile=chosen_profile,
        extras={"skills": skills},
    )
    if (repo_dir / "tests").is_dir():
        _copy_tree(repo_dir / "tests", fixture.pristine_tests_dir)

    if prewarm:
        fixture.extras["uv_sync_rc"] = sync_rc
        if sync_rc:
            fixture.extras["uv_sync_stderr"] = sync_err
    return fixture


def restore_visible_tests(repo_dir: Path, pristine_dir: Path) -> list[str]:
    """Restore shipped visible test files by path (agent-authored new files are left alone)."""
    restored: list[str] = []
    if not pristine_dir.is_dir():
        return restored
    for src in sorted(pristine_dir.rglob("*")):
        if not src.is_file() or "__pycache__" in src.parts:
            continue
        rel = src.relative_to(pristine_dir)
        dst = repo_dir / "tests" / rel
        if not dst.is_file() or dst.read_bytes() != src.read_bytes():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            restored.append(str(Path("tests") / rel))
    return restored


def materialize_grading_tests(spec: TaskSpec, repo_dir: Path, staging_dir: Path | None = None) -> list[str]:
    """Copy held-out tests (+ `holdout/ref`) straight from the task dir into `repo/tests/`."""
    copied: list[str] = []
    tests_dst = repo_dir / "tests"
    tests_dst.mkdir(parents=True, exist_ok=True)
    for sub in ("tests", "ref"):
        src_root = spec.holdout_dir / sub
        if not src_root.is_dir():
            continue
        if staging_dir is not None:
            _copy_tree(src_root, staging_dir / sub)
        for src in sorted(src_root.rglob("*")):
            if not src.is_file() or "__pycache__" in src.parts:
                continue
            rel = src.relative_to(src_root)
            dst = tests_dst / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(str(Path("tests") / rel))
    return copied


def overlay_gold(spec: TaskSpec, repo_dir: Path) -> None:
    """Overlay `gold/` onto the repo (skipping `gold/apply.py`) and run `gold/apply.py` if present."""
    gold = spec.gold_dir
    if not gold.is_dir():
        raise FileNotFoundError(f"{spec.id}: missing gold/ directory")
    for src in sorted(gold.rglob("*")):
        if not src.is_file() or "__pycache__" in src.parts:
            continue
        rel = src.relative_to(gold)
        if rel == Path("apply.py"):
            continue
        dst = repo_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    apply = gold / "apply.py"
    if apply.is_file():
        res = subprocess.run(
            [sys.executable, str(apply), str(repo_dir)],
            cwd=repo_dir,
            env=sanitized_env(),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if res.returncode != 0:
            raise RuntimeError(f"{spec.id}: gold/apply.py failed ({res.returncode}): {res.stderr[-2000:]}")
