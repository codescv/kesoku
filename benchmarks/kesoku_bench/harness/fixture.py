"""Fixture materialisation: per-task workspace (AWD), target repo, held-out tree, rendered config.toml."""

from __future__ import annotations

import copy
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli_w

from benchmarks.kesoku_bench.harness.taskspec import COMMON_DIR, TaskSpec

CONFIG_TEMPLATE = COMMON_DIR / "config.template.toml"
DISTRACTOR_SKILLS_DIR = COMMON_DIR / "distractor_skills"
ROLE_NAME = "eval_engineer"
DEFAULT_WORK_ROOT = Path("/tmp/kesoku-bench")

# Environment variables of the harness's own uv/venv that must not leak into the agent's shell or the
# task repo's `uv run` (they point into the kesoku worktree and would confuse uv).
_LEAKY_ENV_VARS = ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT", "PYTEST_ADDOPTS",
                   "PYTEST_CURRENT_TEST", "UV_RUN_RECURSION_DEPTH")


@dataclass
class Fixture:
    """Paths of one materialised task instance."""

    task_id: str
    work_dir: Path  # <work_root>/<run_id>/<task_id>
    ws_dir: Path  # AWD (holds config.toml)
    repo_dir: Path  # ws/repo
    holdout_dir: Path  # <holdout_root>/<run_id>/<task_id>
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


# --- environment -----------------------------------------------------------------------------------


def sanitized_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return a copy of the environment without the harness's venv/pytest variables.

    PATH entries inside the harness's virtualenv are dropped as well so that `python`/`pytest` inside the
    task repo never resolve to the kesoku environment.
    """
    env = dict(os.environ if base is None else base)
    venv = env.get("VIRTUAL_ENV")
    for key in _LEAKY_ENV_VARS:
        env.pop(key, None)
    if venv:
        venv_bin = os.path.join(venv, "bin")
        env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep) if p and p != venv_bin)
    return env


def sanitize_process_env() -> None:
    """Apply `sanitized_env()` to `os.environ` in place (the agent's shell inherits os.environ)."""
    clean = sanitized_env()
    for key in list(os.environ):
        if key not in clean:
            del os.environ[key]
    os.environ.update(clean)


# --- config rendering ------------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overlay into a copy of base."""
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    """Set `a.b.c = value` in a nested dict, creating tables as needed."""
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot set {dotted!r}: {part!r} is not a table")
    node[parts[-1]] = value


def render_config_dict(
    profile: str,
    overrides: dict[str, Any] | None = None,
    *,
    thinking_level: str | None = None,
    model_name: str | None = None,
    template_path: Path = CONFIG_TEMPLATE,
) -> dict[str, Any]:
    """Build the config dict for a profile.

    Args:
        profile: One of the `[profiles.*]` names in the template.
        overrides: Dotted-key overrides (e.g. {"shell.background_threshold_seconds": 8.0}).
        thinking_level: Optional `gemini.thinking_level` override.
        model_name: Optional model override (applied to model_name and context_model_name).
        template_path: Template file.

    Returns:
        A dict ready for `KesokuConfig.model_validate` / TOML serialisation.
    """
    with open(template_path, "rb") as f:
        template = tomllib.load(f)
    profiles = template.pop("profiles", {})
    if profile not in profiles:
        raise ValueError(f"unknown config profile {profile!r}; known: {sorted(profiles)}")
    data = _deep_merge(template, profiles[profile] or {})
    for key, value in (overrides or {}).items():
        _set_dotted(data, key, value)
    if thinking_level:
        data.setdefault("gemini", {})["thinking_level"] = thinking_level
    if model_name:
        data.setdefault("gemini", {})["model_name"] = model_name
        data["gemini"]["context_model_name"] = model_name
    return data


def render_config_toml(profile: str, overrides: dict[str, Any] | None = None, **kwargs: Any) -> str:
    """Render the config for a profile as TOML text."""
    return tomli_w.dumps(render_config_dict(profile, overrides, **kwargs))


# --- helpers ---------------------------------------------------------------------------------------


def free_port() -> int:
    """Pick a currently free TCP port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def substitute(text: str, port: int, awd: Path | str) -> str:
    """Replace `{PORT}` and `{AWD}` placeholders in a turn text."""
    return text.replace("{PORT}", str(port)).replace("{AWD}", str(awd))


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy a directory tree (merging into dst), skipping caches and virtualenvs."""
    shutil.copytree(
        src,
        dst,
        dirs_exist_ok=True,
        symlinks=False,
        ignore=shutil.ignore_patterns(".venv", "__pycache__", ".pytest_cache", "*.pyc", ".ruff_cache"),
    )


def git_init(repo_dir: Path) -> None:
    """Initialise a git repository with a single `fixture` commit."""
    env = sanitized_env()
    env.update({
        "GIT_AUTHOR_NAME": "kesoku-bench", "GIT_AUTHOR_EMAIL": "bench@example.com",
        "GIT_COMMITTER_NAME": "kesoku-bench", "GIT_COMMITTER_EMAIL": "bench@example.com",
    })
    for cmd in (["git", "init", "-q", "-b", "main"], ["git", "add", "-A"],
                ["git", "commit", "-q", "--no-gpg-sign", "-m", "fixture"]):
        subprocess.run(cmd, cwd=repo_dir, env=env, check=True, capture_output=True, timeout=60)


def uv_sync(repo_dir: Path, timeout: float = 300.0) -> subprocess.CompletedProcess[str]:
    """Pre-warm the task repo's virtualenv so the agent's first `uv run pytest` is instant."""
    return subprocess.run(
        ["uv", "sync", "--quiet"], cwd=repo_dir, env=sanitized_env(), capture_output=True, text=True,
        timeout=timeout, check=False,
    )


def default_holdout_root() -> Path:
    """Create a fresh holdout root outside any work root (XDG_RUNTIME_DIR or the system temp dir)."""
    base = os.environ.get("XDG_RUNTIME_DIR")
    if not base or not os.path.isdir(base) or not os.access(base, os.W_OK):
        base = tempfile.gettempdir()
    return Path(tempfile.mkdtemp(prefix="kb-holdout-", dir=base))


def copy_skills(spec: TaskSpec, skills_dst: Path, distractors_dir: Path = DISTRACTOR_SKILLS_DIR) -> list[str]:
    """Copy task skills + shared distractor skills (Suite D only) into the workspace skills dir.

    Returns:
        Names of the installed skills.
    """
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


# --- materialisation -------------------------------------------------------------------------------


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
    """Materialise `<work_root>/<run_id>/<task_id>/ws` and the held-out tree for one task.

    Args:
        spec: Task specification.
        work_root: Root for run trees (default /tmp/kesoku-bench).
        run_id: Run identifier (unique per repeat).
        holdout_root: Root for held-out material, never under work_root.
        profile: Optional config-profile override (e.g. `compaction_small` for --natural-compaction).
        extra_overrides: Additional dotted config overrides applied after the task's own.
        thinking_level: Optional gemini thinking level override.
        model_name: Optional gemini model override.
        prewarm: Run `uv sync` in the repo.
        port: Optional fixed port (else a free one is picked).

    Returns:
        The Fixture.
    """
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

    # 1. role + repo (+ uv pre-warm, + git)
    _copy_tree(COMMON_DIR / "roles" / ROLE_NAME, ws_dir / "roles" / ROLE_NAME)
    _copy_tree(spec.repo_dir, repo_dir)
    sync_rc: int | None = None
    sync_err = ""
    if prewarm:
        # Pre-warm before `git init` so a generated uv.lock is part of the fixture commit (clean diff).
        res = uv_sync(repo_dir)
        sync_rc, sync_err = res.returncode, res.stderr[-2000:]
    git_init(repo_dir)

    # 2. skills (Suite D)
    skills = copy_skills(spec, ws_dir / "skills")

    # 3. config.toml
    chosen_profile = profile or spec.task.config_profile
    overrides = {**spec.task.config_overrides, **(extra_overrides or {})}
    config_path = ws_dir / "config.toml"
    config_path.write_text(
        render_config_toml(chosen_profile, overrides, thinking_level=thinking_level, model_name=model_name),
        encoding="utf-8",
    )

    # 4. pristine visible tests outside the AWD
    holdout_dir = holdout_root / run_id / spec.id
    if holdout_dir.exists():
        shutil.rmtree(holdout_dir)
    holdout_dir.mkdir(parents=True)
    fixture = Fixture(
        task_id=spec.id, work_dir=work_dir, ws_dir=ws_dir, repo_dir=repo_dir, holdout_dir=holdout_dir,
        config_path=config_path, port=port or free_port(), profile=chosen_profile, extras={"skills": skills},
    )
    if (repo_dir / "tests").is_dir():
        _copy_tree(repo_dir / "tests", fixture.pristine_tests_dir)

    if prewarm:
        fixture.extras["uv_sync_rc"] = sync_rc
        if sync_rc:
            fixture.extras["uv_sync_stderr"] = sync_err
    return fixture


def restore_visible_tests(repo_dir: Path, pristine_dir: Path) -> list[str]:
    """Restore shipped visible test files by path (agent-authored new files are left alone).

    Returns:
        Repo-relative paths that were restored because they differed or were missing.
    """
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
    """Copy held-out tests (+ holdout/ref) straight from the task dir into `repo/tests/`.

    Args:
        spec: Task specification.
        repo_dir: Target repo.
        staging_dir: Optional out-of-AWD staging copy (kept for post-mortems).

    Returns:
        Repo-relative paths of the copied files.
    """
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
    """Overlay `gold/` onto the repo (skipping `gold/apply.py`) and run `gold/apply.py` if present.

    Raises:
        RuntimeError: If apply.py fails.
    """
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
            cwd=repo_dir, env=sanitized_env(), capture_output=True, text=True, timeout=300, check=False,
        )
        if res.returncode != 0:
            raise RuntimeError(f"{spec.id}: gold/apply.py failed ({res.returncode}): {res.stderr[-2000:]}")
