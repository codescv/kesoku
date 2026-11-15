"""Environment sanitisation, config rendering, and workspace helpers for kesoku testing."""

from __future__ import annotations

import copy
import os
import shutil
import socket
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import tomli_w

DEFAULT_CONFIG_TEMPLATE = Path(__file__).with_name("config.template.toml")

# Environment variables of the test harness's own uv/venv that must not leak into isolated subprocesses.
_LEAKY_ENV_VARS = (
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "PYTHONHOME",
    "UV_PROJECT_ENVIRONMENT",
    "PYTEST_ADDOPTS",
    "PYTEST_CURRENT_TEST",
    "UV_RUN_RECURSION_DEPTH",
)


def sanitized_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return a copy of the environment without the harness's venv/pytest variables.

    PATH entries inside the harness's virtualenv are dropped as well so `python`/`pytest` inside a
    target workspace never resolve to the outer kesoku environment.
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
    """Apply `sanitized_env()` to `os.environ` in place."""
    clean = sanitized_env()
    for key in list(os.environ):
        if key not in clean:
            del os.environ[key]
    os.environ.update(clean)


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
    profile: str = "default",
    overrides: dict[str, Any] | None = None,
    *,
    thinking_level: str | None = None,
    model_name: str | None = None,
    template_path: Path = DEFAULT_CONFIG_TEMPLATE,
) -> dict[str, Any]:
    """Build the config dict for a profile.

    Args:
        profile: One of the `[profiles.*]` names in the template.
        overrides: Dotted-key overrides (e.g. `{"shell.background_threshold_seconds": 8.0}`).
        thinking_level: Optional `gemini.thinking_level` override.
        model_name: Optional model override (applied to `model_name` and `context_model_name`).
        template_path: Template TOML file path.

    Returns:
        A dict ready for `KesokuConfig.model_validate` or TOML serialisation.
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


def render_config_toml(profile: str = "default", overrides: dict[str, Any] | None = None, **kwargs: Any) -> str:
    """Render the config for a profile as TOML text."""
    return tomli_w.dumps(render_config_dict(profile, overrides, **kwargs))


def free_port() -> int:
    """Pick a currently free TCP port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def substitute(text: str, port: int, awd: Path | str) -> str:
    """Replace `{PORT}` and `{AWD}` placeholders in a turn text."""
    return text.replace("{PORT}", str(port)).replace("{AWD}", str(awd))


def copy_tree(src: Path, dst: Path) -> None:
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
    env.update(
        {
            "GIT_AUTHOR_NAME": "kesoku-bench",
            "GIT_AUTHOR_EMAIL": "bench@example.com",
            "GIT_COMMITTER_NAME": "kesoku-bench",
            "GIT_COMMITTER_EMAIL": "bench@example.com",
        }
    )
    for cmd in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "--no-gpg-sign", "-m", "fixture"],
    ):
        subprocess.run(cmd, cwd=repo_dir, env=env, check=True, capture_output=True, timeout=60)


def uv_sync(repo_dir: Path, timeout: float = 300.0) -> subprocess.CompletedProcess[str]:
    """Pre-warm the target repo's virtualenv so the agent's first `uv run pytest` is instant."""
    return subprocess.run(
        ["uv", "sync", "--quiet"],
        cwd=repo_dir,
        env=sanitized_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def default_holdout_root() -> Path:
    """Create a fresh holdout root outside any work root (XDG_RUNTIME_DIR or the system temp dir)."""
    base = os.environ.get("XDG_RUNTIME_DIR")
    if not base or not os.path.isdir(base) or not os.access(base, os.W_OK):
        base = tempfile.gettempdir()
    return Path(tempfile.mkdtemp(prefix="kb-holdout-", dir=base))
