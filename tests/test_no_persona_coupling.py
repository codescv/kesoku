"""Guard test: kesoku is a generic agent engine and must not couple to any specific persona or its skills.

Persona-specific behaviour (e.g. a role's dating/roleplay/storytelling skills, planner scripts or
per-turn hooks) belongs in the agent working directory (`roles/<role>/`), never in kesoku's
src/, docs/ or tests/.

Two checks:
1. Absolute: persona-skill / engine-hook identifiers never appear anywhere in src/, docs/, tests/.
2. Incremental: lines added since the generic baseline commit contain no persona names or dating terms.
   (Older pre-existing fixtures that use a persona name as a fake role are grandfathered.)

Forbidden tokens are assembled from fragments so this file never matches itself.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from kesoku.config import AgentConfig

KESOKU_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("src", "docs", "tests")
TEXT_SUFFIXES = {".py", ".md", ".toml", ".txt", ".sh", ".yaml", ".yml", ".json", ".cfg", ".ini"}
BASELINE_COMMIT = "8bf34f6"

# Engine-hook / persona-skill identifiers that must never exist in a generic engine.
ABSOLUTE_FORBIDDEN = re.compile(
    "|".join(
        [
            "date" + "_planner",
            "turn" + "_hook",
            "role_turn" + "_hook",
            "scene" + "_director",
            "<turn" + "_directive",
        ]
    ),
    re.IGNORECASE,
)

# Persona names / dating vocabulary that newly added engine lines must not contain.
INCREMENTAL_FORBIDDEN = re.compile(
    "|".join(["asu" + "ka", "约" + "会", "デー" + "ト", r"\bdate[-_ ]?plan", "date" + "_history"]),
    re.IGNORECASE,
)


def _iter_text_files() -> list[Path]:
    """All text files under the scanned directories, excluding caches."""
    files: list[Path] = []
    for d in SCAN_DIRS:
        base = KESOKU_ROOT / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if "__pycache__" in p.parts or not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES:
                continue
            files.append(p)
    return files


def _git(*args: str) -> str:
    """Run a git command in the kesoku repo and return stdout."""
    return subprocess.run(
        ["git", *args], cwd=KESOKU_ROOT, capture_output=True, text=True, check=True
    ).stdout


def test_no_persona_skill_or_hook_identifiers() -> None:
    """No persona planner/director script names, per-turn hook code or turn directives in the engine."""
    hits: list[str] = []
    for p in _iter_text_files():
        text = p.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), 1):
            if ABSOLUTE_FORBIDDEN.search(line):
                hits.append(f"{p.relative_to(KESOKU_ROOT)}:{lineno}: {line.strip()[:100]}")
    assert not hits, "persona coupling found:\n" + "\n".join(hits[:30])


def test_no_persona_terms_added_since_generic_baseline() -> None:
    """Lines added after the baseline (committed, staged, unstaged or untracked) stay persona-free."""
    if shutil.which("git") is None or not (KESOKU_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    try:
        _git("cat-file", "-e", f"{BASELINE_COMMIT}^{{commit}}")
    except subprocess.CalledProcessError:
        pytest.skip(f"baseline {BASELINE_COMMIT} not available")

    hits: list[str] = []
    diff = _git("diff", "-U0", "--no-color", BASELINE_COMMIT, "--", *SCAN_DIRS)
    current_file = ""
    for line in diff.splitlines():
        if line.startswith("+++ "):
            current_file = line[6:] if line.startswith("+++ b/") else line[4:]
            continue
        if line.startswith("+") and INCREMENTAL_FORBIDDEN.search(line):
            hits.append(f"{current_file}: {line[1:].strip()[:100]}")

    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *SCAN_DIRS).splitlines()
    for rel in untracked:
        p = KESOKU_ROOT / rel
        if p.suffix.lower() not in TEXT_SUFFIXES or not p.is_file():
            continue
        for lineno, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if INCREMENTAL_FORBIDDEN.search(line):
                hits.append(f"{rel}:{lineno}: {line.strip()[:100]}")

    assert not hits, "persona terms added to the generic engine:\n" + "\n".join(hits[:30])


def test_engine_config_has_no_hook_settings() -> None:
    """AgentConfig exposes no per-turn role hook setting."""
    assert not [name for name in AgentConfig.model_fields if "hook" in name.lower()]
