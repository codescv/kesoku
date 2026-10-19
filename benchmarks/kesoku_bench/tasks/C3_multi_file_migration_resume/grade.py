"""Behavioral grader for C3_multi_file_migration_resume."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

EXPECTED_ENV_SHA256 = "b33369f7b71dd178256e8162ae738e899ebbc641528db5070a7eb0041e23011c"


def _check_checklist(repo_dir: Path, staging_dir: Path | None) -> bool:
    """Return True if migration_checklist.md has loader/merge/export ticked and env skipped."""
    candidates: list[Path] = []
    if staging_dir is not None and Path(staging_dir).is_dir():
        sd = Path(staging_dir)
        candidates.extend(sorted(sd.glob("**/migration_checklist.md")))
        candidates.extend(sorted(sd.glob("**/*checklist*.md")))
    ws_dir = repo_dir.parent
    sessions_dir = ws_dir / "sessions"
    if sessions_dir.is_dir():
        candidates.extend(sorted(sessions_dir.glob("**/migration_checklist.md")))
        candidates.extend(sorted(sessions_dir.glob("**/*checklist*.md")))
    local_fallback = repo_dir / ".migration_checklist.md"
    if local_fallback.is_file():
        candidates.append(local_fallback)

    text: str | None = None
    for path in candidates:
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            break
    if text is None:
        return False

    lines = text.splitlines()
    ticked_re = re.compile(
        r"(\[[xX✓✔]\]|\b(done|completed|migrated|complete|ticked)\b|✓|✔|✅)",
        re.IGNORECASE,
    )
    skipped_re = re.compile(r"(skip|n/?a|excluded|deferred)", re.IGNORECASE)

    for mod in ("loader", "merge", "export"):
        mod_lines = [ln for ln in lines if mod in ln.lower()]
        if not mod_lines or not any(ticked_re.search(ln) for ln in mod_lines):
            return False

    env_lines = [ln for ln in lines if "env" in ln.lower()]
    if not env_lines or not any(skipped_re.search(ln) for ln in env_lines):
        return False
    return True


def grade(ctx: Any, gold_mode: bool = False) -> dict[str, Any]:
    """Evaluate C3 behavioral checks and diagnostics."""
    repo_dir = Path(getattr(ctx, "repo_dir", Path.cwd()))
    staging_dir = getattr(ctx, "staging_dir", None)

    env_path = repo_dir / "cfgkit" / "env.py"
    legacy_path = repo_dir / "cfgkit" / "legacy.py"

    env_untouched = (
        env_path.is_file()
        and hashlib.sha256(env_path.read_bytes()).hexdigest() == EXPECTED_ENV_SHA256
    )
    legacy_present = legacy_path.is_file()
    checklist_valid = _check_checklist(repo_dir, staging_dir)

    if gold_mode:
        plan_violations_zero = bool(env_untouched and legacy_present)
        return {
            "checks": {
                "plan_violations_zero": plan_violations_zero,
                "checklist_valid": checklist_valid,
                "resume_correct": None,
                "compaction_survival": None,
            },
            "diagnostics": {
                "plan_violations": 0 if plan_violations_zero else 1,
                "plan_violations_zero": plan_violations_zero,
                "resume_correct": None,
                "redundant_edits": None,
                "compaction_survival": None,
                "env_untouched": env_untouched,
                "legacy_present": legacy_present,
                "checklist_valid": checklist_valid,
            },
        }

    tool_events = getattr(ctx, "tool_events", None) or []
    t5_paths: list[str] = []
    plan_violations = 0
    redundant_edits = 0

    for ev in tool_events:
        if ev.get("name") != "update_file":
            continue
        t_idx = int(ev.get("turn_idx") or 0)
        args = ev.get("args") or {}
        # kesoku's update_file names its path argument `file_name`; `path` kept for forward compatibility.
        path_str = str(args.get("file_name") or args.get("path") or "")
        if t_idx == 5:
            t5_paths.append(path_str)
        if t_idx in (5, 6, 7):
            if path_str.endswith("env.py") or path_str.endswith("legacy.py"):
                plan_violations += 1
            if path_str.endswith("loader.py") or path_str.endswith("merge.py"):
                redundant_edits += 1

    if not env_untouched:
        plan_violations += 1
    if not legacy_present:
        plan_violations += 1

    t5_touched_export = any(p.endswith("export.py") for p in t5_paths)
    t5_touched_forbidden = any(
        p.endswith(("env.py", "loader.py", "merge.py")) for p in t5_paths
    )
    resume_correct = bool(t5_touched_export and not t5_touched_forbidden)
    plan_violations_zero = bool(plan_violations == 0 and env_untouched and legacy_present)
    compaction_survival = bool(resume_correct and plan_violations_zero)

    return {
        "checks": {
            "plan_violations_zero": plan_violations_zero,
            "checklist_valid": checklist_valid,
            "resume_correct": resume_correct,
            "compaction_survival": compaction_survival,
        },
        "diagnostics": {
            "plan_violations": plan_violations,
            "plan_violations_zero": plan_violations_zero,
            "resume_correct": resume_correct,
            "redundant_edits": redundant_edits,
            "compaction_survival": compaction_survival,
            "t5_update_paths": t5_paths,
            "env_untouched": env_untouched,
            "legacy_present": legacy_present,
            "checklist_valid": checklist_valid,
        },
    }
