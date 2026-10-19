"""Behavioral grader and diagnostics for D4_repo_conventions_and_lint_gate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any


def grade(ctx: Any, gold_mode: bool = False) -> dict[str, Any]:
    """Run convention checks and collect AGENTS.md / lint diagnostics for D4.

    Args:
        ctx: Harness GradeContext.
        gold_mode: True when invoked during $0 verify-gold validation.

    Returns:
        Dictionary with `checks` and `diagnostics`.
    """
    repo_dir = Path(getattr(ctx, "repo_dir"))
    lint_script = repo_dir / "scripts" / "lint_conventions.py"
    reg_file = repo_dir / "tests" / "regression" / "test_142.py"

    lint_proc = subprocess.run(
        [sys.executable, str(lint_script)],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    lint_passes = lint_proc.returncode == 0
    reg_exists = reg_file.is_file()

    if gold_mode or getattr(ctx, "gold_mode", False):
        return {
            "checks": {
                "conventions_lint_passes": lint_passes,
                "regression_file_exists": reg_exists,
            },
            "diagnostics": {
                "agents_md_read": None,
                "lint_run_by_agent": None,
                "list_skills_calls": None,
                "llm_calls_before_first_use_skill": None,
                "wrong_skill_loads": None,
            },
        }

    tool_events: list[dict[str, Any]] = getattr(ctx, "tool_events", None) or []
    agents_md_read = False
    lint_run_by_agent = False

    for ev in tool_events:
        args_str = str(ev.get("args", {}))
        res_str = str(ev.get("result", ""))
        if "AGENTS.md" in args_str or "AGENTS.md" in res_str:
            agents_md_read = True
        if ev.get("name") == "run_shell_command" and "lint_conventions.py" in str(
            ev.get("args", {}).get("command", "")
        ):
            lint_run_by_agent = True

    list_skills_calls = sum(1 for ev in tool_events if ev.get("name") == "list_skills")
    use_skill_events = [ev for ev in tool_events if ev.get("name") == "use_skill"]
    loaded_skills = [str(ev.get("args", {}).get("skill_name", "")) for ev in use_skill_events]

    return {
        "checks": {
            "conventions_lint_passes": lint_passes,
            "regression_file_exists": reg_exists,
        },
        "diagnostics": {
            "agents_md_read": agents_md_read,
            "lint_run_by_agent": lint_run_by_agent,
            "list_skills_calls": list_skills_calls,
            "llm_calls_before_first_use_skill": None,
            "wrong_skill_loads": len(loaded_skills),
            "wrong_skill_names": loaded_skills,
        },
    }
