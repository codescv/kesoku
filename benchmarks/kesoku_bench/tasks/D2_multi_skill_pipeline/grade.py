"""Behavioral grader and Suite D skill-discovery diagnostics for D2_multi_skill_pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _count_llm_calls_before_use_skill(ctx: Any, tool_events: list[dict[str, Any]]) -> int:
    """Count how many LLM calls occurred prior to the first `use_skill` call."""
    llm_logs: list[Path] = getattr(ctx, "llm_logs", None) or []
    if llm_logs:
        count = 0
        for log_path in llm_logs:
            try:
                text = Path(log_path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "use_skill" in text:
                return count
            count += 1
        return count

    prior_calls = 0
    for ev in tool_events:
        if ev.get("name") == "use_skill":
            return prior_calls
        prior_calls += 1
    return prior_calls


def grade(ctx: Any, gold_mode: bool = False) -> dict[str, Any]:
    """Run behavioral checks and collect skill-discovery diagnostics for D2.

    Args:
        ctx: Harness GradeContext.
        gold_mode: True when invoked during $0 verify-gold validation.

    Returns:
        Dictionary with `checks` and `diagnostics`.
    """
    repo_dir = Path(getattr(ctx, "repo_dir"))
    q_cust = (repo_dir / "data" / "quarantine" / "customers.csv").is_file()
    q_ord = (repo_dir / "data" / "quarantine" / "orders.csv").is_file()
    quarantine_created = q_cust and q_ord

    if gold_mode or getattr(ctx, "gold_mode", False):
        return {
            "checks": {
                "quarantine_files_created": quarantine_created,
            },
            "diagnostics": {
                "list_skills_calls": None,
                "llm_calls_before_first_use_skill": None,
                "wrong_skill_loads": None,
            },
        }

    tool_events: list[dict[str, Any]] = getattr(ctx, "tool_events", None) or []
    target_skills = {"schema-gen", "fixture-audit"}

    list_skills_calls = sum(1 for ev in tool_events if ev.get("name") == "list_skills")
    use_skill_events = [ev for ev in tool_events if ev.get("name") == "use_skill"]
    loaded_skills = [str(ev.get("args", {}).get("skill_name", "")) for ev in use_skill_events]
    wrong_skills = [name for name in loaded_skills if name not in target_skills]
    llm_calls_before = _count_llm_calls_before_use_skill(ctx, tool_events)

    return {
        "checks": {
            "quarantine_files_created": quarantine_created,
        },
        "diagnostics": {
            "list_skills_calls": list_skills_calls,
            "llm_calls_before_first_use_skill": llm_calls_before,
            "wrong_skill_loads": len(wrong_skills),
            "wrong_skill_names": wrong_skills,
            "loaded_skills": loaded_skills,
            "used_both_target_skills": target_skills.issubset(set(loaded_skills)),
        },
    }
