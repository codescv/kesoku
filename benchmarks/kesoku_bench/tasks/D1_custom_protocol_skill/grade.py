"""Behavioral grader and Suite D skill-discovery diagnostics for D1_custom_protocol_skill."""

from __future__ import annotations

import re
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
    """Run behavioral checks and collect skill-discovery diagnostics for D1.

    Args:
        ctx: Harness GradeContext providing repo_dir, tool_events, messages, and llm_logs.
        gold_mode: True when invoked during $0 verify-gold validation.

    Returns:
        Dictionary with `checks` and `diagnostics`.
    """
    repo_dir = Path(getattr(ctx, "repo_dir"))
    fixed_path = repo_dir / "fixtures" / "session_17.fixed.pcb"
    repaired_exists = fixed_path.is_file()

    if gold_mode or getattr(ctx, "gold_mode", False):
        return {
            "checks": {
                "repaired_file_exists": repaired_exists,
                "corrupted_frame_9_identified": None,
            },
            "diagnostics": {
                "list_skills_calls": None,
                "llm_calls_before_first_use_skill": None,
                "wrong_skill_loads": None,
            },
        }

    tool_events: list[dict[str, Any]] = getattr(ctx, "tool_events", None) or []
    messages: list[dict[str, Any]] = getattr(ctx, "messages", None) or []
    target_skills = {"proto-codec"}

    list_skills_calls = sum(1 for ev in tool_events if ev.get("name") == "list_skills")
    use_skill_events = [ev for ev in tool_events if ev.get("name") == "use_skill"]
    loaded_skills = [str(ev.get("args", {}).get("skill_name", "")) for ev in use_skill_events]
    wrong_skills = [name for name in loaded_skills if name not in target_skills]
    llm_calls_before = _count_llm_calls_before_use_skill(ctx, tool_events)

    assistant_texts = [
        str(m.get("content", ""))
        for m in messages
        if str(m.get("role", "")).upper().endswith("ASSISTANT")
    ]
    combined_reply = "\n".join(assistant_texts)
    frame_9_mentioned = bool(re.search(r"\bframe\s*#?\s*0*9\b|\bindex\s*0*9\b|\b9\b", combined_reply, re.IGNORECASE))

    return {
        "checks": {
            "repaired_file_exists": repaired_exists,
            "corrupted_frame_9_identified": frame_9_mentioned,
        },
        "diagnostics": {
            "list_skills_calls": list_skills_calls,
            "llm_calls_before_first_use_skill": llm_calls_before,
            "wrong_skill_loads": len(wrong_skills),
            "wrong_skill_names": wrong_skills,
            "loaded_skills": loaded_skills,
            "corrupted_frame_9_identified": frame_9_mentioned,
        },
    }
