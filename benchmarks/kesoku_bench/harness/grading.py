"""Re-exports grading and tool-event extraction primitives from `kesoku.testing.grading` for `kesoku_bench`."""

from __future__ import annotations

from kesoku.testing.grading import (
    TAMPER_RE,
    CaseResult,
    GradeContext,
    GradeOutcome,
    PytestRun,
    extract_tool_events,
    group_llm_logs_by_turn,
    load_grade_module,
    message_to_dict,
    normalize_outcome,
    parse_junit,
    purge_pycache,
    required_checks_pass,
    run_behavioral,
    run_pre_cleanup,
    run_pytest,
    tamper_audit,
    turn_index_for,
)

__all__ = [
    "TAMPER_RE",
    "CaseResult",
    "GradeContext",
    "GradeOutcome",
    "PytestRun",
    "extract_tool_events",
    "group_llm_logs_by_turn",
    "load_grade_module",
    "message_to_dict",
    "normalize_outcome",
    "parse_junit",
    "purge_pycache",
    "required_checks_pass",
    "run_behavioral",
    "run_pre_cleanup",
    "run_pytest",
    "tamper_audit",
    "turn_index_for",
]
