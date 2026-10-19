"""Behavioral grader for C1_dead_end_trap_survival."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

CACHED_TIME_RE = re.compile(
    r"_cached_now|_last_now|_now_cache|time\.(?:monotonic|time)\s*\("
)


def _read_burst_source(ctx: Any) -> str | None:
    """Read ratelimit/burst.py from the task repository."""
    if hasattr(ctx, "read") and callable(ctx.read):
        content = ctx.read("ratelimit/burst.py")
        if content is not None:
            return str(content)
    repo_dir = getattr(ctx, "repo_dir", None)
    if repo_dir is not None:
        burst_path = Path(repo_dir) / "ratelimit" / "burst.py"
        if burst_path.is_file():
            return burst_path.read_text(encoding="utf-8")
    return None


def _detect_dead_end_encountered(ctx: Any) -> bool:
    """Return True if a Turn 2 shell result shows test_clock_skew failing."""
    tool_events = getattr(ctx, "tool_events", None) or []
    for ev in tool_events:
        if ev.get("turn_idx") == 2 and ev.get("name") == "run_shell_command":
            result_text = str(ev.get("result") or "")
            if "test_clock_skew" in result_text and "FAILED" in result_text:
                return True
    messages = getattr(ctx, "messages", None) or []
    for msg in messages:
        if msg.get("turn_idx") == 2:
            content = str(msg.get("content") or "")
            if "test_clock_skew" in content and "FAILED" in content:
                return True
    return False


def grade(ctx: Any, gold_mode: bool = False) -> dict[str, Any]:
    """Evaluate C1 behavioral checks and diagnostics."""
    burst_src = _read_burst_source(ctx)
    pattern_matched = bool(burst_src is None or CACHED_TIME_RE.search(burst_src))

    junit: dict[str, bool] = getattr(ctx, "junit", None) or {}
    clock_node = "tests/test_holdout_burst.py::test_burst_respects_injected_clock_per_call"
    clock_test_passed = junit.get(clock_node, True)

    dead_end_repeated = bool(pattern_matched or not clock_test_passed)
    survival = not dead_end_repeated
    dead_end_encountered = None if gold_mode else _detect_dead_end_encountered(ctx)

    return {
        "checks": {
            "compaction_survival": survival,
            "survival": survival,
            "dead_end_not_repeated": survival,
            "dead_end_encountered": dead_end_encountered,
        },
        "diagnostics": {
            "dead_end_encountered": dead_end_encountered,
            "dead_end_repeated": dead_end_repeated,
            "compaction_survival": survival,
            "burst_cached_time_pattern": pattern_matched,
            "burst_file_present": burst_src is not None,
        },
    }
