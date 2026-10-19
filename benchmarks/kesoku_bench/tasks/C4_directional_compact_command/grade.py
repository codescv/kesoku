"""Behavioral grader for C4_directional_compact_command."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

Y_MENTION_RE = re.compile(r"verify\s*\(|subsystem[\s_]*y|webhook", re.IGNORECASE)
X_MENTION_RE = re.compile(r"thumbnail|\blru\b", re.IGNORECASE)


def _uses_compare_digest(repo_dir: Path) -> bool:
    """Return True if svc/subsystem_y/webhook.py references hmac.compare_digest."""
    webhook_path = repo_dir / "svc" / "subsystem_y" / "webhook.py"
    if not webhook_path.is_file():
        return False
    try:
        tree = ast.parse(webhook_path.read_text(encoding="utf-8"), filename=str(webhook_path))
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.Attribute) and node.attr == "compare_digest"
        for node in ast.walk(tree)
    )


def grade(ctx: Any, gold_mode: bool = False) -> dict[str, Any]:
    """Evaluate C4 behavioral checks and P1-4 directional compaction diagnostics."""
    repo_dir = Path(getattr(ctx, "repo_dir", Path.cwd()))
    compare_digest_ok = _uses_compare_digest(repo_dir)

    if gold_mode:
        return {
            "checks": {
                "constant_time_compare_preserved": compare_digest_ok,
                "p1_4_directive": None,
            },
            "diagnostics": {
                "constant_time_compare_preserved": compare_digest_ok,
                "manual_compact_nodes_delta": None,
                "fallback_forced": None,
                "manual_compact_reply": None,
                "scaffold_mentions_y": None,
                "scaffold_mentions_x": None,
                "p1_4_directive": None,
            },
        }

    extras: dict[str, Any] = getattr(ctx, "extras", None) or {}
    summary_nodes: list[dict[str, Any]] = getattr(ctx, "summary_nodes", None) or []
    l0_nodes = sorted(
        [n for n in summary_nodes if int(n.get("level") or 0) == 0],
        key=lambda item: float(item.get("created_at") or 0.0),
    )

    # The first L0 summary node is produced by the /compact command (or its fallback) after T5.
    target_summary = str(l0_nodes[0].get("summary") or "") if l0_nodes else ""
    scaffold_mentions_y = bool(Y_MENTION_RE.search(target_summary))
    scaffold_mentions_x = bool(X_MENTION_RE.search(target_summary))
    p1_4_directive = bool(scaffold_mentions_y and not scaffold_mentions_x)

    return {
        "checks": {
            "constant_time_compare_preserved": compare_digest_ok,
            "p1_4_directive": p1_4_directive,
        },
        "diagnostics": {
            "constant_time_compare_preserved": compare_digest_ok,
            "manual_compact_nodes_delta": extras.get("manual_compact_nodes_delta"),
            "fallback_forced": extras.get("fallback_forced"),
            "manual_compact_reply": extras.get("manual_compact_reply"),
            "scaffold_mentions_y": scaffold_mentions_y,
            "scaffold_mentions_x": scaffold_mentions_x,
            "p1_4_directive": p1_4_directive,
            "l0_summary_count": len(l0_nodes),
        },
    }
