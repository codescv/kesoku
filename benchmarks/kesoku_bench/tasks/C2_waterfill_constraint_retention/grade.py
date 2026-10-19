"""Behavioural grader for C2_waterfill_constraint_retention (spec §4.3).

Survival check: rule 7 of the T2 spec (country aliasing + ``field_path="address.<subfield>"``) is honoured by
``validators/address.py``. Diagnostics separate the survival mechanisms: the T6 scaffold (water-filling of the
compacted T2 message), the LLM summary text, and the agent's own ``notes.md`` pin.

Stdlib only (``yaml`` is used opportunistically when importable).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:  # pragma: no cover - availability depends on the harness environment
    import yaml  # type: ignore[import-untyped]
except Exception:  # noqa: BLE001
    yaml = None

RULE7_FILE = "test_holdout_address_rule7"
CONVENTIONS_FILE = "test_holdout_address_conventions"
TEST_TURN = 6
SCAFFOLD_MARKER = "custom turn-based context management"
SCAFFOLD_RULE_RE = re.compile(r"field_path|COUNTRY_ALIASES")
SUMMARY_RULE_RE = re.compile(r"field_path|ISO")
NOTES_RULE_RE = re.compile(r"field_path|COUNTRY_ALIASES")


def _get(ctx: Any, name: str, default: Any = None) -> Any:
    """Attribute access that tolerates partial stub contexts."""
    value = getattr(ctx, name, default)
    return default if value is None else value


def _junit_verdict(ctx: Any, marker: str) -> bool | None:
    """Return all-passed for junit ids containing ``marker``, or None if none were recorded."""
    junit: dict[str, bool] = _get(ctx, "junit", {}) or {}
    results = [ok for nodeid, ok in junit.items() if marker in nodeid]
    if not results:
        return None
    return all(results)


def _pytest_verdict(ctx: Any, marker: str) -> bool | None:
    """Run ``tests/<marker>.py`` via ``ctx.run`` if it is present in the repo; None if it is not."""
    repo_dir = Path(_get(ctx, "repo_dir", "."))
    rel = f"tests/{marker}.py"
    if not (repo_dir / rel).is_file() or not hasattr(ctx, "run"):
        return None
    proc = ctx.run(["uv", "run", "pytest", "-q", "-p", "no:cacheprovider", rel], timeout=180)
    return proc.returncode == 0


def _static_rule7(ctx: Any) -> bool:
    """Last-resort source inspection of validators/address.py for the two halves of rule 7."""
    repo_dir = Path(_get(ctx, "repo_dir", "."))
    path = repo_dir / "domerr" / "validators" / "address.py"
    if not path.is_file():
        return False
    src = path.read_text(encoding="utf-8", errors="replace")
    return "COUNTRY_ALIASES" in src and "field_path" in src and "address." in src


def _verdict(ctx: Any, marker: str) -> tuple[bool | None, str]:
    """Resolve a holdout-file verdict from junit, then a direct pytest run; returns (verdict, source)."""
    verdict = _junit_verdict(ctx, marker)
    if verdict is not None:
        return verdict, "junit"
    verdict = _pytest_verdict(ctx, marker)
    if verdict is not None:
        return verdict, "pytest"
    return None, "unavailable"


def _scaffold_from_yaml(text: str) -> str | None:
    """Extract the scaffold user message content from a parsed llm-turn log (needs PyYAML)."""
    if yaml is None:
        return None
    try:
        data = yaml.safe_load(text)
    except Exception:  # noqa: BLE001
        return None
    for msg in (data or {}).get("history", []) or []:
        meta = msg.get("metadata") or {}
        if meta.get("is_scaffold") or SCAFFOLD_MARKER in str(msg.get("content") or ""):
            return str(msg.get("content") or "")
    return ""


def _scaffold_heuristic(text: str) -> str:
    """Slice the raw YAML text from the scaffold header to its ``is_scaffold`` metadata flag."""
    start = text.find(SCAFFOLD_MARKER)
    if start < 0:
        return ""
    end = text.find("is_scaffold", start)
    chunk = text[start : end if end > 0 else len(text)]
    # Undo PyYAML double-quoted line folding ("\<newline><indent>\ ") so tokens are contiguous.
    return re.sub(r"\\\n\s*(\\ )?", " ", chunk)


def _scaffold_text(log_path: Path) -> str:
    """Return the scaffold message text of one llm-turn log ("" if the request carried none)."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    parsed = _scaffold_from_yaml(text)
    return parsed if parsed is not None else _scaffold_heuristic(text)


def _scaffold_diagnostics(ctx: Any) -> dict[str, Any]:
    """Inspect the first T6 LLM request: is rule 7 visible in the scaffold / its recent_user_messages block."""
    by_turn: dict[int, list[Path]] = _get(ctx, "llm_log_turn", {}) or {}
    logs = by_turn.get(TEST_TURN) or by_turn.get(str(TEST_TURN)) or []  # type: ignore[call-overload]
    if not logs:
        return {"rule_visible_in_scaffold": None, "rule_in_recent_user_messages": None, "scaffold_chars": None}
    scaffold = _scaffold_text(Path(logs[0]))
    recent = ""
    if "<recent_user_messages>" in scaffold:
        recent = scaffold.split("<recent_user_messages>", 1)[1]
    return {
        "rule_visible_in_scaffold": bool(SCAFFOLD_RULE_RE.search(scaffold)),
        "rule_in_recent_user_messages": bool(SCAFFOLD_RULE_RE.search(recent)),
        "scaffold_chars": len(scaffold),
    }


def _summary_diagnostic(ctx: Any) -> bool | None:
    """Whether any L0/L1 summary node paraphrases rule 7."""
    nodes: list[dict[str, Any]] = _get(ctx, "summary_nodes", []) or []
    if not nodes:
        return None
    return any(SUMMARY_RULE_RE.search(str(n.get("summary") or "")) for n in nodes)


def _notes_diagnostic(ctx: Any) -> bool | None:
    """Whether the agent pinned rule 7 in ``$STAGING_DIR/notes.md``."""
    staging = _get(ctx, "staging_dir")
    if not staging:
        return None
    notes = Path(staging) / "notes.md"
    if not notes.is_file():
        return False
    return bool(NOTES_RULE_RE.search(notes.read_text(encoding="utf-8", errors="replace")))


def grade(ctx: Any, gold_mode: bool = False) -> dict[str, Any]:
    """Grade a C2 run (or the gold fixture when ``gold_mode``).

    Args:
        ctx: GradeContext (attribute access; see the harness).
        gold_mode: True under ``--verify-gold``: live-only diagnostics are reported as None.

    Returns:
        ``{"checks": {...}, "diagnostics": {...}}``. ``rule7_retained`` is the Compaction Survival check.
    """
    rule7, rule7_source = _verdict(ctx, RULE7_FILE)
    if rule7 is None:
        rule7, rule7_source = _static_rule7(ctx), "static"
    conventions, conventions_source = _verdict(ctx, CONVENTIONS_FILE)

    checks: dict[str, bool | None] = {
        "rule7_retained": rule7,
        "address_conventions": conventions,
    }
    diagnostics: dict[str, Any] = {"rule7_source": rule7_source, "conventions_source": conventions_source}
    if gold_mode:
        diagnostics.update(
            {
                "rule_visible_in_scaffold": None,
                "rule_in_recent_user_messages": None,
                "rule_in_summary_text": None,
                "rule_pinned_in_notes": None,
            }
        )
    else:
        diagnostics.update(_scaffold_diagnostics(ctx))
        diagnostics["rule_in_summary_text"] = _summary_diagnostic(ctx)
        diagnostics["rule_pinned_in_notes"] = _notes_diagnostic(ctx)
    return {"checks": checks, "diagnostics": diagnostics}
