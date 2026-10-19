"""$0 re-grading of saved TaskResults: re-run a task's grade.py against the run's preserved workdir and DB.

Used when a behavioural grader or a metric definition changes after a live run. Pytest outcomes are taken from
the saved result (the repo state is unchanged); behavioural checks, event-derived generic diagnostics,
`resolved` and `status` are recomputed.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from benchmarks.kesoku_bench.harness.fixture import substitute
from benchmarks.kesoku_bench.harness.grading import (
    GradeContext,
    extract_tool_events,
    group_llm_logs_by_turn,
    load_grade_module,
    message_to_dict,
    required_checks_pass,
    run_behavioral,
)
from benchmarks.kesoku_bench.harness.runner import decide_status, generic_diagnostics
from benchmarks.kesoku_bench.harness.taskspec import TASKS_DIR, TaskSpec, load_task
from kesoku.constants import MessageRole, MessageType
from kesoku.db import DatabaseManager

EXTRAS_KEYS = ("port", "forced_compaction_turns", "natural_compaction", "pre_cleanup", "manual_compact_nodes_delta",
               "manual_compact_reply", "fallback_forced", "active_jobs_for_session")


def _reconstruct_windows(spec: TaskSpec, result: dict[str, Any], msgs: list[Any], awd: Path) -> list[float]:
    """Turn begin timestamps, from the result (newer runs) or by matching scripted user texts in the DB."""
    saved = [t.get("begin_ts") for t in result.get("turns", [])]
    if saved and all(isinstance(ts, (int, float)) for ts in saved):
        return [float(ts) for ts in saved]
    port = int((result.get("diagnostics") or {}).get("port") or 0)
    users = [m for m in msgs if m.role == MessageRole.USER and m.type == MessageType.TEXT]
    begins: list[float | None] = []
    cursor = 0.0
    for turn in spec.turns[: len(result.get("turns", []))]:
        if turn.kind != "user":
            begins.append(None)
            continue
        text = substitute(turn.text, port, awd)
        hit = next((m for m in users if m.timestamp >= cursor and m.content == text), None)
        if hit is None:
            raise ValueError(f"{spec.id}: cannot locate scripted turn text in the DB")
        begins.append(hit.timestamp)
        cursor = hit.timestamp
    # Command turns (no DB message) start just after the last message preceding the next turn.
    out: list[float] = []
    for i, ts in enumerate(begins):
        if ts is not None:
            out.append(ts)
            continue
        nxt = next((b for b in begins[i + 1:] if b is not None), float("inf"))
        before = [m.timestamp for m in msgs if m.timestamp < nxt]
        out.append((max(before) if before else (out[-1] if out else 0.0)) + 1e-3)
    return out


def regrade_result(result_path: Path, write: bool = True, tasks_root: Path | None = None) -> dict[str, Any]:
    """Re-run the behavioural grader of one saved TaskResult.

    Args:
        result_path: TaskResult JSON.
        write: Rewrite the JSON in place (the original is kept once as `<name>.json.orig`).
        tasks_root: Optional override of the tasks directory.

    Returns:
        The updated TaskResult dict.
    """
    result = json.loads(result_path.read_text(encoding="utf-8"))
    spec = load_task((tasks_root or TASKS_DIR) / result["task_id"])
    ws = Path(result["workdir"]) / "ws"
    db = DatabaseManager(str(ws / "kesoku.db"))
    session = db.get_latest_session()
    if session is None:
        raise ValueError(f"{spec.id}: no session in {ws / 'kesoku.db'}")
    raw_msgs = db.get_messages_by_filters({"session_id": session.id})
    begins = _reconstruct_windows(spec, result, raw_msgs, ws)
    windows = [(i + 1, ts) for i, ts in enumerate(begins)]
    messages = [message_to_dict(m, windows) for m in raw_msgs]
    nodes = [n.model_dump() for n in db.get_all_summary_nodes(session.id)]
    staging = ws / "sessions" / session.workspace_name
    logs = sorted(staging.glob("llm-turn-*.log.yaml"), key=lambda p: int(p.name.split("-")[2].split(".")[0]))
    diag = result.get("diagnostics") or {}
    extras = {k: diag[k] for k in EXTRAS_KEYS if k in diag}
    extras.setdefault("active_jobs_for_session", [])
    port = int(diag.get("port") or 0)
    turns = []
    for i, t in enumerate(result.get("turns", [])):
        turn_spec = spec.turns[i]
        end = begins[i + 1] if i + 1 < len(begins) else None
        turns.append({
            "idx": t["idx"], "kind": t["kind"], "text": substitute(turn_spec.text, port, ws), "start_ts": begins[i],
            "end_ts": t.get("end_ts", end), "status": t.get("status"), "compactions_before": t.get("forced_nodes", 0),
            "command_reply": t.get("command_reply"), "auto_replies": t.get("auto_replies", 0),
        })
    grading = result["grading"]
    f2p, p2p = grading.get("fail_to_pass") or {}, grading.get("pass_to_pass") or {}
    events = extract_tool_events(messages)
    # Event-derived generic metrics are recomputed so every stage uses the current definitions.
    diag = {**diag, **generic_diagnostics(spec, events, None, None)}
    previous = {"status": result.get("status"), "resolved": result.get("resolved"),
                "behavioral": grading.get("behavioral")}
    result["regraded"] = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "previous": previous}
    if spec.grade_path is None:
        result["diagnostics"] = diag
        if write:
            _write(result_path, result)
        return result
    ctx = GradeContext(
        task_id=spec.id, task_dir=spec.task_dir, repo_dir=ws / "repo", gold_mode=False, ws_dir=ws,
        staging_dir=staging, session_id=session.id, db_path=ws / "kesoku.db", turns=turns,
        tool_events=events, messages=messages, summary_nodes=nodes, llm_logs=logs,
        llm_log_turn=group_llm_logs_by_turn(logs, windows), junit={**f2p, **p2p}, extras=extras,
    )
    outcome = run_behavioral(load_grade_module(spec.grade_path), ctx)
    required_ok = required_checks_pass(outcome, spec.grading.required_behavioral)
    resolved = all(f2p.values()) and all(p2p.values()) and not grading.get("tampered") and required_ok
    grading["behavioral"] = outcome.checks
    grading["required_behavioral"] = spec.grading.required_behavioral
    result["diagnostics"] = {**diag, **outcome.diagnostics}
    result["resolved"] = resolved
    result["status"] = decide_status(
        tampered=bool(grading.get("tampered")), error=result.get("error"),
        budget=bool(diag.get("budget_exceeded_reason")),
        timeout=any(t.get("status") == "timeout" for t in result.get("turns", [])), resolved=resolved,
    )
    if write:
        _write(result_path, result)
    return result


def _write(result_path: Path, result: dict[str, Any]) -> None:
    """Rewrite a result JSON, keeping the very first version as `<name>.json.orig`."""
    backup = result_path.with_name(result_path.name + ".orig")
    if not backup.exists():
        shutil.copy2(result_path, backup)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def collect_result_files(paths: list[Path], task_ids: list[str] | None = None) -> list[Path]:
    """Expand files/directories into TaskResult JSON paths (optionally filtered by task id or prefix)."""
    files: list[Path] = []
    for p in paths:
        candidates = sorted(p.rglob("*.json")) if p.is_dir() else [p]
        for f in candidates:
            if f.name in ("scorecard.json", "verify_gold.json"):
                continue
            files.append(f)
    if task_ids:
        def _keep(f: Path) -> bool:
            stem = f.name.split(".")[0]
            return any(stem == t or stem.split("_")[0] == t for t in task_ids)

        files = [f for f in files if _keep(f)]
    return files
