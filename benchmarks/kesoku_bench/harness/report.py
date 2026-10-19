"""Scorecard: aggregate TaskResult JSONs per stage and render Markdown + JSON (spec §3)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from benchmarks.kesoku_bench.harness.taskspec import RESULTS_DIR

LIFT_EXCLUDED = {"C4_directional_compact_command"}  # regression guard (spec §3, M1)


def load_results(stage_dir: Path) -> list[dict[str, Any]]:
    """Load every TaskResult JSON under a stage (or stage/run) directory."""
    out = []
    for path in sorted(Path(stage_dir).rglob("*.json")):
        if path.name in ("scorecard.json", "verify_gold.json"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "task_id" in data:
            out.append(data)
    return out


def _mean(values: list[float]) -> float | None:
    """Mean or None for an empty list."""
    return round(mean(values), 4) if values else None


def _survived(res: dict[str, Any]) -> bool | None:
    """Compaction-survival outcome for Suite C / D3 runs (None = not applicable / not countable).

    Uses the task's own `compaction_survival` (or `survival`) check when its grader reports one; D3 is
    `server_stopped ∧ registry_empty`; otherwise (C2) survival is the resolved rule-retention outcome. C1 runs
    only count when the dead end was actually encountered (spec M2).
    """
    task = res["task_id"]
    if not (res.get("suite") == "C" or task.startswith("D3")) or task in LIFT_EXCLUDED:
        return None
    if not (res.get("totals") or {}).get("forced_compaction_turns"):
        return None
    behavioral = (res.get("grading") or {}).get("behavioral") or {}
    diag = res.get("diagnostics") or {}
    if behavioral.get("dead_end_encountered", diag.get("dead_end_encountered")) is False:
        return None
    if task.startswith("D3"):
        stopped, empty = behavioral.get("server_stopped"), behavioral.get("registry_empty")
        if stopped is None or empty is None:
            return None
        return bool(stopped and empty)
    for key in ("compaction_survival", "survival"):
        if behavioral.get(key) is not None:
            return bool(behavioral[key])
    if "dead_end_repeated" in behavioral or "dead_end_repeated" in diag:
        repeated = behavioral.get("dead_end_repeated", diag.get("dead_end_repeated"))
        return None if repeated is None else not repeated
    return bool(res.get("resolved"))


def stage_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one stage's results into the scorecard metrics."""
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_task[r["task_id"]].append(r)
    per_task = {}
    for task, runs in sorted(by_task.items()):
        per_task[task] = {
            "suite": runs[0].get("suite"),
            "runs": len(runs),
            "resolved_frac": round(sum(1 for r in runs if r.get("status") == "resolved") / len(runs), 4),
            "statuses": [r.get("status") for r in runs],
            "cost_usd": round(sum(float((r.get("totals") or {}).get("cost_usd") or 0) for r in runs), 4),
        }
    suites: dict[str, list[float]] = defaultdict(list)
    for task, info in per_task.items():
        suites[info["suite"] or "?"].append(info["resolved_frac"])
    lift_tasks = [info["resolved_frac"] for t, info in per_task.items() if t not in LIFT_EXCLUDED]
    survival = [s for s in (_survived(r) for r in results) if s is not None]
    c4 = [r for r in results if r["task_id"] in LIFT_EXCLUDED]
    p14 = []
    for r in c4:
        diag = r.get("diagnostics") or {}
        if "scaffold_mentions_y" in diag:
            p14.append(bool(diag.get("scaffold_mentions_y")) and not bool(diag.get("scaffold_mentions_x")))
    totals = [r.get("totals") or {} for r in results]
    diags = [r.get("diagnostics") or {} for r in results]
    upd_calls = sum(int(d.get("update_file_calls") or 0) for d in diags)
    upd_err = sum(int(d.get("update_file_errors") or 0) for d in diags)
    resolved_runs = sum(1 for r in results if r.get("status") == "resolved")
    total_cost = sum(float(t.get("cost_usd") or 0) for t in totals)
    toks = [t.get("tokens") or {} for t in totals]
    cached = sum(int(t.get("cached_in") or 0) for t in toks)
    uncached = sum(int(t.get("uncached_in") or 0) for t in toks)
    suite_b = [d for r, d in zip(results, diags, strict=True) if r.get("suite") == "B"]
    suite_d = [d for r, d in zip(results, diags, strict=True) if r.get("suite") == "D"]
    return {
        "tasks": len(per_task),
        "runs": len(results),
        "resolved_pct": _mean([info["resolved_frac"] for info in per_task.values()]),
        "resolved_pct_lift_basis": _mean(lift_tasks),
        "resolved_pct_by_suite": {s: _mean(v) for s, v in sorted(suites.items())},
        "compaction_survival_pct": _mean([1.0 if s else 0.0 for s in survival]),
        "compaction_survival_n": len(survival),
        "p14_directive_metric": _mean([1.0 if v else 0.0 for v in p14]),
        "avg_llm_calls": _mean([float(t.get("llm_calls") or 0) for t in totals]),
        "avg_summarize_calls": _mean([float(t.get("summarize_calls") or 0) for t in totals]),
        "avg_tool_calls": _mean([float(t.get("tool_calls") or 0) for t in totals]),
        "update_file_error_rate": round(upd_err / upd_calls, 4) if upd_calls else None,
        "shell_log_rereads_suite_b": _mean([float(d.get("shell_log_rereads") or 0) for d in suite_b]),
        "list_skills_calls_suite_d": _mean([float(d.get("list_skills_calls") or 0) for d in suite_d]),
        "llm_calls_before_first_use_skill_suite_d": _mean(
            [float(d["llm_calls_before_first_use_skill"]) for d in suite_d
             if d.get("llm_calls_before_first_use_skill") is not None]),
        "wrong_skill_loads_suite_d": _mean([float(d.get("wrong_skill_loads") or 0) for d in suite_d]),
        "total_cost_usd": round(total_cost, 4),
        "cost_per_resolved_usd": round(total_cost / resolved_runs, 4) if resolved_runs else None,
        "cached_input_share": round(cached / (cached + uncached), 4) if cached + uncached else None,
        "per_task": per_task,
    }


def _fmt(value: Any) -> str:
    """Format a metric cell."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(stages: dict[str, dict[str, Any]]) -> str:
    """Render stages side by side."""
    names = list(stages)
    rows = [
        ("Resolved % (all)", "resolved_pct"), ("Resolved % (lift basis, excl. C4)", "resolved_pct_lift_basis"),
        ("Compaction Survival %", "compaction_survival_pct"), ("Survival n", "compaction_survival_n"),
        ("P1-4 directive metric (C4)", "p14_directive_metric"), ("Avg LLM calls / run", "avg_llm_calls"),
        ("Avg summarize calls / run", "avg_summarize_calls"), ("Avg tool calls / run", "avg_tool_calls"),
        ("update_file error rate", "update_file_error_rate"),
        ("Shell log re-reads (Suite B)", "shell_log_rereads_suite_b"),
        ("list_skills calls (Suite D)", "list_skills_calls_suite_d"),
        ("LLM calls before first use_skill (D)", "llm_calls_before_first_use_skill_suite_d"),
        ("Wrong skill loads (D)", "wrong_skill_loads_suite_d"), ("Total cost USD", "total_cost_usd"),
        ("Cost per resolved run USD", "cost_per_resolved_usd"), ("Cached input share", "cached_input_share"),
        ("Runs", "runs"),
    ]
    lines = ["# kesoku-bench scorecard", "", "| Metric | " + " | ".join(names) + " |",
             "|---|" + "---|" * len(names)]
    for label, key in rows:
        lines.append(f"| {label} | " + " | ".join(_fmt(stages[n].get(key)) for n in names) + " |")
    suites = sorted({s for n in names for s in stages[n].get("resolved_pct_by_suite", {})})
    for s in suites:
        lines.append(f"| Resolved % suite {s} | " + " | ".join(
            _fmt(stages[n].get("resolved_pct_by_suite", {}).get(s)) for n in names) + " |")
    lines += ["", "## Per task (resolved fraction over repeats)", "",
              "| Task | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    tasks = sorted({t for n in names for t in stages[n].get("per_task", {})})
    for t in tasks:
        cells = []
        for n in names:
            info = stages[n].get("per_task", {}).get(t)
            cells.append("—" if not info else f"{info['resolved_frac']:.2f} ({'/'.join(map(str, info['statuses']))})")
        label = f"{t} (regression guard)" if t in LIFT_EXCLUDED else t
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "D1's stage signal is skill discovery (calls before first use_skill / list_skills), not pass/fail."]
    return "\n".join(lines) + "\n"


def build_report(stage_dirs: dict[str, Path], out_dir: Path | None = None) -> tuple[str, dict[str, Any]]:
    """Build the scorecard for one or more stages and optionally write scorecard.md/json.

    Args:
        stage_dirs: Mapping stage label -> results directory (stage or stage/run_id).
        out_dir: Optional output directory.

    Returns:
        (markdown, json-able metrics dict).
    """
    stages = {name: stage_metrics(load_results(path)) for name, path in stage_dirs.items()}
    md = render_markdown(stages)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "scorecard.md").write_text(md, encoding="utf-8")
        (out_dir / "scorecard.json").write_text(json.dumps(stages, indent=2), encoding="utf-8")
    return md, stages


def default_stage_dir(stage: str, run_id: str | None = None) -> Path:
    """results/<stage>[/<run_id>]."""
    path = RESULTS_DIR / stage
    return path / run_id if run_id else path
