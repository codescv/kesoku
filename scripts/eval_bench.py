"""kesoku-bench CLI: run | verify-gold | report | _run_one (internal child entry).

Examples:
    uv run python scripts/eval_bench.py verify-gold --tasks A1,B3
    uv run python scripts/eval_bench.py run --tasks A1,B3,C2,D3 --stage baseline --repeats 1
    uv run python scripts/eval_bench.py report --stage baseline --stage p0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.kesoku_bench.harness.fixture import DEFAULT_WORK_ROOT  # noqa: E402
from benchmarks.kesoku_bench.harness.report import build_report, default_stage_dir  # noqa: E402
from benchmarks.kesoku_bench.harness.runner import RunOptions, result_path, run_suite, run_task  # noqa: E402
from benchmarks.kesoku_bench.harness.taskspec import RESULTS_DIR, TASKS_DIR, load_task, select_tasks  # noqa: E402
from benchmarks.kesoku_bench.harness.verify_gold import (  # noqa: E402
    format_reports,
    reports_to_json,
    verify_tasks,
)

HIGH_VARIANCE_PREFIXES = ("C", "B3", "D3")


def _split(value: str | None) -> list[str] | None:
    """Split a comma-separated CLI list."""
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _add_run_options(p: argparse.ArgumentParser) -> None:
    """Options shared by `run` and `_run_one`."""
    p.add_argument("--stage", default="baseline")
    p.add_argument("--run-id", default=None)
    p.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT))
    p.add_argument("--holdout-root", default=None)
    p.add_argument("--max-tool-iters", type=int, default=25)
    p.add_argument("--max-task-cost-usd", type=float, default=2.0)
    p.add_argument("--max-suite-cost-usd", type=float, default=50.0)
    p.add_argument("--turn-timeout", type=float, default=None)
    p.add_argument("--final-settle", type=float, default=20.0)
    p.add_argument("--llm-call-timeout", type=float, default=300.0)
    p.add_argument("--max-auto-replies", type=int, default=2)
    p.add_argument("--thinking-level", default=None, choices=["minimal", "low", "medium", "high"])
    p.add_argument("--model", default=None)
    p.add_argument("--natural-compaction", action="store_true",
                   help="diagnostic pass: C/D3 use compaction_small and no forced points (never graded)")
    p.add_argument("--no-stub-embeddings", action="store_true")
    p.add_argument("--prices-json", default=None)


def _options(args: argparse.Namespace) -> RunOptions:
    """Build RunOptions from parsed args."""
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    if args.natural_compaction and not run_id.endswith("-natural"):
        run_id += "-natural"
    return RunOptions(
        stage=args.stage, run_id=run_id, work_root=Path(args.work_root),
        holdout_root=Path(args.holdout_root) if args.holdout_root else None,
        max_tool_iters=args.max_tool_iters, max_task_cost_usd=args.max_task_cost_usd,
        max_suite_cost_usd=args.max_suite_cost_usd, turn_timeout_s=args.turn_timeout,
        final_settle_s=args.final_settle, llm_call_timeout_s=args.llm_call_timeout,
        thinking_level=args.thinking_level, model_name=args.model, natural_compaction=args.natural_compaction,
        stub_embeddings=not args.no_stub_embeddings, prices_json=args.prices_json,
        max_auto_replies=args.max_auto_replies,
    )


def cmd_run(args: argparse.Namespace) -> int:
    """Parent orchestrator."""
    specs = select_tasks(_split(args.tasks), _split(args.suites))
    if not specs:
        print("no tasks selected", file=sys.stderr)
        return 2
    opts = _options(args)
    repeats_by_task = None
    if args.recommended_repeats:
        repeats_by_task = {s.id: (3 if s.id.startswith(HIGH_VARIANCE_PREFIXES) else 1) for s in specs}
    results = run_suite(specs, opts, repeats=args.repeats, jobs=args.jobs, repeats_by_task=repeats_by_task)
    out_dir = RESULTS_DIR / opts.stage / opts.run_id
    md, _ = build_report({opts.stage: out_dir}, out_dir=out_dir)
    print(md)
    print(f"results: {out_dir}")
    return 0 if all(r.get("status") != "error" for r in results) else 1


def cmd_run_one(args: argparse.Namespace) -> int:
    """Child entry: run one task in this process and write its TaskResult JSON."""
    spec = load_task(TASKS_DIR / args.task) if (TASKS_DIR / args.task).is_dir() else select_tasks([args.task])[0]
    opts = _options(args)
    opts.suite_spent_usd = args.suite_spent_usd
    opts.repeat = args.repeat
    result = asyncio.run(run_task(spec, opts))
    out = Path(args.out) if args.out else result_path(opts, spec.id, opts.repeat)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"{spec.id}: {result['status']} (${result['totals']['cost_usd']:.4f}) -> {out}")
    return 0


def cmd_verify_gold(args: argparse.Namespace) -> int:
    """$0 fixture validation."""
    specs = select_tasks(_split(args.tasks), _split(args.suites))
    if not specs:
        print("no tasks selected", file=sys.stderr)
        return 2
    reports = verify_tasks(specs, jobs=args.jobs, keep=args.keep)
    print(format_reports(reports))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(reports_to_json(reports), indent=2), encoding="utf-8")
    return 0 if all(r.ok for r in reports) else 1


def cmd_report(args: argparse.Namespace) -> int:
    """Scorecard for one or more stages (label=path or stage names under results/)."""
    stage_dirs: dict[str, Path] = {}
    for item in args.stage:
        if "=" in item:
            label, path = item.split("=", 1)
            stage_dirs[label] = Path(path)
        else:
            stage_dirs[item] = default_stage_dir(item, args.run_id)
    md, _ = build_report(stage_dirs, out_dir=Path(args.out) if args.out else None)
    print(md)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """CLI parser."""
    parser = argparse.ArgumentParser(prog="eval_bench", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run tasks against the live model")
    p_run.add_argument("--tasks", default=None, help="comma-separated ids or prefixes (A1,B3,...)")
    p_run.add_argument("--suites", default=None, help="comma-separated suite letters")
    p_run.add_argument("--repeats", type=int, default=1)
    p_run.add_argument("--recommended-repeats", action="store_true", help="3 repeats for C*, B3, D3; 1 otherwise")
    p_run.add_argument("--jobs", type=int, default=2)
    _add_run_options(p_run)
    p_run.set_defaults(func=cmd_run)

    p_one = sub.add_parser("_run_one", help=argparse.SUPPRESS)
    p_one.add_argument("--task", required=True)
    p_one.add_argument("--out", default=None)
    p_one.add_argument("--repeat", type=int, default=1)
    p_one.add_argument("--suite-spent-usd", type=float, default=0.0)
    _add_run_options(p_one)
    p_one.set_defaults(func=cmd_run_one)

    p_gold = sub.add_parser("verify-gold", help="$0 fixture validation")
    p_gold.add_argument("--tasks", default=None)
    p_gold.add_argument("--suites", default=None)
    p_gold.add_argument("--jobs", type=int, default=4)
    p_gold.add_argument("--keep", action="store_true", help="keep throw-away fixtures")
    p_gold.add_argument("--json-out", default=None)
    p_gold.set_defaults(func=cmd_verify_gold)

    p_rep = sub.add_parser("report", help="scorecard for one or more stages")
    p_rep.add_argument("--stage", action="append", required=True, help="stage name or label=path (repeatable)")
    p_rep.add_argument("--run-id", default=None)
    p_rep.add_argument("--out", default=None)
    p_rep.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
