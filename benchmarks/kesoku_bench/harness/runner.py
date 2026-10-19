"""Task runner: fixture -> session -> scripted turns -> grading -> TaskResult; plus the parent suite orchestrator."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.kesoku_bench.harness.fixture import (
    COMMON_DIR,
    DEFAULT_WORK_ROOT,
    Fixture,
    default_holdout_root,
    materialize,
    materialize_grading_tests,
    restore_visible_tests,
    sanitize_process_env,
    substitute,
)
from benchmarks.kesoku_bench.harness.grading import (
    GradeContext,
    GradeOutcome,
    extract_tool_events,
    group_llm_logs_by_turn,
    load_grade_module,
    message_to_dict,
    required_checks_pass,
    run_behavioral,
    run_pre_cleanup,
    run_pytest,
    tamper_audit,
)
from benchmarks.kesoku_bench.harness.metering import (
    BudgetExceeded,
    Caps,
    CostLedger,
    cross_check_llm_logs,
    load_prices,
)
from benchmarks.kesoku_bench.harness.session import BenchSession, LLMFactory, TurnOutcome
from benchmarks.kesoku_bench.harness.taskspec import RESULTS_DIR, TaskSpec

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_bench.py"
AUTO_REPLY_TEXT = (COMMON_DIR / "auto_reply.md").read_text(encoding="utf-8").strip()
_QUESTION_RE = re.compile(r"```ui\b|\[question:", re.IGNORECASE)
_READ_LIKE_RE = re.compile(r"^\s*(cd\s+\S+\s*&&\s*)?(cat|sed|grep|rg|head|tail|nl|less|awk|wc)\b")
_REREAD_RE = re.compile(r"cmd_output_|hist_tool_|background_job_")
STATUS_ORDER = ("tampered", "error", "budget_exceeded", "timeout", "resolved", "failed")


@dataclass
class RunOptions:
    """Options shared by every task of a run."""

    stage: str = "baseline"
    run_id: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S"))
    work_root: Path = DEFAULT_WORK_ROOT
    holdout_root: Path | None = None
    results_dir: Path = RESULTS_DIR
    max_tool_iters: int = 25
    max_task_cost_usd: float = 2.0
    max_suite_cost_usd: float = 50.0
    suite_spent_usd: float = 0.0
    turn_timeout_s: float | None = None
    final_settle_s: float = 20.0
    llm_call_timeout_s: float = 300.0
    thinking_level: str | None = None
    model_name: str | None = None
    natural_compaction: bool = False
    stub_embeddings: bool = True
    prices_json: str | None = None
    repeat: int = 1
    max_auto_replies: int = 2
    idle_stable_s: float = 1.0
    prewarm: bool = True

    def to_cli_args(self) -> list[str]:
        """Forwardable CLI flags for `_run_one` child processes."""
        args = [
            "--stage", self.stage, "--run-id", self.run_id, "--work-root", str(self.work_root),
            "--max-tool-iters", str(self.max_tool_iters), "--max-task-cost-usd", str(self.max_task_cost_usd),
            "--max-suite-cost-usd", str(self.max_suite_cost_usd), "--final-settle", str(self.final_settle_s),
            "--llm-call-timeout", str(self.llm_call_timeout_s), "--max-auto-replies", str(self.max_auto_replies),
        ]
        if self.holdout_root:
            args += ["--holdout-root", str(self.holdout_root)]
        if self.turn_timeout_s:
            args += ["--turn-timeout", str(self.turn_timeout_s)]
        if self.thinking_level:
            args += ["--thinking-level", self.thinking_level]
        if self.model_name:
            args += ["--model", self.model_name]
        if self.natural_compaction:
            args.append("--natural-compaction")
        if not self.stub_embeddings:
            args.append("--no-stub-embeddings")
        if self.prices_json:
            args += ["--prices-json", self.prices_json]
        return args


def kesoku_sha() -> str:
    """Git SHA of the kesoku worktree under test (with a -dirty suffix if modified)."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                             text=True, timeout=10, check=False).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=10, check=False).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:  # noqa: BLE001
        return "unknown"


def needs_auto_reply(outcome: TurnOutcome) -> bool:
    """Whether the agent asked a question instead of acting (spec §2.4 auto-reply policy)."""
    text = (outcome.final_text or "").strip()
    if not text:
        return False
    if _QUESTION_RE.search(text):
        return True
    return text.endswith(("?", "？")) and outcome.update_file_calls == 0


# --- metrics ---------------------------------------------------------------------------------------


def _turn_tool_metrics(events: list[dict[str, Any]], turn_idx: int) -> dict[str, Any]:
    """Tool-call counters for one scripted turn."""
    evs = [e for e in events if e["turn_idx"] == turn_idx]
    calls = Counter(e["name"] for e in evs)
    errors = Counter(e["name"] for e in evs if e["is_error"])
    not_found = sum(1 for e in evs if e["name"] == "update_file" and "not found" in e["result"])
    return {"tool_calls": dict(calls), "tool_errors": dict(errors), "update_file_not_found": not_found}


def generic_diagnostics(
    spec: TaskSpec, events: list[dict[str, Any]], ledger: CostLedger, skills: list[str]
) -> dict[str, Any]:
    """Suite-independent diagnostics derived from tool events and the ledger (spec §3)."""
    shell = [e for e in events if e["name"] == "run_shell_command"]
    updates = [e for e in events if e["name"] == "update_file"]
    rereads = 0
    for e in shell:
        cmd = str(e["args"].get("command", ""))
        try:
            max_chars = int(e["args"].get("max_output_chars") or 0)
        except (TypeError, ValueError):
            max_chars = 0
        if _REREAD_RE.search(cmd) or max_chars > 5000:
            rereads += 1
    use_skill = [e for e in events if e["name"] == "use_skill"]
    first_use_ts = use_skill[0]["ts"] if use_skill else None
    agent_records = [r for r in ledger.records if r.kind == "agent" and r.blocked is None]
    calls_before_skill = (sum(1 for r in agent_records if r.ts <= first_use_ts) if first_use_ts is not None
                          else None)
    target_skills = {p.name for p in spec.skills_dir.iterdir() if p.is_dir()} if spec.skills_dir.is_dir() else set()
    loaded = [str(e["args"].get("skill_name") or e["args"].get("name") or "") for e in use_skill]
    update_errors = sum(1 for e in updates if e["is_error"])
    return {
        "read_like_shell_calls": sum(1 for e in shell if _READ_LIKE_RE.search(str(e["args"].get("command", "")))),
        "shell_log_rereads": rereads,
        "update_file_calls": len(updates),
        "update_file_errors": update_errors,
        "update_file_error_rate": round(update_errors / len(updates), 4) if updates else None,
        "first_edit_file": str(updates[0]["args"].get("file_name", "")) if updates else None,
        "list_skills_calls": sum(1 for e in events if e["name"] == "list_skills"),
        "use_skill_calls": len(use_skill),
        "llm_calls_before_first_use_skill": calls_before_skill,
        "wrong_skill_loads": sum(1 for n in loaded if target_skills and n not in target_skills),
        "skills_installed": len(skills),
        "tool_calls_missing_result": sum(1 for e in events if not e["has_result"]),
        "tool_iter_cap_turns": sorted({r.turn_idx for r in ledger.records
                                       if r.blocked and r.blocked.startswith("max_tool_iters")}),
    }


def _compaction_attribution(
    nodes: list[dict[str, Any]], turn_records: list[dict[str, Any]]
) -> tuple[dict[int, int], list[int]]:
    """Count summary nodes created per turn and list turns with natural (in-turn) compaction.

    Nodes created between a turn's `begin_ts` and `start_ts` come from the harness's forced point; nodes
    created during a user turn's content window are natural compactions; command turns are manual.
    """
    per_turn: dict[int, int] = {}
    natural: list[int] = []
    for rec in turn_records:
        nxt = [r["begin_ts"] for r in turn_records if r["idx"] == rec["idx"] + 1]
        end = nxt[0] if nxt else float("inf")
        in_turn = [n for n in nodes if rec["begin_ts"] <= n["created_at"] < end]
        per_turn[rec["idx"]] = len(in_turn)
        if rec["kind"] == "user" and any(n["created_at"] >= rec["start_ts"] for n in in_turn):
            natural.append(rec["idx"])
    return per_turn, natural


def decide_status(*, tampered: bool, error: str | None, budget: bool, timeout: bool, resolved: bool) -> str:
    """Status precedence: tampered > error > budget_exceeded > timeout > resolved > failed."""
    if tampered:
        return "tampered"
    if error:
        return "error"
    if budget:
        return "budget_exceeded"
    if timeout:
        return "timeout"
    return "resolved" if resolved else "failed"


# --- single task -----------------------------------------------------------------------------------


async def run_task(spec: TaskSpec, opts: RunOptions, llm_factory: LLMFactory | None = None) -> dict[str, Any]:
    """Run one task end-to-end in this process and return the TaskResult dict.

    Args:
        spec: Task specification.
        opts: Run options.
        llm_factory: Optional inner-LLM factory (tests inject MockLLM); default is the configured Gemini.

    Returns:
        TaskResult (spec §3).
    """
    sanitize_process_env()
    t0 = time.time()
    holdout_root = Path(opts.holdout_root) if opts.holdout_root else default_holdout_root()
    natural = opts.natural_compaction and spec.task.config_profile == "compaction_forced"
    fixture = materialize(
        spec, Path(opts.work_root), opts.run_id, holdout_root,
        profile="compaction_small" if natural else None, thinking_level=opts.thinking_level,
        model_name=opts.model_name, prewarm=opts.prewarm,
    )
    ledger = CostLedger(load_prices(opts.prices_json), Caps(
        max_tool_iters=opts.max_tool_iters, max_task_cost_usd=opts.max_task_cost_usd,
        max_suite_cost_usd=opts.max_suite_cost_usd, suite_spent_usd=opts.suite_spent_usd,
    ))
    session = BenchSession(spec.id, fixture.config_path, ledger, llm_factory=llm_factory,
                           stub_embeddings=opts.stub_embeddings, llm_call_timeout_s=opts.llm_call_timeout_s,
                           idle_stable_s=opts.idle_stable_s)
    turn_timeout = opts.turn_timeout_s or spec.task.turn_timeout_s
    grade_module = load_grade_module(spec.grade_path) if spec.grade_path and spec.grade_path.is_file() else None

    turn_records: list[dict[str, Any]] = []
    extras: dict[str, Any] = {"port": fixture.port, "forced_compaction_turns": [], "natural_compaction": natural}
    error: str | None = None
    timed_out = False
    overhead = 0
    messages: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    llm_logs: list[Path] = []
    pre_ctx: GradeContext | None = None
    try:
        await session.start()
        overhead = await session.overhead_tokens()
        for idx, turn in enumerate(spec.turns, start=1):
            ledger.begin_turn(idx)
            text = substitute(turn.text, fixture.port, fixture.ws_dir)
            rec: dict[str, Any] = {"idx": idx, "kind": turn.kind, "text": text, "begin_ts": time.time(),
                                   "compactions_before": 0, "auto_replies": [], "status": "pending",
                                   "command_reply": None}
            if idx in spec.task.force_compact_before_turn and not natural:
                rec["compactions_before"] = await session.force_compaction(overhead)
                extras["forced_compaction_turns"].append(idx)
            rec["start_ts"] = time.time()
            if turn.kind == "user":
                outcome = await session.send_user_message(text, turn_timeout)
                rec.update(status=outcome.status, msg_id=outcome.msg_id, final_text=outcome.final_text[-2000:])
                replies = 0
                while (outcome.status == "processed" and replies < opts.max_auto_replies
                       and needs_auto_reply(outcome) and not ledger.budget_exceeded):
                    replies += 1
                    outcome = await session.send_user_message(AUTO_REPLY_TEXT, turn_timeout)
                    rec["auto_replies"].append({"msg_id": outcome.msg_id, "status": outcome.status})
                    rec.update(status=outcome.status, final_text=outcome.final_text[-2000:])
            else:
                before = await session.summary_node_count()
                reply = await session.send_command(text)
                delta = await session.summary_node_count() - before
                rec.update(status="processed", command_reply=reply)
                extras.update(manual_compact_nodes_delta=delta, manual_compact_reply=reply)
                if spec.task.fallback_force_if_noop:
                    extras["fallback_forced"] = delta == 0
                    if delta == 0:
                        rec["compactions_before"] += await session.force_compaction(overhead, protect_tail_turns=2)
            rec["end_ts"] = time.time()
            turn_records.append(rec)
            if ledger.budget_exceeded:
                break
            if rec["status"] == "timeout":
                timed_out = True
                if not spec.task.continue_on_timeout:
                    break
        if not ledger.budget_exceeded:
            await session.settle(opts.final_settle_s, turn_timeout)

        # Snapshots + pre-cleanup checks (D3 orphan check) while the session and its jobs are still alive.
        windows = [(r["idx"], r["begin_ts"]) for r in turn_records]
        messages = [message_to_dict(m, windows) for m in await session.all_messages()]
        nodes = await session.all_summary_nodes()
        llm_logs = session.llm_logs()
        extras["active_jobs_for_session"] = session.active_jobs_for_session()
        pre_ctx = _build_context(spec, fixture, session, turn_records, messages, nodes, llm_logs, extras)
        extras["pre_cleanup"] = run_pre_cleanup(grade_module, pre_ctx)
    except BudgetExceeded as exc:
        ledger.budget_exceeded = ledger.budget_exceeded or str(exc)
    except Exception:  # noqa: BLE001 - recorded in the result, the task is still graded
        error = traceback.format_exc()[-4000:]
    finally:
        try:
            if session.session is not None and not messages:
                windows = [(r["idx"], r["begin_ts"]) for r in turn_records]
                messages = [message_to_dict(m, windows) for m in await session.all_messages()]
                nodes = await session.all_summary_nodes()
                llm_logs = session.llm_logs()
        except Exception:  # noqa: BLE001
            pass
        await session.stop()

    result = grade_live_run(spec, fixture, session, turn_records, messages, nodes, llm_logs, extras, grade_module,
                            ledger=ledger)
    totals = ledger.totals()
    per_turn_nodes, natural_turns = _compaction_attribution(nodes, turn_records)
    events = result.pop("_tool_events")
    turns_out = []
    for rec in turn_records:
        led = ledger.turns.get(rec["idx"])
        turns_out.append({
            "idx": rec["idx"], "kind": rec["kind"], "prompt_preview": rec["text"][:200], "status": rec["status"],
            "auto_replies": len(rec["auto_replies"]), "llm_calls": led.llm_calls if led else 0,
            "summarize_calls": led.summarize_calls if led else 0, **_turn_tool_metrics(events, rec["idx"]),
            "tokens": led.tokens() if led else {}, "cost_usd": round(led.cost_usd, 6) if led else 0.0,
            "duration_s": round(rec.get("end_ts", rec["start_ts"]) - rec["begin_ts"], 2),
            "compactions": per_turn_nodes.get(rec["idx"], 0), "forced_nodes": rec["compactions_before"],
            "command_reply": rec["command_reply"], "begin_ts": rec["begin_ts"], "start_ts": rec["start_ts"],
            "end_ts": rec.get("end_ts"),
        })
    totals.update({
        "tool_calls": len(events), "duration_s": round(time.time() - t0, 2),
        "compactions_l0": sum(1 for n in nodes if n["level"] == 0),
        "compactions_l1": sum(1 for n in nodes if n["level"] >= 1),
        "forced_compaction_turns": extras["forced_compaction_turns"], "natural_compaction_turns": natural_turns,
        "overhead_tokens": overhead,
    })
    status = decide_status(tampered=result["tampered"], error=error, budget=bool(ledger.budget_exceeded),
                           timeout=timed_out, resolved=result["resolved"])
    cross = cross_check_llm_logs(llm_logs, ledger)
    diagnostics = {
        **generic_diagnostics(spec, events, ledger, fixture.extras.get("skills", [])),
        "questions_asked": sum(len(r["auto_replies"]) for r in turn_records),
        "budget_exceeded_reason": ledger.budget_exceeded,
        "log_cross_check": asdict(cross),
        "uv_sync_rc": fixture.extras.get("uv_sync_rc"),
        "llm_client_rebuilds": session.llm.client_rebuilds if session.llm else 0,
        **extras,
        **result["behavioral_diagnostics"],
    }
    return {
        "task_id": spec.id, "suite": spec.suite, "stage": opts.stage, "run_id": opts.run_id,
        "kesoku_sha": kesoku_sha(), "model": opts.model_name or "gemini-3.8-flash",
        "thinking_level": opts.thinking_level or "medium", "profile": fixture.profile, "status": status,
        "resolved": result["resolved"], "turns": turns_out, "totals": totals,
        "grading": {k: result[k] for k in ("fail_to_pass", "pass_to_pass", "behavioral", "required_behavioral",
                                            "tampered", "tamper_hits", "restored_tests", "pytest_tail")},
        "diagnostics": diagnostics, "error": error, "repeat": opts.repeat, "workdir": str(fixture.work_dir),
    }


def _build_context(
    spec: TaskSpec,
    fixture: Fixture,
    session: BenchSession,
    turn_records: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    llm_logs: list[Path],
    extras: dict[str, Any],
    junit: dict[str, bool] | None = None,
) -> GradeContext:
    """Assemble the GradeContext for a live run."""
    windows = [(r["idx"], r["begin_ts"]) for r in turn_records]
    turns = [{
        "idx": r["idx"], "kind": r["kind"], "text": r["text"], "start_ts": r["begin_ts"],
        "end_ts": r.get("end_ts"), "status": r["status"], "compactions_before": r["compactions_before"],
        "command_reply": r["command_reply"], "auto_replies": len(r["auto_replies"]),
    } for r in turn_records]
    return GradeContext(
        task_id=spec.id, task_dir=spec.task_dir, repo_dir=fixture.repo_dir, gold_mode=False,
        ws_dir=fixture.ws_dir, staging_dir=session.staging_dir, session_id=session.session.id if session.session
        else None, db_path=Path(session.cfg.workspace.db_path) if session.cfg else None, turns=turns,
        tool_events=extract_tool_events(messages), messages=messages, summary_nodes=nodes, llm_logs=llm_logs,
        llm_log_turn=group_llm_logs_by_turn(llm_logs, windows), junit=junit or {}, extras=extras,
    )


def grade_live_run(
    spec: TaskSpec,
    fixture: Fixture,
    session: BenchSession,
    turn_records: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    llm_logs: list[Path],
    extras: dict[str, Any],
    grade_module: Any,
    ledger: CostLedger | None = None,
) -> dict[str, Any]:
    """Grade a finished run (spec §2.8): audit, restore tests, materialise F2P, pytest, behavioural checks."""
    events = extract_tool_events(messages)
    hits = tamper_audit(events)
    restored = restore_visible_tests(fixture.repo_dir, fixture.pristine_tests_dir)
    materialize_grading_tests(spec, fixture.repo_dir, staging_dir=fixture.grading_dir)
    ids = spec.grading.fail_to_pass + spec.grading.pass_to_pass
    run = run_pytest(fixture.repo_dir, ids, fixture.holdout_dir / "junit.xml", timeout_s=spec.grading.pytest_timeout_s)
    f2p = run.evaluate(spec.grading.fail_to_pass)
    p2p = run.evaluate(spec.grading.pass_to_pass, require_pass=False)
    outcome = GradeOutcome()
    if grade_module is not None:
        ctx = _build_context(spec, fixture, session, turn_records, messages, nodes, llm_logs, extras,
                             junit={**f2p, **p2p})
        outcome = run_behavioral(grade_module, ctx)
    required_ok = required_checks_pass(outcome, spec.grading.required_behavioral)
    resolved = all(f2p.values()) and all(p2p.values()) and not hits and required_ok
    return {
        "fail_to_pass": f2p, "pass_to_pass": p2p, "behavioral": outcome.checks,
        "behavioral_diagnostics": outcome.diagnostics, "required_behavioral": spec.grading.required_behavioral,
        "tampered": bool(hits), "tamper_hits": hits, "restored_tests": restored, "pytest_tail": run.tail[-2000:],
        "resolved": resolved, "_tool_events": events,
    }


# --- parent orchestrator ---------------------------------------------------------------------------


def result_path(opts: RunOptions, task_id: str, repeat: int) -> Path:
    """Result JSON path: results/<stage>/<run_id>/<task_id>[.rN].json."""
    suffix = f".r{repeat}" if repeat > 1 else ""
    return Path(opts.results_dir) / opts.stage / opts.run_id / f"{task_id}{suffix}.json"


def _child_timeout(spec: TaskSpec, opts: RunOptions) -> float:
    """Upper bound on a child process's wall time."""
    per_turn = opts.turn_timeout_s or spec.task.turn_timeout_s
    return per_turn * (len(spec.turns) * (1 + opts.max_auto_replies)) + opts.final_settle_s + 900


def run_suite(specs: list[TaskSpec], opts: RunOptions, repeats: int = 1, jobs: int = 2,
              repeats_by_task: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """Run tasks in child processes (`eval_bench.py _run_one`) with a suite-wide cost breaker.

    Args:
        specs: Tasks to run.
        opts: Shared options (run_id/stage/... forwarded to children).
        repeats: Default number of repeats per task.
        jobs: Parallel child processes.
        repeats_by_task: Optional per-task repeat overrides.

    Returns:
        The TaskResult dicts (failed children produce an `error` result).
    """
    work: list[tuple[TaskSpec, int]] = []
    for spec in specs:
        n = (repeats_by_task or {}).get(spec.id, repeats)
        work.extend((spec, r) for r in range(1, n + 1))
    results: list[dict[str, Any]] = []
    spent = 0.0

    def _launch(item: tuple[TaskSpec, int], suite_spent: float) -> dict[str, Any]:
        spec, rep = item
        out = result_path(opts, spec.id, rep)
        out.parent.mkdir(parents=True, exist_ok=True)
        run_id = opts.run_id if rep == 1 else f"{opts.run_id}.r{rep}"
        child_opts = RunOptions(**{**asdict(opts), "run_id": run_id, "suite_spent_usd": suite_spent, "repeat": rep})
        cmd = [sys.executable, str(EVAL_SCRIPT), "_run_one", "--task", spec.id, "--out", str(out),
               "--repeat", str(rep), "--suite-spent-usd", f"{suite_spent:.6f}",
               *child_opts.to_cli_args()]
        cmd[cmd.index("--run-id") + 1] = run_id
        try:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                                  timeout=_child_timeout(spec, opts), check=False)
            if out.is_file():
                return json.loads(out.read_text(encoding="utf-8"))
            err = f"child exited {proc.returncode} without result\n{proc.stderr[-4000:]}"
        except subprocess.TimeoutExpired:
            err = "child process timed out"
        res = {"task_id": spec.id, "suite": spec.suite, "stage": opts.stage, "run_id": run_id, "status": "error",
               "resolved": False, "error": err, "repeat": rep, "totals": {"cost_usd": 0.0}, "turns": []}
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        return res

    pending = list(work)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        running: dict[Any, tuple[TaskSpec, int]] = {}
        while pending or running:
            while pending and len(running) < max(1, jobs) and spent < opts.max_suite_cost_usd:
                item = pending.pop(0)
                running[pool.submit(_launch, item, spent)] = item
            if not running:
                break
            done_set, _ = wait(list(running), return_when=FIRST_COMPLETED)
            done = next(iter(done_set))
            running.pop(done)
            res = done.result()
            spent += float((res.get("totals") or {}).get("cost_usd") or 0.0)
            results.append(res)
            print(f"[kesoku-bench] {res['task_id']} r{res.get('repeat', 1)}: {res['status']} "
                  f"(${(res.get('totals') or {}).get('cost_usd', 0):.3f}, suite ${spent:.2f})", flush=True)
    for spec, rep in pending:
        results.append({"task_id": spec.id, "suite": spec.suite, "stage": opts.stage, "run_id": opts.run_id,
                        "status": "skipped_suite_budget", "resolved": False, "repeat": rep,
                        "totals": {"cost_usd": 0.0}, "turns": []})
    return results

