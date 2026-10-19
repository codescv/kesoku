"""$0 fixture validation (spec §2.7): visible tests pass, F2P fail before gold, everything passes after gold."""

from __future__ import annotations

import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.kesoku_bench.harness.fixture import (
    materialize_grading_tests,
    overlay_gold,
    restore_visible_tests,
    sanitize_process_env,
    uv_sync,
)
from benchmarks.kesoku_bench.harness.grading import (
    GradeContext,
    load_grade_module,
    run_behavioral,
    run_pytest,
)
from benchmarks.kesoku_bench.harness.taskspec import TaskSpec

GOLD_TESTS_TIME_LIMIT_S = 10.0


@dataclass
class GoldReport:
    """Verification report for one task."""

    task_id: str
    ok: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    steps: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0


def _shutil_ignore() -> Any:
    """Ignore patterns for copying a task repo."""
    return shutil.ignore_patterns(".venv", "__pycache__", ".pytest_cache", "*.pyc", ".ruff_cache")


def verify_task(spec: TaskSpec, work_root: Path | None = None, tests_time_limit_s: float = GOLD_TESTS_TIME_LIMIT_S,
                keep: bool = False) -> GoldReport:
    """Run the four verify-gold steps for one task in a throw-away copy of its repo.

    Args:
        spec: Task specification.
        work_root: Optional parent dir for the throw-away fixture.
        tests_time_limit_s: Max junit test time of the post-gold run.
        keep: Keep the throw-away directory (debugging).

    Returns:
        The GoldReport (ok=True iff no errors).
    """
    report = GoldReport(task_id=spec.id)
    start = time.monotonic()
    tmp = Path(tempfile.mkdtemp(prefix=f"kb-gold-{spec.id}-", dir=str(work_root) if work_root else None))
    repo = tmp / "repo"
    pristine = tmp / "visible_tests_pristine"
    try:
        shutil.copytree(spec.repo_dir, repo, ignore=_shutil_ignore())
        if (repo / "tests").is_dir():
            shutil.copytree(repo / "tests", pristine)
        sync = uv_sync(repo)
        if sync.returncode != 0:
            report.errors.append(f"uv sync failed: {sync.stderr[-1000:]}")
            return report

        # 1. visible tests pass
        visible = run_pytest(repo, ["tests"], tmp / "junit-visible.xml")
        report.steps["visible"] = {"rc": visible.returncode, "summary": visible.summary()}
        if visible.returncode != 0 or not visible.cases:
            report.errors.append(f"visible tests do not pass (rc={visible.returncode}): {visible.tail[-1500:]}")

        # 2. held-out tests: F2P fail, P2P pass
        materialize_grading_tests(spec, repo)
        ids = spec.grading.fail_to_pass + spec.grading.pass_to_pass
        before = run_pytest(repo, ids, tmp / "junit-before.xml")
        f2p_before = before.evaluate(spec.grading.fail_to_pass)
        p2p_before = before.evaluate(spec.grading.pass_to_pass, require_pass=False)
        report.steps["before_gold"] = {"fail_to_pass": f2p_before, "pass_to_pass": p2p_before,
                                       "summary": before.summary()}
        passing_f2p = [k for k, v in f2p_before.items() if v]
        if passing_f2p:
            report.errors.append(f"FAIL_TO_PASS ids already pass before gold: {passing_f2p}")
        failing_p2p = [k for k, v in p2p_before.items() if not v]
        if failing_p2p:
            report.errors.append(f"PASS_TO_PASS ids fail before gold: {failing_p2p}")

        # 3. gold overlay (+ apply.py), restore shipped visible tests by path, everything passes
        try:
            overlay_gold(spec, repo)
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"gold overlay failed: {exc}")
            return report
        restore_visible_tests(repo, pristine)
        after = run_pytest(repo, ["tests"], tmp / "junit-after.xml")
        f2p_after = after.evaluate(spec.grading.fail_to_pass)
        p2p_after = after.evaluate(spec.grading.pass_to_pass, require_pass=False)
        report.steps["after_gold"] = {"fail_to_pass": f2p_after, "pass_to_pass": p2p_after,
                                      "summary": after.summary(), "tests_time_s": round(after.tests_time_s, 2),
                                      "wall_s": round(after.duration_s, 2)}
        not_ok = [k for k, v in {**f2p_after, **p2p_after}.items() if not v]
        if not_ok:
            report.errors.append(f"ids not passing after gold: {not_ok}\n{after.tail[-1500:]}")
        if any(c.outcome in ("failed", "error") for c in after.cases) or after.returncode not in (0,):
            report.errors.append(f"post-gold full test run not green (rc={after.returncode}): {after.tail[-1500:]}")
        if after.tests_time_s > tests_time_limit_s:
            report.errors.append(f"post-gold tests took {after.tests_time_s:.1f}s > {tests_time_limit_s}s")

        # 4. behavioural grader in gold mode (file-based checks only)
        if spec.grade_path:
            if not spec.grade_path.is_file():
                report.errors.append(f"grading.behavioral file missing: {spec.grade_path}")
            else:
                module = load_grade_module(spec.grade_path)
                ctx = GradeContext(task_id=spec.id, task_dir=spec.task_dir, repo_dir=repo, gold_mode=True,
                                   junit={**f2p_after, **p2p_after})
                outcome = run_behavioral(module, ctx)
                report.steps["behavioral_gold"] = {"checks": outcome.checks,
                                                   "diagnostics": _jsonable(outcome.diagnostics)}
                failed = [k for k, v in outcome.checks.items() if v is False]
                if failed:
                    report.errors.append(f"behavioural checks failed in gold mode: {failed}")
                unknown = [r for r in spec.grading.required_behavioral if r not in outcome.checks]
                if unknown:
                    report.errors.append(f"required_behavioral not reported by grade.py: {unknown}")
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"verify-gold crashed: {exc!r}")
    finally:
        report.duration_s = round(time.monotonic() - start, 2)
        report.ok = not report.errors
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            report.steps["workdir"] = str(tmp)
    return report


def _jsonable(value: Any) -> Any:
    """Best-effort conversion to JSON-serialisable data."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def verify_tasks(specs: list[TaskSpec], jobs: int = 4, keep: bool = False,
                 tests_time_limit_s: float = GOLD_TESTS_TIME_LIMIT_S) -> list[GoldReport]:
    """Verify several tasks in parallel threads."""
    sanitize_process_env()
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        return list(pool.map(lambda s: verify_task(s, keep=keep, tests_time_limit_s=tests_time_limit_s), specs))


def format_reports(reports: list[GoldReport]) -> str:
    """Human-readable summary."""
    lines = []
    for r in reports:
        mark = "OK  " if r.ok else "FAIL"
        lines.append(f"{mark} {r.task_id} ({r.duration_s:.1f}s)")
        after = r.steps.get("after_gold", {})
        if after:
            lines.append(f"     post-gold tests: {after.get('summary')} in {after.get('tests_time_s')}s")
        for err in r.errors:
            lines.append("     - " + err.replace("\n", "\n       "))
        for warn in r.warnings:
            lines.append("     ~ " + warn)
    ok = sum(r.ok for r in reports)
    lines.append(f"\nverify-gold: {ok}/{len(reports)} tasks OK")
    return "\n".join(lines)


def reports_to_json(reports: list[GoldReport]) -> list[dict[str, Any]]:
    """Serialisable form."""
    return [_jsonable(asdict(r)) for r in reports]
