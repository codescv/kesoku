"""Grading helpers: pytest/junit runner, tool-event extraction, tamper audit, and behavioural graders."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from kesoku.testing.fixture import sanitized_env

TAMPER_RE = re.compile(r"kesoku_bench|kesoku-wt|kb-holdout|/gold/|\.holdout")
_SHELL_EXIT_CODE_RE = re.compile(r"Command failed with exit code (-?\d+)")


@dataclass
class GradeOutcome:
    """Behavioural checks (True/False, None = skipped_in_gold_mode) and free-form diagnostics."""

    checks: dict[str, bool | None] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class GradeContext:
    """Context object inspected by behavioural `grade.py` modules."""

    task_id: str
    task_dir: Path
    repo_dir: Path
    gold_mode: bool = False
    ws_dir: Path | None = None
    staging_dir: Path | None = None
    session_id: str | None = None
    db_path: Path | None = None
    turns: list[dict[str, Any]] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    summary_nodes: list[dict[str, Any]] = field(default_factory=list)
    llm_logs: list[Path] = field(default_factory=list)
    llm_log_turn: dict[int, list[Path]] = field(default_factory=dict)
    junit: dict[str, bool] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def run(self, cmd: list[str], timeout: float = 60.0, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Run a command in `repo_dir` with a sanitized environment (never raises on non-zero exit)."""
        kwargs.setdefault("env", sanitized_env())
        return subprocess.run(
            cmd,
            cwd=self.repo_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            **kwargs,
        )

    def read(self, relpath: str) -> str | None:
        """Read a repo-relative text file, or return None if absent."""
        path = self.repo_dir / relpath
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8", errors="replace")


def normalize_outcome(raw: Any) -> GradeOutcome:
    """Accept a GradeOutcome-like object, a `{"checks", "diagnostics"}` dict, or a flat dict of checks."""
    if raw is None:
        return GradeOutcome()
    if hasattr(raw, "checks") and not isinstance(raw, dict):
        return GradeOutcome(
            checks=dict(raw.checks or {}),
            diagnostics=dict(getattr(raw, "diagnostics", {}) or {}),
        )
    if isinstance(raw, dict):
        if "checks" in raw or "diagnostics" in raw:
            return GradeOutcome(
                checks=dict(raw.get("checks") or {}),
                diagnostics=dict(raw.get("diagnostics") or {}),
            )
        checks: dict[str, bool | None] = {}
        diagnostics: dict[str, Any] = {}
        for key, value in raw.items():
            if value is None or isinstance(value, bool):
                checks[key] = value
            else:
                diagnostics[key] = value
        return GradeOutcome(checks=checks, diagnostics=diagnostics)
    raise TypeError(f"unsupported grade() return type: {type(raw).__name__}")


def load_grade_module(path: Path) -> ModuleType:
    """Import a task's `grade.py` under a unique module name."""
    name = f"kb_grade_{path.parent.name}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_behavioral(module: ModuleType, ctx: GradeContext) -> GradeOutcome:
    """Call `grade(ctx, gold_mode=...)`, converting exceptions into a failed `grader_error` check."""
    try:
        return normalize_outcome(module.grade(ctx, gold_mode=ctx.gold_mode))
    except Exception as exc:  # noqa: BLE001 - a crashing grader must not crash the run
        return GradeOutcome(checks={"grader_error": False}, diagnostics={"grader_exception": repr(exc)[:2000]})


def run_pre_cleanup(module: ModuleType | None, ctx: GradeContext) -> dict[str, Any]:
    """Call the optional `pre_cleanup(ctx)` hook before session/jobs are stopped."""
    if module is None or not hasattr(module, "pre_cleanup"):
        return {}
    try:
        return dict(module.pre_cleanup(ctx) or {})
    except Exception as exc:  # noqa: BLE001
        return {"pre_cleanup_exception": repr(exc)[:2000]}


def required_checks_pass(outcome: GradeOutcome, required: list[str], gold_mode: bool = False) -> bool:
    """Return True if all required checks are True (None is accepted only in gold mode)."""
    for name in required:
        value = outcome.checks.get(name)
        if value is True or (gold_mode and value is None):
            continue
        return False
    return True


@dataclass
class CaseResult:
    """One junit testcase."""

    nodeid: str
    file: str
    outcome: str  # passed | failed | error | skipped
    message: str = ""


@dataclass
class PytestRun:
    """Result of one pytest invocation."""

    returncode: int
    cases: list[CaseResult]
    duration_s: float
    tests_time_s: float
    tail: str
    junit_path: Path | None = None

    def evaluate(self, ids: list[str], *, require_pass: bool = True) -> dict[str, bool]:
        """Map requested ids (files or node ids) to pass/fail."""
        out: dict[str, bool] = {}
        for wanted in ids:
            matched = [c for c in self.cases if _matches(c, wanted)]
            bad = any(c.outcome in ("failed", "error") for c in matched)
            passed = any(c.outcome == "passed" for c in matched)
            ok = bool(matched) and not bad and (passed or not require_pass)
            out[wanted] = ok
        return out

    def summary(self) -> dict[str, int]:
        """Return outcome counts."""
        counts: dict[str, int] = {}
        for c in self.cases:
            counts[c.outcome] = counts.get(c.outcome, 0) + 1
        return counts


def _matches(case: CaseResult, wanted: str) -> bool:
    """Return True if a junit case belongs to a requested file or node id."""
    wanted = wanted.strip().rstrip("/")
    if "::" not in wanted:
        return case.file == wanted or case.file.startswith(wanted + "/")
    return case.nodeid == wanted or case.nodeid.startswith(wanted + "::") or case.nodeid.startswith(wanted + "[")


def parse_junit(path: Path) -> tuple[list[CaseResult], float]:
    """Parse an xunit1 junit XML into cases."""
    if not path.is_file():
        return [], 0.0
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return [], 0.0
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    tests_time = sum(float(s.get("time") or 0.0) for s in suites)
    cases: list[CaseResult] = []
    for tc in root.iter("testcase"):
        name = tc.get("name") or ""
        classname = tc.get("classname") or ""
        file = (tc.get("file") or "").replace("\\", "/")
        if not file:
            file = classname.replace(".", "/") + ".py" if classname else ""
        module = file[:-3].replace("/", ".") if file.endswith(".py") else ""
        classes = classname[len(module) :].lstrip(".") if module and classname.startswith(module) else ""
        parts = [file] + ([*classes.split(".")] if classes else []) + [name]
        nodeid = "::".join(p for p in parts if p)
        outcome = "passed"
        message = ""
        for child in tc:
            tag = child.tag
            if tag in ("failure", "error", "skipped"):
                outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[tag]
                message = (child.get("message") or child.text or "")[:500]
                break
        cases.append(CaseResult(nodeid=nodeid, file=file, outcome=outcome, message=message))
    return cases, tests_time


def purge_pycache(repo_dir: Path) -> None:
    """Delete `__pycache__` directories of `repo_dir` (pruning its `.venv`)."""
    for dirpath, dirnames, _ in os.walk(repo_dir):
        if ".venv" in dirnames:
            dirnames.remove(".venv")
        if "__pycache__" in dirnames:
            dirnames.remove("__pycache__")
            shutil.rmtree(os.path.join(dirpath, "__pycache__"), ignore_errors=True)


def run_pytest(
    repo_dir: Path,
    targets: list[str],
    junit_path: Path,
    timeout_s: float = 300.0,
    extra_args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> PytestRun:
    """Run `uv run pytest` in `repo_dir` with junitxml (xunit1).

    Args:
        repo_dir: Target repo directory.
        targets: Files / node ids / directories to run.
        junit_path: Destination path for the junit XML.
        timeout_s: Wall-clock timeout.
        extra_args: Extra pytest arguments.
        env: Optional environment (defaults to sanitized `os.environ`).

    Returns:
        The `PytestRun` result.
    """
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    if junit_path.exists():
        junit_path.unlink()
    files = {t for t in targets if "::" not in t}
    uniq: list[str] = []
    for t in targets:
        if "::" in t and t.split("::", 1)[0] in files:
            continue
        if t not in uniq:
            uniq.append(t)
    purge_pycache(repo_dir)
    run_env = dict(env or sanitized_env())
    run_env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [
        "uv",
        "run",
        "--quiet",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "-o",
        "junit_family=xunit1",
        "--continue-on-collection-errors",
        f"--junitxml={junit_path}",
        *(extra_args or []),
        *uniq,
    ]
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=repo_dir,
            env=run_env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        rc, out = proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        rc = -9
        out = f"pytest timed out after {timeout_s}s\n{exc.stdout or ''}"
    duration = time.monotonic() - start
    cases, tests_time = parse_junit(junit_path)
    return PytestRun(
        returncode=rc,
        cases=cases,
        duration_s=duration,
        tests_time_s=tests_time,
        tail=out[-4000:],
        junit_path=junit_path,
    )


def turn_index_for(ts: float, turn_windows: list[tuple[int, float]]) -> int:
    """Map a timestamp to the scripted turn whose start is the latest one `<= ts` (0 before the first turn)."""
    idx = 0
    for turn_idx, start in turn_windows:
        if ts >= start:
            idx = turn_idx
        else:
            break
    return idx


def message_to_dict(msg: Any, turn_windows: list[tuple[int, float]]) -> dict[str, Any]:
    """Convert a kesoku `Message` into a plain dict with `turn_idx`."""
    return {
        "id": msg.id,
        "role": str(msg.role),
        "type": str(msg.type),
        "content": msg.content,
        "metadata": dict(msg.metadata or {}),
        "timestamp": msg.timestamp,
        "status": str(msg.status),
        "parent_id": msg.parent_id,
        "sender": msg.sender,
        "summary_node_id": msg.summary_node_id,
        "turn_idx": turn_index_for(msg.timestamp, turn_windows),
    }


def _infer_exit_code(tool_name: str, result: str, is_error: bool) -> int | None:
    """Infer integer exit code for `run_shell_command` events, or None for non-shell tools."""
    if tool_name != "run_shell_command":
        return None
    if is_error:
        match = _SHELL_EXIT_CODE_RE.search(result)
        return int(match.group(1)) if match else 1
    if result.startswith("Execution denied:"):
        return 1
    return 0


def extract_tool_events(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair TOOL_CALL messages with their TOOL_RESULT children.

    Returns:
        Events `{turn_idx, name, args, result, is_error, exit_code, ts, call_id, has_result}`.
        `update_file` returns errors as text (`"Error: ..."`), which also counts as `is_error`.
    """
    results_by_parent: dict[str, dict[str, Any]] = {}
    for m in messages:
        if m["type"] == "tool_result" and m.get("parent_id"):
            results_by_parent[m["parent_id"]] = m
    events: list[dict[str, Any]] = []
    for m in messages:
        if m["type"] != "tool_call":
            continue
        meta = m.get("metadata") or {}
        name = meta.get("tool_name") or ""
        res = results_by_parent.get(m["id"])
        rmeta = (res or {}).get("metadata") or {}
        if "tool_error" in rmeta:
            result, is_error = str(rmeta.get("tool_error") or ""), True
        else:
            result = str(rmeta.get("tool_result", (res or {}).get("content", "")) or "")
            is_error = False
        if name == "update_file" and result.startswith("Error"):
            is_error = True
        events.append(
            {
                "turn_idx": m["turn_idx"],
                "name": name,
                "args": dict(meta.get("tool_arguments") or {}),
                "result": result,
                "is_error": is_error,
                "exit_code": _infer_exit_code(name, result, is_error),
                "ts": m["timestamp"],
                "call_id": m["id"],
                "has_result": res is not None,
            }
        )
    return events


def tamper_audit(tool_events: list[dict[str, Any]]) -> list[str]:
    """Return tool-call argument strings that reference the benchmark tree or held-out material."""
    hits: list[str] = []
    for ev in tool_events:
        blob = json.dumps(ev.get("args") or {}, ensure_ascii=False)
        if TAMPER_RE.search(blob):
            hits.append(f"turn {ev['turn_idx']} {ev['name']}: {blob[:300]}")
    return hits


def group_llm_logs_by_turn(log_paths: list[Path], turn_windows: list[tuple[int, float]]) -> dict[int, list[Path]]:
    """Assign `llm-turn-*.log.yaml` files to scripted turns by file mtime."""
    out: dict[int, list[Path]] = {}
    for p in log_paths:
        try:
            ts = p.stat().st_mtime
        except OSError:
            continue
        out.setdefault(turn_index_for(ts, turn_windows), []).append(p)
    return out
