"""Behavioural grader for D3_background_server_lifecycle (spec §4.4).

Live checks (``None`` in gold mode):
  * ``integration_ran_live`` - ``requests.log`` shows >= 2 complete integration sequences (ping, echo, sum all 200).
  * ``server_stopped`` - computed in :func:`pre_cleanup` *before* the harness stops jobs: the PID in ``server.pid``
    is not alive, ``{PORT}`` refuses connections, and no process group of a session job is alive.
  * ``registry_empty`` - the harness's ActiveJobsRegistry holds no job for the session.
  * ``job_tracked`` - the server ran as a tracked background job (false for ``nohup ... &``).
  * ``manage_job_used`` - a job-management tool (P2-1) was called.
Gold-mode checks (``None`` live): timing budgets of the unit and (skipped) integration suites.

Stdlib only.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
from pathlib import Path
from typing import Any

PID_FILE = "server.pid"
REQUEST_LOG = "requests.log"
INTEGRATION_SEQUENCE = ("ping", "echo", "sum")
SERVER_CMD_RE = re.compile(r"mockrpc\.server")
BACKGROUND_RESULT_MARKER = "transitioned to background execution"
SYSTEM_ALERT_MARKER = "[System Alert] Background Job"
PROBE_RE = re.compile(
    r"\b(ps|pgrep|pidof|lsof|ss|netstat|fuser|curl|wget|nc)\b|server\.pid|kill\s+-0|/proc/|/health|jobs\b"
)
UNIT_TARGET_S = 1.5
SUITE_FAIL_S = 4.0
PROBE_TURN = 5


def _get(ctx: Any, name: str, default: Any = None) -> Any:
    """Attribute access that tolerates partial stub contexts."""
    value = getattr(ctx, name, default)
    return default if value is None else value


# --------------------------------------------------------------------------------------------- process probes
def _proc_state(pid: int) -> str | None:
    """Return the /proc state letter of ``pid`` (None if unknown / not Linux)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # Format: "pid (comm) S ..."; comm may contain spaces/parens, so split on the last ')'.
    return stat.rsplit(")", 1)[1].split()[0] if ")" in stat else None


def _pid_alive(pid: int) -> bool:
    """Whether ``pid`` exists and is not a zombie."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return _proc_state(pid) != "Z"


def _pgid_alive(pgid: int) -> bool:
    """Whether any non-zombie process is a member of process group ``pgid``."""
    proc = Path("/proc")
    if proc.is_dir():
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text()
            except OSError:
                continue
            fields = stat.rsplit(")", 1)[1].split()
            # fields[0]=state, fields[1]=ppid, fields[2]=pgrp
            if len(fields) > 2 and fields[2] == str(pgid) and fields[0] != "Z":
                return True
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    """Whether something accepts TCP connections on ``host:port``."""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _read_pid(repo_dir: Path) -> int | None:
    """Parse ``server.pid`` (None if missing or malformed)."""
    try:
        return int((repo_dir / PID_FILE).read_text().strip())
    except (OSError, ValueError):
        return None


def pre_cleanup(ctx: Any) -> dict[str, Any]:
    """Orphan check, run after the last turn and BEFORE the harness stops the session's jobs.

    Args:
        ctx: GradeContext; uses ``repo_dir``, ``extras["port"]`` and ``extras["active_jobs_for_session"]``
            (list of ``{pid, pgid, command}``; absent means "not provided").

    Returns:
        Facts consumed by :func:`grade` via ``ctx.extras["pre_cleanup"]``.
    """
    extras: dict[str, Any] = _get(ctx, "extras", {}) or {}
    repo_dir = Path(_get(ctx, "repo_dir", "."))
    pid = _read_pid(repo_dir)
    pid_alive = _pid_alive(pid) if pid is not None else False
    port = extras.get("port")
    port_open = _port_open(int(port)) if port is not None else None
    jobs = extras.get("active_jobs_for_session")
    live_pgids: list[int] = []
    if jobs is not None:
        for job in jobs:
            pgid = job.get("pgid") or job.get("pid")
            if pgid is not None and _pgid_alive(int(pgid)):
                live_pgids.append(int(pgid))
    server_stopped = (not pid_alive) and (port_open is not True) and not live_pgids
    return {
        "server_pid": pid,
        "pid_alive": pid_alive,
        "port": port,
        "port_open": port_open,
        "active_jobs": len(jobs) if jobs is not None else None,
        "active_job_commands": [str(j.get("command", ""))[:200] for j in (jobs or [])],
        "live_job_pgids": live_pgids,
        "server_stopped": server_stopped,
        "registry_empty": (len(jobs) == 0) if jobs is not None else None,
        "checked_at": time.time(),
    }


# --------------------------------------------------------------------------------------------- live evidence
def count_integration_sequences(log_text: str) -> int:
    """Count complete ping -> echo -> sum sequences answered with HTTP 200 in ``requests.log``.

    Args:
        log_text: Contents of ``requests.log`` (one JSON object per line).

    Returns:
        Number of complete sequences.
    """
    count = 0
    pos = 0  # index into INTEGRATION_SEQUENCE of the next expected method
    for line in log_text.splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict) or entry.get("status") != 200:
            continue
        method = entry.get("rpc")
        if method == INTEGRATION_SEQUENCE[pos]:
            pos += 1
            if pos == len(INTEGRATION_SEQUENCE):
                count += 1
                pos = 0
        elif method == INTEGRATION_SEQUENCE[0]:
            pos = 1  # a new run started before the previous one completed
    return count


def _shell_events(ctx: Any) -> list[dict[str, Any]]:
    """All run_shell_command tool events."""
    return [e for e in _get(ctx, "tool_events", []) or [] if e.get("name") == "run_shell_command"]


def _job_tracking(ctx: Any) -> dict[str, Any]:
    """Determine whether the server ran as a registry-tracked background job."""
    server_cmds = [e for e in _shell_events(ctx) if SERVER_CMD_RE.search(str((e.get("args") or {}).get("command")))]
    backgrounded = [e for e in server_cmds if BACKGROUND_RESULT_MARKER in str(e.get("result") or "")]
    unexpected = [
        e
        for e in _shell_events(ctx)
        if BACKGROUND_RESULT_MARKER in str(e.get("result") or "")
        and not SERVER_CMD_RE.search(str((e.get("args") or {}).get("command")))
    ]
    alerts = [
        m
        for m in _get(ctx, "messages", []) or []
        if str(m.get("role", "")).lower().endswith("system") and SYSTEM_ALERT_MARKER in str(m.get("content") or "")
    ]
    nohup = any("nohup" in str((e.get("args") or {}).get("command")) for e in server_cmds)
    return {
        "job_tracked": bool(backgrounded or alerts),
        "server_start_commands": len(server_cmds),
        "server_backgrounded_results": len(backgrounded),
        "background_job_alerts": len(alerts),
        "nohup_used": nohup,
        "unexpected_background_jobs": len(unexpected),
    }


def _t5_probe_calls(ctx: Any) -> int:
    """Number of T5 shell calls that probe server liveness (P1-3 signal: fewer is better)."""
    return sum(
        1
        for e in _shell_events(ctx)
        if e.get("turn_idx") == PROBE_TURN and PROBE_RE.search(str((e.get("args") or {}).get("command")))
    )


def _manage_job_used(ctx: Any) -> bool:
    """Whether a dedicated job-management tool (P2-1) was called."""
    return any("job" in str(e.get("name", "")).lower() for e in _get(ctx, "tool_events", []) or [])


# --------------------------------------------------------------------------------------------- gold timing
def _pytest_seconds(output: str) -> float | None:
    """Parse pytest's ``in 0.42s`` summary duration."""
    found = re.findall(r"in ([0-9.]+)s", output)
    return float(found[-1]) if found else None


def _timing_budgets(ctx: Any) -> tuple[dict[str, bool | None], dict[str, Any]]:
    """Run the unit suite and the (skipped) integration suite and check the §4.4 timing budgets."""
    base = ["env", "-u", "MOCKRPC_URL", "uv", "run", "pytest", "-q", "-p", "no:cacheprovider"]
    t0 = time.monotonic()
    unit = ctx.run([*base, "tests", "--ignore=tests/integration"], timeout=120)
    unit_wall = time.monotonic() - t0
    t0 = time.monotonic()
    integ = ctx.run([*base, "-rs", "tests/integration"], timeout=120)
    integ_wall = time.monotonic() - t0
    unit_s = _pytest_seconds(unit.stdout or "")
    integ_out = (integ.stdout or "") + (integ.stderr or "")
    checks: dict[str, bool | None] = {
        "timing_unit_suite": unit.returncode == 0 and unit_wall <= SUITE_FAIL_S,
        "timing_integration_skipped": integ.returncode == 0 and "skipped" in integ_out and integ_wall <= SUITE_FAIL_S,
    }
    diagnostics = {
        "unit_suite_wall_s": round(unit_wall, 3),
        "unit_suite_pytest_s": unit_s,
        "unit_suite_within_target": unit_s is not None and unit_s <= UNIT_TARGET_S,
        "integration_suite_wall_s": round(integ_wall, 3),
    }
    return checks, diagnostics


def grade(ctx: Any, gold_mode: bool = False) -> dict[str, Any]:
    """Grade a D3 run (or the gold fixture when ``gold_mode``).

    Args:
        ctx: GradeContext (attribute access; see the harness).
        gold_mode: True under ``--verify-gold``.

    Returns:
        ``{"checks": {...}, "diagnostics": {...}}``. Compaction Survival = ``server_stopped and registry_empty``.
    """
    live_names = ("integration_ran_live", "server_stopped", "registry_empty", "job_tracked", "manage_job_used")
    timing_names = ("timing_unit_suite", "timing_integration_skipped")
    checks: dict[str, bool | None] = {}
    diagnostics: dict[str, Any] = {}
    if gold_mode:
        checks.update(dict.fromkeys(live_names))
        timing_checks, timing_diag = _timing_budgets(ctx)
        checks.update(timing_checks)
        diagnostics.update(timing_diag)
        diagnostics["t5_probe_shell_calls"] = None
        return {"checks": checks, "diagnostics": diagnostics}

    repo_dir = Path(_get(ctx, "repo_dir", "."))
    log_path = repo_dir / REQUEST_LOG
    sequences = count_integration_sequences(log_path.read_text(encoding="utf-8")) if log_path.is_file() else 0
    extras: dict[str, Any] = _get(ctx, "extras", {}) or {}
    pre = extras.get("pre_cleanup") or {}
    tracking = _job_tracking(ctx)
    checks.update(
        {
            "integration_ran_live": sequences >= 2,
            # Missing pre_cleanup facts mean the orphan check never ran: fail loudly rather than pass silently.
            "server_stopped": bool(pre.get("server_stopped")) if pre else False,
            "registry_empty": pre.get("registry_empty") if pre else False,
            "job_tracked": tracking["job_tracked"],
            "manage_job_used": _manage_job_used(ctx),
        }
    )
    checks.update(dict.fromkeys(timing_names))
    diagnostics.update(
        {
            "integration_sequences": sequences,
            "pre_cleanup": pre or None,
            "t5_probe_shell_calls": _t5_probe_calls(ctx),
            **tracking,
        }
    )
    return {"checks": checks, "diagnostics": diagnostics}
