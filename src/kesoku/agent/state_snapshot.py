"""Deterministic per-turn state snapshots injected as `<context>` blocks.

Two snapshots are rendered next to the session notes in the latest user message:

- `running_jobs`: background shell jobs of the session (from `ActiveJobsRegistry.snapshot`), naming each job's
  process group and the exact command that stops the whole group.
- `staging_artifacts`: a deterministic listing of the files the agent created in `$STAGING_DIR` (internal logs
  omitted), rendered by the caller only once the session has summary nodes.

Cache rule: the rendered text must be byte-identical across all LLM calls of one turn, so it never contains
wall-clock-relative fields (no "elapsed"), timestamps are absolute, and staging mtimes are truncated to the minute.
The caller computes both snapshots once per turn.

This module has no dependency on the tools package (the shell tool imports the dataclasses from here).
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, tzinfo

RUNNING_JOBS_BLOCK_NAME = "running_jobs"
STAGING_ARTIFACTS_BLOCK_NAME = "staging_artifacts"

# Maximum stored/rendered command length per job.
JOB_COMMAND_MAX_CHARS = 160

RUNNING_JOBS_HEADER = (
    "Background jobs of this session (snapshot at the start of this turn; "
    "jobs you started or stopped during this turn are not reflected)."
)
RUNNING_JOBS_FOOTER = "Use the log paths to inspect output; stop a job only when the user asks."
STAGING_ARTIFACTS_HEADER = (
    "Files you created in $STAGING_DIR earlier in this session (deterministic listing; internal logs omitted):"
)

# Name prefixes of files written by kesoku itself into the staging directory.
STAGING_EXCLUDED_PREFIXES: tuple[str, ...] = (
    "llm-turn-",
    "cmd_output_",
    "hist_",
    "background_",
    "error_",
    "turn_",
)
# Exact names excluded from the staging listing (pinned notes and the tool-maintained context directory).
STAGING_EXCLUDED_NAMES: frozenset[str] = frozenset({"notes.md", "context"})
# Upper bound on files counted inside one first-level directory (keeps huge trees cheap to list).
DIR_FILE_COUNT_CAP = 1000

_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
_MINUTE_FORMAT = "%Y-%m-%d %H:%M"


@dataclass(frozen=True)
class JobInfo:
    """A running background job, as captured at registration time."""

    job_id: str
    command: str
    pid: int
    pgid: int
    started_at: float
    stdout_path: str
    stderr_path: str = ""


@dataclass(frozen=True)
class FinishedJob:
    """A background job that finished (or was stopped) earlier in the session."""

    job_id: str
    command: str
    return_code: int | None
    finished_at: float
    reason: str = "exited"  # "exited" (process ended on its own) or "stopped" (terminated by kesoku)


@dataclass(frozen=True)
class JobsSnapshot:
    """Point-in-time view of a session's background jobs."""

    running: list[JobInfo] = field(default_factory=list)
    finished: list[FinishedJob] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True when there is neither a running nor a finished job."""
        return not self.running and not self.finished


def _format_ts(ts: float, tz: tzinfo | None, fmt: str = _TIMESTAMP_FORMAT) -> str:
    """Format an epoch timestamp in `tz` (local time when None)."""
    return datetime.fromtimestamp(ts, tz).strftime(fmt)


def _one_line(command: str) -> str:
    """Collapse all whitespace (including newlines) so a command renders on a single line."""
    collapsed = " ".join(command.split())
    return collapsed or "(unknown)"


def _finished_status(job: FinishedJob) -> str:
    """Render the outcome of a finished job ("exit 0", "stopped", ...)."""
    if job.reason == "stopped":
        return "stopped"
    if job.return_code is None:
        return "exit unknown"
    return f"exit {job.return_code}"


def render_running_jobs(snapshot: JobsSnapshot, tz: tzinfo | None = None) -> str | None:
    """Render the `running_jobs` context block content.

    Jobs run under `setsid` through `sh -c`, so the recorded pid is only the shell leader; the block therefore
    names the process group and tells the model to kill the whole group (`kill -TERM -- -<pgid>`).

    Args:
        snapshot: Jobs snapshot of the session, taken once at the start of the turn.
        tz: Timezone for absolute timestamps; local time when None.

    Returns:
        The block content, or None when there are neither running nor finished jobs.
    """
    if snapshot.is_empty:
        return None

    lines = [RUNNING_JOBS_HEADER]
    if snapshot.running:
        lines.append("Running:")
        for job in snapshot.running:
            lines.append(
                f"- {job.job_id} | pgid {job.pgid} | started {_format_ts(job.started_at, tz)} | "
                f"command: {_one_line(job.command)}"
            )
            log_line = f"  stdout: {job.stdout_path}"
            if job.stderr_path:
                log_line += f" | stderr: {job.stderr_path}"
            lines.append(log_line)
            lines.append(
                f"  stop the whole job with: kill -TERM -- -{job.pgid}   "
                "(never kill only the pid: child processes would be orphaned)"
            )
    if snapshot.finished:
        lines.append("Finished earlier in this session:")
        for fin in snapshot.finished:
            lines.append(
                f"- {fin.job_id} | {_finished_status(fin)} | finished {_format_ts(fin.finished_at, tz)} | "
                f"command: {_one_line(fin.command)}"
            )
    lines.append(RUNNING_JOBS_FOOTER)
    return "\n".join(lines)


def human_readable_size(num_bytes: int) -> str:
    """Format a byte count as `412 B`, `1.3 KB`, `2.0 MB`, ... (1024-based)."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    size = float(num_bytes)
    for unit in ("KB", "MB", "GB", "TB"):
        size /= 1024.0
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}"
    return f"{size:.1f} TB"  # pragma: no cover - loop always returns


def is_internal_staging_name(name: str) -> bool:
    """Return True for staging entries that kesoku writes itself (or that are hidden) and must not be listed."""
    if name.startswith(".") or name in STAGING_EXCLUDED_NAMES:
        return True
    return name.startswith(STAGING_EXCLUDED_PREFIXES)


def _count_files(path: str, cap: int | None = None) -> tuple[int, bool]:
    """Count regular files under `path` recursively (dot-directories skipped), stopping at `cap`.

    Args:
        path: Directory to count.
        cap: Maximum count; defaults to `DIR_FILE_COUNT_CAP`.

    Returns:
        `(count, capped)` where `capped` is True when counting stopped at the cap.
    """
    limit = DIR_FILE_COUNT_CAP if cap is None else cap
    count = 0
    for root, dirs, files in os.walk(path):
        # Deterministic, and skip hidden directories such as .git / .venv.
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in files:
            if os.path.isfile(os.path.join(root, name)):
                count += 1
                if count >= limit:
                    return count, True
    return count, False


def _plural_files(count: int, capped: bool) -> str:
    """Render a file count such as `1 file`, `3 files`, `1000+ files`."""
    if capped:
        return f"{count}+ files"
    return "1 file" if count == 1 else f"{count} files"


def render_staging_artifacts(staging_dir: str, *, max_entries: int = 30) -> str | None:
    """Render the `staging_artifacts` context block content (synchronous filesystem access).

    Lists top-level files (size + mtime to the minute) and first-level directories (`dir/ (N files)`), sorted by
    name, excluding kesoku's internal files (`llm-turn-*`, `cmd_output_*`, `hist_*`, `background_*`, `error_*`,
    `turn_*`), `notes.md`, `context/` and dotfiles. At most `max_entries` entries are listed, followed by
    `… and N more`.

    Args:
        staging_dir: Absolute path of the session staging directory.
        max_entries: Maximum number of listed entries.

    Returns:
        The block content, or None when the directory is missing or has no listable entries.
    """
    if not staging_dir or not os.path.isdir(staging_dir):
        return None
    try:
        with os.scandir(staging_dir) as it:
            candidates = [e for e in it if not is_internal_staging_name(e.name)]
    except OSError:
        return None

    # Case-insensitive name order, ties broken by the exact name for full determinism.
    candidates.sort(key=lambda e: (e.name.lower(), e.name))

    rendered: list[str] = []
    for entry in candidates:
        try:
            if entry.is_dir(follow_symlinks=False):
                count, capped = _count_files(entry.path)
                rendered.append(f"- {entry.name}/ ({_plural_files(count, capped)})")
            elif entry.is_file():
                st = entry.stat()
                mtime = _format_ts(st.st_mtime, None, _MINUTE_FORMAT)
                rendered.append(f"- {entry.name} ({human_readable_size(st.st_size)}, modified {mtime})")
        except OSError:
            # Entry vanished or is unreadable between listing and stat; skip it.
            continue

    if not rendered:
        return None
    lines = [STAGING_ARTIFACTS_HEADER, *rendered[: max(max_entries, 0)]]
    remaining = len(rendered) - max(max_entries, 0)
    if remaining > 0:
        lines.append(f"… and {remaining} more")
    return "\n".join(lines)


def sorted_running(jobs: Sequence[JobInfo]) -> list[JobInfo]:
    """Order running jobs deterministically (start time, then job id)."""
    return sorted(jobs, key=lambda j: (j.started_at, j.job_id))
