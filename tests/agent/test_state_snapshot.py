"""Unit tests for deterministic per-turn state snapshots (kesoku.agent.state_snapshot)."""

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kesoku.agent import state_snapshot
from kesoku.agent.state_snapshot import (
    RUNNING_JOBS_FOOTER,
    RUNNING_JOBS_HEADER,
    STAGING_ARTIFACTS_HEADER,
    FinishedJob,
    JobInfo,
    JobsSnapshot,
    human_readable_size,
    is_internal_staging_name,
    render_running_jobs,
    render_staging_artifacts,
    sorted_running,
)

T0 = datetime(2026, 10, 19, 13, 51, 4, tzinfo=UTC).timestamp()
T1 = datetime(2026, 10, 19, 13, 58, 10, tzinfo=UTC).timestamp()


def _job(job_id: str = "job_1_aaaa", started_at: float = T0, **kwargs: object) -> JobInfo:
    """Build a running JobInfo with sensible defaults."""
    defaults: dict[str, object] = {
        "command": "uv run python -m mockrpc.server --port 35151",
        "pid": 12345,
        "pgid": 12345,
        "stdout_path": "/staging/background_job_1_aaaa.stdout",
        "stderr_path": "/staging/background_job_1_aaaa.stderr",
    }
    defaults.update(kwargs)
    return JobInfo(job_id=job_id, started_at=started_at, **defaults)  # type: ignore[arg-type]


# --- render_running_jobs -------------------------------------------------------------------------------------------


def test_render_running_jobs_none_when_empty() -> None:
    """No running and no finished jobs -> the block is omitted."""
    assert render_running_jobs(JobsSnapshot()) is None
    assert JobsSnapshot().is_empty


def test_render_running_and_finished() -> None:
    """Both sections render with pgid, absolute timestamps, log paths and the process-group stop command."""
    snap = JobsSnapshot(
        running=[_job(pgid=4242, pid=4242)],
        finished=[
            FinishedJob(job_id="job_0_bbbb", command="make test", return_code=0, finished_at=T1),
            FinishedJob(job_id="job_0_cccc", command="false", return_code=1, finished_at=T1),
        ],
    )
    text = render_running_jobs(snap, UTC)
    assert text is not None
    lines = text.split("\n")
    assert lines[0] == RUNNING_JOBS_HEADER
    assert "snapshot at the start of this turn" in lines[0]
    assert lines[1] == "Running:"
    assert lines[2] == (
        "- job_1_aaaa | pgid 4242 | started 2026-10-19 13:51:04 | command: uv run python -m mockrpc.server --port 35151"
    )
    assert lines[3] == (
        "  stdout: /staging/background_job_1_aaaa.stdout | stderr: /staging/background_job_1_aaaa.stderr"
    )
    assert "kill -TERM -- -4242" in lines[4]
    assert "never kill only the pid" in lines[4]
    assert lines[5] == "Finished earlier in this session:"
    assert lines[6] == "- job_0_bbbb | exit 0 | finished 2026-10-19 13:58:10 | command: make test"
    assert lines[7] == "- job_0_cccc | exit 1 | finished 2026-10-19 13:58:10 | command: false"
    assert lines[-1] == RUNNING_JOBS_FOOTER


def test_render_running_only() -> None:
    """Without finished jobs the finished section is absent."""
    text = render_running_jobs(JobsSnapshot(running=[_job()]), UTC)
    assert text is not None
    assert "Running:" in text
    assert "Finished earlier" not in text
    assert "kill -TERM -- -12345" in text


def test_render_finished_only() -> None:
    """Finished-only snapshots render without a Running section or stop hint."""
    snap = JobsSnapshot(finished=[FinishedJob("job_x", "sleep 1", None, T1)])
    text = render_running_jobs(snap, UTC)
    assert text is not None
    assert "Running:" not in text
    assert "kill -TERM" not in text
    assert "- job_x | exit unknown | finished" in text


def test_render_stopped_and_unknown_command() -> None:
    """Stopped jobs say 'stopped'; an empty command renders as (unknown); a missing stderr path is omitted."""
    snap = JobsSnapshot(
        running=[_job(command="", stderr_path="")],
        finished=[FinishedJob("job_s", "srv", None, T1, reason="stopped")],
    )
    text = render_running_jobs(snap, UTC)
    assert text is not None
    assert "command: (unknown)" in text
    assert "| stderr:" not in text
    assert "- job_s | stopped | finished" in text


def test_render_collapses_multiline_command() -> None:
    """Commands with newlines stay on one line."""
    text = render_running_jobs(JobsSnapshot(running=[_job(command="echo a\n  echo b\tc")]), UTC)
    assert text is not None
    assert "command: echo a echo b c" in text


def test_render_is_deterministic_and_has_no_elapsed_field() -> None:
    """Rendering the same snapshot twice is byte-identical and contains no wall-clock-relative fields."""
    snap = JobsSnapshot(running=[_job()], finished=[FinishedJob("j", "c", 0, T1)])
    a = render_running_jobs(snap, UTC)
    b = render_running_jobs(snap, UTC)
    assert a == b
    assert a is not None
    assert "elapsed" not in a.lower()
    assert " ago" not in a


def test_render_uses_local_time_by_default() -> None:
    """tz=None renders local time."""
    text = render_running_jobs(JobsSnapshot(running=[_job()]))
    assert text is not None
    expected = datetime.fromtimestamp(T0).strftime("%Y-%m-%d %H:%M:%S")
    assert f"started {expected}" in text


def test_sorted_running_orders_by_start_then_id() -> None:
    """Running jobs are ordered by start time, ties by job id."""
    jobs = [_job("b", T1), _job("c", T0), _job("a", T0)]
    assert [j.job_id for j in sorted_running(jobs)] == ["a", "c", "b"]


# --- render_staging_artifacts --------------------------------------------------------------------------------------


def _touch(path: Path, content: str = "x", mtime: float | None = None) -> None:
    """Create a file with content and an optional fixed mtime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def test_human_readable_size() -> None:
    """Sizes are 1024-based with one decimal above bytes."""
    assert human_readable_size(0) == "0 B"
    assert human_readable_size(412) == "412 B"
    assert human_readable_size(1023) == "1023 B"
    assert human_readable_size(1331) == "1.3 KB"
    assert human_readable_size(2 * 1024 * 1024) == "2.0 MB"
    assert human_readable_size(3 * 1024**3) == "3.0 GB"
    assert human_readable_size(5 * 1024**5) == "5120.0 TB"


def test_is_internal_staging_name() -> None:
    """Kesoku's own files, notes, context/ and dotfiles are internal."""
    for name in [
        "llm-turn-3.log.yaml",
        "cmd_output_1_ab.txt",
        "hist_x.txt",
        "background_job_1.stdout",
        "error_42.txt",
        "turn_1.json",
        "notes.md",
        "context",
        ".hidden",
    ]:
        assert is_internal_staging_name(name), name
    for name in ["STATUS.md", "migration_checklist.md", "notes.md.bak", "contexts", "my_turn.md"]:
        assert not is_internal_staging_name(name), name


def test_render_staging_artifacts_missing_or_empty(tmp_path: Path) -> None:
    """Missing directory, empty string or only internal files -> None."""
    assert render_staging_artifacts("") is None
    assert render_staging_artifacts(str(tmp_path / "nope")) is None
    assert render_staging_artifacts(str(tmp_path)) is None
    _touch(tmp_path / "llm-turn-1.log.yaml")
    _touch(tmp_path / "notes.md")
    _touch(tmp_path / "context" / "10-a.md")
    assert render_staging_artifacts(str(tmp_path)) is None


def test_render_staging_artifacts_filters_and_sorts(tmp_path: Path) -> None:
    """Listing is name-sorted (case-insensitive), filtered, with size and minute-precision mtime."""
    mtime = datetime(2026, 10, 19, 13, 42, 37).timestamp()
    _touch(tmp_path / "migration_checklist.md", "a" * 412, mtime)
    _touch(tmp_path / "STATUS.md", "b" * 1331, mtime)
    _touch(tmp_path / "background_job_1.stdout", "log")
    _touch(tmp_path / "cmd_output_1.txt", "log")
    _touch(tmp_path / "error_1.txt", "log")
    _touch(tmp_path / ".secret", "x")
    _touch(tmp_path / "notes.md", "x")

    text = render_staging_artifacts(str(tmp_path))
    assert text == (
        f"{STAGING_ARTIFACTS_HEADER}\n"
        "- migration_checklist.md (412 B, modified 2026-10-19 13:42)\n"
        "- STATUS.md (1.3 KB, modified 2026-10-19 13:42)"
    )


def test_render_staging_artifacts_directories(tmp_path: Path) -> None:
    """First-level directories render as `dir/ (N files)`, counting recursively and skipping dot-dirs."""
    _touch(tmp_path / "out" / "a.txt")
    _touch(tmp_path / "out" / "sub" / "b.txt")
    _touch(tmp_path / "out" / ".git" / "objects" / "c")
    _touch(tmp_path / "one" / "only.txt")
    (tmp_path / "empty").mkdir()
    (tmp_path / ".venv").mkdir()

    text = render_staging_artifacts(str(tmp_path))
    assert text is not None
    lines = text.split("\n")[1:]
    assert lines == ["- empty/ (0 files)", "- one/ (1 file)", "- out/ (2 files)"]


def test_render_staging_artifacts_dir_count_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Huge directories stop counting at the cap."""
    for i in range(5):
        _touch(tmp_path / "big" / f"{i}.txt")
    monkeypatch.setattr(state_snapshot, "DIR_FILE_COUNT_CAP", 3)
    text = render_staging_artifacts(str(tmp_path))
    assert text is not None
    assert text.split("\n")[1] == "- big/ (3+ files)"


def test_render_staging_artifacts_max_entries(tmp_path: Path) -> None:
    """Entries beyond max_entries collapse into `… and N more`."""
    for i in range(7):
        _touch(tmp_path / f"f{i}.txt")
    text = render_staging_artifacts(str(tmp_path), max_entries=3)
    assert text is not None
    lines = text.split("\n")
    assert len(lines) == 1 + 3 + 1
    assert lines[1].startswith("- f0.txt ")
    assert lines[3].startswith("- f2.txt ")
    assert lines[-1] == "… and 4 more"

    full = render_staging_artifacts(str(tmp_path))
    assert full is not None
    assert "more" not in full


def test_render_staging_artifacts_stable_within_minute(tmp_path: Path) -> None:
    """A content-preserving touch within the same minute does not change the listing."""
    base = datetime(2026, 10, 19, 13, 41, 5).timestamp()
    target = tmp_path / "plan.md"
    _touch(target, "abc", base)
    first = render_staging_artifacts(str(tmp_path))
    os.utime(target, (base + 30, base + 30))
    assert render_staging_artifacts(str(tmp_path)) == first


def test_render_staging_artifacts_skips_broken_symlink(tmp_path: Path) -> None:
    """Dangling symlinks are neither files nor dirs and are skipped."""
    (tmp_path / "dangling").symlink_to(tmp_path / "missing")
    _touch(tmp_path / "real.txt")
    text = render_staging_artifacts(str(tmp_path))
    assert text is not None
    assert "dangling" not in text
    assert "real.txt" in text
