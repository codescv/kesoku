"""Generic deferred media attachment protocol helpers for Kesoku chatbot adapters.

Supports two-phase producers that perform synchronous validation and then spawn a
detached background worker to generate large media files while the assistant text
is streamed and delivered first:

- ``<target>.pending.json`` marks an in-progress attachment with ``job_id``,
  ``created_at``, ``timeout_sec``, ``pid``, optional ``systemd_unit``, and optional
  ``placeholder`` text.
- ``<target>.error.json`` records a detached worker failure (``job_id``, ``error``).
- Readiness requires ``os.path.isfile(target)`` and either ``<target>.pending.json``
  is absent or ``os.path.getmtime(target) >= marker.created_at - 0.05`` so reused
  output filenames never deliver stale files from prior turns.
"""

import asyncio
import copy
import difflib
import json
import os
import re
import shutil
import signal
import time
from collections.abc import Callable
from typing import Any

from kesoku.constants import MessageRole
from kesoku.logger import setup_logger
from kesoku.utils.path import PathResolver

logger = setup_logger(__name__)

DEFERRED_EXCLUDED_SUFFIXES: tuple[str, ...] = (
    ".pending.json",
    ".error.json",
    ".job.json",
    ".part",
)

DEFAULT_DEFERRED_PLACEHOLDER = "⏳ 附件生成中…"
MAX_PLACEHOLDER_CHARS = 120
MAX_ERROR_REASON_CHARS = 200
DEFAULT_TIMEOUT_SEC = 240.0
MIN_TIMEOUT_SEC = 10.0
MAX_TIMEOUT_SEC = 600.0
READY_MTIME_EPSILON_SEC = 0.05
DEAD_WORKER_GRACE_SEC = 3.0
RECOVERY_LOOKBACK_SEC = 86400.0
DEFERRED_FUZZY_THRESHOLD = 0.80

_SAFE_SYSTEMD_UNIT_RE = re.compile(r"^[A-Za-z0-9_.@-]+$")
_ENV_SECRET_ASSIGN_RE = re.compile(
    r"(?i)\b(?:[A-Z0-9_]*(?:API_KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|AUTH)[A-Z0-9_]*)\s*[=:]\s*\S+"
)
_BEARER_TOKEN_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_RAW_TOKEN_RE = re.compile(r"\b(?:AIza[0-9A-Za-z\-_]{20,}|ya29\.[0-9A-Za-z\-_]+|sk-[A-Za-z0-9\-_]{16,})\b")


def is_deferred_sidecar_name(name: str) -> bool:
    """Return True if ``name`` is a deferred-media sidecar or temporary partial file."""
    return name.endswith(DEFERRED_EXCLUDED_SUFFIXES)


def read_json_dict(path: str) -> dict[str, Any] | None:
    """Read a JSON dictionary from ``path``.

    Returns:
        The parsed dictionary when ``path`` exists and contains a JSON object,
        ``{}`` when ``path`` exists on disk but cannot be parsed as a dict, or
        ``None`` when ``path`` does not exist.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError):
        return {} if os.path.isfile(path) else None


def safe_unlink(path: str) -> None:
    """Unlink ``path`` ignoring missing-file and OS errors."""
    try:
        os.unlink(path)
    except OSError:
        pass


def unlink_if_job_matches(sidecar_path: str, expected_job_id: str | None) -> bool:
    """Unlink ``sidecar_path`` only if it belongs to ``expected_job_id`` (or has no job_id).

    Prevents an older watcher or cleanup routine from unlinking a newer ``.pending.json``
    or ``.error.json`` created for the same output filename.
    """
    data = read_json_dict(sidecar_path)
    if data is None:
        return False
    expected = (expected_job_id or "").strip()
    live_job_id = str(data.get("job_id") or "").strip()
    if expected and live_job_id and live_job_id != expected:
        return False
    safe_unlink(sidecar_path)
    return True


def is_deferred_target_ready(
    target: str,
    pending_path: str | None = None,
    marker: dict[str, Any] | None = None,
) -> bool:
    """Return True when ``target`` exists as a regular file and satisfies the mtime gate."""
    if not os.path.isfile(target):
        return False
    p_path = pending_path or f"{target}.pending.json"
    if not os.path.exists(p_path):
        return True
    effective_marker = marker
    if effective_marker is None:
        effective_marker = read_json_dict(p_path)
        if effective_marker is None:
            return os.path.isfile(target)
    if not effective_marker:
        # Sidecar exists on disk but could not be parsed yet; wait for next poll.
        try:
            created_at = os.path.getmtime(p_path)
        except OSError:
            return False
    else:
        try:
            created_at = float(effective_marker.get("created_at", 0) or 0)
        except (TypeError, ValueError):
            created_at = 0.0
    try:
        return os.path.getmtime(target) >= created_at - READY_MTIME_EPSILON_SEC
    except OSError:
        return False


def is_pid_alive(pid: int) -> bool:
    """Return True if POSIX process ``pid`` is alive (and not a Linux zombie).

    Note:
        ``pid <= 0`` returns ``True`` because Phase 1 writes ``"pid": 0`` before
        the detached worker backfills its own PID. ``PermissionError`` from
        ``os.kill(pid, 0)`` indicates the PID exists.
    """
    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True

    proc_stat = f"/proc/{pid}/stat"
    if os.path.exists("/proc"):
        try:
            with open(proc_stat, encoding="utf-8", errors="replace") as fh:
                stat_text = fh.read()
            rparen = stat_text.rfind(")")
            if rparen != -1 and len(stat_text) > rparen + 2:
                state = stat_text[rparen + 2]
                if state in ("Z", "X"):
                    return False
        except FileNotFoundError:
            return False
        except OSError:
            pass
    return True


def sanitize_placeholder(marker: dict[str, Any] | None) -> str:
    """Extract and validate placeholder text from ``marker`` (capped at 120 chars)."""
    raw = marker.get("placeholder") if isinstance(marker, dict) else None
    if not isinstance(raw, str):
        return DEFAULT_DEFERRED_PLACEHOLDER
    cleaned = " ".join(raw.splitlines()).strip()
    if not cleaned:
        return DEFAULT_DEFERRED_PLACEHOLDER
    if len(cleaned) > MAX_PLACEHOLDER_CHARS:
        return cleaned[: MAX_PLACEHOLDER_CHARS - 1] + "…"
    return cleaned


def sanitize_error_reason(raw_reason: Any, default: str = "后台生成失败") -> str:
    """Sanitize and truncate a producer error reason for safe user-facing display.

    Strips raw multi-line tracebacks, redacts credential/env assignments and bearer
    tokens, collapses whitespace, and caps output at ``MAX_ERROR_REASON_CHARS``.
    """
    if not isinstance(raw_reason, str):
        return default
    text = raw_reason.strip()
    if not text:
        return default

    if "Traceback (most recent call last):" in text:
        non_empty_lines = [
            ln.strip()
            for ln in text.splitlines()
            if ln.strip() and not ln.lstrip().startswith("File ")
        ]
        text = non_empty_lines[-1] if non_empty_lines else default

    text = " ".join(text.split())
    text = _BEARER_TOKEN_RE.sub("Bearer [REDACTED]", text)
    text = _RAW_TOKEN_RE.sub("[REDACTED]", text)
    text = _ENV_SECRET_ASSIGN_RE.sub("[REDACTED]", text)

    if not text:
        return default
    if len(text) > MAX_ERROR_REASON_CHARS:
        text = text[: MAX_ERROR_REASON_CHARS - 1] + "…"
    return text


def format_deferred_failure_message(raw_path: str, reason: str) -> str:
    """Format a user-facing attachment generation failure notice."""
    safe_reason = sanitize_error_reason(reason)
    name = os.path.basename(raw_path.strip()) if isinstance(raw_path, str) else ""
    if name:
        return f"⚠️ 附件生成失败 ({name}): {safe_reason}"
    return f"⚠️ 附件生成失败：{safe_reason}"


def clamp_timeout_sec(marker: dict[str, Any] | None, default: float = DEFAULT_TIMEOUT_SEC) -> float:
    """Return ``marker['timeout_sec']`` clamped to ``[10.0, 600.0]`` seconds."""
    raw = marker.get("timeout_sec", default) if isinstance(marker, dict) else default
    try:
        val = float(raw)
    except (TypeError, ValueError):
        val = default
    return max(MIN_TIMEOUT_SEC, min(MAX_TIMEOUT_SEC, val))


def _dedup_paths(paths: list[str]) -> list[str]:
    """Deduplicate candidate file paths while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def build_candidate_targets(raw_path: str, staging_dir: str | None) -> list[str]:
    """Build ordered candidate target file paths for ``raw_path``."""
    cleaned = raw_path.strip()
    if not cleaned:
        return []
    raw_candidates: list[str] = []
    if os.path.isabs(cleaned):
        raw_candidates.append(cleaned)
        raw_candidates.append(os.path.realpath(cleaned))
    else:
        resolved_awd = PathResolver.resolve(cleaned)
        raw_candidates.append(resolved_awd)
        raw_candidates.append(os.path.realpath(resolved_awd))
        if staging_dir:
            joined_staging = os.path.join(staging_dir, cleaned)
            raw_candidates.append(joined_staging)
            raw_candidates.append(os.path.realpath(joined_staging))
    return _dedup_paths(raw_candidates)


def resolve_deferred_sync(
    raw_path: str,
    staging_dir: str | None,
) -> tuple[str, str, dict[str, Any]] | None:
    """Inspect deferred sidecar state for ``raw_path`` synchronously.

    Must run strictly before ``resolve_outbound_path`` fuzzy matching so an in-progress
    or fast-failed deferred attachment never fuzzy-matches a similar filename from an
    earlier turn or triggers ``DeliveryAbortedError`` self-healing.

    Returns:
        - ``("ready", target_path, marker)`` when ``.pending.json`` exists and ``target``
          already passes the readiness rule.
        - ``("pending", target_path, marker)`` when ``.pending.json`` exists and ``target``
          is not ready yet.
        - ``("failed", target_path, error_dict)`` when ``.pending.json`` and ``target`` are
          absent and ``.error.json`` exists.
        - ``None`` when no deferred sidecars apply (fall through to normal delivery).
    """
    cleaned = raw_path.strip()
    if not cleaned:
        return None

    candidates = build_candidate_targets(cleaned, staging_dir)

    for cand in candidates:
        pending_path = f"{cand}.pending.json"
        error_path = f"{cand}.error.json"
        marker = read_json_dict(pending_path)
        if marker is not None:
            if is_deferred_target_ready(cand, pending_path, marker):
                job_id = str(marker.get("job_id") or "")
                unlink_if_job_matches(pending_path, job_id)
                unlink_if_job_matches(error_path, job_id)
                return ("ready", os.path.realpath(cand), marker)
            return ("pending", cand, marker)
        if os.path.isfile(cand):
            return None
        err = read_json_dict(error_path)
        if err is not None:
            return ("failed", cand, err)

    # Fallback [C-6]: recursive basename lookup inside staging_dir for <basename>.pending.json / .error.json
    if staging_dir and os.path.isdir(staging_dir):
        base = os.path.basename(cleaned)
        if base:
            want_pending = f"{base}.pending.json"
            want_error = f"{base}.error.json"
            pending_hits: list[tuple[float, str]] = []
            error_hits: list[tuple[float, str]] = []
            for root, dirs, filenames in os.walk(staging_dir):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for fname in filenames:
                    if fname == want_pending:
                        full_p = os.path.join(root, fname)
                        try:
                            mtime = os.path.getmtime(full_p)
                        except OSError:
                            mtime = 0.0
                        pending_hits.append((mtime, full_p))
                    elif fname == want_error:
                        full_e = os.path.join(root, fname)
                        try:
                            mtime = os.path.getmtime(full_e)
                        except OSError:
                            mtime = 0.0
                        error_hits.append((mtime, full_e))

            if pending_hits:
                pending_hits.sort(key=lambda item: item[0], reverse=True)
                best_pending = pending_hits[0][1]
                sub_target = best_pending[: -len(".pending.json")]
                marker = read_json_dict(best_pending) or {}
                if is_deferred_target_ready(sub_target, best_pending, marker):
                    unlink_if_job_matches(best_pending, str(marker.get("job_id") or ""))
                    return ("ready", os.path.realpath(sub_target), marker)
                return ("pending", sub_target, marker)

            if error_hits:
                error_hits.sort(key=lambda item: item[0], reverse=True)
                best_error = error_hits[0][1]
                sub_target = best_error[: -len(".error.json")]
                if not os.path.isfile(sub_target):
                    err = read_json_dict(best_error) or {}
                    return ("failed", sub_target, err)

            # Fallback [C-7]: fuzzy match (>= 0.80) against active *.pending.json targets in staging_dir
            now = time.time()
            fuzzy_hits: list[tuple[float, float, str]] = []
            for root, dirs, filenames in os.walk(staging_dir):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for fname in filenames:
                    if not fname.endswith(".pending.json"):
                        continue
                    full_p = os.path.join(root, fname)
                    try:
                        mtime = os.path.getmtime(full_p)
                    except OSError:
                        continue
                    marker = read_json_dict(full_p) or {}
                    if now - mtime > clamp_timeout_sec(marker):
                        continue
                    sub_target = full_p[: -len(".pending.json")]
                    target_base = os.path.basename(sub_target)
                    if not target_base:
                        continue
                    ratio = difflib.SequenceMatcher(None, base, target_base).ratio()
                    if ratio >= DEFERRED_FUZZY_THRESHOLD:
                        fuzzy_hits.append((ratio, mtime, full_p))

            if fuzzy_hits:
                fuzzy_hits.sort(key=lambda item: (item[0], item[1]), reverse=True)
                best_ratio, _, best_pending = fuzzy_hits[0]
                sub_target = best_pending[: -len(".pending.json")]
                marker = read_json_dict(best_pending) or {}
                logger.warning(
                    f"Fuzzy-resolved deferred attachment '{raw_path}' -> '{sub_target}' (score={best_ratio:.2f})"
                )
                if is_deferred_target_ready(sub_target, best_pending, marker):
                    job_id = str(marker.get("job_id") or "")
                    unlink_if_job_matches(best_pending, job_id)
                    unlink_if_job_matches(f"{sub_target}.error.json", job_id)
                    return ("ready", os.path.realpath(sub_target), marker)
                return ("pending", sub_target, marker)

    return None


async def terminate_deferred_worker(target: str, stored_marker: dict[str, Any] | None) -> None:
    """Terminate a detached deferred-media worker (PID and/or transient systemd unit) safely.

    Verifies ``job_id`` ownership against any live ``.pending.json`` before sending
    signals or unlinking sidecars so a newly spawned worker sharing the same output
    path is never harmed.
    """
    pending_path = f"{target}.pending.json"
    live_marker = await asyncio.to_thread(read_json_dict, pending_path)
    expected_job_id = str((stored_marker or {}).get("job_id") or "").strip()

    if isinstance(live_marker, dict) and live_marker:
        live_job_id = str(live_marker.get("job_id") or "").strip()
        if expected_job_id and live_job_id and live_job_id != expected_job_id:
            return
        effective = live_marker
    else:
        effective = stored_marker or {}

    systemd_unit = str(effective.get("systemd_unit") or "").strip()
    try:
        pid = int(effective.get("pid") or 0)
    except (TypeError, ValueError):
        pid = 0

    if systemd_unit and _SAFE_SYSTEMD_UNIT_RE.match(systemd_unit) and shutil.which("systemctl"):
        try:
            proc = await asyncio.create_subprocess_exec(
                "systemctl",
                "--user",
                "stop",
                systemd_unit,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except Exception as exc:
            logger.debug(f"Best-effort systemctl stop '{systemd_unit}' failed: {exc}")

    if pid > 0:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception as exc:
            logger.debug(f"Best-effort SIGTERM to deferred worker pid {pid} failed: {exc}")

    await asyncio.to_thread(unlink_if_job_matches, pending_path, expected_job_id)


async def cancel_session_deferred_entries_in_db(db: Any, session_id: str) -> None:
    """Terminate pending deferred workers recorded in SQLite for ``session_id`` and mark cancelled."""
    get_msgs_fn = getattr(db, "get_messages_by_filters", None)
    update_meta_fn = getattr(db, "update_message_metadata", None)
    if not callable(get_msgs_fn) or not callable(update_meta_fn):
        return
    try:
        messages = await get_msgs_fn({"session_id": session_id, "role": MessageRole.ASSISTANT})
    except Exception as exc:
        logger.debug(f"Failed to query deferred messages for session {session_id}: {exc}")
        return

    if not isinstance(messages, list):
        return

    for msg in messages:
        meta = getattr(msg, "metadata", None)
        if not isinstance(meta, dict):
            continue
        entries = meta.get("deferred_media")
        if not isinstance(entries, list):
            continue
        changed = False
        for entry in entries:
            if isinstance(entry, dict) and entry.get("state") == "pending":
                target = str(entry.get("path") or "")
                marker = entry.get("marker") if isinstance(entry.get("marker"), dict) else {}
                if target:
                    await terminate_deferred_worker(target, marker)
                entry["state"] = "cancelled"
                changed = True
        if changed:
            try:
                await update_meta_fn(msg.id, copy.deepcopy(meta))
            except Exception as exc:
                logger.warning(f"Failed to mark deferred media cancelled on message {msg.id}: {exc}")


async def wait_for_deferred_file(
    candidates: list[str],
    poll_interval: float = 0.5,
    max_wait_cap: float = 240.0,
    grace_sec: float = DEAD_WORKER_GRACE_SEC,
    interrupt_check: Callable[[], bool] | None = None,
) -> tuple[str | None, str | None]:
    """Wait asynchronously if any candidate in ``candidates`` has an active ``.pending.json``.

    Used by chained tool consumers such as ``analyze_media`` when inspecting an
    attachment whose detached worker is still rendering.

    Returns:
        ``(ready_path, error_reason)``:
        - ``(path, None)`` if a deferred candidate became ready (or was already ready).
        - ``(None, reason)`` if the deferred worker failed, died, or timed out.
        - ``(None, None)`` if no candidate had a deferred ``.pending.json`` / ``.error.json``.
    """
    active_cand: str | None = None
    initial_marker: dict[str, Any] | None = None

    for cand in candidates:
        pending_path = f"{cand}.pending.json"
        error_path = f"{cand}.error.json"
        marker = await asyncio.to_thread(read_json_dict, pending_path)
        if marker is not None:
            if await asyncio.to_thread(is_deferred_target_ready, cand, pending_path, marker):
                return (cand, None)
            active_cand = cand
            initial_marker = marker
            break
        if not await asyncio.to_thread(os.path.isfile, cand):
            err = await asyncio.to_thread(read_json_dict, error_path)
            if err is not None:
                return (None, sanitize_error_reason(err.get("error"), default="后台生成失败"))

    if active_cand is None or initial_marker is None:
        return (None, None)

    pending_path = f"{active_cand}.pending.json"
    error_path = f"{active_cand}.error.json"
    expected_job_id = str(initial_marker.get("job_id") or "").strip()
    timeout_sec = min(clamp_timeout_sec(initial_marker), max_wait_cap)
    deadline = time.monotonic() + timeout_sec
    last_pid = 0
    try:
        last_pid = int(initial_marker.get("pid") or 0)
    except (TypeError, ValueError):
        last_pid = 0

    dead_since: float | None = None
    missing_since: float | None = None

    while True:
        if interrupt_check and interrupt_check():
            return (None, "等待附件生成被中断")

        live_marker = await asyncio.to_thread(read_json_dict, pending_path)
        if isinstance(live_marker, dict) and live_marker:
            live_job_id = str(live_marker.get("job_id") or "").strip()
            if expected_job_id and live_job_id and live_job_id != expected_job_id:
                # Superseded by a newer same-name task: switch to tracking the new job!
                expected_job_id = live_job_id
                initial_marker = live_marker
                dead_since = None
                missing_since = None
            try:
                live_pid = int(live_marker.get("pid") or 0)
                if live_pid > 0:
                    last_pid = live_pid
            except (TypeError, ValueError):
                pass
            missing_since = None

        if await asyncio.to_thread(is_deferred_target_ready, active_cand, pending_path, live_marker or initial_marker):
            return (active_cand, None)

        err = await asyncio.to_thread(read_json_dict, error_path)
        if err is not None:
            err_job_id = str(err.get("job_id") or "").strip()
            if not expected_job_id or not err_job_id or err_job_id == expected_job_id:
                return (None, sanitize_error_reason(err.get("error"), default="后台生成失败"))

        now_mono = time.monotonic()
        if last_pid > 0 and not await asyncio.to_thread(is_pid_alive, last_pid):
            if dead_since is None:
                dead_since = now_mono
            elif now_mono - dead_since >= grace_sec:
                return (None, "后台生成进程异常退出")
        else:
            dead_since = None

        if live_marker is None:
            if missing_since is None:
                missing_since = now_mono
            elif now_mono - missing_since >= grace_sec:
                return (None, "生成任务中断")

        if now_mono >= deadline:
            return (None, "生成超时")

        sleep_dur = min(poll_interval, 0.2) if (dead_since is not None or missing_since is not None) else poll_interval
        await asyncio.sleep(sleep_dur)
