"""Generic staging-state mirror: forwards `game/*.json` and `context/*.md` changes as `state.sync`.

The watcher has zero knowledge of file names, fields or rules: changed JSON files are forwarded parsed,
Markdown context files as (truncated) text. Clients decide what to render.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from kesoku.gateway.chatbot.app.host import realpath_within
from kesoku.logger import setup_logger

logger = setup_logger(__name__)

STATE_JSON_DIR = "game"
STATE_TEXT_DIR = "context"
CONTEXT_TEXT_MAX_BYTES = 8 * 1024
STATE_JSON_MAX_BYTES = 512 * 1024
MAX_STATE_FILES = 64


@dataclass
class _SessionState:
    """Last observed file signatures for one session."""

    signatures: dict[str, tuple[int, int]] = field(default_factory=dict)
    rev: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _read_state_files_sync(staging_dir: str) -> dict[str, tuple[tuple[int, int], str]]:
    """List candidate state files `{rel: ((mtime_ns, size), abs_path)}` jailed inside `staging_dir`."""
    out: dict[str, tuple[tuple[int, int], str]] = {}
    for sub, suffix in ((STATE_JSON_DIR, ".json"), (STATE_TEXT_DIR, ".md")):
        base = os.path.join(staging_dir, sub)
        if not os.path.isdir(base) or not realpath_within(staging_dir, base):
            continue
        try:
            names = sorted(os.listdir(base))
        except OSError:
            continue
        for name in names:
            if not name.endswith(suffix) or name.startswith("."):
                continue
            path = os.path.join(base, name)
            try:
                if not os.path.isfile(path) or not realpath_within(staging_dir, path):
                    continue
                st = os.stat(path)
            except OSError:
                continue
            out[f"{sub}/{name}"] = ((st.st_mtime_ns, st.st_size), path)
            if len(out) >= MAX_STATE_FILES:
                return out
    return out


def _load_state_file_sync(rel: str, path: str) -> tuple[bool, Any]:
    """Load one state file; returns `(ok, value)` (JSON parse failures are not ok, so they retry later)."""
    try:
        if rel.endswith(".json"):
            if os.path.getsize(path) > STATE_JSON_MAX_BYTES:
                return False, None
            with open(path, encoding="utf-8") as f:
                return True, json.load(f)
        with open(path, "rb") as f:
            raw = f.read(CONTEXT_TEXT_MAX_BYTES + 1)
        text = raw[:CONTEXT_TEXT_MAX_BYTES].decode("utf-8", errors="ignore")
        return True, text
    except (OSError, ValueError):
        return False, None


class StateWatcher:
    """Tracks per-session staging state files and emits incremental `state.sync` events."""

    def __init__(
        self,
        staging_dir_for: Callable[[str], Awaitable[str | None]],
        emit: Callable[[str, dict[str, Any]], Awaitable[Any]],
    ) -> None:
        """Initialize the watcher.

        Args:
            staging_dir_for: Async resolver `session_id -> staging dir` (None when unknown).
            emit: Async callback `(session_id, payload)` that emits a sequenced `state.sync` event.
        """
        self._staging_dir_for = staging_dir_for
        self._emit = emit
        self._sessions: dict[str, _SessionState] = {}

    def _state(self, session_id: str) -> _SessionState:
        st = self._sessions.get(session_id)
        if st is None:
            st = _SessionState()
            self._sessions[session_id] = st
        return st

    async def snapshot(self, session_id: str) -> tuple[dict[str, Any], int]:
        """Return all current state files and the current `staging_rev`; primes change detection."""
        staging = await self._staging_dir_for(session_id)
        st = self._state(session_id)
        if not staging:
            return {}, st.rev
        async with st.lock:
            listing = await asyncio.to_thread(_read_state_files_sync, staging)
            files: dict[str, Any] = {}
            sigs: dict[str, tuple[int, int]] = {}
            for rel, (sig, path) in listing.items():
                ok, value = await asyncio.to_thread(_load_state_file_sync, rel, path)
                if ok:
                    files[rel] = value
                    sigs[rel] = sig
            st.signatures = sigs
            return files, st.rev

    async def scan(self, session_id: str) -> dict[str, Any] | None:
        """Emit `state.sync{files, removed, staging_rev}` when any state file changed; returns the payload."""
        staging = await self._staging_dir_for(session_id)
        if not staging:
            return None
        st = self._state(session_id)
        async with st.lock:
            listing = await asyncio.to_thread(_read_state_files_sync, staging)
            changed: dict[str, Any] = {}
            new_sigs = dict(st.signatures)
            for rel, (sig, path) in listing.items():
                if st.signatures.get(rel) == sig:
                    continue
                ok, value = await asyncio.to_thread(_load_state_file_sync, rel, path)
                if not ok:
                    continue
                changed[rel] = value
                new_sigs[rel] = sig
            removed = sorted(rel for rel in st.signatures if rel not in listing)
            for rel in removed:
                new_sigs.pop(rel, None)
            if not changed and not removed:
                return None
            st.signatures = new_sigs
            st.rev += 1
            payload = {"files": changed, "removed": removed, "staging_rev": st.rev}
        try:
            await self._emit(session_id, payload)
        except Exception as e:
            logger.warning(f"Failed emitting state.sync for session {session_id}: {e}")
        return payload

    def forget(self, session_id: str) -> None:
        """Drop cached signatures of a deleted session."""
        self._sessions.pop(session_id, None)
