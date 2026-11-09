"""Attachment manager for sanitizing and saving incoming media and file attachments."""

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

from kesoku.config import get_config
from kesoku.utils.async_fs import async_exists, async_realpath, async_write_binary_file

logger = logging.getLogger(__name__)


class AttachmentManager:
    """Central manager for handling session attachment persistence and collision resolution."""

    def __init__(self, sessions_dir: str | None = None) -> None:
        """Initialize the AttachmentManager.

        Args:
            sessions_dir: Optional base directory for sessions. If not provided,
                          resolves from system configuration.
        """
        self._sessions_dir = sessions_dir
        self._reserved_paths: set[str] = set()
        self._reserve_lock: asyncio.Lock | None = None

    def _get_reserve_lock(self) -> asyncio.Lock:
        lock = self._reserve_lock
        if lock is not None and not lock.locked():
            bound_loop = getattr(lock, "_loop", None)
            if bound_loop is not None and bound_loop is not asyncio.get_running_loop():
                lock = None
        if lock is None:
            lock = asyncio.Lock()
            self._reserve_lock = lock
        return lock

    def _get_sessions_dir(self) -> str:
        if self._sessions_dir:
            return self._sessions_dir
        return get_config().workspace.sessions_dir

    async def save_attachment(
        self,
        filename: str,
        workspace_name: str,
        data: bytes | None = None,
        save_callback: Callable[[str], Awaitable[None]] | None = None,
        collision_id: str | None = None,
    ) -> dict[str, Any]:
        """Sanitize, resolve collisions, and save attachment to session workspace.

        Args:
            filename: Original filename.
            workspace_name: Session workspace name.
            data: Raw bytes to save (optional).
            save_callback: Async callback that takes a filepath and saves the file (optional).
            collision_id: Optional unique ID to append in case of collision (e.g., attachment ID).

        Returns:
            Dict containing path and sanitized filename.
        """
        sessions_dir = self._get_sessions_dir()
        session_staging_dir = await async_realpath(os.path.join(sessions_dir, workspace_name))
        os.makedirs(session_staging_dir, exist_ok=True)

        # Sanitize filename to prevent path traversal
        safe_filename = "".join(c for c in filename if c.isalnum() or c in "._-")
        if not safe_filename:
            safe_filename = f"attachment_{collision_id or 'unnamed'}"

        # Atomically reserve target path across concurrent in-flight downloads before slow I/O
        async with self._get_reserve_lock():
            filepath = os.path.join(session_staging_dir, safe_filename)
            if filepath in self._reserved_paths or await async_exists(filepath):
                base, ext = os.path.splitext(safe_filename)
                suffix = collision_id or "collision"
                safe_filename = f"{base}_{suffix}{ext}"
                filepath = os.path.join(session_staging_dir, safe_filename)
                counter = 1
                while filepath in self._reserved_paths or await async_exists(filepath):
                    safe_filename = f"{base}_{suffix}_{counter}{ext}"
                    filepath = os.path.join(session_staging_dir, safe_filename)
                    counter += 1
            self._reserved_paths.add(filepath)
        try:
            if data is not None:
                await async_write_binary_file(filepath, data)
            elif save_callback is not None:
                await save_callback(filepath)
            else:
                raise ValueError("Either 'data' or 'save_callback' must be provided.")
        finally:
            self._reserved_paths.discard(filepath)

        logger.info(f"Saved attachment {filename} as {safe_filename} in {workspace_name}")
        return {
            "path": filepath,
            "filename": safe_filename,
        }
