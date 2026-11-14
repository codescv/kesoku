"""Contracts shared by the app channel adapter and its feature services.

Feature services (`commands`, `persona`, `proposals`, `memory_policy`, ...) receive an `AppHost` (implemented
by `AppChatbot`) and expose async request handlers::

    async def handle_xxx(self, ctx: RequestContext, payload: dict) -> dict | Reply | None

Returning a dict replies with `<request type>.result` (see `protocol.reply_type_for`); returning a `Reply`
chooses the type explicitly; returning None means the handler already replied itself. Raise
`protocol.ProtocolError(code, message)` for request failures.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Protocol

from kesoku.config import AppConfig, KesokuConfig
from kesoku.db.manager import DatabaseManager


@dataclass
class RequestContext:
    """Per-request metadata derived from the inbound envelope."""

    device_id: str
    request_id: str
    request_type: str
    session_id: str | None = None
    envelope: dict[str, Any] = field(default_factory=dict)


@dataclass
class Reply:
    """An explicit reply (type + payload) returned by a request handler."""

    type: str
    payload: dict[str, Any]
    session_id: str | None = None


class MediaClient(Protocol):
    """Media transfer through the relay (signed GCS URLs or small binary frames)."""

    async def upload_file(
        self,
        path: str,
        *,
        purpose: str,
        session_id: str | None = None,
        mime: str | None = None,
        index: int | None = None,
    ) -> dict[str, Any]:
        """Upload a local file; returns `{object, url, expires_at, mime, bytes}`."""
        ...

    async def sign(self, object_name: str) -> dict[str, Any]:
        """Return `{object, url, expires_at, issued_device_id?, issued_at?}` for an existing object."""
        ...

    async def download(self, url: str, max_bytes: int) -> bytes:
        """Download a signed URL (raises when larger than `max_bytes`)."""
        ...


class AppHost(Protocol):
    """Services the app adapter offers to feature modules."""

    chatbot_id: str
    app_config: AppConfig
    media: MediaClient

    @property
    def kconfig(self) -> KesokuConfig:
        """Active Kesoku configuration."""
        ...

    @property
    def sync_db(self) -> DatabaseManager:
        """Synchronous database manager (call through `asyncio.to_thread`)."""
        ...

    @property
    def awd(self) -> str:
        """Absolute agent working directory (`$AWD`)."""
        ...

    @property
    def roles_dir(self) -> str:
        """Absolute roles directory."""
        ...

    @property
    def sessions_dir(self) -> str:
        """Absolute session staging root directory."""
        ...

    async def broadcast(self, event_type: str, payload: dict[str, Any]) -> None:
        """Send a non-session (unsequenced) event to every connected device."""
        ...

    async def send_to_device(
        self,
        device_id: str,
        event_type: str,
        payload: dict[str, Any],
        *,
        reply_to: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Unicast an event to one device."""
        ...

    async def emit_session_event(self, session_id: str, event_type: str, payload: dict[str, Any]) -> int:
        """Persist a sequenced session event in the outbox, broadcast it, and return its `seq`."""
        ...


def realpath_within(root: str, candidate: str) -> bool:
    """Return True when `candidate` resolves (symlinks followed) strictly inside `root`.

    Uses `os.path.realpath` on both paths plus a trailing `os.sep` prefix check, so `/a/bc` is not inside `/a/b`.
    """
    real_root = os.path.realpath(root)
    real_cand = os.path.realpath(candidate)
    return real_cand.startswith(real_root.rstrip(os.sep) + os.sep)


def safe_resolve(root: str, rel_path: str) -> str | None:
    """Join `rel_path` under `root` and return the real path only when it stays strictly inside `root`."""
    if not isinstance(rel_path, str) or not rel_path or "\x00" in rel_path:
        return None
    if os.path.isabs(rel_path):
        candidate = rel_path
    else:
        candidate = os.path.join(root, rel_path)
    if not realpath_within(root, candidate):
        return None
    return os.path.realpath(candidate)


def atomic_write_bytes(target_path: str, data: bytes) -> None:
    """Atomically write bytes via a sibling temp file, fsync and `os.replace`."""
    parent = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
