"""Wire protocol (V1) shared by the app channel adapter, the relay and app clients.

Envelope::

    {"v": 1, "type": "...", "id": "evt_<ULID>", "ts": 1763100000.123,
     "session_id"?: str, "seq"?: int, "reply_to"?: str, "device_id"?: str, "payload": {...}}

`device_id` is a relay routing field: on client->engine frames the relay always sets it to the authenticated
device; on engine->client frames it unicasts to that device (absent = broadcast to every connected device).

Request/response rule: every client request receives exactly one reply whose envelope carries
`reply_to = request.id` (and the requesting `device_id`). The reply type is `<request type>.result` unless
listed in `REPLY_TYPE_OVERRIDES`; failures reply with `error{code, message, reply_to}`.
"""

from __future__ import annotations

import json
import re
import secrets
import struct
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

PROTOCOL_VERSION = 1
MAX_TEXT_FRAME_BYTES = 1_048_576
MAX_BINARY_FRAME_BYTES = 1_048_576

# Session events that are persisted in `app_outbox` and carry a per-session `seq`.
SEQ_EVENT_TYPES = frozenset(
    {
        "message.user",
        "message.final",
        "turn.event",
        "media.pending",
        "media.ready",
        "media.failed",
        "state.sync",
        "session.updated",
        "memory.proposal",
        "memory.proposal.resolved",
    }
)

REPLY_TYPE_OVERRIDES: dict[str, str] = {
    "session.open": "session.snapshot",
    "session.create": "session.snapshot",
    "message.send": "message.ack",
    "avatar.generate": "avatar.job",
    "media.hydrate": "media.ready",
    "voice.preview": "media.ready",
}

ERROR_CODES = frozenset(
    {
        "UNAUTHORIZED",
        "FORBIDDEN_EMAIL",
        "ENGINE_OFFLINE",
        "RATE_LIMITED",
        "BAD_REQUEST",
        "NOT_FOUND",
        "TURN_BUSY",
        "PERSONA_BUILTIN",
        "AVATAR_SOURCE_INVALID",
        "PROPOSAL_NOT_PENDING",
        "INTERNAL",
    }
)

MEMORY_POLICIES = ("full", "confirm_only", "isolated")
PROPOSAL_STATUSES = ("pending", "applied", "dismissed", "expired")

SESSION_ID_RE = re.compile(r"^[0-9a-f]{8}$")
SESSION_MEDIA_OBJECT_RE = re.compile(r"^media/[0-9a-f]{8}/[0-9a-f-]{36}\.[a-z0-9]{1,5}$")
AVATAR_MEDIA_OBJECT_RE = re.compile(r"^media/avatar/[0-9a-f-]{36}(-cand[0-3])?\.(jpg|png|webp|heic)$")
AVATAR_SOURCE_OBJECT_RE = re.compile(r"^media/avatar/[0-9a-f-]{36}\.(jpg|png|webp|heic)$")
ASSET_MEDIA_OBJECT_RE = re.compile(r"^media/asset/[0-9a-f-]{36}\.[a-z0-9]{1,5}$")
AVATAR_REF_RE = re.compile(r"^(roles/[a-z0-9_]{2,32}|user)/avatar\.webp$")
ROLE_ID_RE = re.compile(r"^[a-z0-9_]{2,32}$")
MODE_ID_RE = re.compile(r"^[a-z0-9_]{1,40}$")

LIMITS: dict[str, int] = {
    "MODE_PREFS_MAX_BYTES": 2048,
    "PROMPT_HINT_MAX": 80,
    "DISPLAY_NAME_MAX": 32,
    "TAGLINE_MAX": 80,
    "PERSONA_FILE_MAX_BYTES": 64 * 1024,
    "BIO_MAX_BYTES": 2048,
    "PROPOSAL_TEXT_MAX": 500,
    "MESSAGE_MEDIA_MAX_BYTES": 25 * 1024 * 1024,
    "AVATAR_SOURCE_MAX_BYTES": 12 * 1024 * 1024,
    "HISTORY_PAGE_MAX": 200,
    "SESSION_LIST_MAX": 200,
    "REPLAY_LIMIT_MAX": 500,
    "MESSAGE_TEXT_MAX_CHARS": 32000,
    "ATTACHMENTS_MAX": 10,
}

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


class ProtocolError(Exception):
    """A request-level failure mapped to an `error{code, message}` reply."""

    def __init__(self, code: str, message: str = "") -> None:
        """Initialize with a protocol error code (unknown codes become `INTERNAL`)."""
        self.code = code if code in ERROR_CODES else "INTERNAL"
        self.message = message or self.code
        super().__init__(f"{self.code}: {self.message}")


class Envelope(BaseModel):
    """Pydantic model of the V1 envelope header (payload validated by each handler)."""

    model_config = ConfigDict(extra="allow")

    v: int = Field(PROTOCOL_VERSION)
    type: str = Field(..., min_length=1, max_length=64)
    id: str = Field(..., min_length=1, max_length=64)
    ts: float = 0.0
    session_id: str | None = Field(default=None, max_length=64)
    seq: int | None = Field(default=None, ge=0)
    reply_to: str | None = Field(default=None, max_length=64)
    device_id: str | None = Field(default=None, max_length=64)
    payload: dict[str, Any] = Field(default_factory=dict)


def ulid(now_ms: int | None = None) -> str:
    """Return a 26-character time-sortable ULID (Crockford base32)."""
    t = int(now_ms if now_ms is not None else time.time() * 1000)
    time_part = ""
    for _ in range(10):
        time_part = _CROCKFORD[t % 32] + time_part
        t //= 32
    rand = "".join(_CROCKFORD[b % 32] for b in secrets.token_bytes(16))
    return time_part + rand


def new_id(prefix: str = "evt") -> str:
    """Return an `<prefix>_<ULID>` identifier."""
    return f"{prefix}_{ulid()}"


def reply_type_for(request_type: str) -> str:
    """Return the reply envelope type for a client request type."""
    return REPLY_TYPE_OVERRIDES.get(request_type, f"{request_type}.result")


def make_envelope(
    event_type: str,
    payload: dict[str, Any] | None = None,
    *,
    session_id: str | None = None,
    seq: int | None = None,
    reply_to: str | None = None,
    device_id: str | None = None,
    event_id: str | None = None,
    ts: float | None = None,
) -> dict[str, Any]:
    """Build a V1 envelope dict, omitting unset optional header fields."""
    env: dict[str, Any] = {
        "v": PROTOCOL_VERSION,
        "type": event_type,
        "id": event_id or new_id("evt"),
        "ts": float(ts if ts is not None else time.time()),
        "payload": payload if payload is not None else {},
    }
    if session_id is not None:
        env["session_id"] = session_id
    if seq is not None:
        env["seq"] = int(seq)
    if reply_to is not None:
        env["reply_to"] = reply_to
    if device_id is not None:
        env["device_id"] = device_id
    return env


def make_error(code: str, message: str, *, reply_to: str | None = None, device_id: str | None = None) -> dict:
    """Build an `error` envelope."""
    safe_code = code if code in ERROR_CODES else "INTERNAL"
    payload: dict[str, Any] = {"code": safe_code, "message": message[:500]}
    if reply_to is not None:
        payload["reply_to"] = reply_to
    return make_envelope("error", payload, reply_to=reply_to, device_id=device_id)


def parse_envelope(raw: str | bytes | dict[str, Any]) -> dict[str, Any]:
    """Parse and validate an inbound text frame / dict into an envelope dict.

    Raises:
        ProtocolError: `BAD_REQUEST` for oversize, malformed JSON, wrong version or invalid header fields.
    """
    data: Any = raw
    if isinstance(raw, (bytes, bytearray)):
        if len(raw) > MAX_TEXT_FRAME_BYTES:
            raise ProtocolError("BAD_REQUEST", "frame too large")
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        if len(raw.encode("utf-8")) > MAX_TEXT_FRAME_BYTES:
            raise ProtocolError("BAD_REQUEST", "frame too large")
        try:
            data = json.loads(raw)
        except ValueError as e:
            raise ProtocolError("BAD_REQUEST", "invalid json") from e
    if not isinstance(data, dict):
        raise ProtocolError("BAD_REQUEST", "envelope must be an object")
    if data.get("v") != PROTOCOL_VERSION:
        raise ProtocolError("BAD_REQUEST", "unsupported protocol version")
    if "payload" in data and not isinstance(data.get("payload"), dict):
        raise ProtocolError("BAD_REQUEST", "payload must be an object")
    try:
        env = Envelope.model_validate(data)
    except ValidationError as e:
        raise ProtocolError("BAD_REQUEST", f"invalid envelope: {e.errors()[0].get('msg', 'error')}") from e
    return env.model_dump(exclude_none=True)


def encode_binary_frame(header: dict[str, Any], data: bytes) -> bytes:
    """Encode a binary frame: `uint32 BE header_len | header JSON (utf-8) | bytes` (≤ 1 MiB total)."""
    head = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    frame = struct.pack(">I", len(head)) + head + data
    if len(frame) > MAX_BINARY_FRAME_BYTES:
        raise ProtocolError("BAD_REQUEST", "binary frame too large")
    return frame


def decode_binary_frame(frame: bytes) -> tuple[dict[str, Any], bytes]:
    """Decode a binary frame produced by `encode_binary_frame`."""
    if len(frame) < 4 or len(frame) > MAX_BINARY_FRAME_BYTES:
        raise ProtocolError("BAD_REQUEST", "invalid binary frame size")
    (head_len,) = struct.unpack(">I", frame[:4])
    if head_len <= 0 or 4 + head_len > len(frame):
        raise ProtocolError("BAD_REQUEST", "invalid binary frame header length")
    try:
        header = json.loads(frame[4 : 4 + head_len].decode("utf-8"))
    except ValueError as e:
        raise ProtocolError("BAD_REQUEST", "invalid binary frame header") from e
    if not isinstance(header, dict):
        raise ProtocolError("BAD_REQUEST", "binary frame header must be an object")
    return header, frame[4 + head_len :]


def require_str(payload: dict[str, Any], key: str, *, max_len: int = 4096, allow_empty: bool = False) -> str:
    """Return `payload[key]` as a string or raise `BAD_REQUEST`."""
    value = payload.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()) or len(value) > max_len:
        raise ProtocolError("BAD_REQUEST", f"'{key}' must be a non-empty string (≤ {max_len} chars)")
    return value


def optional_str(payload: dict[str, Any], key: str, *, max_len: int = 4096) -> str | None:
    """Return `payload[key]` as a string, None when absent/null, or raise `BAD_REQUEST`."""
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > max_len:
        raise ProtocolError("BAD_REQUEST", f"'{key}' must be a string (≤ {max_len} chars)")
    return value


def require_session_id(payload: dict[str, Any], key: str = "session_id") -> str:
    """Return a syntactically valid session id from `payload[key]` or raise `BAD_REQUEST`."""
    value = payload.get(key)
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise ProtocolError("BAD_REQUEST", f"'{key}' must be a valid session id")
    return value


def bounded_int(payload: dict[str, Any], key: str, default: int, lo: int, hi: int) -> int:
    """Return an int within `[lo, hi]` (default when absent) or raise `BAD_REQUEST`."""
    value = payload.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError("BAD_REQUEST", f"'{key}' must be a number")
    return max(lo, min(hi, int(value)))
