"""Media transfer between the engine host and app clients through the relay.

Outbound files are uploaded either as a small binary frame (`media.put`, < 1 MiB) relayed to object storage,
or via a signed PUT URL from `media.upload.ticket`. Inbound client uploads are fetched through `media.sign`.
Only short-lived signed URLs ever leave the engine; nothing is published.
"""

from __future__ import annotations

import asyncio
import inspect
import mimetypes
import os
import time
import wave
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

import aiohttp

from kesoku.gateway.chatbot.app.protocol import (
    MAX_BINARY_FRAME_BYTES,
    ProtocolError,
    encode_binary_frame,
    make_envelope,
    new_id,
)
from kesoku.logger import setup_logger

logger = setup_logger(__name__)

BINARY_UPLOAD_MAX_BYTES = MAX_BINARY_FRAME_BYTES - 16 * 1024
UPLOAD_CACHE_TTL_SECONDS = 6 * 86400.0
UPLOAD_CACHE_MAX_ENTRIES = 512
HTTP_TIMEOUT_SECONDS = 120.0
MEDIA_PURPOSES = ("message", "asset", "avatar_candidate")


class RelayRequester(Protocol):
    """Subset of `EngineLink` used for media requests."""

    async def request(
        self, envelope: dict[str, Any], timeout: float | None = None  # noqa: ASYNC109
    ) -> dict[str, Any]:
        """Send a request envelope and await its reply envelope."""
        ...

    async def request_binary(
        self, frame: bytes, request_id: str, timeout: float | None = None  # noqa: ASYNC109
    ) -> dict[str, Any]:
        """Send a binary frame and await the reply envelope for `request_id`."""
        ...


def guess_mime(path: str, default: str = "application/octet-stream") -> str:
    """Guess a MIME type from a filename."""
    mime, _ = mimetypes.guess_type(path)
    if mime:
        return mime
    ext = os.path.splitext(path)[1].lower()
    return {".webp": "image/webp", ".heic": "image/heic", ".opus": "audio/ogg", ".m4a": "audio/mp4"}.get(ext, default)


class MediaManager:
    """Uploads local files and fetches client uploads using relay-issued signed URLs."""

    def __init__(
        self,
        requester_getter: Callable[[], RelayRequester | None],
        http_put: Callable[..., Awaitable[None]] | None = None,
        http_get: Callable[[str, int], Awaitable[bytes]] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Initialize the media manager.

        Args:
            requester_getter: Returns the live relay requester (`EngineLink`) or None when offline.
            http_put: Optional override `(url, data, mime[, extra_headers])` for signed PUT uploads (tests).
                Callables accepting a 4th positional argument receive the relay ticket's required headers.
            http_get: Optional override `(url, max_bytes) -> bytes` for signed GET downloads (tests).
            clock: Injectable time source.
        """
        self._requester_getter = requester_getter
        self._http_put = http_put or self._default_put
        self._http_put_takes_headers = _accepts_extra_headers(self._http_put)
        self._http_get = http_get or self._default_get
        self._clock = clock
        self._upload_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}

    def _requester(self) -> RelayRequester:
        req = self._requester_getter()
        if req is None:
            raise ProtocolError("ENGINE_OFFLINE", "relay link is not connected")
        return req

    @staticmethod
    async def _default_put(url: str, data: bytes, mime: str, extra_headers: dict[str, str] | None = None) -> None:
        req_headers = build_put_headers(mime, extra_headers)
        timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.put(url, data=data, headers=req_headers) as resp:
                if resp.status >= 300:
                    raise ConnectionError(f"signed PUT failed with HTTP {resp.status}")

    @staticmethod
    async def _default_get(url: str, max_bytes: int) -> bytes:
        timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status >= 300:
                    raise ConnectionError(f"signed GET failed with HTTP {resp.status}")
                declared = resp.content_length
                if declared is not None and declared > max_bytes:
                    raise ValueError("download exceeds size limit")
                chunks: list[bytes] = []
                total = 0
                async for chunk in resp.content.iter_chunked(65536):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("download exceeds size limit")
                    chunks.append(chunk)
                return b"".join(chunks)

    async def upload_file(
        self,
        path: str,
        *,
        purpose: str,
        session_id: str | None = None,
        mime: str | None = None,
        index: int | None = None,
    ) -> dict[str, Any]:
        """Upload a local file and return `{object, url, expires_at, mime, bytes}`.

        Re-uploads of an unchanged file reuse the existing object (re-signed) while it is younger than the
        storage lifecycle.
        """
        if purpose not in MEDIA_PURPOSES:
            raise ValueError(f"Unknown media purpose '{purpose}'")
        real = await asyncio.to_thread(os.path.realpath, path)
        st = await asyncio.to_thread(os.stat, real)
        if not await asyncio.to_thread(os.path.isfile, real):
            raise FileNotFoundError(path)
        eff_mime = mime or guess_mime(real)
        cache_key = (real, int(st.st_mtime_ns), int(st.st_size), f"{purpose}:{session_id or ''}:{index}")
        cached = self._upload_cache.get(cache_key)
        if cached is not None and self._clock() - cached["uploaded_at"] < UPLOAD_CACHE_TTL_SECONDS:
            try:
                signed = await self.sign(cached["object"])
                return {
                    "object": cached["object"],
                    "url": signed["url"],
                    "expires_at": signed.get("expires_at"),
                    "mime": eff_mime,
                    "bytes": int(st.st_size),
                }
            except Exception as e:
                logger.debug(f"Re-sign of cached media object failed, re-uploading: {e}")

        data = await asyncio.to_thread(_read_bytes, real)
        filename = os.path.basename(real)
        requester = self._requester()
        if len(data) <= BINARY_UPLOAD_MAX_BYTES:
            req_id = new_id("req")
            header: dict[str, Any] = {
                "v": 1,
                "type": "media.put",
                "id": req_id,
                "purpose": purpose,
                "mime": eff_mime,
                "filename": filename,
            }
            if session_id:
                header["session_id"] = session_id
            if index is not None:
                header["index"] = int(index)
            reply = await requester.request_binary(encode_binary_frame(header, data), req_id)
            body = reply.get("payload") or {}
            result = {
                "object": body.get("object"),
                "url": body.get("url"),
                "expires_at": body.get("expires_at"),
                "mime": body.get("mime") or eff_mime,
                "bytes": int(body.get("bytes") or len(data)),
            }
        else:
            ticket_payload: dict[str, Any] = {
                "purpose": purpose,
                "mime": eff_mime,
                "bytes": len(data),
                "filename": filename,
            }
            if session_id:
                ticket_payload["session_id"] = session_id
            if index is not None:
                ticket_payload["index"] = int(index)
            reply = await requester.request(
                make_envelope("media.upload.ticket", ticket_payload, event_id=new_id("req"))
            )
            body = reply.get("payload") or {}
            put_url = body.get("put_url")
            max_bytes = int(body.get("max_bytes") or len(data))
            if not isinstance(put_url, str) or len(data) > max_bytes:
                raise ProtocolError("BAD_REQUEST", "relay refused the upload ticket")
            raw_headers = body.get("headers")
            ticket_headers = raw_headers if isinstance(raw_headers, dict) else None
            if self._http_put_takes_headers:
                await self._http_put(put_url, data, eff_mime, ticket_headers)
            else:
                await self._http_put(put_url, data, eff_mime)
            result = {
                "object": body.get("object"),
                "url": body.get("get_url"),
                "expires_at": body.get("get_expires_at"),
                "mime": eff_mime,
                "bytes": len(data),
            }
            if not result["url"] and result["object"]:
                signed = await self.sign(str(result["object"]))
                result["url"] = signed.get("url")
                result["expires_at"] = signed.get("expires_at")
        if not result.get("object") or not result.get("url"):
            raise ProtocolError("INTERNAL", "relay returned an incomplete media reply")
        self._remember(cache_key, str(result["object"]))
        return result

    def _remember(self, key: tuple[str, int, int, str], object_name: str) -> None:
        if len(self._upload_cache) >= UPLOAD_CACHE_MAX_ENTRIES:
            oldest = min(self._upload_cache, key=lambda k: self._upload_cache[k]["uploaded_at"])
            self._upload_cache.pop(oldest, None)
        self._upload_cache[key] = {"object": object_name, "uploaded_at": self._clock()}

    async def sign(self, object_name: str) -> dict[str, Any]:
        """Return `{object, url, expires_at, issued_device_id?, issued_at?}` for an existing object."""
        reply = await self._requester().request(
            make_envelope("media.sign", {"object": object_name}, event_id=new_id("req"))
        )
        body = dict(reply.get("payload") or {})
        if not isinstance(body.get("url"), str):
            raise ProtocolError("NOT_FOUND", "media object could not be signed")
        return body

    async def download(self, url: str, max_bytes: int) -> bytes:
        """Download a signed URL, enforcing `max_bytes`."""
        if not isinstance(url, str) or not url.startswith(("https://", "http://127.0.0.1", "http://localhost")):
            raise ProtocolError("BAD_REQUEST", "refusing to download from an untrusted URL scheme")
        return await self._http_get(url, int(max_bytes))

    async def fetch_object(self, object_name: str, max_bytes: int) -> tuple[bytes, dict[str, Any]]:
        """Sign then download a relay object; returns `(data, signed_payload)`."""
        signed = await self.sign(object_name)
        data = await self.download(str(signed["url"]), max_bytes)
        return data, signed


def probe_audio_duration_ms(file_path: str) -> int | None:
    """Return the duration of a PCM WAV file in milliseconds (header only), or None when not parseable."""
    try:
        with wave.open(file_path, "rb") as w:
            frames, rate = w.getnframes(), w.getframerate()
    except Exception:  # noqa: BLE001 - non-WAV / truncated / unreadable files simply have no duration
        return None
    return int(frames * 1000 / rate) if rate else None


def build_put_headers(mime: str, extra_headers: dict[str, Any] | None = None) -> dict[str, str]:
    """Build signed-PUT request headers: the relay ticket's headers plus `Content-Type`.

    Signed V4 URLs may bind extension headers (e.g. `x-goog-content-length-range`); omitting them makes the
    object store reject the upload, so every string header from the ticket is forwarded verbatim.
    """
    req_headers: dict[str, str] = {}
    if isinstance(extra_headers, dict):
        req_headers.update(
            {str(k): str(v) for k, v in extra_headers.items() if isinstance(k, str) and isinstance(v, str)}
        )
    for key in [k for k in req_headers if k.lower() == "content-type"]:
        req_headers.pop(key)
    req_headers["Content-Type"] = mime
    return req_headers


def _accepts_extra_headers(fn: Callable[..., Any]) -> bool:
    """Return True when `fn` can take a 4th positional `extra_headers` argument."""
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return True
    positional = [
        p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 4


def _read_bytes(path: str) -> bytes:
    """Read a whole file."""
    with open(path, "rb") as f:
        return f.read()
