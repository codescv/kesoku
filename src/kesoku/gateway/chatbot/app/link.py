"""Engine leg of the app channel: a resilient WebSocket client from the engine host to the relay.

`EngineLink` keeps exactly one *primary* leg to the relay's engine endpoint and transparently survives:

* network drops / relay restarts — exponential backoff reconnect (reset after a healthy connection);
* `relay.rotate{grace_ms}` — opens a new leg (`conn_seq + 1`, fresh `engine.hello`), switches all outbound
  traffic to it immediately and keeps reading the old leg for `grace_ms` before closing it, so no inbound
  frame is lost;
* `relay.restarting{retry_ms}` — reconnects after `retry_ms`;
* silent peers — an app-level `ping` envelope every `ping_interval`; any inbound frame counts as liveness and
  two missed intervals drop the leg and reconnect.

Inbound frames are delivered to `on_frame` sequentially, in arrival order, from a single dispatcher task that is
independent of the socket readers: a slow handler never stalls liveness or reply matching, and a handler may
itself `await link.request(...)` without deadlocking. Long-running work should still be offloaded to a task.
Frames whose `reply_to` matches a pending `request` resolve that request instead of reaching `on_frame`.
There is no offline queue: `send` returns False when no leg is live.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp

from kesoku.gateway.chatbot.app.protocol import (
    MAX_BINARY_FRAME_BYTES,
    MAX_TEXT_FRAME_BYTES,
    ProtocolError,
    make_envelope,
    parse_envelope,
)

logger = logging.getLogger(__name__)

CLOSE_UNAUTHORIZED = 4401
CLOSE_SUPERSEDED = 4409
DEFAULT_ROTATE_GRACE_MS = 3000
MAX_ROTATE_GRACE_MS = 60_000
DEFAULT_RESTART_RETRY_MS = 1000
MAX_RESTART_RETRY_MS = 300_000
_EXPIRED_REQUEST_MEMORY = 256


def _bounded_ms(payload: dict[str, Any], key: str, default: int, hi: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return max(0, min(hi, int(value)))


def _describe(exc: BaseException) -> str:
    """Return a short, secret-free description of a connection error."""
    if isinstance(exc, aiohttp.WSServerHandshakeError):
        return f"handshake rejected (HTTP {exc.status})"
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


class _Leg:
    """One WebSocket connection to the relay."""

    __slots__ = ("ws", "conn_seq", "opened_at", "last_rx", "send_lock", "reader")

    def __init__(self, ws: aiohttp.ClientWebSocketResponse, conn_seq: int) -> None:
        self.ws = ws
        self.conn_seq = conn_seq
        self.opened_at = time.monotonic()
        self.last_rx = self.opened_at
        self.send_lock = asyncio.Lock()
        self.reader: asyncio.Task[None] | None = None


class _ConnectRejected(Exception):
    """The relay refused the engine credentials during the HTTP handshake."""


class EngineLink:
    """Resilient engine→relay WebSocket client (see module docstring)."""

    def __init__(
        self,
        relay_url: str,
        token: str,
        *,
        on_frame: Callable[[dict], Awaitable[None]],
        on_connected: Callable[[], Awaitable[None]] | None = None,
        on_disconnected: Callable[[], Awaitable[None]] | None = None,
        hello_extra: Callable[[], dict] | None = None,
        engine_version: str = "",
        ping_interval: float = 25.0,
        backoff_initial: float = 1.0,
        backoff_max: float = 30.0,
        request_timeout: float = 15.0,
    ) -> None:
        """Configure the link; nothing connects until `run()` is awaited.

        Args:
            relay_url: `ws(s)://<relay>/engine` endpoint.
            token: Engine token sent as `Authorization: Bearer <token>` (never logged).
            on_frame: Receives every inbound envelope not consumed internally or by a pending request.
            on_connected: Awaited after the hello on each fresh (non-rotation) connection.
            on_disconnected: Awaited when a previously connected link has no live leg any more.
            hello_extra: Returns extra `engine.hello` payload fields (e.g. `revoked_device_ids`, `last_inbound_id`).
            engine_version: Reported in `engine.hello`.
            ping_interval: Seconds between app-level pings; 2 silent intervals drop the leg.
            backoff_initial: First reconnect delay in seconds.
            backoff_max: Reconnect delay ceiling in seconds (also used after an auth rejection).
            request_timeout: Default `request()` timeout and connect timeout in seconds.
        """
        self._relay_url = relay_url
        self._token = token
        self._on_frame = on_frame
        self._on_connected = on_connected
        self._on_disconnected = on_disconnected
        self._hello_extra = hello_extra
        self._engine_version = engine_version
        self._ping_interval = float(ping_interval)
        self._backoff_initial = float(backoff_initial)
        self._backoff_max = max(float(backoff_max), self._backoff_initial)
        self._request_timeout = float(request_timeout)
        self._close_timeout = max(0.2, min(5.0, self._ping_interval))
        self._healthy_after = self._ping_interval

        self.boot_id: int = int(time.time() * 1000)
        self.conn_seq: int = 0

        self._session: aiohttp.ClientSession | None = None
        self._primary: _Leg | None = None
        self._legs: set[_Leg] = set()
        self._rotation: asyncio.Task[None] | None = None
        self._dispatcher: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[dict] = asyncio.Queue()
        self._pending: dict[str, asyncio.Future[dict]] = {}
        self._expired: deque[str] = deque(maxlen=_EXPIRED_REQUEST_MEMORY)

        self._stopping = asyncio.Event()
        self._lost = asyncio.Event()
        self._lost_code: int | None = None
        self._restart_delay: float | None = None
        self._announced = False
        self._running = False
        self._run_done = asyncio.Event()

    # ------------------------------------------------------------------ public API

    @property
    def connected(self) -> bool:
        """True while a primary leg is open."""
        leg = self._primary
        return leg is not None and not leg.ws.closed

    async def run(self) -> None:
        """Connect and keep reconnecting until `stop()` is called."""
        if self._running:
            raise RuntimeError("EngineLink.run() is already running")
        self._running = True
        self._run_done.clear()
        self._session = aiohttp.ClientSession()
        self._dispatcher = asyncio.create_task(self._dispatch_loop(), name="engine-link-dispatch")
        backoff = self._backoff_initial
        try:
            while not self._stopping.is_set():
                code: int | None = None
                uptime = 0.0
                self._restart_delay = None
                try:
                    leg = await self._open_leg()
                except asyncio.CancelledError:
                    raise
                except _ConnectRejected:
                    code = CLOSE_UNAUTHORIZED
                    leg = None
                except Exception as e:
                    logger.warning("engine link: connect to relay failed (%s)", _describe(e))
                    leg = None
                if leg is not None:
                    started = time.monotonic()
                    await self._serve(leg)
                    uptime = time.monotonic() - started
                    code = self._lost_code
                if self._stopping.is_set():
                    break
                delay, backoff = self._next_delay(code, uptime, backoff)
                logger.info("engine link: reconnecting in %.2fs", delay)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stopping.wait(), delay)
        finally:
            await self._teardown()
            if self._dispatcher is not None:
                self._dispatcher.cancel()
                await asyncio.gather(self._dispatcher, return_exceptions=True)
                self._dispatcher = None
            if self._session is not None:
                await self._session.close()
                self._session = None
            self._running = False
            self._run_done.set()

    async def stop(self) -> None:
        """Stop reconnecting, close every leg and wait for `run()` to return."""
        self._stopping.set()
        self._lost.set()
        if self._running:
            await self._run_done.wait()

    async def send(self, envelope: dict) -> bool:
        """Send a text envelope on the primary leg; False when no leg is live (nothing is queued).

        Raises:
            ProtocolError: `BAD_REQUEST` when the encoded frame exceeds 1 MiB.
        """
        text = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
        if len(text.encode("utf-8")) > MAX_TEXT_FRAME_BYTES:
            raise ProtocolError("BAD_REQUEST", "text frame too large")
        leg = self._primary
        if leg is None:
            return False
        return await self._send_on(leg, text)

    async def send_binary(self, frame: bytes) -> bool:
        """Send a binary frame (≤ 1 MiB) on the primary leg; False when no leg is live.

        Raises:
            ProtocolError: `BAD_REQUEST` when the frame exceeds 1 MiB.
        """
        if len(frame) > MAX_BINARY_FRAME_BYTES:
            raise ProtocolError("BAD_REQUEST", "binary frame too large")
        leg = self._primary
        if leg is None:
            return False
        return await self._send_on(leg, bytes(frame))

    async def request(self, envelope: dict, timeout: float | None = None) -> dict:  # noqa: ASYNC109
        """Send `envelope` and return the reply envelope whose `reply_to` equals `envelope["id"]`.

        Raises:
            ProtocolError: The relay answered with an `error{code, message}` envelope.
            TimeoutError: No reply within `timeout` (default `request_timeout`).
            ConnectionError: No live leg, or every leg was lost before the reply arrived.
        """
        req_id = envelope.get("id")
        if not isinstance(req_id, str) or not req_id:
            raise ValueError("request envelope needs a non-empty 'id'")
        return await self._request(req_id, lambda: self.send(envelope), timeout)

    async def request_binary(
        self,
        frame: bytes,
        request_id: str,
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> dict:
        """Send a binary frame whose header id is `request_id` and return the matching reply envelope.

        Raises:
            ProtocolError: Oversize frame, or the relay answered with an `error` envelope.
            TimeoutError: No reply within `timeout` (default `request_timeout`).
            ConnectionError: No live leg, or every leg was lost before the reply arrived.
        """
        if not request_id:
            raise ValueError("request_binary needs a non-empty request_id")
        if len(frame) > MAX_BINARY_FRAME_BYTES:
            raise ProtocolError("BAD_REQUEST", "binary frame too large")
        return await self._request(request_id, lambda: self.send_binary(frame), timeout)

    # ------------------------------------------------------------------ connection lifecycle

    async def _open_leg(self) -> _Leg:
        """Connect, send `engine.hello` and start the leg reader."""
        assert self._session is not None
        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    self._relay_url,
                    headers={"Authorization": f"Bearer {self._token}"},
                    autoping=True,
                    heartbeat=None,
                    timeout=aiohttp.ClientWSTimeout(ws_receive=None, ws_close=self._close_timeout),
                    max_msg_size=MAX_TEXT_FRAME_BYTES + 4096,
                ),
                self._request_timeout,
            )
        except aiohttp.WSServerHandshakeError as e:
            if e.status in (401, 403):
                raise _ConnectRejected() from None
            raise
        self.conn_seq += 1
        leg = _Leg(ws, self.conn_seq)
        try:
            await ws.send_str(json.dumps(self._hello_envelope(leg.conn_seq), separators=(",", ":")))
        except BaseException:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(ws.close(), self._close_timeout)
            raise
        leg.reader = asyncio.create_task(self._read_loop(leg), name=f"engine-link-reader-{leg.conn_seq}")
        self._legs.add(leg)
        logger.info("engine link: leg %d connected to %s", leg.conn_seq, self._relay_url)
        return leg

    def _hello_envelope(self, conn_seq: int) -> dict:
        payload: dict[str, Any] = {}
        if self._hello_extra is not None:
            try:
                extra = self._hello_extra()
                if isinstance(extra, dict):
                    payload.update(extra)
            except Exception:
                logger.exception("engine link: hello_extra() failed; sending a plain hello")
        payload.update(boot_id=self.boot_id, conn_seq=conn_seq, engine_version=self._engine_version)
        return make_envelope("engine.hello", payload)

    async def _serve(self, leg: _Leg) -> None:
        """Run one logical connection (spanning rotations) until it is lost or the link stops."""
        self._lost_code = None
        self._lost.clear()
        if self._stopping.is_set():
            self._lost.set()
        self._primary = leg
        pinger = asyncio.create_task(self._ping_loop(), name="engine-link-ping")
        try:
            self._announced = True
            if self._on_connected is not None:
                try:
                    await self._on_connected()
                except Exception:
                    logger.exception("engine link: on_connected handler failed")
            await self._lost.wait()
        finally:
            pinger.cancel()
            await asyncio.gather(pinger, return_exceptions=True)
            await self._teardown()

    async def _teardown(self) -> None:
        """Close every leg, fail pending requests and announce the disconnect (idempotent)."""
        self._primary = None
        rotation, self._rotation = self._rotation, None
        if rotation is not None and not rotation.done():
            rotation.cancel()
            await asyncio.gather(rotation, return_exceptions=True)
        for leg in list(self._legs):
            await self._close_leg(leg)
        pending, self._pending = self._pending, {}
        for fut in pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("engine link lost"))
        if self._announced:
            self._announced = False
            if self._on_disconnected is not None:
                try:
                    await self._on_disconnected()
                except Exception:
                    logger.exception("engine link: on_disconnected handler failed")

    async def _close_leg(self, leg: _Leg, code: int = aiohttp.WSCloseCode.OK) -> None:
        if not leg.ws.closed:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(leg.ws.close(code=code), self._close_timeout)
        if leg.reader is not None:
            if not leg.reader.done():
                leg.reader.cancel()
            await asyncio.gather(leg.reader, return_exceptions=True)
        self._legs.discard(leg)

    def _next_delay(self, code: int | None, uptime: float, backoff: float) -> tuple[float, float]:
        """Return `(delay before the next attempt, next backoff)`."""
        if self._restart_delay is not None:
            return self._restart_delay, self._backoff_initial
        if code == CLOSE_UNAUTHORIZED:
            logger.error(
                "engine link: relay rejected the engine token; retrying in %.1fs (check the configured token)",
                self._backoff_max,
            )
            return self._backoff_max, self._backoff_max
        if code == CLOSE_SUPERSEDED:
            logger.warning("engine link: superseded by another engine connection; retrying in %.1fs", self._backoff_max)
            return self._backoff_max, self._backoff_max
        if uptime >= self._healthy_after:
            backoff = self._backoff_initial
        return backoff, min(backoff * 2, self._backoff_max)

    async def _ping_loop(self) -> None:
        while True:
            await asyncio.sleep(self._ping_interval)
            leg = self._primary
            if leg is None:
                return
            if time.monotonic() - leg.last_rx > 2 * self._ping_interval:
                logger.warning("engine link: leg %d silent for 2 ping intervals; reconnecting", leg.conn_seq)
                self._lost_code = None
                self._lost.set()
                return
            await self.send(make_envelope("ping", {}))

    # ------------------------------------------------------------------ rotation

    def _start_rotation(self, leg: _Leg, env: dict) -> None:
        if leg is not self._primary or (self._rotation is not None and not self._rotation.done()):
            return
        grace_ms = _bounded_ms(env.get("payload") or {}, "grace_ms", DEFAULT_ROTATE_GRACE_MS, MAX_ROTATE_GRACE_MS)
        self._rotation = asyncio.create_task(self._rotate(leg, grace_ms / 1000.0), name="engine-link-rotate")

    async def _rotate(self, old: _Leg, grace: float) -> None:
        try:
            new = await self._open_leg()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("engine link: rotation connect failed (%s); keeping leg %d", _describe(e), old.conn_seq)
            return
        if self._stopping.is_set() or self._lost.is_set() or self._primary is not old:
            await self._close_leg(new)
            return
        self._primary = new
        logger.info("engine link: rotated leg %d -> %d (grace %.1fs)", old.conn_seq, new.conn_seq, grace)
        await asyncio.sleep(grace)
        await self._close_leg(old)

    # ------------------------------------------------------------------ inbound

    async def _read_loop(self, leg: _Leg) -> None:
        try:
            async for msg in leg.ws:
                leg.last_rx = time.monotonic()
                if msg.type == aiohttp.WSMsgType.TEXT:
                    self._handle_text(leg, msg.data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    logger.warning("engine link: ignoring unexpected binary frame (%d bytes)", len(msg.data))
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.warning("engine link: leg %d error (%s)", leg.conn_seq, leg.ws.exception())
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("engine link: leg %d reader failed", leg.conn_seq)
        finally:
            self._leg_closed(leg)

    def _leg_closed(self, leg: _Leg) -> None:
        self._legs.discard(leg)
        code = leg.ws.close_code
        if leg is self._primary:
            if code not in (CLOSE_UNAUTHORIZED, CLOSE_SUPERSEDED):
                logger.info("engine link: leg %d closed (code=%s)", leg.conn_seq, code)
            self._lost_code = code
            self._lost.set()
        else:
            # A rotated-out leg (typically closed 4409 by the relay after its drain): expected.
            logger.debug("engine link: previous leg %d closed (code=%s)", leg.conn_seq, code)

    def _handle_text(self, leg: _Leg, data: str) -> None:
        try:
            env = parse_envelope(data)
        except ProtocolError as e:
            logger.warning("engine link: skipping malformed frame (%s)", e.message)
            return
        etype = env["type"]
        payload = env.get("payload") or {}
        reply_to = env.get("reply_to")
        if not reply_to and etype == "error":
            reply_to = payload.get("reply_to")
        if isinstance(reply_to, str) and reply_to:
            fut = self._pending.pop(reply_to, None)
            if fut is not None:
                if not fut.done():
                    fut.set_result(env)
                return
            if reply_to in self._expired:
                logger.debug("engine link: dropping late reply %s to %s", etype, reply_to)
                return
        if etype == "pong":
            return
        if etype == "relay.rotate":
            self._start_rotation(leg, env)
            return
        if etype == "relay.restarting":
            if leg is self._primary:
                retry_ms = _bounded_ms(payload, "retry_ms", DEFAULT_RESTART_RETRY_MS, MAX_RESTART_RETRY_MS)
                logger.info("engine link: relay restarting; reconnecting in %dms", retry_ms)
                self._restart_delay = retry_ms / 1000.0
                self._lost_code = None
                self._lost.set()
            return
        self._queue.put_nowait(env)

    async def _dispatch_loop(self) -> None:
        while True:
            env = await self._queue.get()
            try:
                await self._on_frame(env)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("engine link: on_frame handler failed for %s", env.get("type"))

    # ------------------------------------------------------------------ outbound helpers

    async def _send_on(self, leg: _Leg, data: str | bytes) -> bool:
        if leg.ws.closed:
            return False
        try:
            async with leg.send_lock:
                if isinstance(data, bytes):
                    await leg.ws.send_bytes(data)
                else:
                    await leg.ws.send_str(data)
        except (ConnectionError, aiohttp.ClientError, RuntimeError) as e:
            logger.debug("engine link: send on leg %d failed (%s)", leg.conn_seq, _describe(e))
            return False
        return True

    async def _request(self, req_id: str, sender: Callable[[], Awaitable[bool]], wait_s: float | None) -> dict:
        if req_id in self._pending:
            raise ValueError(f"duplicate pending request id {req_id}")
        fut: asyncio.Future[dict] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        try:
            if not await sender():
                raise ConnectionError("engine link is not connected")
            try:
                env = await asyncio.wait_for(fut, self._request_timeout if wait_s is None else wait_s)
            except TimeoutError:
                self._expired.append(req_id)
                raise
        finally:
            if self._pending.get(req_id) is fut:
                del self._pending[req_id]
        if env.get("type") == "error":
            payload = env.get("payload") or {}
            raise ProtocolError(str(payload.get("code") or "INTERNAL"), str(payload.get("message") or ""))
        return env
