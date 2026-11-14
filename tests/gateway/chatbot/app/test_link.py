"""Tests for the engine→relay WebSocket link (`EngineLink`) against an in-process fake relay."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from aiohttp import WSMsgType, web

from kesoku.gateway.chatbot.app.link import EngineLink
from kesoku.gateway.chatbot.app.protocol import (
    ProtocolError,
    decode_binary_frame,
    encode_binary_frame,
    make_envelope,
    make_error,
)

TOKEN = "s3cret-engine-token"


@dataclass
class Conn:
    ws: web.WebSocketResponse
    auth: str | None
    opened_at: float
    frames: list[dict] = field(default_factory=list)
    binaries: list[bytes] = field(default_factory=list)
    inbox: asyncio.Queue = field(default_factory=asyncio.Queue)
    closed: asyncio.Event = field(default_factory=asyncio.Event)

    async def next_frame(self, etype: str | None = None, timeout: float = 2.0) -> dict:
        async with asyncio.timeout(timeout):
            while True:
                env = await self.inbox.get()
                if etype is None or env.get("type") == etype:
                    return env


Responder = Callable[["Conn", dict], Awaitable[None]]


class FakeRelay:
    """Minimal relay engine endpoint bound to 127.0.0.1 on an ephemeral port."""

    def __init__(self) -> None:
        self.conns: list[Conn] = []
        self.new_conns: asyncio.Queue[Conn] = asyncio.Queue()
        self.rejected = 0
        self.reply_pong = True
        self.on_hello: Responder | None = None
        self.on_text: Responder | None = None
        self.on_binary: Callable[[Conn, bytes], Awaitable[None]] | None = None
        self._runner: web.AppRunner | None = None
        self.url = ""

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/engine", self._handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        host, port = self._runner.addresses[0][:2]
        self.url = f"ws://{host}:{port}/engine"

    async def close(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    async def next_conn(self, timeout: float = 2.0) -> Conn:
        async with asyncio.timeout(timeout):
            return await self.new_conns.get()

    async def _handler(self, request: web.Request) -> web.StreamResponse:
        if request.headers.get("Authorization") != f"Bearer {TOKEN}":
            self.rejected += 1
            raise web.HTTPUnauthorized()
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        conn = Conn(ws=ws, auth=request.headers.get("Authorization"), opened_at=time.monotonic())
        self.conns.append(conn)
        self.new_conns.put_nowait(conn)
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    env = json.loads(msg.data)
                    conn.frames.append(env)
                    conn.inbox.put_nowait(env)
                    if env["type"] == "engine.hello" and self.on_hello is not None:
                        await self.on_hello(conn, env)
                    elif env["type"] == "ping" and self.reply_pong:
                        await ws.send_json(make_envelope("pong", reply_to=env["id"]))
                    elif self.on_text is not None:
                        await self.on_text(conn, env)
                elif msg.type == WSMsgType.BINARY:
                    conn.binaries.append(msg.data)
                    if self.on_binary is not None:
                        await self.on_binary(conn, msg.data)
        finally:
            conn.closed.set()
        return ws


@pytest.fixture
async def relay():
    r = FakeRelay()
    await r.start()
    yield r
    await r.close()


class Harness:
    def __init__(self, relay: FakeRelay, token: str = TOKEN, **kw: Any) -> None:
        self.frames: asyncio.Queue[dict] = asyncio.Queue()
        self.connected = 0
        self.disconnected = 0
        kw.setdefault("ping_interval", 5.0)
        kw.setdefault("backoff_initial", 0.02)
        kw.setdefault("backoff_max", 0.5)
        kw.setdefault("request_timeout", 2.0)
        self.on_frame_hook: Callable[[dict], Awaitable[None]] | None = None

        async def on_frame(env: dict) -> None:
            if self.on_frame_hook is not None:
                await self.on_frame_hook(env)
            self.frames.put_nowait(env)

        async def on_connected() -> None:
            self.connected += 1

        async def on_disconnected() -> None:
            self.disconnected += 1

        self.link = EngineLink(
            relay.url,
            token,
            on_frame=on_frame,
            on_connected=on_connected,
            on_disconnected=on_disconnected,
            **kw,
        )
        self.task: asyncio.Task | None = None

    def start(self) -> Harness:
        self.task = asyncio.create_task(self.link.run())
        return self

    async def stop(self) -> None:
        await self.link.stop()
        if self.task is not None:
            await asyncio.wait_for(self.task, 2.0)

    async def next_frame(self, etype: str | None = None, timeout: float = 2.0) -> dict:
        async with asyncio.timeout(timeout):
            while True:
                env = await self.frames.get()
                if etype is None or env.get("type") == etype:
                    return env


@pytest.fixture
async def harness_factory(relay):
    made: list[Harness] = []

    def make(**kw: Any) -> Harness:
        h = Harness(relay, **kw).start()
        made.append(h)
        return h

    yield make
    for h in made:
        await h.stop()


async def wait_until(pred: Callable[[], bool], timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not pred():
            await asyncio.sleep(0.005)


def link_tasks() -> list[asyncio.Task]:
    return [t for t in asyncio.all_tasks() if t.get_name().startswith("engine-link") and not t.done()]


async def test_auth_header_and_hello_fields(relay, harness_factory, caplog):
    caplog.set_level(logging.DEBUG)
    h = harness_factory(engine_version="9.9.9", hello_extra=lambda: {"revoked_device_ids": ["d1"], "boot_id": "x"})
    conn = await relay.next_conn()
    hello = await conn.next_frame()
    assert conn.auth == f"Bearer {TOKEN}"
    assert hello["type"] == "engine.hello" and hello["v"] == 1
    p = hello["payload"]
    assert p["boot_id"] == h.link.boot_id and isinstance(p["boot_id"], int)  # core fields win over extras
    assert p["conn_seq"] == 1 == h.link.conn_seq
    assert p["engine_version"] == "9.9.9"
    assert p["revoked_device_ids"] == ["d1"]
    await wait_until(lambda: h.connected == 1)
    assert h.link.connected
    await conn.ws.send_json(make_envelope("engine.welcome", {"primary": True}))
    assert (await h.next_frame())["type"] == "engine.welcome"
    assert TOKEN not in caplog.text


async def test_ping_pong_keeps_leg_alive(relay, harness_factory):
    h = harness_factory(ping_interval=0.05)
    conn = await relay.next_conn()
    await asyncio.sleep(0.4)
    assert len(relay.conns) == 1
    assert sum(1 for f in conn.frames if f["type"] == "ping") >= 3
    assert h.frames.empty()  # pongs are consumed internally


async def test_liveness_timeout_reconnects(relay, harness_factory):
    relay.reply_pong = False
    h = harness_factory(ping_interval=0.05)
    first = await relay.next_conn()
    second = await relay.next_conn(timeout=2.0)
    assert (await second.next_frame("engine.hello"))["payload"]["conn_seq"] == 2
    await wait_until(lambda: first.closed.is_set())
    assert h.disconnected == 1
    await wait_until(lambda: h.connected == 2)


async def test_request_response_error_and_timeout(relay, harness_factory):
    async def on_text(conn: Conn, env: dict) -> None:
        if env["type"] == "media.sign":
            await conn.ws.send_json(make_envelope("media.signed", {"url": "https://x/y"}, reply_to=env["id"]))
        elif env["type"] == "media.upload.ticket":
            await conn.ws.send_json(make_error("NOT_FOUND", "no such object", reply_to=env["id"]))
        elif env["type"] == "slow":
            await asyncio.sleep(0.3)
            await conn.ws.send_json(make_envelope("slow.result", reply_to=env["id"]))

    relay.on_text = on_text
    h = harness_factory()
    await relay.next_conn()
    await wait_until(lambda: h.link.connected)

    reply = await h.link.request(make_envelope("media.sign", {"object": "media/a/b.png"}))
    assert reply["type"] == "media.signed" and reply["payload"]["url"] == "https://x/y"

    with pytest.raises(ProtocolError) as ei:
        await h.link.request(make_envelope("media.upload.ticket", {}))
    assert ei.value.code == "NOT_FOUND"

    with pytest.raises(TimeoutError):
        await h.link.request(make_envelope("slow", {}), timeout=0.05)
    await asyncio.sleep(0.4)  # the late reply must not leak into on_frame
    assert h.frames.empty()


async def test_binary_frame_round_trip(relay, harness_factory):
    async def on_binary(conn: Conn, data: bytes) -> None:
        header, body = decode_binary_frame(data)
        await conn.ws.send_json(
            make_envelope("media.put.ok", {"object": "media/asset/x.bin", "bytes": len(body)}, reply_to=header["id"])
        )

    relay.on_binary = on_binary
    h = harness_factory()
    conn = await relay.next_conn()
    await wait_until(lambda: h.link.connected)
    body = bytes(range(256)) * 40
    frame = encode_binary_frame({"v": 1, "type": "media.put", "id": "req_bin1", "purpose": "asset"}, body)
    reply = await h.link.request_binary(frame, "req_bin1")
    assert reply["payload"]["bytes"] == len(body)
    assert conn.binaries == [frame]
    with pytest.raises(ProtocolError):
        await h.link.send_binary(b"\x00" * (1_048_576 + 1))


async def test_send_without_leg_returns_false(relay):
    h = Harness(relay)
    assert await h.link.send(make_envelope("x")) is False
    assert await h.link.send_binary(b"abc") is False
    with pytest.raises(ConnectionError):
        await h.link.request(make_envelope("x"))
    assert not h.link.connected


async def test_seamless_rotation(relay, harness_factory, caplog):
    rotated = asyncio.Event()

    async def on_hello(conn: Conn, env: dict) -> None:
        if env["payload"]["conn_seq"] == 1:
            await conn.ws.send_json(make_envelope("relay.rotate", {"grace_ms": 300}))
        else:
            rotated.set()

    relay.on_hello = on_hello
    h = harness_factory()
    old = await relay.next_conn()
    new = await relay.next_conn()
    assert (await new.next_frame("engine.hello"))["payload"]["conn_seq"] == 2
    await asyncio.wait_for(rotated.wait(), 2)
    await wait_until(lambda: h.link.conn_seq == 2)

    # Frames still arriving on the old leg during the grace period are delivered.
    await old.ws.send_json(make_envelope("client.frame", {"n": 1}))
    assert (await h.next_frame("client.frame"))["payload"]["n"] == 1
    # Outbound traffic goes to the new leg only.
    assert await h.link.send(make_envelope("engine.out", {"k": 1}))
    assert (await new.next_frame("engine.out"))["payload"]["k"] == 1
    assert not any(f["type"] == "engine.out" for f in old.frames)
    # The old leg is closed after the grace period; the link stays up throughout.
    await asyncio.wait_for(old.closed.wait(), 2)
    await new.ws.send_json(make_envelope("client.frame", {"n": 2}))
    assert (await h.next_frame("client.frame"))["payload"]["n"] == 2
    assert h.link.connected and h.connected == 1 and h.disconnected == 0
    assert len(relay.conns) == 2
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


async def test_superseded_old_leg_close_is_silent(relay, harness_factory, caplog):
    async def on_hello(conn: Conn, env: dict) -> None:
        if env["payload"]["conn_seq"] == 1:
            await conn.ws.send_json(make_envelope("relay.rotate", {"grace_ms": 5000}))
        else:
            await relay.conns[0].ws.close(code=4409, message=b"superseded")

    relay.on_hello = on_hello
    h = harness_factory()
    old = await relay.next_conn()
    await relay.next_conn()
    await asyncio.wait_for(old.closed.wait(), 2)
    await asyncio.sleep(0.1)
    assert h.link.connected and h.link.conn_seq == 2 and len(relay.conns) == 2
    assert h.disconnected == 0
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


async def test_reconnect_with_exponential_backoff(relay, harness_factory):
    async def on_hello(conn: Conn, env: dict) -> None:
        if env["payload"]["conn_seq"] <= 3:
            await conn.ws.close()

    relay.on_hello = on_hello
    h = harness_factory(backoff_initial=0.05, backoff_max=1.0)
    conns = [await relay.next_conn() for _ in range(4)]
    gaps = [b.opened_at - a.opened_at for a, b in zip(conns, conns[1:], strict=False)]
    assert gaps[0] >= 0.05 and gaps[1] >= 0.1 and gaps[2] >= 0.2
    assert gaps[0] < gaps[1] < gaps[2]
    await wait_until(lambda: h.connected == 4)
    assert h.disconnected == 3 and h.link.connected


async def test_4401_close_backs_off_at_max(relay, harness_factory, caplog):
    async def on_hello(conn: Conn, env: dict) -> None:
        if env["payload"]["conn_seq"] == 1:
            await conn.ws.close(code=4401, message=b"unauthorized")

    relay.on_hello = on_hello
    h = harness_factory(backoff_initial=0.01, backoff_max=0.3)
    first = await relay.next_conn()
    second = await relay.next_conn()
    assert second.opened_at - first.opened_at >= 0.3
    assert any(r.levelno == logging.ERROR and "rejected" in r.getMessage() for r in caplog.records)
    assert TOKEN not in caplog.text


async def test_http_401_handshake_backs_off_at_max(relay, harness_factory, caplog):
    h = harness_factory(token="wrong-token", backoff_initial=0.01, backoff_max=0.3)
    await wait_until(lambda: relay.rejected >= 1)
    await asyncio.sleep(0.15)
    assert relay.rejected == 1 and not relay.conns
    assert any(r.levelno == logging.ERROR for r in caplog.records)
    assert "wrong-token" not in caplog.text


async def test_relay_restarting_reconnects_after_retry(relay, harness_factory):
    async def on_hello(conn: Conn, env: dict) -> None:
        if env["payload"]["conn_seq"] == 1:
            await conn.ws.send_json(make_envelope("relay.restarting", {"retry_ms": 200}))

    relay.on_hello = on_hello
    h = harness_factory(backoff_initial=0.01)
    first = await relay.next_conn()
    second = await relay.next_conn()
    assert second.opened_at - first.opened_at >= 0.2
    assert (await second.next_frame("engine.hello"))["payload"]["conn_seq"] == 2
    assert h.frames.empty()


async def test_malformed_frames_and_handler_errors_do_not_kill_reader(relay, harness_factory):
    async def hook(env: dict) -> None:
        if env["type"] == "boom":
            raise RuntimeError("handler failure")

    h = harness_factory()
    h.on_frame_hook = hook
    conn = await relay.next_conn()
    await conn.ws.send_str("not json")
    await conn.ws.send_str(json.dumps({"v": 2, "type": "x", "id": "a"}))
    await conn.ws.send_bytes(b"\x00\x01")
    await conn.ws.send_json(make_envelope("boom"))
    await conn.ws.send_json(make_envelope("after", {"ok": True}))
    assert (await h.next_frame("after"))["payload"]["ok"] is True
    assert h.link.connected and len(relay.conns) == 1


async def test_on_frame_may_await_a_request(relay, harness_factory):
    async def on_text(conn: Conn, env: dict) -> None:
        if env["type"] == "media.sign":
            await conn.ws.send_json(make_envelope("media.signed", {"url": "u"}, reply_to=env["id"]))

    relay.on_text = on_text
    results: list[dict] = []
    h = harness_factory()

    async def hook(env: dict) -> None:
        if env["type"] == "message.send":
            results.append(await h.link.request(make_envelope("media.sign", {"object": "o"})))

    h.on_frame_hook = hook
    conn = await relay.next_conn()
    await conn.ws.send_json(make_envelope("message.send", {}))
    await h.next_frame("message.send")
    assert results and results[0]["payload"]["url"] == "u"


async def test_stop_is_clean_and_fails_pending_requests(relay):
    h = Harness(relay).start()
    conn = await relay.next_conn()
    await wait_until(lambda: h.link.connected)
    pending = asyncio.create_task(h.link.request(make_envelope("never.answered"), timeout=5))
    await conn.next_frame("never.answered")
    t0 = time.monotonic()
    await h.stop()
    assert time.monotonic() - t0 < 1.0
    with pytest.raises(ConnectionError):
        await pending
    assert h.task.done() and h.task.exception() is None
    assert not h.link.connected and h.disconnected == 1
    await asyncio.wait_for(conn.closed.wait(), 1)
    assert link_tasks() == []


async def test_stop_during_backoff_returns_quickly(relay):
    relay_url_down = relay.url
    await relay.close()
    h = Harness(relay, backoff_initial=5.0, backoff_max=5.0)
    h.link._relay_url = relay_url_down
    h.start()
    await asyncio.sleep(0.1)
    t0 = time.monotonic()
    await h.stop()
    assert time.monotonic() - t0 < 1.0
    assert h.connected == 0 and h.disconnected == 0
    assert link_tasks() == []
