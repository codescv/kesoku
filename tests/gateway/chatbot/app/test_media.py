"""Tests for app media uploads (`app/media.py`) and assistant sender records (`app/segments.py`)."""

import time
import wave
from typing import Any

import pytest
from aiohttp import web

from kesoku.constants import MessageRole
from kesoku.db import Message
from kesoku.gateway.chatbot.app import media as media_mod
from kesoku.gateway.chatbot.app.media import (
    BINARY_UPLOAD_MAX_BYTES,
    MediaManager,
    build_put_headers,
    probe_audio_duration_ms,
)
from kesoku.gateway.chatbot.app.segments import message_to_record

RANGE_HEADER = "x-goog-content-length-range"
RANGE_VALUE = "0,10485760"


class TicketRelay:
    """Relay stub issuing upload tickets that bind an extension header (like signed V4 URLs)."""

    def __init__(self, put_url: str = "https://store.invalid/put/obj", headers: Any = None) -> None:
        self.put_url = put_url
        self.headers = headers if headers is not None else {"Content-Type": "image/png", RANGE_HEADER: RANGE_VALUE}
        self.requests: list[dict[str, Any]] = []

    async def request(self, envelope: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        self.requests.append(envelope)
        assert envelope["type"] == "media.upload.ticket"
        payload = envelope["payload"]
        body: dict[str, Any] = {
            "object": "media/s1/big.png",
            "put_url": self.put_url,
            "max_bytes": payload["bytes"],
            "get_url": "https://store.invalid/media/s1/big.png?sig=1",
            "get_expires_at": time.time() + 900,
        }
        if self.headers is not False:
            body["headers"] = self.headers
        return {"type": "media.ticket.ok", "payload": body}

    async def request_binary(self, frame: bytes, request_id: str, timeout: float | None = None) -> dict[str, Any]:
        raise AssertionError("large files must use the upload ticket")


def _big_file(tmp_path) -> str:
    path = tmp_path / "cg.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * (BINARY_UPLOAD_MAX_BYTES + 10))
    return str(path)


def test_build_put_headers_merges_ticket_headers() -> None:
    hdrs = build_put_headers(
        "image/png", {RANGE_HEADER: RANGE_VALUE, "content-type": "text/plain", "x-bad": 3, 7: "x"}
    )
    assert hdrs == {RANGE_HEADER: RANGE_VALUE, "Content-Type": "image/png"}
    assert build_put_headers("audio/wav", None) == {"Content-Type": "audio/wav"}
    assert build_put_headers("audio/wav", ["not", "a", "dict"]) == {"Content-Type": "audio/wav"}


async def test_large_upload_forwards_ticket_headers_to_4_arg_put(tmp_path) -> None:
    relay = TicketRelay()
    calls: list[tuple[str, int, str, Any]] = []

    async def put(url: str, data: bytes, mime: str, extra_headers: dict[str, str] | None = None) -> None:
        calls.append((url, len(data), mime, extra_headers))

    mgr = MediaManager(lambda: relay, http_put=put)
    res = await mgr.upload_file(_big_file(tmp_path), purpose="message", session_id="s1")
    assert res["object"] == "media/s1/big.png" and res["url"].startswith("https://store.invalid/")
    assert len(calls) == 1
    url, size, mime, extra = calls[0]
    assert url == relay.put_url and size > BINARY_UPLOAD_MAX_BYTES and mime == "image/png"
    assert extra == {"Content-Type": "image/png", RANGE_HEADER: RANGE_VALUE}


async def test_large_upload_supports_legacy_3_arg_and_varargs_put(tmp_path) -> None:
    legacy_calls: list[tuple[str, int, str]] = []

    async def legacy_put(url: str, data: bytes, mime: str) -> None:
        legacy_calls.append((url, len(data), mime))

    mgr = MediaManager(lambda: TicketRelay(), http_put=legacy_put)
    await mgr.upload_file(_big_file(tmp_path), purpose="asset")
    assert len(legacy_calls) == 1 and legacy_calls[0][2] == "image/png"

    var_calls: list[tuple[Any, ...]] = []

    async def var_put(*args: Any) -> None:
        var_calls.append(args)

    mgr2 = MediaManager(lambda: TicketRelay(), http_put=var_put)
    await mgr2.upload_file(_big_file(tmp_path), purpose="asset", index=1)
    assert len(var_calls[0]) == 4 and var_calls[0][3][RANGE_HEADER] == RANGE_VALUE


async def test_large_upload_without_ticket_headers_passes_none(tmp_path) -> None:
    seen: list[Any] = []

    async def put(url: str, data: bytes, mime: str, extra_headers: dict[str, str] | None = None) -> None:
        seen.append(extra_headers)

    mgr = MediaManager(lambda: TicketRelay(headers=False), http_put=put)
    await mgr.upload_file(_big_file(tmp_path), purpose="asset")
    mgr2 = MediaManager(lambda: TicketRelay(headers="junk"), http_put=put)
    await mgr2.upload_file(_big_file(tmp_path), purpose="message", session_id="s2")
    assert seen == [None, None]


@pytest.fixture
async def put_server():
    """Local HTTP server that records signed PUT requests and enforces the bound range header."""
    received: list[dict[str, Any]] = []

    async def handle(request: web.Request) -> web.Response:
        body = await request.read()
        received.append({"headers": dict(request.headers), "bytes": len(body)})
        if request.headers.get(RANGE_HEADER) != RANGE_VALUE:
            return web.Response(status=400, text="missing extension header")
        return web.Response(status=200)

    app = web.Application(client_max_size=4 * 1024 * 1024)
    app.router.add_put("/put/{name}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # noqa: SLF001
    yield f"http://127.0.0.1:{port}", received
    await runner.cleanup()


async def test_default_put_sends_ticket_headers_end_to_end(tmp_path, put_server) -> None:
    base, received = put_server
    relay = TicketRelay(put_url=f"{base}/put/big.png")
    mgr = MediaManager(lambda: relay)
    res = await mgr.upload_file(_big_file(tmp_path), purpose="message", session_id="s1")
    assert res["object"] == "media/s1/big.png"
    assert len(received) == 1
    assert received[0]["headers"][RANGE_HEADER] == RANGE_VALUE
    assert received[0]["headers"]["Content-Type"] == "image/png"
    assert received[0]["bytes"] > BINARY_UPLOAD_MAX_BYTES


async def test_default_put_without_bound_header_is_rejected(put_server) -> None:
    base, _ = put_server
    with pytest.raises(ConnectionError, match="HTTP 400"):
        await MediaManager._default_put(f"{base}/put/x.png", b"data", "image/png")
    await MediaManager._default_put(f"{base}/put/x.png", b"data", "image/png", {RANGE_HEADER: RANGE_VALUE})


def _write_wav(path, seconds: float = 0.5, rate: int = 8000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))


def test_probe_audio_duration_ms(tmp_path) -> None:
    wav = tmp_path / "v.wav"
    _write_wav(wav, 1.25)
    assert probe_audio_duration_ms(str(wav)) == 1250
    junk = tmp_path / "v.ogg"
    junk.write_bytes(b"OggS" + b"\x00" * 64)
    assert probe_audio_duration_ms(str(junk)) is None
    assert probe_audio_duration_ms(str(tmp_path / "missing.wav")) is None
    assert media_mod.probe_audio_duration_ms is probe_audio_duration_ms


def _msg(role: MessageRole, sender: str) -> Message:
    return Message(session_id="s1", chatbot_id="app", channel_id="app:s1", sender=sender, role=role, content="hello")


def test_message_to_record_attaches_persona_to_assistant_sender() -> None:
    rec = message_to_record(
        _msg(MessageRole.ASSISTANT, "Kesoku"), "app", role_id="hana", role_display_name="Hana Example"
    )
    assert rec["sender"] == {
        "kind": "assistant", "name": "Kesoku", "chatbot_id": "app", "role_id": "hana", "display_name": "Hana Example",
    }
    legacy = message_to_record(_msg(MessageRole.ASSISTANT, "Kesoku"), "app")
    assert "role_id" not in legacy["sender"] and "display_name" not in legacy["sender"]
    note = message_to_record(_msg(MessageRole.ASSISTANT, "Notification"), "app", role_id="hana",
                             role_display_name="Hana Example")
    assert "display_name" not in note["sender"] and "role_id" not in note["sender"]
    user = message_to_record(_msg(MessageRole.USER, "dev-1"), "app", role_id="hana", role_display_name="Hana Example")
    assert user["sender"]["kind"] == "user" and "display_name" not in user["sender"]
