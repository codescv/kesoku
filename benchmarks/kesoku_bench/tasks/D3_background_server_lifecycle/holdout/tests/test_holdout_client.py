"""Held-out tests for the mockrpc client fixes: JSON content type, bounded 503 retry, CLI --timeout."""

import io
import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from email.message import Message
from pathlib import Path
from typing import Any

import pytest
from mockrpc import client as cl
from mockrpc import server as srv


class _Response(io.BytesIO):
    """urlopen() response stand-in."""

    status = 200


class _Opener:
    """Fake urlopen: raises the queued HTTP errors, then returns the JSON body."""

    def __init__(self, errors: list[int], body: dict[str, Any] | None = None) -> None:
        """Queue HTTP error codes to raise before answering with ``body``."""
        self.errors = list(errors)
        self.body = body if body is not None else {"jsonrpc": "2.0", "id": 1, "result": "pong"}
        self.requests: list[urllib.request.Request] = []
        self.timeouts: list[float | None] = []

    def __call__(self, request: urllib.request.Request, timeout: float | None = None, **_: Any) -> _Response:
        """Record the call and answer."""
        self.requests.append(request)
        self.timeouts.append(timeout)
        if self.errors:
            code = self.errors.pop(0)
            raise urllib.error.HTTPError(
                request.full_url, code, "warming", Message(), io.BytesIO(b'{"error": "warming"}')
            )
        return _Response(json.dumps(self.body).encode())


def _headers(request: urllib.request.Request) -> dict[str, str]:
    """Header dict with lower-cased names."""
    return {k.lower(): v for k, v in request.header_items()}


@pytest.fixture
def live_url(tmp_path: Path) -> Iterator[str]:
    """In-process server that answers 503 to its first 2 requests (like the real one after warm-up)."""
    server = srv.build_server(0, state_dir=tmp_path, warming_requests=2)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.005}, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_content_type_is_json() -> None:
    """Requests declare Content-Type: application/json."""
    opener = _Opener([])
    cl.RpcClient("http://x", opener=opener).call("ping")
    ctype = _headers(opener.requests[0]).get("content-type", "")
    assert ctype.split(";")[0].strip().lower() == "application/json"


def test_retries_503_then_succeeds() -> None:
    """Two 503 answers are retried transparently with a short backoff."""
    opener = _Opener([503, 503])
    start = time.monotonic()
    assert cl.RpcClient("http://x", opener=opener).call("ping") == "pong"
    assert len(opener.requests) == 3
    assert time.monotonic() - start < 1.5


def test_503_retry_is_bounded() -> None:
    """A server that keeps answering 503 is given up on after a bounded number of attempts."""
    opener = _Opener([503] * 50)
    start = time.monotonic()
    with pytest.raises(Exception):
        cl.RpcClient("http://x", opener=opener).call("ping")
    assert 3 <= len(opener.requests) <= 5
    assert time.monotonic() - start < 3.0


def test_cli_timeout_flag(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """`--timeout 2.5` reaches the HTTP layer."""
    opener = _Opener([])
    monkeypatch.setattr(urllib.request, "urlopen", opener)
    monkeypatch.setattr(cl, "urlopen", opener, raising=False)
    rc = cl.main(["--url", "http://x", "--timeout", "2.5", "ping"])
    assert rc in (0, None)
    assert opener.timeouts and all(t == 2.5 for t in opener.timeouts)


def test_end_to_end_against_warming_server(live_url: str) -> None:
    """The client survives the warm-up 503s and the server accepts its requests."""
    client = cl.RpcClient(live_url, timeout=5.0)
    assert client.call("ping") == "pong"
    assert client.call("echo", text="ok") == "ok"
    assert client.call("sum", values=[1, 2]) == 3
