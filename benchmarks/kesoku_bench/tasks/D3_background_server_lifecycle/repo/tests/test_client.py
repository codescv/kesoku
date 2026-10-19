"""Unit tests for mockrpc.client (fake opener, no network)."""

import io
import json
import urllib.request
from typing import Any

import pytest
from mockrpc import client as cl


class FakeResponse(io.BytesIO):
    """A urlopen() response stand-in usable as a context manager."""

    status = 200


class FakeOpener:
    """Records requests and replays canned JSON responses."""

    def __init__(self, *bodies: dict[str, Any]) -> None:
        self.bodies = list(bodies)
        self.requests: list[urllib.request.Request] = []
        self.timeouts: list[float | None] = []

    def __call__(self, request: urllib.request.Request, timeout: float | None = None) -> FakeResponse:
        self.requests.append(request)
        self.timeouts.append(timeout)
        return FakeResponse(json.dumps(self.bodies.pop(0)).encode())


def test_call_returns_result() -> None:
    """call() returns the result member."""
    opener = FakeOpener({"jsonrpc": "2.0", "id": 1, "result": "pong"})
    assert cl.RpcClient("http://x", opener=opener).call("ping") == "pong"


def test_call_posts_to_rpc_endpoint() -> None:
    """Requests go to <base>/rpc via POST; a trailing slash is tolerated."""
    opener = FakeOpener({"result": 1})
    cl.RpcClient("http://x:1/", opener=opener).call("ping")
    req = opener.requests[0]
    assert req.full_url == "http://x:1/rpc"
    assert req.get_method() == "POST"


def test_request_body_is_jsonrpc() -> None:
    """The body is a JSON-RPC 2.0 request with the keyword params."""
    opener = FakeOpener({"result": "hi"})
    cl.RpcClient("http://x", opener=opener).call("echo", text="hi")
    body = json.loads(opener.requests[0].data)
    assert body["jsonrpc"] == "2.0"
    assert body["method"] == "echo"
    assert body["params"] == {"text": "hi"}


def test_request_ids_increment() -> None:
    """Each call uses a fresh id."""
    opener = FakeOpener({"result": 1}, {"result": 2})
    client = cl.RpcClient("http://x", opener=opener)
    client.ping()
    client.ping()
    assert [json.loads(r.data)["id"] for r in opener.requests] == [1, 2]


def test_timeout_passed_to_opener() -> None:
    """The client's timeout is used for every HTTP request."""
    opener = FakeOpener({"result": 1})
    cl.RpcClient("http://x", timeout=1.25, opener=opener).ping()
    assert opener.timeouts == [1.25]


def test_rpc_error_raised() -> None:
    """A JSON-RPC error becomes RpcError."""
    opener = FakeOpener({"error": {"code": -32601, "message": "method not found"}})
    with pytest.raises(cl.RpcError) as info:
        cl.RpcClient("http://x", opener=opener).call("nope")
    assert info.value.code == -32601


def test_helpers() -> None:
    """echo/sum helpers pass their params."""
    opener = FakeOpener({"result": "a"}, {"result": 3})
    client = cl.RpcClient("http://x", opener=opener)
    assert client.echo("a") == "a"
    assert client.sum([1, 2]) == 3
    assert json.loads(opener.requests[1].data)["params"] == {"values": [1, 2]}


def test_cli_prints_result(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI prints the JSON result."""
    monkeypatch.setattr(urllib.request, "urlopen", FakeOpener({"result": "hi"}))
    assert cl.main(["--url", "http://x", "echo", '{"text": "hi"}']) == 0
    assert capsys.readouterr().out.strip() == '"hi"'


def test_cli_rejects_non_object_params(capsys: pytest.CaptureFixture[str]) -> None:
    """params must be a JSON object."""
    assert cl.main(["--url", "http://x", "echo", "[1]"]) == 2


def test_cli_reports_rpc_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """RPC errors exit with status 1."""
    monkeypatch.setattr(urllib.request, "urlopen", FakeOpener({"error": {"code": -1, "message": "boom"}}))
    assert cl.main(["--url", "http://x", "ping"]) == 1
    assert "boom" in capsys.readouterr().err
