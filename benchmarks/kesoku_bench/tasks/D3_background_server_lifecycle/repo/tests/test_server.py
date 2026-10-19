"""Unit tests for mockrpc.server (in-process, no warm-up)."""

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from mockrpc import server as srv


@pytest.fixture
def start_server(tmp_path: Path) -> Iterator[Callable[..., str]]:
    """Start servers on free ports in background threads; returns a factory yielding the base URL."""
    started: list[srv.RpcHTTPServer] = []

    def _start(warming_requests: int = 0) -> str:
        server = srv.build_server(0, state_dir=tmp_path, warming_requests=warming_requests)
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.005}, daemon=True).start()
        started.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield _start
    for server in started:
        server.shutdown()
        server.server_close()


def _post(url: str, payload: object, content_type: str = "application/json") -> tuple[int, dict]:
    """POST ``payload`` and return (status, decoded body)."""
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    req = urllib.request.Request(url + "/rpc", data=data, method="POST", headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _rpc(method: str, **params: object) -> dict:
    """Build a JSON-RPC request body."""
    return {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}


def test_ping(start_server: Callable[..., str]) -> None:
    """ping returns pong."""
    status, body = _post(start_server(), _rpc("ping"))
    assert status == 200
    assert body["result"] == "pong"


def test_echo(start_server: Callable[..., str]) -> None:
    """echo returns its text."""
    _, body = _post(start_server(), _rpc("echo", text="héllo"))
    assert body["result"] == "héllo"


def test_sum(start_server: Callable[..., str]) -> None:
    """sum adds numbers."""
    _, body = _post(start_server(), _rpc("sum", values=[1, 2, 3.5]))
    assert body["result"] == 6.5


def test_response_echoes_request_id(start_server: Callable[..., str]) -> None:
    """The response id matches the request id."""
    payload = _rpc("ping")
    payload["id"] = 42
    _, body = _post(start_server(), payload)
    assert body["id"] == 42


def test_unknown_method(start_server: Callable[..., str]) -> None:
    """Unknown methods yield JSON-RPC -32601."""
    status, body = _post(start_server(), _rpc("nope"))
    assert status == 200
    assert body["error"]["code"] == -32601


def test_bad_params(start_server: Callable[..., str]) -> None:
    """Bad params yield JSON-RPC -32602."""
    _, body = _post(start_server(), _rpc("sum", values="x"))
    assert body["error"]["code"] == -32602


def test_invalid_json(start_server: Callable[..., str]) -> None:
    """Malformed bodies are a 400 parse error."""
    status, body = _post(start_server(), b"{not json")
    assert status == 400
    assert body["error"]["code"] == -32700


def test_wrong_content_type_rejected(start_server: Callable[..., str]) -> None:
    """Only application/json is accepted."""
    status, _ = _post(start_server(), _rpc("ping"), content_type="text/plain")
    assert status == 415


def test_content_type_with_charset_accepted(start_server: Callable[..., str]) -> None:
    """Parameters on the media type are fine."""
    status, _ = _post(start_server(), _rpc("ping"), content_type="application/json; charset=utf-8")
    assert status == 200


def test_first_requests_get_503_while_warming(start_server: Callable[..., str]) -> None:
    """The first N requests after bind are answered with 503."""
    url = start_server(warming_requests=2)
    assert _post(url, _rpc("ping"))[0] == 503
    assert _post(url, _rpc("ping"))[0] == 503
    assert _post(url, _rpc("ping"))[0] == 200


def test_health(start_server: Callable[..., str]) -> None:
    """GET /health reports ok."""
    with urllib.request.urlopen(start_server() + "/health", timeout=5) as resp:
        assert json.loads(resp.read()) == {"status": "ok"}


def test_unknown_path_404(start_server: Callable[..., str]) -> None:
    """Other paths are 404."""
    with pytest.raises(urllib.error.HTTPError) as info:
        urllib.request.urlopen(start_server() + "/nope", timeout=5)
    assert info.value.code == 404


def test_requests_are_logged(start_server: Callable[..., str], tmp_path: Path) -> None:
    """Each request appends one JSON line to requests.log."""
    url = start_server(warming_requests=1)
    _post(url, _rpc("ping"))
    _post(url, _rpc("echo", text="x"))
    lines = [json.loads(line) for line in (tmp_path / srv.REQUEST_LOG).read_text().splitlines()]
    assert [(e["rpc"], e["status"]) for e in lines] == [("ping", 503), ("echo", 200)]
    assert [e["n"] for e in lines] == [1, 2]
    assert all(e["http"] == "POST /rpc" for e in lines)


def test_main_requires_port() -> None:
    """The CLI refuses to start without --port."""
    with pytest.raises(SystemExit):
        srv.main([])
