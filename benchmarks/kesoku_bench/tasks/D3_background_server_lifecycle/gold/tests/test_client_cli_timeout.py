"""Unit test for the client CLI --timeout option (reference solution of turn 4)."""

import io
import json
import urllib.request

import pytest
from mockrpc import client as cl


def test_cli_timeout_option(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """--timeout is parsed as float and used for the HTTP request."""
    seen: list[float | None] = []

    def opener(request: urllib.request.Request, timeout: float | None = None) -> io.BytesIO:
        seen.append(timeout)
        return io.BytesIO(json.dumps({"result": "pong"}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", opener)
    assert cl.main(["--url", "http://x", "--timeout", "0.75", "ping"]) == 0
    assert seen == [0.75]
    assert cl.build_parser().parse_args(["ping"]).timeout == cl.DEFAULT_TIMEOUT
