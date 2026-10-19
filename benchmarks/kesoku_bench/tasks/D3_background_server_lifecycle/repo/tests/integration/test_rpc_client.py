"""Integration tests against a running mockrpc server (set MOCKRPC_URL, e.g. http://127.0.0.1:8765)."""

import os

import pytest
from mockrpc.client import RpcClient

URL = os.environ.get("MOCKRPC_URL")

pytestmark = pytest.mark.skipif(not URL, reason="MOCKRPC_URL not set (start the server first)")


@pytest.fixture
def client() -> RpcClient:
    """Client pointed at the live server."""
    assert URL
    return RpcClient(URL, timeout=5.0)


def test_ping(client: RpcClient) -> None:
    """The server answers ping."""
    assert client.ping() == "pong"


def test_echo(client: RpcClient) -> None:
    """Round-trip of a unicode string."""
    assert client.echo("grüße, 世界") == "grüße, 世界"


def test_sum(client: RpcClient) -> None:
    """Numeric round trip."""
    assert client.sum([1, 2, 3, 4.5]) == 10.5
