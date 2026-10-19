"""HTTP client for the mockrpc JSON-RPC 2.0 server."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

DEFAULT_URL = "http://127.0.0.1:8765"
DEFAULT_TIMEOUT = 5.0
MAX_ATTEMPTS = 3
RETRY_BACKOFF_S = (0.05, 0.1)  # sleeps between attempts; total <= 0.2 s

Opener = Callable[..., Any]


class RpcError(Exception):
    """A JSON-RPC error returned by the server."""

    def __init__(self, code: int, message: str) -> None:
        """Store the JSON-RPC error code and message.

        Args:
            code: JSON-RPC error code.
            message: Error message from the server.
        """
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class RpcClient:
    """Minimal JSON-RPC 2.0 client over HTTP POST."""

    def __init__(self, base_url: str, timeout: float = DEFAULT_TIMEOUT, opener: Opener | None = None) -> None:
        """Create a client.

        Args:
            base_url: Server base URL, e.g. ``http://127.0.0.1:8765``.
            timeout: Socket timeout in seconds for each HTTP request.
            opener: ``urlopen``-compatible callable (injected by tests).
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._opener = opener or urllib.request.urlopen
        self._ids = itertools.count(1)

    def _build_request(self, method: str, params: dict[str, Any]) -> urllib.request.Request:
        """Build the HTTP request for one RPC call."""
        payload = {"jsonrpc": "2.0", "id": next(self._ids), "method": method, "params": params}
        return urllib.request.Request(
            f"{self.base_url}/rpc",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )

    def call(self, method: str, **params: Any) -> Any:
        """Invoke ``method`` with keyword ``params`` and return its result.

        Args:
            method: RPC method name.
            **params: Method parameters.

        Returns:
            The ``result`` member of the response.

        Raises:
            RpcError: If the server returns a JSON-RPC error.
            urllib.error.HTTPError: On a non-2xx HTTP status (503 only after MAX_ATTEMPTS attempts).
        """
        request = self._build_request(method, params)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                with self._opener(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                # 503 = server still warming up: retry with a short backoff, give up after MAX_ATTEMPTS.
                if exc.code != 503 or attempt == MAX_ATTEMPTS:
                    raise
                time.sleep(RETRY_BACKOFF_S[min(attempt - 1, len(RETRY_BACKOFF_S) - 1)])
        if "error" in data:
            error = data["error"] or {}
            raise RpcError(int(error.get("code", -32000)), str(error.get("message", "unknown error")))
        return data.get("result")

    def ping(self) -> Any:
        """Call ``ping``."""
        return self.call("ping")

    def echo(self, text: str) -> Any:
        """Call ``echo``."""
        return self.call("echo", text=text)

    def sum(self, values: list[float]) -> Any:
        """Call ``sum``."""
        return self.call("sum", values=values)


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(prog="mockrpc.client", description="Call a mockrpc server method.")
    parser.add_argument("--url", default=os.environ.get("MOCKRPC_URL", DEFAULT_URL), help="server base URL")
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT, help=f"HTTP timeout in seconds (default {DEFAULT_TIMEOUT})"
    )
    parser.add_argument("method", help="RPC method name")
    parser.add_argument("params", nargs="?", default="{}", help="JSON object with the method parameters")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: call one method and print its JSON result.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    params = json.loads(args.params)
    if not isinstance(params, dict):
        print("params must be a JSON object", file=sys.stderr)
        return 2
    client = RpcClient(args.url, timeout=args.timeout)
    try:
        result = client.call(args.method, **params)
    except RpcError as exc:
        print(f"rpc error {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
