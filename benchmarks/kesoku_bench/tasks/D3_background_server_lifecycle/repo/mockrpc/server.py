"""JSON-RPC 2.0 mock server (stdlib ``http.server``).

Run with ``python -m mockrpc.server --port N``. The server warms up for :data:`WARMUP_SECONDS` before it binds,
answers ``503 warming`` to the first :data:`WARMING_REQUESTS` requests after binding, writes its own PID to
``server.pid`` and appends one JSON line per request to ``requests.log`` (both in the repository root).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import FrameType
from typing import Any

WARMUP_SECONDS = 12.0
WARMING_REQUESTS = 2
REPO_ROOT = Path(__file__).resolve().parent.parent
PID_FILE = "server.pid"
REQUEST_LOG = "requests.log"


class RpcHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying the mock's request counter and log location."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], state_dir: Path, warming_requests: int) -> None:
        """Bind the server.

        Args:
            address: ``(host, port)``; port 0 picks a free port.
            state_dir: Directory receiving ``requests.log``.
            warming_requests: Number of initial requests answered with 503.
        """
        super().__init__(address, RpcHandler)
        self.state_dir = Path(state_dir)
        self.warming_requests = warming_requests
        self.request_count = 0
        self._lock = threading.Lock()

    def next_request_number(self) -> int:
        """Atomically count a request and return its 1-based number."""
        with self._lock:
            self.request_count += 1
            return self.request_count

    def log_request_line(self, entry: dict[str, Any]) -> None:
        """Append one JSON line to ``requests.log``."""
        line = json.dumps(entry, sort_keys=True)
        with self._lock, open(self.state_dir / REQUEST_LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _dispatch(method: str, params: Any) -> Any:
    """Evaluate one RPC method; raises LookupError for unknown methods and ValueError for bad params."""
    if method == "ping":
        return "pong"
    if method == "echo":
        if not isinstance(params, dict) or not isinstance(params.get("text"), str):
            raise ValueError("echo needs params.text (string)")
        return params["text"]
    if method == "sum":
        values = params.get("values") if isinstance(params, dict) else None
        if not isinstance(values, list) or not all(isinstance(v, int | float) for v in values):
            raise ValueError("sum needs params.values (list of numbers)")
        return sum(values)
    raise LookupError(method)


class RpcHandler(BaseHTTPRequestHandler):
    """Request handler for ``/rpc`` and ``/health``."""

    server: RpcHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        """Silence the default stderr access log (requests.log is the record)."""

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        """Send a JSON response."""
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _finish(self, n: int, status: int, payload: dict[str, Any], rpc_method: str | None) -> None:
        """Record the request, then send the response (log first so clients never observe a missing line)."""
        self.server.log_request_line(
            {
                "n": n,
                "ts": round(time.time(), 3),
                "http": f"{self.command} {self.path}",
                "content_type": self.headers.get("Content-Type"),
                "rpc": rpc_method,
                "status": status,
            }
        )
        self._send_json(status, payload)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        """Handle ``GET /health``."""
        n = self.server.next_request_number()
        if n <= self.server.warming_requests:
            self._finish(n, 503, {"error": "warming"}, None)
        elif self.path == "/health":
            self._finish(n, 200, {"status": "ok"}, None)
        else:
            self._finish(n, 404, {"error": "not found"}, None)

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        """Handle ``POST /rpc`` (JSON-RPC 2.0)."""
        n = self.server.next_request_number()
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        rpc_method: str | None = None
        try:
            request = json.loads(raw.decode("utf-8")) if raw else None
            if isinstance(request, dict) and isinstance(request.get("method"), str):
                rpc_method = request["method"]
        except (UnicodeDecodeError, json.JSONDecodeError):
            request = None
        if n <= self.server.warming_requests:
            self._finish(n, 503, {"error": "warming"}, rpc_method)
            return
        if self.path != "/rpc":
            self._finish(n, 404, {"error": "not found"}, rpc_method)
            return
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if content_type != "application/json":
            self._finish(n, 415, {"error": f"unsupported content type {content_type!r}"}, rpc_method)
            return
        if not isinstance(request, dict) or rpc_method is None:
            self._finish(
                n, 400, {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}, rpc_method
            )
            return
        req_id = request.get("id")
        try:
            result = _dispatch(rpc_method, request.get("params", {}))
        except LookupError:
            error = {"code": -32601, "message": f"method not found: {rpc_method}"}
            self._finish(n, 200, {"jsonrpc": "2.0", "id": req_id, "error": error}, rpc_method)
            return
        except ValueError as exc:
            error = {"code": -32602, "message": str(exc)}
            self._finish(n, 200, {"jsonrpc": "2.0", "id": req_id, "error": error}, rpc_method)
            return
        self._finish(n, 200, {"jsonrpc": "2.0", "id": req_id, "result": result}, rpc_method)


def build_server(
    port: int, host: str = "127.0.0.1", state_dir: Path | None = None, warming_requests: int = WARMING_REQUESTS
) -> RpcHTTPServer:
    """Create (and bind) a server without warming up or writing a PID file.

    Args:
        port: TCP port (0 picks a free one).
        host: Interface to bind.
        state_dir: Directory for ``requests.log`` (defaults to the repository root).
        warming_requests: Number of initial requests answered with 503.

    Returns:
        The bound, not yet serving, server.
    """
    return RpcHTTPServer((host, port), state_dir or REPO_ROOT, warming_requests)


def _stop(signum: int, frame: FrameType | None) -> None:
    """Signal handler: unwind ``serve_forever`` via KeyboardInterrupt."""
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    """Warm up, bind, write ``server.pid`` and serve until SIGTERM/SIGINT.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(prog="mockrpc.server", description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, required=True, help="TCP port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default 127.0.0.1)")
    args = parser.parse_args(argv)

    signal.signal(signal.SIGTERM, _stop)
    print(f"mockrpc: warming up fixtures (~{WARMUP_SECONDS:.0f}s) ...", flush=True)
    try:
        time.sleep(WARMUP_SECONDS)
        server = build_server(args.port, host=args.host)
    except KeyboardInterrupt:
        print("mockrpc: interrupted during warm-up", flush=True)
        return 0
    (REPO_ROOT / PID_FILE).write_text(f"{os.getpid()}\n", encoding="utf-8")
    print(f"mockrpc: listening on http://{args.host}:{args.port} (pid {os.getpid()})", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("mockrpc: stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
