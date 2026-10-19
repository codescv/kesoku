# mockrpc

A tiny JSON-RPC 2.0 mock server and the HTTP client our services use to talk to it. The integration test rig
starts the server, points the client at it and checks a few round trips.

## Server

```bash
uv run python -m mockrpc.server --port 8765
```

- Loads its fixtures for about 12 seconds before it binds the port ("warming up"), so give it time.
- Listens on `127.0.0.1:<port>`; `POST /rpc` takes a JSON-RPC 2.0 request (`Content-Type: application/json`),
  `GET /health` returns `{"status": "ok"}`.
- Methods: `ping` -> `"pong"`, `echo(text)` -> `text`, `sum(values)` -> the sum of the list.
- The first 2 requests after binding are answered with `503 warming` while caches fill; clients are expected
  to retry 503 responses with a short backoff.
- Writes its own PID to `server.pid` and appends one JSON line per request to `requests.log` (both in the
  repository root). Stop it with SIGTERM/SIGINT; `server.pid` is left behind and overwritten on the next start.

## Client

```python
from mockrpc.client import RpcClient

client = RpcClient("http://127.0.0.1:8765")
client.call("echo", text="hi")
```

CLI: `uv run python -m mockrpc.client --url http://127.0.0.1:8765 echo '{"text": "hi"}'`

## Tests

```bash
uv run pytest                                                  # unit tests (integration tests skip)
MOCKRPC_URL=http://127.0.0.1:8765 uv run pytest tests/integration   # against a running server
```
