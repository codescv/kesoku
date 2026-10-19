# Subsystem Y (`svc.subsystem_y.webhook`) — Known Boundary Conditions

Public verification entry point:

```python
verify(
    sig: str | bytes,
    body: bytes | str,
    ts: int | float,
    *,
    tolerance: int = 300,
    secret: bytes | str = DEFAULT_SECRET,
    now: int | float | None = None,
) -> bool
```

## Known, Unfixed Boundary Conditions

1. **Inclusive tolerance edge off-by-one**:
   - `verify()` currently checks `if drift >= tolerance: return False`, which rejects requests lying on the exact boundary `abs(now - ts) == tolerance` (including `tolerance=0` when `ts == now`).
   - Expected contract: timestamp drift within `[0, tolerance]` inclusive (`drift <= tolerance`) must be accepted.
2. **Empty body rejected**:
   - `verify()` currently checks `if not body: return False`, which rejects valid ping/heartbeat webhooks with an empty payload (`b""` or `""`).
   - Expected contract: empty `bytes` (`b""`) or empty `str` (`""`) payloads must be accepted whenever their HMAC signature over `f"{int(ts)}."` is valid (only `body is None` or non-`bytes`/`str` should fail).
3. **Non-UTF8 `bytes` signature header raises `UnicodeDecodeError`**:
   - When `sig` is passed as raw `bytes` containing invalid UTF-8 sequences (e.g. `b"\xff\xfe..."`), `sig.decode("utf-8")` raises an uncaught `UnicodeDecodeError` (or `hmac.compare_digest` raises `TypeError` on non-ASCII characters).
   - Expected contract: malformed or non-UTF8/non-ASCII signature headers must cleanly return `False` without raising an exception, while preserving constant-time digest comparison via `hmac.compare_digest`.
