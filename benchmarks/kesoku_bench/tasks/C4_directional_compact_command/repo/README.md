# svc

Monorepo containing two independent backend subsystems:

- **Subsystem X (`svc.subsystem_x`)**: Byte-budgeted thumbnail LRU cache (`ThumbnailLRUCache`) and image geometry/aspect-ratio calculator (`resizer.py`).
- **Subsystem Y (`svc.subsystem_y`)**: Webhook HMAC-SHA256 signature verification (`verify(sig, body, ts, *, tolerance=300)` in `webhook.py`; see `svc/subsystem_y/BOUNDARIES.md` for known edge cases).

## Development

```bash
uv run pytest
uv run python -m svc.lint
```
