# ratelimit

Deterministic, injectable-clock rate limiting library for Python services.

## Overview

`ratelimit` provides per-client token-bucket rate limiters and HTTP/RPC middleware helpers:

- `ratelimit.clock`: `Clock` protocol, `SystemClock`, and `ManualClock` for deterministic testing.
- `ratelimit.limiter`: `TokenBucketLimiter(rate, capacity, clock=None)` with `allow(client_id, cost=1.0)`, `remaining(client_id)`, and `reset(client_id=None)`.
- `ratelimit.middleware`: `RateLimitMiddleware` wrapping a limiter around request handlers, plus route-level configuration via `routes.toml`.
- `ratelimit.bench`: Synthetic throughput and latency benchmark runner (`uv run python -m ratelimit.bench`).

## Clock Contract

All limiters accept an optional `clock: Clock | None = None` (defaulting to `SystemClock()`).
To support deterministic testing and multi-tenant simulation, limiters must read `self.clock.now()` on every `allow()` invocation and clamp negative time deltas (`max(0.0, now - last_refill)`) when a non-monotonic clock steps backwards.

## Running Tests

```bash
uv run pytest
```
