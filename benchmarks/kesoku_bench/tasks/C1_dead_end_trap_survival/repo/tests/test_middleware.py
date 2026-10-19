"""Unit tests for RateLimitMiddleware."""

from __future__ import annotations

from ratelimit.clock import ManualClock
from ratelimit.limiter import TokenBucketLimiter
from ratelimit.middleware import RateLimitMiddleware


def test_middleware_dispatch_and_check() -> None:
    """Middleware returns 200 when tokens are available and 429 when exhausted."""
    clock = ManualClock(50.0)
    limiter = TokenBucketLimiter(rate=1.0, capacity=2.0, clock=clock)
    mw = RateLimitMiddleware(limiter)

    status, body = mw.dispatch("ip-1", lambda: "ok", cost=1.0)
    assert status == 200
    assert body == "ok"

    decision = mw.check("ip-1", cost=1.0)
    assert decision.allowed is True
    assert decision.remaining == 0.0

    status_denied, body_denied = mw.dispatch("ip-1", lambda: "never", cost=1.0)
    assert status_denied == 429
    assert body_denied == "rate limit exceeded"
