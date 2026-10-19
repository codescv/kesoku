"""Clock injection and clock-skew tests for TokenBucketLimiter."""

from __future__ import annotations

import pytest
from ratelimit.clock import ManualClock
from ratelimit.limiter import TokenBucketLimiter


def test_clock_called_once_per_allow() -> None:
    """Every allow() invocation must query self.clock.now() fresh (no cached timestamp)."""
    clock = ManualClock(1000.0)
    limiter = TokenBucketLimiter(rate=5.0, capacity=10.0, clock=clock)

    assert clock.call_count == 0
    for step in range(1, 9):
        clock.advance(0.05)
        limiter.allow("tenant-alpha", cost=1.0)
        assert clock.call_count == step, (
            f"Expected clock.now() to be called on every allow() (step={step}, "
            f"actual={clock.call_count})"
        )


def test_two_limiters_independent_injected_clocks() -> None:
    """Two limiters with distinct injected clocks must maintain independent timelines."""
    clock_fast = ManualClock(500.0)
    clock_frozen = ManualClock(500.0)

    limiter_fast = TokenBucketLimiter(rate=10.0, capacity=2.0, clock=clock_fast)
    limiter_frozen = TokenBucketLimiter(rate=10.0, capacity=2.0, clock=clock_frozen)

    assert limiter_fast.allow("u1", cost=2.0) is True
    assert limiter_frozen.allow("u1", cost=2.0) is True
    assert limiter_fast.allow("u1", cost=1.0) is False
    assert limiter_frozen.allow("u1", cost=1.0) is False

    clock_fast.advance(0.25)
    assert limiter_fast.allow("u1", cost=2.0) is True
    assert limiter_frozen.allow("u1", cost=1.0) is False


def test_clock_stepping_backwards_does_not_grant_tokens() -> None:
    """A clock that steps backwards must not grant extra tokens or corrupt future refills."""
    clock = ManualClock(200.0)
    limiter = TokenBucketLimiter(rate=10.0, capacity=5.0, clock=clock)

    assert limiter.allow("client-1", cost=4.0) is True
    assert limiter.remaining("client-1") == pytest.approx(1.0)

    # Step clock backwards by 15 seconds.
    clock.set(185.0)
    assert limiter.allow("client-1", cost=2.0) is False
    assert limiter.allow("client-1", cost=1.0) is True
    assert limiter.allow("client-1", cost=0.1) is False

    # Advancing forward from 185.0 by 0.5s should refill 5.0 tokens.
    clock.advance(0.5)
    assert limiter.allow("client-1", cost=5.0) is True
