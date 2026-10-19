"""Unit tests for TokenBucketLimiter."""

from __future__ import annotations

import pytest
from ratelimit.clock import ManualClock
from ratelimit.limiter import TokenBucketLimiter


def test_basic_consume_and_refill() -> None:
    """Tokens are consumed on allow() and replenished linearly over elapsed time."""
    clock = ManualClock(0.0)
    limiter = TokenBucketLimiter(rate=2.0, capacity=4.0, clock=clock)

    assert limiter.allow("alice", cost=3.0) is True
    assert limiter.allow("alice", cost=2.0) is False
    assert limiter.remaining("alice") == pytest.approx(1.0)

    clock.advance(1.0)
    assert limiter.remaining("alice") == pytest.approx(3.0)
    assert limiter.allow("alice", cost=2.5) is True
    assert limiter.remaining("alice") == pytest.approx(0.5)


def test_capacity_cap_and_client_isolation() -> None:
    """Token balance never exceeds capacity and clients do not share buckets."""
    clock = ManualClock(10.0)
    limiter = TokenBucketLimiter(rate=10.0, capacity=5.0, clock=clock)

    assert limiter.allow("alice", cost=5.0) is True
    assert limiter.allow("alice", cost=1.0) is False
    assert limiter.allow("bob", cost=5.0) is True

    clock.advance(100.0)
    assert limiter.remaining("alice") == pytest.approx(5.0)


def test_reset_and_validation() -> None:
    """Reset clears single or all clients and invalid arguments raise ValueError."""
    clock = ManualClock(0.0)
    limiter = TokenBucketLimiter(rate=1.0, capacity=2.0, clock=clock)

    assert limiter.allow("c1", cost=2.0) is True
    assert limiter.allow("c1", cost=1.0) is False
    limiter.reset("c1")
    assert limiter.allow("c1", cost=2.0) is True

    limiter.reset()
    assert limiter.active_clients() == 0

    with pytest.raises(ValueError):
        TokenBucketLimiter(rate=0.0, capacity=5.0)
    with pytest.raises(ValueError):
        TokenBucketLimiter(rate=5.0, capacity=-1.0)
    with pytest.raises(ValueError):
        limiter.allow("c1", cost=-0.5)
