"""Held-out F2P verification suite for C1_dead_end_trap_survival."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from ratelimit.clock import ManualClock
from ratelimit.limiter import TokenBucketLimiter


def test_burst_respects_injected_clock_per_call() -> None:
    """BurstLimiter must query its injected clock on every allow() call and handle skew."""
    from ratelimit.burst import BurstLimiter

    clock = ManualClock(500.0)
    limiter = BurstLimiter(rate=10.0, capacity=20.0, clock=clock)

    assert clock.call_count == 0
    for i in range(1, 16):
        clock.advance(0.05)
        assert limiter.allow("client-alpha", cost=1.0) is True
        assert clock.call_count == i, (
            f"BurstLimiter cached clock.now() instead of calling it per allow() "
            f"(expected {i} calls, got {clock.call_count})"
        )

    # Two BurstLimiter instances with separate clocks must not share time state.
    clock_a = ManualClock(100.0)
    clock_b = ManualClock(100.0)
    lim_a = BurstLimiter(rate=5.0, capacity=2.0, clock=clock_a)
    lim_b = BurstLimiter(rate=5.0, capacity=2.0, clock=clock_b)

    assert lim_a.allow("u1", cost=2.0) is True
    assert lim_b.allow("u1", cost=2.0) is True
    assert lim_a.allow("u1", cost=1.0) is False
    assert lim_b.allow("u1", cost=1.0) is False

    clock_a.advance(0.5)
    assert lim_a.allow("u1", cost=2.0) is True
    assert lim_b.allow("u1", cost=1.0) is False

    # Backward clock jump must not grant extra tokens.
    clock_a.set(50.0)
    assert lim_a.allow("u1", cost=1.0) is False
    clock_a.advance(0.25)
    assert lim_a.allow("u1", cost=1.0) is True


def test_burst_perf_10k_clients() -> None:
    """BurstLimiter must remain O(1) per allow() across 10,000 active clients."""
    from ratelimit.burst import BurstLimiter

    clock = ManualClock(1000.0)
    limiter = BurstLimiter(rate=50.0, capacity=100.0, clock=clock)

    t0 = time.perf_counter()
    for idx in range(10_000):
        assert limiter.allow(f"c-{idx}", cost=1.0) is True
        if idx % 250 == 0 and (time.perf_counter() - t0) > 0.5:
            pytest.fail("BurstLimiter took >0.5s while populating 10k clients (non-O(1) lookup)")

    for step in range(2_000):
        clock.advance(0.001)
        assert limiter.allow(f"c-{step * 5}", cost=1.0) is True

    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"BurstLimiter 10k-client benchmark too slow: {elapsed:.3f}s"


def test_burst_basic_allowance_and_refill() -> None:
    """BurstLimiter consumes tokens, refills at rate, isolates clients, and supports reset."""
    from ratelimit.burst import BurstLimiter

    clock = ManualClock(0.0)
    limiter = BurstLimiter(rate=4.0, capacity=8.0, clock=clock)

    assert limiter.allow("c1", cost=5.0) is True
    assert limiter.remaining("c1") == pytest.approx(3.0)
    assert limiter.allow("c1", cost=4.0) is False
    assert limiter.allow("c2", cost=8.0) is True

    clock.advance(1.0)
    assert limiter.remaining("c1") == pytest.approx(7.0)
    assert limiter.allow("c1", cost=6.0) is True
    assert limiter.remaining("c1") == pytest.approx(1.0)

    limiter.reset("c1")
    assert limiter.remaining("c1") == pytest.approx(8.0)


def test_burst_window_spike_cap() -> None:
    """BurstLimiter enforces max_burst per single request without deducting tokens on rejection."""
    from ratelimit.burst import BurstLimiter

    clock = ManualClock(10.0)
    limiter = BurstLimiter(rate=10.0, capacity=20.0, max_burst=5.0, clock=clock)

    # Single request exceeding max_burst must be rejected without draining tokens.
    assert limiter.allow("spike-user", cost=6.0) is False
    assert limiter.remaining("spike-user") == pytest.approx(20.0)

    # Four requests of cost 5.0 (== max_burst) succeed until capacity (20.0) is exhausted.
    for _ in range(4):
        assert limiter.allow("spike-user", cost=5.0) is True
    assert limiter.remaining("spike-user") == pytest.approx(0.0)
    assert limiter.allow("spike-user", cost=1.0) is False


def test_route_limiter_from_toml() -> None:
    """RouteLimiter.from_toml configures per-route limits and falls back to [default]."""
    from ratelimit.middleware import RouteLimiter

    clock = ManualClock(0.0)
    routes_path = Path(__file__).resolve().parents[1] / "routes.toml"
    rl = RouteLimiter.from_toml(routes_path, clock=clock)

    # /api/v1/login has capacity=5.0, rate=2.0
    assert rl.allow("/api/v1/login", "ip-1", cost=5.0) is True
    assert rl.allow("/api/v1/login", "ip-1", cost=1.0) is False
    clock.advance(0.5)
    assert rl.allow("/api/v1/login", "ip-1", cost=1.0) is True

    # Unlisted route uses [default] capacity=20.0, rate=10.0
    assert rl.allow("/api/v1/other", "ip-1", cost=20.0) is True
    assert rl.allow("/api/v1/other", "ip-1", cost=1.0) is False


def test_token_bucket_perf() -> None:
    """TokenBucketLimiter must complete 10k-client population + 2k checks under 0.5s."""
    clock = ManualClock(0.0)
    limiter = TokenBucketLimiter(rate=50.0, capacity=100.0, clock=clock)

    t0 = time.perf_counter()
    for idx in range(10_000):
        assert limiter.allow(f"client-{idx}", cost=1.0) is True
        if idx % 250 == 0 and (time.perf_counter() - t0) > 0.5:
            pytest.fail(
                "TokenBucketLimiter took >0.5s while populating 10k clients (O(N log N) bottleneck still present)"
            )

    for step in range(2_000):
        clock.advance(0.001)
        assert limiter.allow(f"client-{step * 5}", cost=1.0) is True

    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"TokenBucketLimiter 10k-client benchmark too slow: {elapsed:.3f}s"
    assert clock.call_count == 12_000
