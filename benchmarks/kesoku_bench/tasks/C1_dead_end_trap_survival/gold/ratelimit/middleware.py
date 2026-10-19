"""Request middleware wrapper and TOML-configured route rate limiter."""

from __future__ import annotations

import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from ratelimit.clock import Clock, SystemClock
from ratelimit.limiter import TokenBucketLimiter

T = TypeVar("T")


@dataclass(frozen=True)
class RateLimitDecision:
    """Outcome of a middleware rate limit check."""

    allowed: bool
    client_id: str
    remaining: float


class RateLimitMiddleware:
    """Wrap a ``TokenBucketLimiter`` to gate request handler execution."""

    def __init__(self, limiter: TokenBucketLimiter) -> None:
        """Initialize middleware with a backing limiter."""
        self.limiter = limiter

    def check(self, client_id: str, cost: float = 1.0) -> RateLimitDecision:
        """Evaluate rate limit for ``client_id`` and return a decision record."""
        allowed = self.limiter.allow(client_id, cost=cost)
        rem = self.limiter.remaining(client_id)
        return RateLimitDecision(allowed=allowed, client_id=client_id, remaining=rem)

    def dispatch(
        self,
        client_id: str,
        handler: Callable[[], T],
        cost: float = 1.0,
    ) -> tuple[int, T | str]:
        """Execute ``handler`` if allowed (200), else return 429 Too Many Requests."""
        if not self.limiter.allow(client_id, cost=cost):
            return 429, "rate limit exceeded"
        return 200, handler()


class RouteLimiter:
    """Per-route rate limiter backed by ``TokenBucketLimiter`` instances."""

    def __init__(
        self,
        default_rate: float = 10.0,
        default_capacity: float = 20.0,
        routes: Mapping[str, tuple[float, float]] | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Initialize route limiter with default and route-specific token buckets."""
        self.clock: Clock = clock if clock is not None else SystemClock()
        self.default_rate = float(default_rate)
        self.default_capacity = float(default_capacity)
        self._default_limiter = TokenBucketLimiter(
            rate=self.default_rate,
            capacity=self.default_capacity,
            clock=self.clock,
        )
        self._route_limiters: dict[str, TokenBucketLimiter] = {}
        if routes:
            for route, (rate, capacity) in routes.items():
                self._route_limiters[route] = TokenBucketLimiter(
                    rate=float(rate),
                    capacity=float(capacity),
                    clock=self.clock,
                )

    @classmethod
    def from_toml(
        cls,
        path: str | Path = "routes.toml",
        clock: Clock | None = None,
    ) -> RouteLimiter:
        """Load route rate limits from a TOML configuration file."""
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        default_cfg = raw.get("default", {})
        default_rate = float(default_cfg.get("rate", 10.0))
        default_capacity = float(default_cfg.get("capacity", 20.0))
        routes_cfg = raw.get("routes", {})
        routes: dict[str, tuple[float, float]] = {}
        for route_name, spec in routes_cfg.items():
            routes[route_name] = (
                float(spec["rate"]),
                float(spec["capacity"]),
            )
        return cls(
            default_rate=default_rate,
            default_capacity=default_capacity,
            routes=routes,
            clock=clock,
        )

    def limiter_for(self, route: str) -> TokenBucketLimiter:
        """Return the ``TokenBucketLimiter`` assigned to ``route``."""
        return self._route_limiters.get(route, self._default_limiter)

    def allow(self, route: str, client_id: str, cost: float = 1.0) -> bool:
        """Check whether ``client_id`` is allowed on ``route``."""
        return self.limiter_for(route).allow(client_id, cost=cost)
