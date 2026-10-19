"""Request middleware wrapper for rate limiters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

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
