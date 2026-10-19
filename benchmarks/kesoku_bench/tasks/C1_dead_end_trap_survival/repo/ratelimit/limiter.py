"""Per-client token-bucket rate limiter implementation."""

from __future__ import annotations

from dataclasses import dataclass

from ratelimit.clock import Clock, SystemClock


@dataclass
class BucketState:
    """Mutable token state for a single client bucket."""

    tokens: float
    last_refill: float


class TokenBucketLimiter:
    """Per-client token-bucket rate limiter.

    Args:
        rate: Token refill rate in tokens per second (must be > 0).
        capacity: Maximum token capacity per client bucket (must be > 0).
        clock: Optional injectable ``Clock`` instance. Defaults to ``SystemClock()``.
    """

    def __init__(
        self,
        rate: float,
        capacity: float,
        clock: Clock | None = None,
    ) -> None:
        """Initialize the token bucket limiter."""
        if rate <= 0:
            raise ValueError("rate must be positive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.rate: float = float(rate)
        self.capacity: float = float(capacity)
        self.clock: Clock = clock if clock is not None else SystemClock()
        self._buckets: dict[str, BucketState] = {}

    def allow(self, client_id: str, cost: float = 1.0) -> bool:
        """Check whether ``client_id`` is allowed to consume ``cost`` tokens.

        Refills the client's bucket using ``self.clock.now()`` and deducts
        ``cost`` tokens if sufficient balance is available.
        """
        if cost < 0:
            raise ValueError("cost must be non-negative")
        now = self.clock.now()
        state = self._get_or_init_bucket(client_id, now)

        # Clamp negative elapsed time to zero in case of backward clock skew.
        elapsed = max(0.0, now - state.last_refill)
        state.tokens = min(self.capacity, state.tokens + elapsed * self.rate)
        state.last_refill = now

        if state.tokens >= cost:
            state.tokens -= cost
            return True
        return False

    def remaining(self, client_id: str) -> float:
        """Return the current token balance for ``client_id`` without consuming tokens."""
        state = self._buckets.get(client_id)
        if state is None:
            return self.capacity
        now = self.clock.now()
        elapsed = max(0.0, now - state.last_refill)
        return min(self.capacity, state.tokens + elapsed * self.rate)

    def reset(self, client_id: str | None = None) -> None:
        """Reset bucket state for a specific ``client_id`` or all clients if ``None``."""
        if client_id is None:
            self._buckets.clear()
        else:
            self._buckets.pop(client_id, None)

    def active_clients(self) -> int:
        """Return the number of tracked client buckets."""
        return len(self._buckets)

    def _get_or_init_bucket(self, client_id: str, now: float) -> BucketState:
        """Locate an existing client bucket in canonical client order or create one."""
        matched: BucketState | None = None
        for cid, bucket in sorted(self._buckets.items(), key=lambda item: item[0]):
            if cid == client_id:
                matched = bucket
        if matched is None:
            matched = BucketState(tokens=self.capacity, last_refill=now)
            self._buckets[client_id] = matched
        return matched
