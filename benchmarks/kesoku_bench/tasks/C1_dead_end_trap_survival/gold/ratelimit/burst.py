"""Per-client burst-capped token-bucket rate limiter."""

from __future__ import annotations

from ratelimit.clock import Clock, SystemClock
from ratelimit.limiter import BucketState


class BurstLimiter:
    """Token-bucket limiter with an optional per-request burst ceiling.

    Follows the same architecture and injectable ``Clock`` contract as
    ``TokenBucketLimiter``: ``self.clock.now()`` is queried on every ``allow()``
    call and client buckets are indexed in O(1) via a dictionary.
    """

    def __init__(
        self,
        rate: float,
        capacity: float,
        max_burst: float | Clock | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Initialize the burst limiter."""
        if isinstance(max_burst, Clock) and clock is None:
            clock = max_burst
            max_burst = None
        if rate <= 0:
            raise ValueError("rate must be positive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.rate: float = float(rate)
        self.capacity: float = float(capacity)
        self.max_burst: float = float(max_burst) if max_burst is not None else float(capacity)
        if self.max_burst <= 0:
            raise ValueError("max_burst must be positive")
        self.clock: Clock = clock if clock is not None else SystemClock()
        self._buckets: dict[str, BucketState] = {}

    def allow(self, client_id: str, cost: float = 1.0) -> bool:
        """Refill ``client_id``'s bucket using ``self.clock.now()`` and check burst/balance."""
        if cost < 0:
            raise ValueError("cost must be non-negative")
        now = self.clock.now()
        state = self._buckets.get(client_id)
        if state is None:
            state = BucketState(tokens=self.capacity, last_refill=now)
            self._buckets[client_id] = state
        else:
            elapsed = max(0.0, now - state.last_refill)
            state.tokens = min(self.capacity, state.tokens + elapsed * self.rate)
            state.last_refill = now

        if cost > self.max_burst:
            return False
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
        """Reset bucket state for ``client_id`` or all clients if ``None``."""
        if client_id is None:
            self._buckets.clear()
        else:
            self._buckets.pop(client_id, None)

    def active_clients(self) -> int:
        """Return the number of tracked client buckets."""
        return len(self._buckets)
