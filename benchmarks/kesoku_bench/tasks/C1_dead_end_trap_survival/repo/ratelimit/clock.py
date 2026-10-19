"""Clock abstractions for deterministic rate limiter testing."""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Protocol for monotonic or simulated time sources."""

    def now(self) -> float:
        """Return the current timestamp in seconds."""


class SystemClock:
    """Production clock backed by ``time.monotonic``."""

    def now(self) -> float:
        """Return the current monotonic timestamp in seconds."""
        return time.monotonic()


class ManualClock:
    """Deterministic manually controlled clock for unit and skew tests."""

    def __init__(self, initial: float = 0.0) -> None:
        """Initialize the manual clock with a starting timestamp."""
        self._now: float = float(initial)
        self.call_count: int = 0

    def now(self) -> float:
        """Return the current simulated timestamp and increment call_count."""
        self.call_count += 1
        return self._now

    def advance(self, delta: float) -> float:
        """Advance the simulated clock by ``delta`` seconds."""
        self._now += float(delta)
        return self._now

    def set(self, timestamp: float) -> None:
        """Set the simulated clock to an explicit timestamp (may step backwards)."""
        self._now = float(timestamp)
