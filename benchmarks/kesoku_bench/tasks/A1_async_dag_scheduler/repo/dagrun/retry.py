"""Retry/backoff policy for node execution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff retry policy.

    A node attempt that raises an exception matching ``retry_on`` is retried up
    to ``max_retries`` times; before retry number ``n`` (1-based) the scheduler
    sleeps ``delay(n)`` seconds. ``max_retries=0`` disables retries.

    Attributes:
        max_retries: Number of retries after the first attempt.
        base_delay: Delay before the first retry, in seconds.
        factor: Multiplier applied for each further retry.
        max_delay: Upper bound for a single delay.
        retry_on: Exception types that are retried.
    """

    max_retries: int = 3
    base_delay: float = 0.05
    factor: float = 2.0
    max_delay: float = 2.0
    retry_on: tuple[type[BaseException], ...] = (Exception,)

    def __post_init__(self) -> None:
        """Validate the policy parameters."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("delays must be >= 0")
        if self.factor < 1.0:
            raise ValueError("factor must be >= 1.0")

    def delay(self, retry_number: int) -> float:
        """Return the sleep before retry ``retry_number`` (1-based)."""
        if retry_number < 1:
            raise ValueError("retry_number is 1-based")
        return min(self.max_delay, self.base_delay * self.factor ** (retry_number - 1))

    def should_retry(self, exc: BaseException, retries_done: int) -> bool:
        """Return True if an attempt that raised ``exc`` should be retried.

        Args:
            exc: The exception raised by the attempt.
            retries_done: How many retries have already been performed.
        """
        return retries_done < self.max_retries and isinstance(exc, self.retry_on)

    def total_backoff(self) -> float:
        """Return the cumulative sleep time if every retry is used."""
        return sum(self.delay(n) for n in range(1, self.max_retries + 1))


NO_RETRY = RetryPolicy(max_retries=0, base_delay=0.0)
