"""Synthetic benchmark runner for rate limiter throughput and latency."""

from __future__ import annotations

import random
import time

from ratelimit.clock import ManualClock
from ratelimit.limiter import TokenBucketLimiter


def run_benchmark(num_clients: int = 2000, operations: int = 1500) -> str:
    """Run synthetic workload across simulated endpoints and format a report."""
    rng = random.Random(42)
    clock = ManualClock(1_000_000.0)
    limiter = TokenBucketLimiter(rate=25.0, capacity=50.0, clock=clock)

    for idx in range(num_clients):
        limiter.allow(f"client-{idx:05d}", cost=1.0)

    lines: list[str] = [
        "========================================================================",
        "RATELIMIT SYNTHETIC BENCHMARK SUITE v1.2",
        f"Pre-populated active clients: {num_clients} | Timed ops: {operations}",
        "========================================================================",
    ]

    t0 = time.perf_counter()
    allowed_count = 0
    denied_count = 0
    for step in range(operations):
        cid = f"client-{rng.randrange(num_clients):05d}"
        clock.advance(0.002)
        if limiter.allow(cid, cost=1.0):
            allowed_count += 1
        else:
            denied_count += 1
        if (step + 1) % 25 == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            qps = (step + 1) / max(1e-6, elapsed_ms / 1000.0)
            lines.append(
                f"[window {step + 1:04d}/{operations:04d}] "
                f"client={cid} allowed={allowed_count} denied={denied_count} "
                f"elapsed_ms={elapsed_ms:8.2f} throughput_ops_s={qps:10.1f} "
                f"buckets={limiter.active_clients()}"
            )

    total_ms = (time.perf_counter() - t0) * 1000.0
    lines.append("------------------------------------------------------------------------")
    lines.append(
        f"SUMMARY: total_ops={operations} allowed={allowed_count} denied={denied_count} "
        f"total_ms={total_ms:.2f} clock_calls={clock.call_count}"
    )
    return "\n".join(lines)


def main() -> None:
    """CLI entry point for ``python -m ratelimit.bench``."""
    print(run_benchmark())


if __name__ == "__main__":
    main()
