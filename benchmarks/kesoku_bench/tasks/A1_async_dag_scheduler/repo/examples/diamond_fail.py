"""Diamond pipeline where both branches fail at the same time.

    A --> B --> D
    |           ^
    +---> C ----+

Run with ``uv run python examples/diamond_fail.py`` (or
``uv run python -m dagrun.cli examples/diamond_fail.py``).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dagrun import DAG, RetryPolicy, Scheduler  # noqa: E402


async def fetch() -> str:
    """Pretend to download the input."""
    await asyncio.sleep(0.02)
    return "payload"


async def transform_left() -> str:
    """Left branch: the upstream service is down, always fails."""
    await asyncio.sleep(0.05)
    raise ConnectionError("left service unavailable")


async def transform_right() -> str:
    """Right branch: same service, fails at the same moment."""
    await asyncio.sleep(0.05)
    raise ConnectionError("right service unavailable")


async def publish() -> str:
    """Join node: needs both branches."""
    await asyncio.sleep(0.01)
    return "published"


def build() -> Scheduler:
    """Return a scheduler for the diamond pipeline."""
    dag = DAG()
    dag.add("A", fetch)
    dag.add("B", transform_left, deps=["A"])
    dag.add("C", transform_right, deps=["A"])
    dag.add("D", publish, deps=["B", "C"])
    return Scheduler(dag, concurrency=4, retry=RetryPolicy(max_retries=3, base_delay=0.05, factor=2.0), verbose=True)


async def main() -> None:
    """Run the pipeline and print the outcome."""
    result = await build().run()
    print(result.summary())
    for name, node in result.nodes.items():
        print(f"  {name}: {node.state.value} cause={node.cause}")


if __name__ == "__main__":
    asyncio.run(main())
