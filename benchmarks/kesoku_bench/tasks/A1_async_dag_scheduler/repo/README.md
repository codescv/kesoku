# dagrun

`dagrun` runs a directed acyclic graph of async Python tasks with:

- deterministic topological ordering and cycle detection (`dagrun/graph.py`);
- bounded concurrency (a shared semaphore) and one worker task per node that
  waits on the `asyncio.Event` of each upstream node (`dagrun/scheduler.py`);
- retries with exponential backoff (`dagrun/retry.py`);
- a timestamped event log of everything that happened (`dagrun/events.py`);
- cancellation of a run in progress (`Scheduler.cancel()`).

Pure standard library, Python >= 3.12.

## Usage

```python
import asyncio
from dagrun import DAG, RetryPolicy, Scheduler

async def extract(): ...
async def load(): ...

dag = DAG()
dag.add("extract", extract)
dag.add("load", load, deps=["extract"])
result = asyncio.run(Scheduler(dag, concurrency=4, retry=RetryPolicy(max_retries=3)).run())
print(result.summary())
print(result.events.format())
```

### Semantics

- A node runs only if **all** of its upstream nodes succeeded.
- A failing attempt is retried up to `RetryPolicy.max_retries` times with
  `base_delay * factor ** (n - 1)` seconds of backoff before retry `n`.
  When retries are exhausted the node is `FAILED`.
- A node that cannot run because an upstream node failed (or was itself
  skipped) is `SKIPPED`; it is never executed and never retried, and its
  `cause` is the upstream failure that happened **first** (by failure
  timestamp), followed transitively through skipped nodes.
- `Scheduler.run()` returns as soon as every node is in a terminal state
  (`SUCCESS`, `FAILED`, `SKIPPED`, `CANCELLED`).

Event kinds: `run_start`, `attempt`, `retry`, `success`, `failed`, `skipped`,
`cancelled`, `run_end` (see `dagrun/events.py`).

## CLI

```bash
uv run python -m dagrun.cli examples/diamond_fail.py --timeout 10
```

A pipeline file defines `build() -> Scheduler`; see `examples/`.

## Development

```bash
uv run pytest
```
