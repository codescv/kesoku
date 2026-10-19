"""Asyncio runner for a :class:`dagrun.graph.DAG`.

Execution model
---------------
* Every node gets its own worker task. A worker first waits for the
  ``asyncio.Event`` of each upstream node, then executes the node's function
  under a shared semaphore (``concurrency``), retrying failed attempts with the
  scheduler's :class:`dagrun.retry.RetryPolicy`.
* A node whose upstream nodes did not all succeed is never executed; it ends in
  state ``SKIPPED`` and records the failure that caused the skip (see
  :class:`NodeResult`).
* ``run()`` returns once every node is in a terminal state (or the run is
  cancelled with :meth:`Scheduler.cancel`).
"""

from __future__ import annotations

import asyncio
import enum
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .errors import SchedulerStateError, UpstreamFailed
from .events import EventLog, format_event
from .graph import DAG, Node
from .retry import RetryPolicy

SleepFn = Callable[[float], Awaitable[None]]


class NodeState(enum.Enum):
    """Lifecycle state of a node during a run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        """Return True for states a node never leaves."""
        return self not in (NodeState.PENDING, NodeState.RUNNING)


@dataclass
class NodeResult:
    """Outcome of one node after a run.

    Attributes:
        name: Node name.
        state: Final state.
        result: Return value of the node function (``SUCCESS`` only).
        error: Exception of the last attempt (``FAILED`` only).
        cause: For ``SKIPPED`` nodes, the name of the *failed* node responsible
            for the skip: among all failures upstream of the node (followed
            transitively through skipped upstream nodes) the one that failed
            first, by failure timestamp. ``None`` for nodes that ran.
        attempts: How many times the node function was called (0 for nodes
            that never ran).
        started_at: Clock reading of the first attempt, if any.
        finished_at: Clock reading when the node reached its final state.
    """

    name: str
    state: NodeState
    result: Any = None
    error: BaseException | None = None
    cause: str | None = None
    attempts: int = 0
    started_at: float | None = None
    finished_at: float | None = None


@dataclass
class RunResult:
    """Outcome of a whole run.

    Attributes:
        nodes: Per-node results keyed by name (topological order).
        events: The run's event log.
        duration_s: Wall time of the run in seconds.
        cancelled: True if the run was cancelled.
    """

    nodes: dict[str, NodeResult]
    events: EventLog
    duration_s: float
    cancelled: bool = False

    @property
    def ok(self) -> bool:
        """Return True if every node succeeded."""
        return all(r.state is NodeState.SUCCESS for r in self.nodes.values())

    def state(self, name: str) -> NodeState:
        """Return the final state of node ``name``."""
        return self.nodes[name].state

    def with_state(self, state: NodeState) -> list[str]:
        """Return the names of the nodes that ended in ``state``."""
        return [n for n, r in self.nodes.items() if r.state is state]

    def summary(self) -> str:
        """Return a one-line human readable summary."""
        parts = [f"{s.value}={len(self.with_state(s))}" for s in NodeState if self.with_state(s)]
        return f"{len(self.nodes)} nodes in {self.duration_s:.2f}s: " + ", ".join(parts)


@dataclass
class _NodeRun:
    """Mutable per-node bookkeeping used while a run is in progress."""

    name: str
    state: NodeState = NodeState.PENDING
    event: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None
    result: Any = None
    error: BaseException | None = None
    cause: str | None = None
    attempts: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    failed_at: float | None = None


class Scheduler:
    """Runs a DAG of async tasks with bounded concurrency and retries."""

    def __init__(
        self,
        dag: DAG,
        *,
        concurrency: int = 4,
        retry: RetryPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: SleepFn = asyncio.sleep,
        verbose: bool = False,
    ) -> None:
        """Configure a scheduler for one run of ``dag``.

        Args:
            dag: The graph to execute.
            concurrency: Maximum number of node functions running at once.
            retry: Retry policy for failed attempts (default: ``RetryPolicy()``).
            clock: Time source for timestamps.
            sleep: Coroutine used for backoff sleeps (injectable for tests).
            verbose: Print every event to stdout as it is emitted.
        """
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self._dag = dag
        self._concurrency = concurrency
        self._retry = retry if retry is not None else RetryPolicy()
        self._clock = clock
        self._sleep = sleep
        self._log = EventLog(clock)
        if verbose:
            origin = clock()
            self._log.subscribe(lambda e: print(format_event(e, origin), flush=True))
        self._runs: dict[str, _NodeRun] = {}
        self._sem: asyncio.Semaphore | None = None
        self._done_event: asyncio.Event | None = None
        self._remaining = 0
        self._started = False
        self._cancelled = False

    @property
    def events(self) -> EventLog:
        """The event log of this scheduler's run."""
        return self._log

    @property
    def retry_policy(self) -> RetryPolicy:
        """The retry policy applied to node attempts."""
        return self._retry

    # ------------------------------------------------------------------ public

    async def run(self) -> RunResult:
        """Execute the whole graph and return the per-node results.

        Raises:
            SchedulerStateError: If called more than once.
            CycleError: If the graph contains a cycle.
            UnknownNodeError: If a dependency does not exist.
        """
        if self._started:
            raise SchedulerStateError("a Scheduler can only run once")
        self._started = True
        order = self._dag.topo_order()
        self._runs = {name: _NodeRun(name) for name in order}
        self._sem = asyncio.Semaphore(self._concurrency)
        self._done_event = asyncio.Event()
        self._remaining = len(order)
        t0 = self._clock()
        self._log.emit("run_start", "*", nodes=len(order), concurrency=self._concurrency)
        if not order:
            self._done_event.set()
        for name in order:
            self._runs[name].task = asyncio.create_task(self._node_main(name), name=f"dagrun:{name}")
        try:
            await self._done_event.wait()
        finally:
            await self._reap_tasks()
        result = self._build_result(self._clock() - t0)
        self._log.emit("run_end", "*", ok=result.ok, cancelled=result.cancelled)
        return result

    def cancel(self) -> None:
        """Cancel a run in progress: unfinished nodes end in ``CANCELLED``."""
        if not self._started or self._cancelled:
            return
        self._cancelled = True
        for name, run in self._runs.items():
            if run.state.terminal:
                continue
            run.state = NodeState.CANCELLED
            run.finished_at = self._clock()
            self._log.emit("cancelled", name)
            run.event.set()
            if run.task is not None and not run.task.done():
                run.task.cancel()
        if self._done_event is not None:
            self._done_event.set()

    # ----------------------------------------------------------------- workers

    async def _node_main(self, name: str) -> None:
        """Worker coroutine for one node: wait for upstream nodes, then execute."""
        try:
            await self._wait_upstream(name)
            await self._run_with_retry(name)
        except asyncio.CancelledError:
            if self._runs[name].state is NodeState.CANCELLED:
                return  # cancelled through Scheduler.cancel(): already recorded
            raise

    async def _wait_upstream(self, name: str) -> None:
        """Block until every upstream node of ``name`` reached a terminal state."""
        for dep in self._dag.deps(name):
            await self._runs[dep].event.wait()

    def _check_upstream(self, name: str) -> None:
        """Raise UpstreamFailed if an upstream node of ``name`` did not succeed."""
        for dep in self._dag.deps(name):
            if self._runs[dep].state is not NodeState.SUCCESS:
                raise UpstreamFailed(name, dep)

    async def _run_with_retry(self, name: str) -> None:
        """Execute node ``name`` with the retry policy and record the outcome."""
        run = self._runs[name]
        node = self._dag.node(name)
        retries = 0
        while True:
            run.attempts += 1
            try:
                self._check_upstream(name)
                value = await self._attempt(node, run)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._retry.should_retry(exc, retries):
                    retries += 1
                    delay = self._retry.delay(retries)
                    self._log.emit("retry", name, attempt=run.attempts, delay=round(delay, 4), error=repr(exc))
                    await self._sleep(delay)
                    continue
                if isinstance(exc, UpstreamFailed):
                    st = run
                    if st.cause is None:
                        upstream = self._runs[exc.upstream]
                        st.cause = upstream.cause or exc.upstream
                    if st.state is NodeState.PENDING:
                        st.state = NodeState.SKIPPED
                        st.finished_at = self._clock()
                        self._log.emit("skipped", name, cause=st.cause)
                        st.event.set()
                        self._remaining -= 1
                        self._check_done()
                    return
                self._on_failure(name, exc)
                return
            self._on_success(name, value)
            return

    async def _attempt(self, node: Node, run: _NodeRun) -> Any:
        """Call the node function once, under the concurrency semaphore."""
        assert self._sem is not None
        async with self._sem:
            if run.started_at is None:
                run.started_at = self._clock()
            run.state = NodeState.RUNNING
            self._log.emit("attempt", node.name, attempt=run.attempts)
            if node.timeout_s is None:
                return await node.fn()
            async with asyncio.timeout(node.timeout_s):
                return await node.fn()

    # -------------------------------------------------------------- accounting

    def _on_success(self, name: str, value: Any) -> None:
        """Record a successful node and wake its downstream nodes."""
        run = self._runs[name]
        run.state = NodeState.SUCCESS
        run.result = value
        run.finished_at = self._clock()
        self._log.emit("success", name, attempts=run.attempts)
        run.event.set()
        self._remaining -= 1
        self._check_done()

    def _on_failure(self, failed: str, exc: BaseException) -> None:
        """Record a failed node (retries exhausted) and notify downstream nodes."""
        run = self._runs[failed]
        run.state = NodeState.FAILED
        run.error = exc
        run.finished_at = run.failed_at = self._clock()
        self._log.emit("failed", failed, attempts=run.attempts, error=repr(exc))
        run.event.set()
        self._remaining -= 1
        # Tell downstream nodes why they will not run. A join node whose every
        # upstream has now failed can never run either, so settle it right away
        # instead of waiting for its worker to notice.
        for name in self._dag.successors(failed):
            st = self._runs[name]
            st.cause = failed
            deps = self._dag.deps(name)
            if st.state is NodeState.PENDING and len(deps) > 1:
                if all(self._runs[d].state is NodeState.FAILED for d in deps):
                    if st.state is NodeState.PENDING:
                        st.state = NodeState.SKIPPED
                        st.finished_at = self._clock()
                        self._log.emit("skipped", name, cause=st.cause)
                        st.event.set()
        self._check_done()

    def _check_done(self) -> None:
        """Wake ``run()`` once every node has been accounted for."""
        if self._remaining == 0 and self._done_event is not None:
            self._done_event.set()

    async def _reap_tasks(self) -> None:
        """Cancel and await any worker task that is still alive."""
        tasks = [r.task for r in self._runs.values() if r.task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _build_result(self, duration_s: float) -> RunResult:
        """Freeze the per-node bookkeeping into a RunResult."""
        nodes = {
            name: NodeResult(
                name=name,
                state=run.state,
                result=run.result,
                error=run.error,
                cause=run.cause if run.state is NodeState.SKIPPED else None,
                attempts=run.attempts,
                started_at=run.started_at,
                finished_at=run.finished_at,
            )
            for name, run in self._runs.items()
        }
        return RunResult(nodes=nodes, events=self._log, duration_s=duration_s, cancelled=self._cancelled)
