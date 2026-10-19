import asyncio

from dagrun import DAG, NO_RETRY, NodeState, Scheduler


def ok(value: object = None, delay: float = 0.0):
    """Return a task function that sleeps then returns ``value``."""

    async def fn() -> object:
        await asyncio.sleep(delay)
        return value

    return fn


def boom(delay: float = 0.0):
    """Return a task function that sleeps then raises."""

    async def fn() -> None:
        await asyncio.sleep(delay)
        raise RuntimeError("boom")

    return fn


def test_all_succeed():
    """A diamond without failures runs every node once."""
    dag = DAG()
    dag.add("A", ok(1))
    dag.add("B", ok(2), deps=["A"])
    dag.add("C", ok(3), deps=["A"])
    dag.add("D", ok(4), deps=["B", "C"])
    res = asyncio.run(Scheduler(dag, retry=NO_RETRY).run())
    assert res.ok
    assert [res.nodes[n].result for n in "ABCD"] == [1, 2, 3, 4]
    assert all(res.nodes[n].attempts == 1 for n in "ABCD")
    assert res.events.count("success") == 4


def test_leaf_failure():
    """A failing leaf does not affect its siblings."""
    dag = DAG()
    dag.add("A", ok())
    dag.add("B", boom(), deps=["A"])
    dag.add("C", ok(), deps=["A"])
    res = asyncio.run(Scheduler(dag, retry=NO_RETRY).run())
    assert res.state("A") is NodeState.SUCCESS
    assert res.state("B") is NodeState.FAILED
    assert isinstance(res.nodes["B"].error, RuntimeError)
    assert res.state("C") is NodeState.SUCCESS
    assert not res.ok


def test_chain_failure_skips_downstream():
    """A failure in the middle of a chain skips the rest of the chain."""
    dag = DAG()
    dag.add("A", ok())
    dag.add("B", boom(), deps=["A"])
    dag.add("C", ok(), deps=["B"])
    res = asyncio.run(Scheduler(dag, retry=NO_RETRY).run())
    assert res.state("B") is NodeState.FAILED
    assert res.state("C") is NodeState.SKIPPED
    assert res.nodes["C"].cause == "B"
    assert res.events.count("skipped", "C") == 1
    assert res.events.count("attempt", "C") == 0


def test_concurrency_limit():
    """No more than ``concurrency`` node functions run at once."""
    running = 0
    peak = 0

    async def work() -> None:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1

    dag = DAG()
    for i in range(8):
        dag.add(f"n{i}", work)
    res = asyncio.run(Scheduler(dag, concurrency=3, retry=NO_RETRY).run())
    assert res.ok
    assert peak == 3


def test_cancel_marks_unfinished_nodes():
    """Scheduler.cancel() stops the run and marks unfinished nodes CANCELLED."""
    dag = DAG()
    dag.add("fast", ok())
    dag.add("slow", ok(delay=5.0))
    dag.add("after", ok(), deps=["slow"])
    sched = Scheduler(dag, retry=NO_RETRY)

    async def main():
        task = asyncio.create_task(sched.run())
        await asyncio.sleep(0.05)
        sched.cancel()
        return await asyncio.wait_for(task, 1.0)

    res = asyncio.run(main())
    assert res.cancelled
    assert res.state("fast") is NodeState.SUCCESS
    assert res.state("slow") is NodeState.CANCELLED
    assert res.state("after") is NodeState.CANCELLED
