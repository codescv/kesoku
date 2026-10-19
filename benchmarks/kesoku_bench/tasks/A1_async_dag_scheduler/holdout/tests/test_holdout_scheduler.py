"""Hidden tests for A1_async_dag_scheduler (copied into repo/tests/ at grading time)."""

import asyncio

from dagrun import DAG, NodeState, RetryPolicy, Scheduler

FAST = RetryPolicy(max_retries=3, base_delay=0.001, factor=1.0)
NONE = RetryPolicy(max_retries=0, base_delay=0.0)


def ok(value: object = None, delay: float = 0.0):
    """Return a task function that sleeps then returns ``value``."""

    async def fn() -> object:
        await asyncio.sleep(delay)
        return value

    return fn


def boom(delay: float = 0.0, msg: str = "boom"):
    """Return a task function that sleeps then raises ConnectionError."""

    async def fn() -> None:
        await asyncio.sleep(delay)
        raise ConnectionError(msg)

    return fn


def run(sched: Scheduler, timeout: float = 2.0):
    """Run the scheduler, failing (TimeoutError) instead of hanging."""
    return asyncio.run(asyncio.wait_for(sched.run(), timeout))


def assert_skipped_once(res, name: str, cause: str | None = None) -> None:
    """Assert ``name`` was skipped exactly once and never executed or retried."""
    node = res.nodes[name]
    assert node.state is NodeState.SKIPPED, f"{name} is {node.state}"
    assert res.events.count("skipped", name) == 1, res.events.format()
    assert res.events.count("attempt", name) == 0, res.events.format()
    assert res.events.count("retry", name) == 0, res.events.format()
    assert node.attempts == 0
    if cause is not None:
        assert node.cause == cause, res.events.format()


def test_diamond_concurrent_failure_terminates():
    """Both branches of a diamond fail: the join is skipped and run() returns."""
    for left_delay, right_delay in [(0.01, 0.01), (0.0, 0.0), (0.01, 0.04)]:
        _check_diamond(left_delay, right_delay)


def _check_diamond(left_delay: float, right_delay: float) -> None:
    """Run one diamond where B and C fail after the given delays."""
    dag = DAG()
    dag.add("A", ok())
    dag.add("B", boom(left_delay), deps=["A"])
    dag.add("C", boom(right_delay), deps=["A"])
    dag.add("D", ok(), deps=["B", "C"])
    res = run(Scheduler(dag, retry=FAST))
    assert res.state("A") is NodeState.SUCCESS
    assert res.state("B") is NodeState.FAILED
    assert res.state("C") is NodeState.FAILED
    assert res.nodes["B"].attempts == 4 and res.nodes["C"].attempts == 4
    assert_skipped_once(res, "D")
    assert res.nodes["D"].cause in {"B", "C"}
    if left_delay < right_delay:
        assert res.nodes["D"].cause == "B"
    assert res.events.count("run_end") == 1


def test_diamond_concurrent_failure_without_retries():
    """Same diamond with retries disabled (simultaneous failure)."""
    dag = DAG()
    dag.add("A", ok())
    dag.add("B", boom(), deps=["A"])
    dag.add("C", boom(), deps=["A"])
    dag.add("D", ok(), deps=["B", "C"])
    dag.add("E", ok(), deps=["D"])
    res = run(Scheduler(dag, retry=NONE))
    assert_skipped_once(res, "D")
    assert_skipped_once(res, "E", cause=res.nodes["D"].cause)


def test_skipped_cause_is_first_failure():
    """A skipped node's cause is the earliest upstream failure, followed through skipped nodes."""
    dag = DAG()
    dag.add("A", boom(0.01))  # fails first
    dag.add("B", ok(), deps=["A"])  # skipped, root cause A
    dag.add("C", boom(0.06))  # fails later
    dag.add("OK", ok(delay=0.10))
    dag.add("J", ok(), deps=["C", "B", "OK"])
    dag.add("X", boom(0.08))  # fails late
    dag.add("Y", boom(0.02))  # fails early
    dag.add("K", ok(), deps=["X", "Y", "OK"])
    res = run(Scheduler(dag, retry=NONE))
    assert res.state("OK") is NodeState.SUCCESS
    assert_skipped_once(res, "B", cause="A")
    assert_skipped_once(res, "J", cause="A")
    assert_skipped_once(res, "K", cause="Y")


def test_skipped_cause_with_retries():
    """Failure timestamps (after retries are exhausted) decide the cause."""
    dag = DAG()
    dag.add("late", boom(0.03))  # 4 attempts -> fails at ~0.12s
    dag.add("early", boom(0.005))  # 4 attempts -> fails at ~0.02s
    dag.add("good", ok(delay=0.2))
    dag.add("J", ok(), deps=["late", "early", "good"])
    res = run(Scheduler(dag, retry=FAST))
    assert_skipped_once(res, "J", cause="early")


def test_skipped_nodes_are_not_retried():
    """Skipped nodes never enter the retry loop; failing nodes keep their retries."""
    calls = {"flaky": 0}

    async def flaky() -> str:
        calls["flaky"] += 1
        if calls["flaky"] < 3:
            raise ConnectionError("transient")
        return "ok"

    dag = DAG()
    dag.add("A", boom())
    dag.add("B", ok(), deps=["A"])
    dag.add("C", ok(), deps=["B"])
    dag.add("F", flaky)
    dag.add("G", ok(), deps=["F"])
    res = run(Scheduler(dag, retry=FAST))
    assert res.state("A") is NodeState.FAILED
    assert res.nodes["A"].attempts == 4
    assert res.events.count("retry", "A") == 3
    assert_skipped_once(res, "B", cause="A")
    assert_skipped_once(res, "C", cause="A")
    assert res.state("F") is NodeState.SUCCESS and res.nodes["F"].attempts == 3
    assert res.events.count("retry", "F") == 2
    assert res.state("G") is NodeState.SUCCESS


def test_skipped_nodes_do_not_sleep_backoff():
    """A skipped node must not consume backoff time."""
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    dag = DAG()
    dag.add("A", boom())
    dag.add("B", ok(), deps=["A"])
    res = run(Scheduler(dag, retry=RetryPolicy(max_retries=2, base_delay=0.5), sleep=fake_sleep))
    assert sleeps == [0.5, 1.0]  # only A's two retries
    assert_skipped_once(res, "B", cause="A")


def test_three_way_fanin_concurrent_failures():
    """Three failing branches into one join, plus a node downstream of the join."""
    dag = DAG()
    dag.add("A", ok())
    dag.add("B", boom(0.03), deps=["A"])
    dag.add("C", boom(0.01), deps=["A"])
    dag.add("E", boom(0.02), deps=["A"])
    dag.add("D", ok(), deps=["B", "C", "E"])
    dag.add("F", ok(), deps=["D"])
    dag.add("G", ok(), deps=["A"])
    res = run(Scheduler(dag, retry=FAST))
    for name in "BCE":
        assert res.state(name) is NodeState.FAILED
        assert res.events.count("retry", name) == 3
    assert_skipped_once(res, "D", cause="C")
    assert_skipped_once(res, "F", cause="C")
    assert res.state("G") is NodeState.SUCCESS
    assert res.duration_s < 1.5
