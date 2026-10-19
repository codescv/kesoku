import asyncio

import pytest
from dagrun import DAG, NodeState, RetryPolicy, Scheduler


def test_policy_delays():
    """Exponential backoff capped by max_delay."""
    policy = RetryPolicy(max_retries=4, base_delay=0.1, factor=2.0, max_delay=0.5)
    assert [policy.delay(n) for n in range(1, 5)] == [0.1, 0.2, 0.4, 0.5]
    assert policy.total_backoff() == pytest.approx(1.2)


def test_policy_should_retry():
    """Retries stop after max_retries and only for matching exceptions."""
    policy = RetryPolicy(max_retries=2, retry_on=(ConnectionError,))
    assert policy.should_retry(ConnectionError(), 0)
    assert policy.should_retry(ConnectionError(), 1)
    assert not policy.should_retry(ConnectionError(), 2)
    assert not policy.should_retry(ValueError(), 0)


def test_policy_validation():
    """Invalid parameters are rejected."""
    with pytest.raises(ValueError):
        RetryPolicy(max_retries=-1)
    with pytest.raises(ValueError):
        RetryPolicy(factor=0.5)


def test_success_on_second_attempt():
    """A flaky node succeeds on its second attempt after one backoff sleep."""
    calls = 0
    sleeps: list[float] = []

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("transient")
        return "ok"

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    dag = DAG()
    dag.add("fetch", flaky)
    dag.add("use", flaky, deps=["fetch"])
    sched = Scheduler(dag, retry=RetryPolicy(max_retries=3, base_delay=0.25), sleep=fake_sleep)
    res = asyncio.run(sched.run())
    assert res.ok
    assert res.nodes["fetch"].attempts == 2
    assert res.nodes["use"].attempts == 1
    assert sleeps == [0.25]
    assert res.events.kinds("fetch") == ["attempt", "retry", "attempt", "success"]


def test_retries_exhausted():
    """A node that keeps failing is attempted max_retries + 1 times."""

    async def always_fails() -> None:
        raise ConnectionError("down")

    dag = DAG()
    dag.add("x", always_fails)
    res = asyncio.run(Scheduler(dag, retry=RetryPolicy(max_retries=2, base_delay=0.001)).run())
    assert res.state("x") is NodeState.FAILED
    assert res.nodes["x"].attempts == 3
    assert res.events.count("retry", "x") == 2
    assert res.events.count("failed", "x") == 1


def test_non_retryable_exception_fails_immediately():
    """Exceptions outside retry_on are not retried."""

    async def bad() -> None:
        raise ValueError("bad input")

    dag = DAG()
    dag.add("x", bad)
    policy = RetryPolicy(max_retries=5, base_delay=0.001, retry_on=(ConnectionError,))
    res = asyncio.run(Scheduler(dag, retry=policy).run())
    assert res.state("x") is NodeState.FAILED
    assert res.nodes["x"].attempts == 1


def test_attempt_timeout_is_retried():
    """A per-node timeout counts as a failed attempt."""
    calls = 0

    async def slow_then_fast() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(1.0)
        return "done"

    dag = DAG()
    dag.add("x", slow_then_fast, timeout_s=0.05)
    res = asyncio.run(Scheduler(dag, retry=RetryPolicy(max_retries=1, base_delay=0.001)).run())
    assert res.state("x") is NodeState.SUCCESS
    assert res.nodes["x"].attempts == 2
