"""Tests for prices, cost math, breakers and the MeteredLLM proxy."""

import asyncio
from typing import Any

import pytest

from benchmarks.kesoku_bench.harness.metering import (
    BUDGET_STOP_TEXT,
    BudgetExceeded,
    Caps,
    CostLedger,
    MeteredLLM,
    Prices,
    cross_check_llm_logs,
    load_prices,
)
from kesoku.agent.llm import BaseLLM, LLMResponse, MockLLM
from kesoku.constants import MessageRole
from kesoku.db import Message

PRICES = Prices(input=0.75, output=3.75, cached_input=0.075)


def _res(prompt: int = 1000, cached: int = 0, total: int | None = 1500, cand: int = 100) -> LLMResponse:
    return LLMResponse(content="ok", prompt_tokens=prompt, cached_tokens=cached, total_tokens=total,
                       candidates_tokens=cand)


def _hist() -> list[Message]:
    return [Message(session_id="s", chatbot_id="c", channel_id="ch", sender="u", role=MessageRole.USER, content="hi")]


class _Scripted(BaseLLM):
    """LLM returning scripted responses or raising scripted exceptions."""

    def __init__(self, items: list[Any]) -> None:
        self.items = list(items)
        self.calls = 0
        self.model_name = "scripted-model"
        self.config = {"marker": 1}

    async def generate(self, prompt: str | None = None, system_prompt: str | None = None,
                       history: list[Message] | None = None, tools: Any = None,
                       cached_content: str | None = None) -> LLMResponse:
        self.calls += 1
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        if item == "sleep":
            await asyncio.sleep(10)
        return item


async def _nosleep(_: float) -> None:
    return None


def test_prices_pinned_from_compression_sim() -> None:
    p = load_prices()
    assert (p.input, p.output, p.cached_input) == (0.75, 3.75, 0.075)


def test_prices_json_override(tmp_path: Any) -> None:
    path = tmp_path / "p.json"
    path.write_text('{"input": 1, "output": 2, "cached_input": 0.5}')
    assert load_prices(path) == Prices(1.0, 2.0, 0.5)


def test_cost_math_and_token_split() -> None:
    led = CostLedger(PRICES)
    led.begin_turn(1)
    rec = led.book("agent", _res(prompt=1_000_000, cached=400_000, total=1_200_000), 0.1)
    assert (rec.uncached_in, rec.cached_in, rec.out) == (600_000, 400_000, 200_000)
    assert rec.cost_usd == pytest.approx(0.6 * 0.75 + 0.4 * 0.075 + 0.2 * 3.75)
    # without total_tokens, candidates are used
    rec2 = led.book("agent", _res(prompt=10, cached=0, total=None, cand=7), 0.1)
    assert rec2.out == 7
    t = led.totals()
    assert t["llm_calls"] == 2 and t["summarize_calls"] == 0
    assert led.turns[1].llm_calls == 2


def test_classify() -> None:
    assert MeteredLLM.classify("summarize this", None) == "summarize"
    assert MeteredLLM.classify(None, _hist()) == "agent"
    assert MeteredLLM.classify("x", _hist()) == "agent"


@pytest.mark.asyncio
async def test_proxy_books_and_forwards() -> None:
    inner = _Scripted([_res(), _res(prompt=50, total=60)])
    led = CostLedger(PRICES)
    llm = MeteredLLM(inner, led)
    assert isinstance(llm, BaseLLM)
    assert llm.model_name == "scripted-model"
    assert llm.config == {"marker": 1}  # forwarded via __getattr__
    led.begin_turn(1)
    await llm.generate(history=_hist())
    await llm.generate(prompt="summarize")
    t = led.totals()
    assert t["llm_calls"] == 1 and t["summarize_calls"] == 1
    assert led.turns[1].summarize_calls == 1


@pytest.mark.asyncio
async def test_max_tool_iters_returns_stop_response_and_excludes_summaries() -> None:
    inner = _Scripted([_res() for _ in range(10)])
    led = CostLedger(PRICES, Caps(max_tool_iters=2))
    llm = MeteredLLM(inner, led)
    led.begin_turn(1)
    await llm.generate(history=_hist())
    await llm.generate(prompt="summary")  # does not count toward the cap
    await llm.generate(history=_hist())
    res = await llm.generate(history=_hist())
    assert res.content == BUDGET_STOP_TEXT and res.tool_calls == []
    assert inner.calls == 3
    assert led.budget_exceeded is None  # tool-iteration cap is not a budget breach
    led.begin_turn(2)
    res = await llm.generate(history=_hist())
    assert res.content == "ok"


@pytest.mark.asyncio
async def test_task_cost_breaker_agent_and_summarize() -> None:
    inner = _Scripted([_res(prompt=1_000_000, total=1_000_000)] * 3)
    led = CostLedger(PRICES, Caps(max_task_cost_usd=0.5))
    llm = MeteredLLM(inner, led)
    led.begin_turn(1)
    await llm.generate(history=_hist())  # $0.75 spent
    res = await llm.generate(history=_hist())
    assert res.content == BUDGET_STOP_TEXT
    assert led.budget_exceeded and "max_task_cost_usd" in led.budget_exceeded
    with pytest.raises(BudgetExceeded):
        await llm.generate(prompt="summarize")
    assert inner.calls == 1


@pytest.mark.asyncio
async def test_suite_cost_breaker() -> None:
    led = CostLedger(PRICES, Caps(max_suite_cost_usd=10.0, suite_spent_usd=10.0))
    llm = MeteredLLM(_Scripted([_res()]), led)
    res = await llm.generate(history=_hist())
    assert res.content == BUDGET_STOP_TEXT
    assert "max_suite_cost_usd" in (led.budget_exceeded or "")


@pytest.mark.asyncio
async def test_retries_transient_errors() -> None:
    inner = _Scripted([RuntimeError("429 RESOURCE_EXHAUSTED"), RuntimeError("503 UNAVAILABLE"), _res()])
    led = CostLedger(PRICES)
    llm = MeteredLLM(inner, led, sleep=_nosleep)
    res = await llm.generate(history=_hist())
    assert res.content == "ok"
    assert led.records[-1].retries == 2
    assert led.totals()["llm_retries"] == 2


@pytest.mark.asyncio
async def test_non_transient_error_is_booked_and_raised() -> None:
    led = CostLedger(PRICES)
    llm = MeteredLLM(_Scripted([ValueError("bad request")]), led, sleep=_nosleep)
    with pytest.raises(ValueError):
        await llm.generate(history=_hist())
    assert led.records[-1].error and "bad request" in led.records[-1].error


@pytest.mark.asyncio
async def test_call_timeout_retries_then_raises() -> None:
    led = CostLedger(PRICES)
    llm = MeteredLLM(_Scripted(["sleep", "sleep"]), led, call_timeout_s=0.01, max_retries=1, sleep=_nosleep)
    with pytest.raises(TimeoutError):
        await llm.generate(history=_hist())


@pytest.mark.asyncio
async def test_mockllm_passthrough_methods() -> None:
    llm = MeteredLLM(MockLLM(mock_response="hello"), CostLedger(PRICES))
    res = await llm.generate(history=_hist())
    assert res.content == "hello"
    assert llm.estimate_tokens_fallback(prompt="abcd" * 10) == 10
    assert llm.count_tokens(prompt="abcd") >= 1
    assert await llm.create_cache([]) is None
    await llm.delete_cache("x")


def test_cross_check_llm_logs(tmp_path: Any) -> None:
    led = CostLedger(PRICES)
    led.begin_turn(1)
    led.book("agent", _res(prompt=1000, cached=200, total=1100), 0.1)
    log = tmp_path / "llm-turn-1.log.yaml"
    log.write_text("response:\n  metrics:\n    prompt_tokens: 1000\n    cached_tokens: 200\n    total_tokens: 1100\n")
    cc = cross_check_llm_logs([log], led)
    assert cc.log_files == 1 and cc.within_1pct is True
    log.write_text("response:\n  metrics:\n    prompt_tokens: 5000\n")
    assert cross_check_llm_logs([log], led).within_1pct is False
