"""LLM metering: price table, cost ledger with circuit breakers, and the MeteredLLM proxy."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from kesoku.agent.llm import BaseLLM, LLMResponse
from kesoku.db import Message

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPRESSION_SIM_PATH = REPO_ROOT / "scripts" / "compression_sim.py"
BUDGET_STOP_TEXT = "⚠️ [kesoku-bench] budget exhausted; stopping this turn."
_RETRYABLE_MARKERS = ("429", "503", "RESOURCE_EXHAUSTED", "ResourceExhausted", "UNAVAILABLE", "Too Many Requests")


@dataclass(frozen=True)
class Prices:
    """USD per 1M tokens."""

    input: float
    output: float
    cached_input: float


def load_prices(prices_json: str | Path | None = None) -> Prices:
    """Load prices from `scripts/compression_sim.py::Prices` (or a JSON override file).

    `scripts/` is not a package, so the module is loaded by file location.

    Args:
        prices_json: Optional JSON file with keys input/output/cached_input.

    Returns:
        The price table.
    """
    if prices_json:
        data = json.loads(Path(prices_json).read_text(encoding="utf-8"))
        return Prices(input=float(data["input"]), output=float(data["output"]),
                      cached_input=float(data["cached_input"]))
    spec = importlib.util.spec_from_file_location("kb_compression_sim", COMPRESSION_SIM_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {COMPRESSION_SIM_PATH}")
    module = importlib.util.module_from_spec(spec)
    # dataclass field resolution needs the module registered while it executes
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    p = module.Prices()
    return Prices(input=float(p.input), output=float(p.output), cached_input=float(p.cached_input))


class BudgetExceeded(RuntimeError):
    """Raised by MeteredLLM when a summarize call would breach a budget."""


@dataclass
class Caps:
    """Circuit-breaker limits."""

    max_tool_iters: int = 25
    max_task_cost_usd: float = 2.0
    max_suite_cost_usd: float = 50.0
    suite_spent_usd: float = 0.0  # spend of already-finished tasks (parent-provided)
    max_summarize_calls_per_turn: int = 20


@dataclass
class CallRecord:
    """One metered LLM call."""

    kind: str  # agent | summarize
    turn_idx: int
    uncached_in: int = 0
    cached_in: int = 0
    out: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0
    retries: int = 0
    error: str | None = None
    blocked: str | None = None  # breaker reason if the call was not sent
    ts: float = field(default_factory=time.time)  # wall-clock time the call finished (or was blocked)


@dataclass
class TurnLedger:
    """Aggregates for one scripted turn."""

    idx: int
    llm_calls: int = 0
    summarize_calls: int = 0
    uncached_in: int = 0
    cached_in: int = 0
    out: int = 0
    cost_usd: float = 0.0
    llm_retries: int = 0
    blocked_calls: int = 0

    def tokens(self) -> dict[str, int]:
        """Token breakdown dict."""
        return {"uncached_in": self.uncached_in, "cached_in": self.cached_in, "out": self.out}


class CostLedger:
    """Books every LLM call, enforces the breakers, and aggregates per-turn and per-task totals."""

    def __init__(self, prices: Prices, caps: Caps | None = None) -> None:
        """Initialize the ledger.

        Args:
            prices: Price table (USD per 1M tokens).
            caps: Breaker limits.
        """
        self.prices = prices
        self.caps = caps or Caps()
        self.records: list[CallRecord] = []
        self.turns: dict[int, TurnLedger] = {}
        self.current_turn = 0
        self._turn_agent_calls = 0
        self._turn_summarize_calls = 0
        self.budget_exceeded: str | None = None  # reason of the first breach
        self.turns[0] = TurnLedger(idx=0)

    # --- turn bookkeeping ------------------------------------------------------------------------
    def begin_turn(self, idx: int) -> None:
        """Start a scripted turn: resets the per-turn agent-call counter (--max-tool-iters)."""
        self.current_turn = idx
        self._turn_agent_calls = 0
        self._turn_summarize_calls = 0
        self.turns.setdefault(idx, TurnLedger(idx=idx))

    def reset_turn_iters(self) -> None:
        """Reset the agent-call counter without changing the turn index (e.g. for an auto-reply)."""
        self._turn_agent_calls = 0

    @property
    def task_cost(self) -> float:
        """Total task spend so far."""
        return sum(r.cost_usd for r in self.records)

    def cost(self, uncached_in: int, cached_in: int, out: int) -> float:
        """Compute USD cost of a call."""
        p = self.prices
        return uncached_in * p.input / 1e6 + cached_in * p.cached_input / 1e6 + out * p.output / 1e6

    # --- breakers --------------------------------------------------------------------------------
    def check(self, kind: str) -> str | None:
        """Return the breaker reason that blocks a call of this kind, or None if it may proceed."""
        caps = self.caps
        if kind == "agent" and self._turn_agent_calls >= caps.max_tool_iters:
            return f"max_tool_iters ({caps.max_tool_iters}) reached in turn {self.current_turn}"
        if kind == "summarize" and self._turn_summarize_calls >= caps.max_summarize_calls_per_turn:
            return f"max_summarize_calls_per_turn ({caps.max_summarize_calls_per_turn}) reached"
        spent = self.task_cost
        if spent >= caps.max_task_cost_usd:
            return f"max_task_cost_usd ({caps.max_task_cost_usd:.2f}) reached (spent {spent:.4f})"
        if caps.suite_spent_usd + spent >= caps.max_suite_cost_usd:
            return f"max_suite_cost_usd ({caps.max_suite_cost_usd:.2f}) reached"
        return None

    def note_blocked(self, kind: str, reason: str) -> None:
        """Record a call that was refused by a breaker."""
        is_cost = "cost" in reason
        if is_cost and self.budget_exceeded is None:
            self.budget_exceeded = reason
        if kind == "summarize" and self.budget_exceeded is None:
            self.budget_exceeded = reason
        self.records.append(CallRecord(kind=kind, turn_idx=self.current_turn, blocked=reason))
        self.turns.setdefault(self.current_turn, TurnLedger(idx=self.current_turn)).blocked_calls += 1

    # --- booking ---------------------------------------------------------------------------------
    def book(self, kind: str, res: LLMResponse | None, duration_s: float, retries: int = 0,
             error: str | None = None) -> CallRecord:
        """Book a finished (or failed) call.

        Token accounting: uncached_in = prompt - cached; out = total - prompt (includes thinking tokens)
        when total is known, else candidates.
        """
        prompt = (res.prompt_tokens or 0) if res else 0
        cached = (res.cached_tokens or 0) if res else 0
        total = (res.total_tokens or 0) if res else 0
        cand = (res.candidates_tokens or 0) if res else 0
        uncached = max(0, prompt - cached)
        out = (total - prompt) if total else cand
        out = max(0, out)
        rec = CallRecord(kind=kind, turn_idx=self.current_turn, uncached_in=uncached, cached_in=cached, out=out,
                         cost_usd=self.cost(uncached, cached, out), duration_s=duration_s, retries=retries,
                         error=error)
        self.records.append(rec)
        t = self.turns.setdefault(self.current_turn, TurnLedger(idx=self.current_turn))
        if kind == "agent":
            self._turn_agent_calls += 1
            t.llm_calls += 1
        else:
            self._turn_summarize_calls += 1
            t.summarize_calls += 1
        t.uncached_in += uncached
        t.cached_in += cached
        t.out += out
        t.cost_usd += rec.cost_usd
        t.llm_retries += retries
        return rec

    # --- reporting -------------------------------------------------------------------------------
    def totals(self) -> dict[str, Any]:
        """Task-level totals."""
        ok = [r for r in self.records if r.blocked is None]
        kinds = Counter(r.kind for r in ok)
        return {
            "llm_calls": kinds.get("agent", 0),
            "summarize_calls": kinds.get("summarize", 0),
            "tokens": {
                "uncached_in": sum(r.uncached_in for r in ok),
                "cached_in": sum(r.cached_in for r in ok),
                "out": sum(r.out for r in ok),
            },
            "agent_tokens": {
                "uncached_in": sum(r.uncached_in for r in ok if r.kind == "agent"),
                "cached_in": sum(r.cached_in for r in ok if r.kind == "agent"),
                "out": sum(r.out for r in ok if r.kind == "agent"),
            },
            "cost_usd": round(sum(r.cost_usd for r in ok), 6),
            "llm_retries": sum(r.retries for r in ok),
            "llm_errors": sum(1 for r in ok if r.error),
            "blocked_calls": sum(1 for r in self.records if r.blocked),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialisable dump of all records."""
        return {"records": [asdict(r) for r in self.records], "budget_exceeded": self.budget_exceeded}


def _is_retryable(exc: BaseException) -> bool:
    """Whether an LLM exception is a transient quota/availability error."""
    text = f"{type(exc).__name__}: {exc}"
    return any(marker in text for marker in _RETRYABLE_MARKERS)


class MeteredLLM(BaseLLM):
    """BaseLLM proxy that meters every call and enforces the ledger's breakers.

    `prompt=` calls without history are summarize (compaction/consolidation) calls; everything else is an
    agent call. A breach during an agent call returns a stop response (no tool calls) so TurnExecutor ends
    the turn cleanly; a breach during a summarize call raises BudgetExceeded so no bogus summary is persisted.
    """

    def __init__(
        self,
        inner: BaseLLM,
        ledger: CostLedger,
        call_timeout_s: float = 300.0,
        max_retries: int = 3,
        backoff_s: float = 2.0,
        sleep: Callable[[float], Any] | None = None,
    ) -> None:
        """Wrap an LLM.

        Args:
            inner: The real LLM (GeminiLLM, MockLLM, ...).
            ledger: Shared cost ledger.
            call_timeout_s: Per-call timeout.
            max_retries: Retries for transient 429/503 errors.
            backoff_s: Base backoff (exponential).
            sleep: Injectable async sleep (tests).
        """
        self._inner = inner
        self.ledger = ledger
        self.call_timeout_s = call_timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._sleep = sleep or asyncio.sleep
        self.model_name = getattr(inner, "model_name", None)
        self.context_window_limit = getattr(inner, "context_window_limit", BaseLLM.context_window_limit)

    @property
    def inner(self) -> BaseLLM:
        """The wrapped LLM."""
        return self._inner

    def __getattr__(self, name: str) -> Any:
        """Forward any other attribute (config, client, ...) to the wrapped LLM."""
        if name.startswith("__") or name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    @staticmethod
    def classify(prompt: str | None, history: list[Message] | None) -> str:
        """Classify a call as `summarize` (prompt only) or `agent`."""
        return "summarize" if prompt is not None and not history else "agent"

    async def generate(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        history: list[Message] | None = None,
        tools: list[Callable] | None = None,
        cached_content: str | None = None,
    ) -> LLMResponse:
        """Meter and forward a generation call (see class docstring for breaker semantics)."""
        kind = self.classify(prompt, history)
        reason = self.ledger.check(kind)
        if reason:
            self.ledger.note_blocked(kind, reason)
            if kind == "summarize":
                raise BudgetExceeded(reason)
            return LLMResponse(content=BUDGET_STOP_TEXT, tool_calls=[], finish_reason="KB_BUDGET")

        retries = 0
        start = time.monotonic()
        while True:
            try:
                res = await asyncio.wait_for(
                    self._inner.generate(
                        prompt=prompt, system_prompt=system_prompt, history=history, tools=tools,
                        cached_content=cached_content,
                    ),
                    timeout=self.call_timeout_s,
                )
                self.ledger.book(kind, res, time.monotonic() - start, retries=retries)
                return res
            except BudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 - re-raised below unless transient
                retryable = _is_retryable(exc) or isinstance(exc, TimeoutError)
                if retryable and retries < self.max_retries:
                    retries += 1
                    await self._sleep(self.backoff_s * (2 ** (retries - 1)))
                    continue
                self.ledger.book(kind, None, time.monotonic() - start, retries=retries,
                                 error=f"{type(exc).__name__}: {exc}"[:500])
                raise

    # --- explicit forwards (BaseLLM defines these, so __getattr__ would not reach the inner LLM) ----
    async def create_cache(
        self,
        contents: list[Message],
        system_prompt: str | None = None,
        tools: list[Callable] | None = None,
        display_name: str | None = None,
        ttl_seconds: int = 300,
    ) -> str | None:
        """Forward to the wrapped LLM."""
        return await self._inner.create_cache(contents, system_prompt, tools, display_name, ttl_seconds)

    async def delete_cache(self, cache_name: str) -> None:
        """Forward to the wrapped LLM."""
        await self._inner.delete_cache(cache_name)

    def count_tokens(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        history: list[Message] | None = None,
        tools: list[Callable] | None = None,
    ) -> int:
        """Forward to the wrapped LLM."""
        return self._inner.count_tokens(prompt, system_prompt, history, tools)

    def estimate_tokens_fallback(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        history: list[Message] | None = None,
    ) -> int:
        """Forward to the wrapped LLM."""
        return self._inner.estimate_tokens_fallback(prompt, system_prompt, history)


@dataclass
class LogCrossCheck:
    """Comparison of ledger agent tokens against TurnLogger YAML metrics."""

    log_files: int = 0
    log_tokens: dict[str, int] = field(default_factory=dict)
    ledger_tokens: dict[str, int] = field(default_factory=dict)
    within_1pct: bool | None = None


def cross_check_llm_logs(log_paths: list[Path], ledger: CostLedger) -> LogCrossCheck:
    """Sum `response.metrics` of llm-turn-*.log.yaml files and compare with the ledger's agent calls."""
    totals = {"prompt_tokens": 0, "cached_tokens": 0, "total_tokens": 0}
    n = 0
    for path in log_paths:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001 - a corrupt log only weakens the cross-check
            continue
        metrics = (data.get("response") or {}).get("metrics") or {}
        n += 1
        for key in totals:
            totals[key] += int(metrics.get(key) or 0)
    agent = ledger.totals()["agent_tokens"]
    ledger_prompt = agent["uncached_in"] + agent["cached_in"]
    ok: bool | None = None
    if n:
        denom = max(1, ledger_prompt)
        ok = abs(totals["prompt_tokens"] - ledger_prompt) / denom <= 0.01
    return LogCrossCheck(log_files=n, log_tokens=totals,
                         ledger_tokens={"prompt_tokens": ledger_prompt, **agent}, within_1pct=ok)
