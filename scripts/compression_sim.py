"""Offline simulator comparing Kesoku context-compression policies on a real session.

Replays every logical turn (and every LLM call inside each turn) of a stored session and
models, per call, the prompt that would be sent to Gemini under a given compression
policy. It accounts for:

* prepared-history sanitization (thoughts dropped, historical tool results capped at 500 chars),
* the hierarchical summary scaffold (L0 nodes + consolidation),
* Kesoku's explicit Gemini context cache (created once over the prefix before the current
  user message, TTL without refresh, deleted on every compaction),
* summarization / consolidation LLM calls.

No LLM is called: summary sizes are modeled, and real token counts are approximated with a
linear ASCII / non-ASCII model calibrated against Gemini ``count_tokens``.

Usage:
    uv run python scripts/compression_sim.py --db ../band/kesoku.db --session de73ad46
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
from dataclasses import dataclass, field
from typing import Protocol

from kesoku.agent.compressor import CONSOLIDATE_SUMMARIES_PROMPT, SUMMARIZE_TURN_PROMPT

_NON_ASCII_RE = re.compile(r"[^\x00-\x7f]")

# Calibrated on 400 random de73ad46 messages via gemini-3.8-flash count_tokens
# (real = 0.2424 * ascii + 0.8663 * non_ascii; fit error on total < 2%).
REAL_ASCII_COEF = 0.2424
REAL_NON_ASCII_COEF = 0.8663

HIST_TOOL_CHARS = 500
# Header added by prepare_history_for_llm when a historical tool result is truncated.
TRUNC_HEADER = (
    "Tool `run_shell_command` output:\n```\nOutput truncated for history (total length 12345 chars). "
    "Full output saved to: `sessions/xxxxxxxx/hist_tool_run_shell_command_1760000000_abcdef12.txt`.\n\n"
    "Preview (last 500 chars):\n\n```"
)
SCAFFOLD_HEADER = (
    "[Note: This conversation uses custom turn-based context management. Earlier turns have been "
    "compacted into hierarchical summaries below. You have access to search past messages and "
    "memories using standard search tools.]\n"
)
SCAFFOLD_NODE_LABEL = "\n[Session Arc Summary (Level 1, node 00000000-0000-0000-0000-000000000000)]\n"
SCAFFOLD_ACK = "Understood. I have the compacted summaries of earlier turns."
TS_PREFIX = "[2026-10-13 23:58:38 CST] ASSISTANT: "


def split_chars(text: str) -> tuple[int, int]:
    """Return (ascii_chars, non_ascii_chars) of a text."""
    non_ascii = len(_NON_ASCII_RE.findall(text))
    return len(text) - non_ascii, non_ascii


def real_tokens(text: str) -> int:
    """Approximate Gemini token count of a text using the calibrated linear model."""
    a, n = split_chars(text)
    return int(a * REAL_ASCII_COEF + n * REAL_NON_ASCII_COEF)


def fallback_est_tokens(text: str) -> int:
    """Replicate BaseLLM.estimate_tokens_fallback (ascii/4 + non_ascii*2)."""
    a, n = split_chars(text)
    return max(1, int(a / 4 + n * 2.0))


# --------------------------------------------------------------------------------------
# Session loading
# --------------------------------------------------------------------------------------


@dataclass
class Call:
    """One LLM call inside a turn."""

    partial_in: int  # real tokens of current-turn messages sent before this call (full detail)
    visible_out: int  # real tokens of the visible output (thought text + tool calls + text)


@dataclass
class Turn:
    """One logical turn (user/system message plus the agent's work until the next one)."""

    start_ts: float
    end_ts: float
    prepared_real: int  # real tokens once completed & sanitized (what later calls see)
    prepared_est: int  # fallback estimate used by the current compressor's trigger
    summary_src_real: int  # real tokens of format_turn_for_summary(turn)
    calls: list[Call]
    metrics: dict | None  # turn_metrics stored on the final assistant message, if any


def _prepared_content(role: str, mtype: str, content: str) -> str | None:
    """Return the content of a message as it appears in a completed (historical) turn."""
    if role == "assistant" and mtype == "thought":
        return None
    if role == "tool" and mtype == "tool_result" and len(content) > HIST_TOOL_CHARS:
        return TRUNC_HEADER + content[-HIST_TOOL_CHARS:]
    return content


def load_turns(db_path: str, session_id: str) -> list[Turn]:
    """Load a session from kesoku.db and segment it into logical turns with per-call stats."""
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT role, type, sender, content, metadata, timestamp FROM messages "
        "WHERE session_id = ? ORDER BY timestamp, rowid",
        (session_id,),
    ).fetchall()
    # Same exclusion and segmentation rules as build_history / segment_logical_turns.
    rows = [r for r in rows if not (r[0] == "assistant" and r[2] == "Notification")]
    groups: list[list[tuple]] = []
    for r in rows:
        if r[0] in ("user", "system") or not groups:
            groups.append([r])
        else:
            groups[-1].append(r)

    turns: list[Turn] = []
    for g in groups:
        prepared_real = prepared_est = summary_src = 0
        metrics = None
        for role, mtype, _sender, content, meta, _ts in g:
            content = content or ""
            prep = _prepared_content(role, mtype, content)
            if prep is not None:
                prepared_real += real_tokens(prep)
            # The current compressor estimates with content[-500:] for tool results.
            if not (role == "assistant" and mtype == "thought"):
                est_text = content[-HIST_TOOL_CHARS:] if (role == "tool" and mtype == "tool_result") else content
                prepared_est += fallback_est_tokens(est_text)
                summary_src += real_tokens(TS_PREFIX + content)
            if role == "assistant" and mtype == "text" and meta:
                tm = json.loads(meta).get("turn_metrics")
                if tm:
                    metrics = tm

        # Split the turn into LLM calls: a call's output is a run of assistant/tool_call
        # messages; tool results feed the next call.
        calls: list[Call] = []
        partial = real_tokens(g[0][3] or "")
        cur_out = 0
        in_output = False
        for role, mtype, _sender, content, _meta, _ts in g[1:]:
            t = real_tokens(content or "")
            if role == "tool" and mtype == "tool_result":
                if in_output:
                    calls.append(Call(partial_in=partial, visible_out=cur_out))
                    partial += cur_out
                    cur_out, in_output = 0, False
                partial += t
            else:
                cur_out += t
                in_output = True
        if in_output or not calls:
            calls.append(Call(partial_in=partial, visible_out=cur_out))

        turns.append(
            Turn(
                start_ts=g[0][5],
                end_ts=g[-1][5],
                prepared_real=prepared_real,
                prepared_est=prepared_est,
                summary_src_real=summary_src,
                calls=calls,
                metrics=metrics,
            )
        )
    return turns


# --------------------------------------------------------------------------------------
# Simulation state and policies
# --------------------------------------------------------------------------------------


@dataclass
class Node:
    """A summary node in the scaffold forest."""

    level: int
    tokens: int
    start_ts: float


@dataclass
class Prices:
    """USD per 1M tokens (Vertex gemini-3.8-flash introductory pricing, through 2026-12-31)."""

    input: float = 0.75
    output: float = 3.75
    cached_input: float = 0.075
    cache_storage_per_hour: float = 0.50


@dataclass
class SimConfig:
    """Workload-wide assumptions shared by all policies."""

    sys_tools_tokens: int = 0  # cacheable system prompt + tool schemas
    inject_tokens: int = 0  # per-call uncached injection (instructions, notes, auto-RAG)
    output_scale: float = 1.0  # real output (incl. thinking) / visible output
    summary_thinking_tokens: int = 1000  # thinking tokens per summarization call
    cache_ttl: float = 3600.0
    cache_min_tokens: int = 4096
    prices: Prices = field(default_factory=Prices)


@dataclass
class State:
    """Mutable simulation state."""

    turns: list[Turn]
    compressed: list[bool]
    roots: list[Node] = field(default_factory=list)
    summary_calls: int = 0
    summary_in: int = 0
    summary_out: int = 0
    l0_nodes: int = 0

    def scaffold_tokens(self) -> int:
        """Real tokens of the scaffold user message + assistant ack."""
        if not self.roots:
            return 0
        per_node = real_tokens(SCAFFOLD_NODE_LABEL)
        return (
            real_tokens(SCAFFOLD_HEADER)
            + real_tokens(SCAFFOLD_ACK)
            + sum(n.tokens + per_node for n in self.roots)
        )


_SUMMARIZE_PROMPT_TOKENS = real_tokens(SUMMARIZE_TURN_PROMPT)
_CONSOLIDATE_PROMPT_TOKENS = real_tokens(CONSOLIDATE_SUMMARIES_PROMPT)


def make_l0(state: State, idxs: list[int], size_fn, cfg: SimConfig) -> int:
    """Compress the given turn indices into a Level-0 node; return the node's tokens."""
    src = sum(state.turns[i].prepared_real for i in idxs)
    tokens = size_fn(src)
    for i in idxs:
        state.compressed[i] = True
    state.roots.append(Node(level=0, tokens=tokens, start_ts=state.turns[idxs[0]].start_ts))
    state.roots.sort(key=lambda n: n.start_ts)
    state.summary_calls += 1
    state.l0_nodes += 1
    state.summary_in += _SUMMARIZE_PROMPT_TOKENS + sum(state.turns[i].summary_src_real for i in idxs)
    state.summary_out += tokens + cfg.summary_thinking_tokens
    return tokens


def consolidate(state: State, k: int, size_fn, cfg: SimConfig) -> None:
    """Replicate HistoryCompressor.consolidate_forest (merge oldest K when a level has >= 2K roots)."""
    level = 0
    while True:
        roots = [n for n in state.roots if n.level == level]
        if len(roots) < 2 * k:
            break
        i = 0
        while len(roots) - i >= 2 * k:
            chunk = roots[i : i + k]
            tokens = size_fn([n.tokens for n in chunk])
            for n in chunk:
                state.roots.remove(n)
            state.roots.append(Node(level=level + 1, tokens=tokens, start_ts=chunk[0].start_ts))
            state.summary_calls += 1
            state.summary_in += _CONSOLIDATE_PROMPT_TOKENS + sum(n.tokens + 20 for n in chunk)
            state.summary_out += tokens + cfg.summary_thinking_tokens
            i += k
        state.roots.sort(key=lambda n: n.start_ts)
        level += 1


class Policy(Protocol):
    """A compression policy."""

    name: str
    protect_front: int

    def compact(self, state: State, cur: int, cfg: SimConfig, history_tokens: int) -> bool:
        """Run compaction at the start of turn `cur`; return True if anything was compressed."""

    def should_refresh_cache(self, uncached_history: int) -> bool:
        """Whether to rebuild the explicit cache to absorb the uncached history suffix."""


@dataclass
class CurrentPolicy:
    """Exact replica of today's HistoryCompressor.auto_compact_session trigger."""

    name: str = "current"
    protect_front: int = 1
    protect_tail: int = 3
    base_node_min_tokens: int = 25000
    base_node_turns: int = 4
    k: int = 4
    l0_tokens: int = 460  # measured avg real tokens of L0 nodes in de73ad46
    merged_tokens: int = 500  # measured avg real tokens of L1+ nodes

    def compact(self, state: State, cur: int, cfg: SimConfig, history_tokens: int) -> bool:
        """Compress chunks of >= base_node_turns and >= base_node_min_tokens (fallback estimate)."""
        n_turns = cur + 1  # history includes the current turn
        if n_turns <= self.protect_front + self.protect_tail:
            return False
        chunk: list[int] = []
        est = 0
        occurred = False
        for i in range(self.protect_front, n_turns - self.protect_tail):
            if state.compressed[i]:
                continue
            chunk.append(i)
            est += state.turns[i].prepared_est
            if len(chunk) >= self.base_node_turns and est >= self.base_node_min_tokens:
                make_l0(state, chunk, lambda _src: self.l0_tokens, cfg)
                chunk, est, occurred = [], 0, True
        if occurred:
            consolidate(state, self.k, lambda _c: self.merged_tokens, cfg)
        return occurred

    def should_refresh_cache(self, uncached_history: int) -> bool:
        """Never refresh: today's code only rebuilds the cache after compaction or expiry."""
        return False


@dataclass
class WatermarkPolicy:
    """Proposed policy: compact only above a high watermark, down to a low watermark.

    * Trigger on the real size of the whole prompt (system + tools + scaffold + history).
    * Protect a tail by tokens (and a minimum number of turns), not just by turn count.
    * Summary length scales with the source size instead of a fixed ~1200-char cap.
    * Optionally rebuild the explicit cache when the uncached suffix grows large.
    """

    name: str = "watermark"
    protect_front: int = 1
    high: int = 128_000
    low: int = 64_000
    tail_min_tokens: int = 24_000
    tail_min_turns: int = 4
    chunk_tokens: int = 24_000
    chunk_min_turns: int = 2
    k: int = 4
    l0_ratio: float = 0.04
    l0_min: int = 400
    l0_max: int = 1500
    merge_ratio: float = 0.5
    merge_max: int = 2000
    cache_refresh_tokens: int | None = 16_000

    def _l0_size(self, src: int) -> int:
        """Summary length proportional to the source, clamped."""
        return int(min(self.l0_max, max(self.l0_min, src * self.l0_ratio)))

    def _merge_size(self, children: list[int]) -> int:
        """Consolidated length proportional to the children, clamped."""
        return int(min(self.merge_max, max(max(children), sum(children) * self.merge_ratio)))

    def compact(self, state: State, cur: int, cfg: SimConfig, history_tokens: int) -> bool:
        """Compress oldest unprotected turns until the prompt drops below the low watermark."""
        total = cfg.sys_tools_tokens + cfg.inject_tokens + state.scaffold_tokens() + history_tokens
        if total <= self.high:
            return False

        # Determine the protected tail (walk back from the current turn).
        tail_start = cur + 1
        tail_tokens = 0
        while tail_start > self.protect_front and (
            tail_tokens < self.tail_min_tokens or (cur + 1 - tail_start) < self.tail_min_turns
        ):
            tail_start -= 1
            if not state.compressed[tail_start]:
                tail_tokens += state.turns[tail_start].prepared_real

        candidates = [i for i in range(self.protect_front, tail_start) if not state.compressed[i]]
        occurred = False
        while total > self.low and candidates:
            chunk: list[int] = []
            src = 0
            while candidates and (src < self.chunk_tokens or len(chunk) < self.chunk_min_turns):
                i = candidates.pop(0)
                chunk.append(i)
                src += state.turns[i].prepared_real
            node_tokens = make_l0(state, chunk, self._l0_size, cfg)
            total += node_tokens + real_tokens(SCAFFOLD_NODE_LABEL) - src
            occurred = True
        if occurred:
            consolidate(state, self.k, self._merge_size, cfg)
        return occurred

    def should_refresh_cache(self, uncached_history: int) -> bool:
        """Rebuild the cache once the uncached history suffix exceeds the threshold."""
        return self.cache_refresh_tokens is not None and uncached_history > self.cache_refresh_tokens


# --------------------------------------------------------------------------------------
# Simulation loop
# --------------------------------------------------------------------------------------


@dataclass
class Result:
    """Aggregated simulation output."""

    name: str
    llm_calls: int = 0
    input_uncached: int = 0
    input_cached: int = 0
    output: int = 0
    summary_calls: int = 0
    summary_in: int = 0
    summary_out: int = 0
    compaction_events: int = 0
    l0_nodes: int = 0
    cache_creations: int = 0
    cache_create_tokens: int = 0
    cache_storage_token_hours: float = 0.0
    prompt_sizes: list[int] = field(default_factory=list)
    raw_history_sizes: list[int] = field(default_factory=list)  # uncompressed history per call
    per_turn_last_prompt: list[int] = field(default_factory=list)
    per_turn_cached: list[int] = field(default_factory=list)
    per_turn_input: list[int] = field(default_factory=list)
    final_scaffold: int = 0

    def cost(self, p: Prices) -> dict[str, float]:
        """Return a USD cost breakdown."""
        c = {
            "input": self.input_uncached * p.input / 1e6,
            "cached_input": self.input_cached * p.cached_input / 1e6,
            "output": self.output * p.output / 1e6,
            "cache_create": self.cache_create_tokens * p.input / 1e6,
            "cache_storage": self.cache_storage_token_hours * p.cache_storage_per_hour / 1e6,
            "summarize": (self.summary_in * p.input + self.summary_out * p.output) / 1e6,
        }
        c["total"] = sum(c.values())
        return c


def simulate(turns: list[Turn], policy: Policy, cfg: SimConfig) -> Result:
    """Replay a session under a policy and return aggregated token / cost statistics."""
    state = State(turns=turns, compressed=[False] * len(turns))
    res = Result(name=policy.name)
    cache_prefix: int | None = None  # cached history tokens (excluding sys/tools)
    cache_created = cache_expires = 0.0

    def close_cache(now: float) -> None:
        nonlocal cache_prefix
        if cache_prefix is not None:
            alive = min(now, cache_expires) - cache_created
            res.cache_storage_token_hours += (cfg.sys_tools_tokens + cache_prefix) * max(alive, 0) / 3600
        cache_prefix = None

    for cur, turn in enumerate(turns):
        now = turn.start_ts
        hist_before = sum(t.prepared_real for i, t in enumerate(turns[:cur]) if not state.compressed[i])

        compacted = policy.compact(state, cur, cfg, hist_before)
        if compacted:
            res.compaction_events += 1
            close_cache(now)  # compaction invalidates the prefix -> cache deleted

        # Prepared history before the current turn: head + scaffold + uncompressed turns.
        raw_hist = sum(t.prepared_real for i, t in enumerate(turns[:cur]) if not state.compressed[i])
        scaffold = state.scaffold_tokens()
        hist = raw_hist + scaffold

        if cache_prefix is not None and now >= cache_expires:
            close_cache(now)  # expired; Kesoku retries uncached and recreates it
        if cache_prefix is not None and policy.should_refresh_cache(hist - cache_prefix):
            close_cache(now)
        if cache_prefix is None and cfg.sys_tools_tokens + hist >= cfg.cache_min_tokens and cur > 0:
            cache_prefix = hist
            cache_created, cache_expires = now, now + cfg.cache_ttl
            res.cache_creations += 1
            res.cache_create_tokens += cfg.sys_tools_tokens + hist

        turn_input = 0
        for call in turn.calls:
            prompt = cfg.sys_tools_tokens + hist + cfg.inject_tokens + call.partial_in
            cached = cfg.sys_tools_tokens + cache_prefix if cache_prefix is not None else 0
            res.llm_calls += 1
            res.input_cached += cached
            res.input_uncached += prompt - cached
            res.output += int(call.visible_out * cfg.output_scale)
            res.prompt_sizes.append(prompt)
            res.raw_history_sizes.append(raw_hist)
            turn_input += prompt
        res.per_turn_last_prompt.append(prompt)
        res.per_turn_cached.append(cached)
        res.per_turn_input.append(turn_input)

    close_cache(turns[-1].end_ts)
    res.summary_calls, res.summary_in, res.summary_out = state.summary_calls, state.summary_in, state.summary_out
    res.l0_nodes = state.l0_nodes
    res.final_scaffold = state.scaffold_tokens()
    return res


# --------------------------------------------------------------------------------------
# Calibration against real turn_metrics and reporting
# --------------------------------------------------------------------------------------


def calibrate(turns: list[Turn], since_ts: float) -> SimConfig:
    """Fit fixed overheads and output scale from real turn_metrics of recent turns.

    Runs the current policy with zero overheads, then:
    * sys_tools = median(real cached_tokens - simulated cached history) over cache-hit turns,
    * inject   = median(real context_tokens - simulated last prompt) - sys_tools,
    * output_scale = sum(real turn_tokens - simulated inputs) / sum(visible outputs).
    """
    base = simulate(turns, CurrentPolicy(), SimConfig())
    d_cached, d_ctx = [], []
    out_real = out_vis = 0
    for i, t in enumerate(turns):
        m = t.metrics
        if not m or t.start_ts < since_ts:
            continue
        if m.get("cached_tokens") and base.per_turn_cached[i]:
            d_cached.append(m["cached_tokens"] - base.per_turn_cached[i])
        if m.get("context_tokens"):
            d_ctx.append(m["context_tokens"] - base.per_turn_last_prompt[i])
    sys_tools = int(statistics.median(d_cached)) if d_cached else 0
    inject = max(0, int(statistics.median(d_ctx)) - sys_tools) if d_ctx else 0
    for i, t in enumerate(turns):
        m = t.metrics
        if not m or t.start_ts < since_ts or not m.get("turn_tokens"):
            continue
        n_calls = len(t.calls)
        sim_in = base.per_turn_input[i] + (sys_tools + inject) * n_calls
        out_real += max(0, m["turn_tokens"] - sim_in)
        out_vis += sum(c.visible_out for c in t.calls)
    scale = out_real / out_vis if out_vis else 1.0
    return SimConfig(sys_tools_tokens=sys_tools, inject_tokens=inject, output_scale=max(1.0, scale))


def validate(turns: list[Turn], res: Result, since_ts: float) -> dict[str, float]:
    """Compare simulated (current policy) per-turn prompt sizes against real metrics."""
    ratios_ctx, ratios_cached = [], []
    for i, t in enumerate(turns):
        m = t.metrics
        if not m or t.start_ts < since_ts:
            continue
        if m.get("context_tokens"):
            ratios_ctx.append(res.per_turn_last_prompt[i] / m["context_tokens"])
        if m.get("cached_tokens") and res.per_turn_cached[i]:
            ratios_cached.append(res.per_turn_cached[i] / m["cached_tokens"])
    return {
        "turns": len(ratios_ctx),
        "ctx_ratio_median": statistics.median(ratios_ctx) if ratios_ctx else float("nan"),
        "ctx_ratio_p10": statistics.quantiles(ratios_ctx, n=10)[0] if len(ratios_ctx) > 10 else float("nan"),
        "ctx_ratio_p90": statistics.quantiles(ratios_ctx, n=10)[-1] if len(ratios_ctx) > 10 else float("nan"),
        "cached_ratio_median": statistics.median(ratios_cached) if ratios_cached else float("nan"),
    }


def _pct(xs: list[int], q: float) -> int:
    """Return the q-quantile of a list."""
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))] if s else 0


def report(results: list[Result], prices: Prices) -> str:
    """Render a markdown comparison table."""
    lines = [
        "| metric | " + " | ".join(r.name for r in results) + " |",
        "|---|" + "---|" * len(results),
    ]

    def row(label: str, fn) -> None:
        lines.append(f"| {label} | " + " | ".join(fn(r) for r in results) + " |")

    row("compaction events", lambda r: f"{r.compaction_events}")
    row("L0 nodes / summary LLM calls", lambda r: f"{r.l0_nodes} / {r.summary_calls}")
    row("cache creations", lambda r: f"{r.cache_creations}")
    row("main LLM calls", lambda r: f"{r.llm_calls}")
    row("prompt tokens p50 / p95", lambda r: f"{_pct(r.prompt_sizes, .5):,} / {_pct(r.prompt_sizes, .95):,}")
    row("raw history p50 / min-after-compaction", lambda r: f"{_pct(r.raw_history_sizes, .5):,} / "
        f"{_pct(r.raw_history_sizes, .05):,} (p5)")
    row("final scaffold tokens", lambda r: f"{r.final_scaffold:,}")
    row("input uncached (M)", lambda r: f"{r.input_uncached / 1e6:.1f}")
    row("input cached (M)", lambda r: f"{r.input_cached / 1e6:.1f}")
    row("total input (M)", lambda r: f"{(r.input_uncached + r.input_cached) / 1e6:.1f}")
    row("output (M)", lambda r: f"{r.output / 1e6:.1f}")
    row("summarize in/out (M)", lambda r: f"{r.summary_in / 1e6:.2f} / {r.summary_out / 1e6:.2f}")
    row("cache create (M)", lambda r: f"{r.cache_create_tokens / 1e6:.1f}")
    for k in ["input", "cached_input", "output", "cache_create", "cache_storage", "summarize", "total"]:
        row(f"${k}", lambda r, k=k: f"{r.cost(prices)[k]:.2f}")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default="../band/kesoku.db")
    ap.add_argument("--session", default="de73ad46")
    ap.add_argument("--calib-since", default="2026-10-01", help="calibrate on turns after this date")
    args = ap.parse_args()

    import datetime

    since = datetime.datetime.fromisoformat(args.calib_since).timestamp()
    turns = load_turns(args.db, args.session)
    cfg = calibrate(turns, since)
    print(f"turns={len(turns)} calls={sum(len(t.calls) for t in turns)} "
          f"sys_tools={cfg.sys_tools_tokens} inject={cfg.inject_tokens} output_scale={cfg.output_scale:.2f}")

    cur = simulate(turns, CurrentPolicy(), cfg)
    print("validation (sim/real, current policy):", validate(turns, cur, since))

    policies: list[Policy] = [
        CurrentPolicy(),
        CurrentPolicy(name="current-tuned(80K,tail8)", base_node_min_tokens=80_000, protect_tail=8),
        WatermarkPolicy(name="watermark-128K/64K(no-refresh)", cache_refresh_tokens=None),
        WatermarkPolicy(name="watermark-128K/64K"),
        WatermarkPolicy(name="watermark-64K/32K", high=64_000, low=32_000, tail_min_tokens=16_000),
        WatermarkPolicy(name="watermark-256K/128K", high=256_000, low=128_000, tail_min_tokens=32_000),
    ]
    results = [simulate(turns, p, cfg) for p in policies]
    print(report(results, cfg.prices))


if __name__ == "__main__":
    main()
