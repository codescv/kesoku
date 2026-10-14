"""Tests for the offline compression-policy simulator (scripts/compression_sim.py)."""

import os

import pytest

from scripts.compression_sim import (
    Call,
    CurrentPolicy,
    Prices,
    Result,
    SimConfig,
    Turn,
    WatermarkPolicy,
    fallback_est_tokens,
    load_turns,
    real_tokens,
    simulate,
)

BAND_DB = os.path.join(os.path.dirname(__file__), "..", "..", "..", "band", "kesoku.db")


def make_turns(n: int, prepared: int, est: int | None = None, gap: float = 60.0, calls: int = 2) -> list[Turn]:
    """Build n synthetic turns of a fixed prepared size, spaced `gap` seconds apart."""
    return [
        Turn(
            start_ts=i * gap,
            end_ts=i * gap + 10,
            prepared_real=prepared,
            prepared_est=est if est is not None else prepared,
            summary_src_real=prepared * 2,
            calls=[Call(partial_in=100 * (j + 1), visible_out=50) for j in range(calls)],
            metrics=None,
        )
        for i in range(n)
    ]


def test_token_estimators() -> None:
    """Fallback counts non-ASCII as 2 tokens; calibrated model is much lower for CJK."""
    assert fallback_est_tokens("abcd" * 100) == 100
    assert fallback_est_tokens("あ" * 100) == 200
    assert real_tokens("あ" * 100) == 86
    assert real_tokens("abcd" * 100) == 96


def test_current_policy_needs_both_turns_and_tokens() -> None:
    """A chunk is cut only when it has >= base_node_turns AND >= base_node_min_tokens."""
    # 10K est per turn: chunk closes at 4 turns (40K >= 25K).
    res = simulate(make_turns(12, 10_000), CurrentPolicy(), SimConfig())
    assert res.l0_nodes >= 1
    # 1K est per turn: never reaches 25K within 12 turns -> no compaction.
    res = simulate(make_turns(12, 1_000), CurrentPolicy(), SimConfig())
    assert res.l0_nodes == 0
    assert res.compaction_events == 0


def test_current_policy_protects_tail_turns() -> None:
    """The last protect_tail turns (incl. the current one) are never compressed."""
    res = simulate(make_turns(30, 10_000), CurrentPolicy(), SimConfig())
    # Raw history never drops below the protected completed turns (2 turns before current).
    assert min(res.raw_history_sizes[-10:]) >= 2 * 10_000


def test_watermark_does_not_compact_below_high() -> None:
    """With the whole session below the high watermark, nothing is compressed."""
    res = simulate(make_turns(20, 2_000), WatermarkPolicy(high=100_000, low=50_000), SimConfig())
    assert res.compaction_events == 0
    assert res.summary_calls == 0


def test_watermark_compacts_to_low() -> None:
    """Once above high, history is compacted to roughly the low watermark in one event."""
    pol = WatermarkPolicy(high=60_000, low=30_000, tail_min_tokens=10_000, chunk_tokens=10_000)
    res = simulate(make_turns(40, 5_000), pol, SimConfig())
    assert res.compaction_events >= 1
    assert max(res.prompt_sizes) <= 60_000 + 5_000 + 1_000  # at most one turn overshoot
    # Fewer events than turns compressed: compaction is batched.
    assert res.compaction_events < res.l0_nodes


def test_cache_invalidated_by_compaction_and_expiry() -> None:
    """Cache is recreated after compaction and after TTL expiry."""
    cfg = SimConfig(cache_min_tokens=1)
    # No compaction, gaps shorter than TTL -> one cache for the whole session.
    res = simulate(make_turns(10, 1_000, gap=60), CurrentPolicy(base_node_min_tokens=10**9), cfg)
    assert res.cache_creations == 1
    # Gaps longer than TTL -> a new cache every turn after the first.
    res = simulate(make_turns(10, 1_000, gap=4000), CurrentPolicy(base_node_min_tokens=10**9), cfg)
    assert res.cache_creations == 9


def test_cache_refresh_reduces_uncached_input() -> None:
    """Refreshing the cache moves suffix tokens from uncached to cached input."""
    cfg = SimConfig(cache_min_tokens=1)
    turns = make_turns(30, 3_000, gap=30)
    no = simulate(turns, WatermarkPolicy(high=10**9, cache_refresh_tokens=None), cfg)
    yes = simulate(turns, WatermarkPolicy(high=10**9, cache_refresh_tokens=6_000), cfg)
    assert yes.input_uncached < no.input_uncached
    assert yes.input_uncached + yes.input_cached == no.input_uncached + no.input_cached
    assert yes.cache_creations > no.cache_creations


def test_cost_breakdown() -> None:
    """Cost uses per-1M prices for each bucket."""
    r = Result(name="x", input_uncached=1_000_000, input_cached=1_000_000, output=1_000_000,
               cache_create_tokens=1_000_000, cache_storage_token_hours=1_000_000)
    c = r.cost(Prices(input=1, output=2, cached_input=0.1, cache_storage_per_hour=0.5))
    assert c == pytest.approx(
        {"input": 1, "cached_input": 0.1, "output": 2, "cache_create": 1, "cache_storage": 0.5,
         "summarize": 0, "total": 4.6}
    )


@pytest.mark.skipif(not os.path.exists(BAND_DB), reason="band/kesoku.db not available")
def test_real_session_de73ad46_comparison() -> None:
    """Replay asuka's de73ad46 session: watermark compacts far less often but costs more."""
    turns = load_turns(BAND_DB, "de73ad46")
    if not turns:
        pytest.skip("session de73ad46 not found")
    cfg = SimConfig(sys_tools_tokens=4797, inject_tokens=762, output_scale=2.52)
    cur = simulate(turns, CurrentPolicy(), cfg)
    new = simulate(turns, WatermarkPolicy(high=64_000, low=24_000, cache_refresh_tokens=8_000), cfg)
    assert new.compaction_events < cur.compaction_events
    assert sorted(new.raw_history_sizes)[len(new.raw_history_sizes) // 2] > sorted(cur.raw_history_sizes)[
        len(cur.raw_history_sizes) // 2
    ]
    assert new.output == cur.output  # outputs are identical by assumption
