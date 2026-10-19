"""Tests for compression strategies (legacy chunk vs. watermark), forced compaction and related helpers."""

import asyncio
import time
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from kesoku.agent import compressor as compressor_module
from kesoku.agent.compressor import (
    DEFAULT_SUMMARY_MAX_CHARS,
    MAX_L0_SUMMARY_CHARS,
    MIN_L0_SUMMARY_CHARS,
    RECENT_USER_MESSAGE_OMISSION,
    HistoryCompressor,
    estimate_tokens_calibrated,
    summary_max_chars,
    water_fill_allocation,
)
from kesoku.agent.llm import LLMResponse
from kesoku.config import AgentConfig, KesokuConfig
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message, SummaryNode


@pytest.fixture(autouse=True)
def _reset_compaction_state() -> Iterator[None]:
    """Tests reuse session/message ids, so clear the module-level lock and link caches around each test."""
    compressor_module._session_locks.clear()
    compressor_module._session_links.clear()
    yield
    compressor_module._session_locks.clear()
    compressor_module._session_links.clear()


SUMMARY_JSON = '{"timeline": ["2026-01-01 18:00 - 18:30: did things"], "learnings": null}'


def make_history(n_turns: int, reply_chars: int) -> list[Message]:
    """Build n user/assistant turns with fixed-size ASCII replies."""
    msgs: list[Message] = []
    base = time.time() - 10_000
    for i in range(n_turns):
        for role, content in [(MessageRole.USER, f"user {i}"), (MessageRole.ASSISTANT, "x" * reply_chars)]:
            msgs.append(
                Message(
                    id=f"m{i}_{role}",
                    session_id="s",
                    chatbot_id="cb",
                    channel_id="ch",
                    sender=str(role),
                    role=role,
                    type=MessageType.TEXT,
                    content=content,
                    status=MessageStatus.RESPONDED,
                    timestamp=base + i * 10 + (1 if role == MessageRole.ASSISTANT else 0),
                )
            )
    return msgs


def make_llm() -> AsyncMock:
    """LLM mock returning a fixed summary and the legacy fallback estimator."""
    llm = AsyncMock()
    llm.generate.return_value = LLMResponse(content=SUMMARY_JSON)
    llm.estimate_tokens_fallback = MagicMock(side_effect=lambda prompt=None, **_: max(1, len(prompt or "") // 4))
    return llm


def make_db() -> AsyncMock:
    """DB mock with no existing summary nodes; get_all_summary_nodes returns the nodes inserted so far."""
    db = AsyncMock()
    db.get_root_summary_nodes.return_value = []
    db.get_all_summary_nodes.side_effect = lambda session_id: [c[0][0] for c in db.insert_summary_node.call_args_list]
    return db


def test_estimate_tokens_calibrated() -> None:
    """Calibrated estimator: ~0.25 per ASCII char and ~0.9 per non-ASCII char."""
    assert estimate_tokens_calibrated("a" * 400) == 100
    assert estimate_tokens_calibrated("あ" * 100) == 90
    assert estimate_tokens_calibrated("") == 0
    assert estimate_tokens_calibrated(None) == 0


def test_summary_max_chars_scales_and_clamps() -> None:
    """Level-0 budget is ~8% of source tokens, clamped to [1500, 3000] chars."""
    assert MIN_L0_SUMMARY_CHARS == 1500
    assert summary_max_chars(1000) == MIN_L0_SUMMARY_CHARS
    assert summary_max_chars(24000) == 1920
    assert summary_max_chars(10**6) == MAX_L0_SUMMARY_CHARS


def test_watermark_config_validation() -> None:
    """Invalid watermark settings are rejected only when the watermark strategy is selected."""
    with pytest.raises(ValueError):
        AgentConfig(compression_strategy="watermark", context_high_watermark=10000, context_low_watermark=20000)
    with pytest.raises(ValueError):
        AgentConfig(compression_strategy="watermark", context_low_watermark=10000, protect_tail_min_tokens=12000)
    # Same values are fine for the legacy strategy (fields unused).
    AgentConfig(compression_strategy="chunk", context_high_watermark=10000, context_low_watermark=20000)


@pytest.mark.asyncio
async def test_chunk_strategy_is_default_and_unchanged() -> None:
    """Default strategy is the legacy chunk algorithm (4 turns and 25K fallback-estimated tokens per node)."""
    cfg = KesokuConfig()
    assert cfg.agent.compression_strategy == "chunk"
    db, llm = make_db(), make_llm()
    # 12 turns x 8000 chars (~2000 fallback tokens): never reaches 25K in one 4+ turn chunk -> no compaction.
    assert not await HistoryCompressor(db).auto_compact_session("s", make_history(12, 8000), llm, cfg)
    # 12 turns x 40000 chars (~10000 tokens): chunks close at 4 turns.
    db, llm = make_db(), make_llm()
    assert await HistoryCompressor(db).auto_compact_session("s", make_history(12, 40000), llm, cfg)
    prompt = llm.generate.call_args_list[0][1]["prompt"]
    assert f"< {DEFAULT_SUMMARY_MAX_CHARS} chars" in prompt


@pytest.mark.asyncio
async def test_watermark_no_compaction_below_high() -> None:
    """Below the high watermark, the watermark strategy never compacts."""
    cfg = KesokuConfig()
    cfg.agent.compression_strategy = "watermark"
    db, llm = make_db(), make_llm()
    # 20 turns x 4000 chars = ~20K calibrated tokens < 64K
    assert not await HistoryCompressor(db).auto_compact_session("s", make_history(20, 4000), llm, cfg)
    llm.generate.assert_not_called()


@pytest.mark.asyncio
async def test_watermark_compacts_down_to_low_and_protects_tail() -> None:
    """Above high, oldest turns are compacted until the estimate is below low; the tail stays raw."""
    cfg = KesokuConfig()
    cfg.agent.compression_strategy = "watermark"
    db, llm = make_db(), make_llm()
    history = make_history(40, 8000)  # ~2000 tokens/turn, ~80K total
    compacted = await HistoryCompressor(db).auto_compact_session("s", history, llm, cfg, overhead_tokens=1000)
    assert compacted
    compressed_turns = {m.id.split("_")[0] for m in history if m.summary_node_id}
    remaining = 40 - len(compressed_turns)
    # Remaining raw history must be <= low watermark and >= protected tail (12K tokens => >= 6 turns).
    assert remaining * 2000 + 1000 <= cfg.agent.context_low_watermark + 2000
    assert remaining >= 6
    # The front turn is protected.
    assert "m0" not in compressed_turns
    # Chunks target ~24K tokens => ~12 turns per Level-0 node.
    first_chunk_prompt = llm.generate.call_args_list[0][1]["prompt"]
    assert "user 1\n" in first_chunk_prompt and "user 12\n" in first_chunk_prompt


@pytest.mark.asyncio
async def test_watermark_does_not_thrash_on_small_candidates() -> None:
    """If less than one chunk of tokens sits outside the tail, nothing is compacted even above high."""
    cfg = KesokuConfig()
    cfg.agent.compression_strategy = "watermark"
    db, llm = make_db(), make_llm()
    # Huge fixed overhead pushes the total over high, but only a few small turns exist.
    history = make_history(8, 4000)
    assert not await HistoryCompressor(db).auto_compact_session("s", history, llm, cfg, overhead_tokens=100_000)
    llm.generate.assert_not_called()


def test_recent_user_messages_block() -> None:
    """Only user messages from compressed turns are kept, most recent `limit`, oldest first."""
    history = make_history(5, 10)
    compressor = HistoryCompressor(make_db())
    turns = compressor.segment_turns(history)
    for t in turns[:3]:
        for m in t:
            m.summary_node_id = "node"
    block = HistoryCompressor.build_recent_user_messages_block(turns, limit=2)
    assert block.startswith("<recent_user_messages>")
    assert "user 0" not in block
    assert block.index("user 1") < block.index("user 2")
    assert "user 3" not in block  # not compressed
    assert HistoryCompressor.build_recent_user_messages_block(turns, limit=0) == ""



# --- P1-2: water-filling ---------------------------------------------------------------------------------


def test_water_fill_all_fit() -> None:
    """Everything fits: every item gets its full length."""
    assert water_fill_allocation([10, 20, 30], 100) == [10, 20, 30]


def test_water_fill_one_long_many_short() -> None:
    """Short items stay intact; the long one gets the pool minus the short ones."""
    assert water_fill_allocation([100, 100, 100, 5000], 2000) == [100, 100, 100, 1700]


def test_water_fill_all_long_remainder_to_latest() -> None:
    """All items too long: equal shares, the division remainder goes to the latest items."""
    assert water_fill_allocation([500, 500, 500], 1000) == [333, 333, 334]
    assert water_fill_allocation([900, 50, 900, 900], 1052) == [334, 50, 334, 334]


def test_water_fill_edge_cases() -> None:
    """Empty input, zero/negative budget and negative lengths are handled."""
    assert water_fill_allocation([], 1000) == []
    assert water_fill_allocation([10, 20], 0) == [0, 0]
    assert water_fill_allocation([10, 20], -5) == [0, 0]
    assert water_fill_allocation([-3, 10], 100) == [0, 10]


def test_water_fill_c2_worked_example() -> None:
    """Spec worked example: spec 2890, paste (ceiling-capped) 6000, two short messages; pool 12000."""
    assert water_fill_allocation([2890, 6000, 100, 150], 12000) == [2890, 6000, 100, 150]
    # Never exceeds the pool and never exceeds a length.
    alloc = water_fill_allocation([7000, 7000, 7000], 12000)
    assert sum(alloc) == 12000 and all(a <= 7000 for a in alloc)


def _compressed_turns_with_user_texts(texts: list[str]) -> list[list[Message]]:
    """One compressed turn per user text (user message + short assistant reply)."""
    base = time.time() - 10_000
    turns: list[list[Message]] = []
    for i, text in enumerate(texts):
        turn = []
        for role, content in [(MessageRole.USER, text), (MessageRole.ASSISTANT, "ok")]:
            turn.append(
                Message(
                    id=f"u{i}_{role}",
                    session_id="s",
                    chatbot_id="cb",
                    channel_id="ch",
                    sender=str(role),
                    role=role,
                    type=MessageType.TEXT,
                    content=content,
                    status=MessageStatus.RESPONDED,
                    timestamp=base + i * 10,
                    summary_node_id="node",
                )
            )
        turns.append(turn)
    return turns


def _block_entries(block: str) -> list[str]:
    """Split a rendered block into its per-message texts (timestamp prefix removed)."""
    body = block.split("-->\n", 1)[1].rsplit("\n</recent_user_messages>", 1)[0]
    entries: list[str] = []
    for line in body.split("\n"):
        if line.startswith("[") and "] " in line[:20]:
            entries.append(line.split("] ", 1)[1])
        else:
            entries[-1] += "\n" + line
    return entries


def test_recent_user_messages_c2_shape() -> None:
    """C2: the 2.9K spec stays intact; the 11K paste is middle-cut with both ends kept."""
    spec = "SPEC_START\n" + "\n".join(f"rule {i}: " + "r" * 90 for i in range(28)) + "\nrule 7: E_DOM_CORE_007 END"
    assert 2800 < len(spec) < 3000
    paste = "PASTE_HEAD\n" + "\n".join("p" * 99 for _ in range(112)) + "\nPASTE_TAIL what is wrong here?"
    assert len(paste) > 11000
    turns = _compressed_turns_with_user_texts([spec, paste, "short one", "another short message"])
    block = HistoryCompressor.build_recent_user_messages_block(turns, limit=100)
    entries = _block_entries(block)
    assert entries[0] == spec
    assert entries[2:] == ["short one", "another short message"]
    cut = entries[1]
    assert cut.startswith("PASTE_HEAD") and cut.endswith("PASTE_TAIL what is wrong here?")
    assert RECENT_USER_MESSAGE_OMISSION.strip() in cut
    assert len(cut) <= 6000
    assert "(long messages may be shortened in the middle)" in block


def test_recent_user_messages_ceiling_applies_with_spare_budget() -> None:
    """A single long message is capped at the ceiling even if the pool has room."""
    turns = _compressed_turns_with_user_texts(["HEAD\n" + "x\n" * 4000 + "TAIL"])
    block = HistoryCompressor.build_recent_user_messages_block(turns, 100, total_chars=12000, ceiling_chars=6000)
    (entry,) = _block_entries(block)
    assert len(entry) <= 6000 and entry.startswith("HEAD") and entry.endswith("TAIL")
    big = HistoryCompressor.build_recent_user_messages_block(turns, 100, total_chars=12000, ceiling_chars=12000)
    assert _block_entries(big)[0] == "HEAD\n" + "x\n" * 4000 + "TAIL"


def test_recent_user_messages_limit_bounds_count_and_unchanged_when_fitting() -> None:
    """`limit` still bounds the count; short messages render verbatim, exactly as before."""
    texts = [f"message {i}" for i in range(10)]
    turns = _compressed_turns_with_user_texts(texts)
    block = HistoryCompressor.build_recent_user_messages_block(turns, limit=3)
    assert _block_entries(block) == ["message 7", "message 8", "message 9"]
    assert "…" not in block


def _watermark_history_with_big_compressed_user_messages() -> list[Message]:
    """Front turn, 3 compressed turns with 20K-char user messages, then 10 raw turns of ~500 tokens."""
    history: list[Message] = []
    base = time.time() - 10_000
    for i in range(14):
        compressed = 1 <= i <= 3
        user_text = ("Q" * 20000) if compressed else f"user {i}"
        reply = "ok" if compressed else "x" * 2000
        for role, content in [(MessageRole.USER, user_text), (MessageRole.ASSISTANT, reply)]:
            history.append(
                Message(
                    id=f"w{i}_{role}",
                    session_id="s",
                    chatbot_id="cb",
                    channel_id="ch",
                    sender=str(role),
                    role=role,
                    type=MessageType.TEXT,
                    content=content,
                    status=MessageStatus.RESPONDED,
                    timestamp=base + i * 10 + (1 if role == MessageRole.ASSISTANT else 0),
                    summary_node_id="old" if compressed else None,
                )
            )
    return history


@pytest.mark.asyncio
async def test_watermark_estimate_uses_recent_user_message_pool() -> None:
    """The watermark estimate includes the water-filled block, so the pool size changes the decision."""
    results = {}
    for pool, ceiling in [(12000, 6000), (600, 200)]:
        cfg = KesokuConfig()
        agent = cfg.agent
        agent.compression_strategy = "watermark"
        agent.context_high_watermark = 7000
        agent.context_low_watermark = 3000
        agent.protect_tail_min_tokens = 500
        agent.protect_front_turns = 1
        agent.protect_tail_turns = 1
        agent.watermark_chunk_tokens = 1000
        agent.recent_user_messages_total_chars = pool
        agent.recent_user_message_ceiling_chars = ceiling
        db, llm = make_db(), make_llm()
        history = _watermark_history_with_big_compressed_user_messages()
        results[pool] = await HistoryCompressor(db).auto_compact_session("s", history, llm, cfg)
    # ~5K raw tokens + ~3K block tokens > 7K with the big pool; ~5.2K with the small pool.
    assert results == {12000: True, 600: False}


# --- P1-4: forced compaction ---------------------------------------------------------------------------------


def _force_cfg() -> KesokuConfig:
    """Config with 1 protected head turn and 3 protected tail turns."""
    cfg = KesokuConfig()
    cfg.agent.protect_front_turns = 1
    cfg.agent.protect_tail_turns = 3
    return cfg


def _compressed_ids(history: list[Message]) -> set[str]:
    """Turn ids (e.g. "m3") whose messages carry a summary node."""
    return {m.id.split("_")[0] for m in history if m.summary_node_id}


@pytest.mark.asyncio
async def test_force_compact_compresses_exactly_middle_turns() -> None:
    """All middle turns are summarised regardless of the watermark; head and tail stay raw."""
    db, llm = make_db(), make_llm()
    history = make_history(10, 100)  # tiny: far below any watermark
    nodes = await HistoryCompressor(db).force_compact_session("s", history, llm, _force_cfg())
    assert nodes == 1
    assert _compressed_ids(history) == {f"m{i}" for i in range(1, 7)}
    prompt = llm.generate.call_args_list[0][1]["prompt"]
    assert "user 1\n" in prompt and "user 6\n" in prompt
    assert "user 0\n" not in prompt and "user 7\n" not in prompt
    assert "Compaction focus" not in prompt
    db.insert_summary_node.assert_called_once()
    db.update_messages_summary_node.assert_called_once()


@pytest.mark.asyncio
async def test_force_compact_chunks_and_skips_already_compressed() -> None:
    """Already-compressed middle turns are skipped; chunking follows watermark_chunk_tokens/min_turns."""
    cfg = _force_cfg()
    cfg.agent.watermark_chunk_tokens = 1
    cfg.agent.watermark_chunk_min_turns = 2
    db, llm = make_db(), make_llm()
    history = make_history(12, 100)
    for m in history:
        if m.id.split("_")[0] in ("m1", "m2"):
            m.summary_node_id = "old"
    nodes = await HistoryCompressor(db).force_compact_session("s", history, llm, cfg)
    # Middle turns m3..m8 (6 turns) in chunks of 2.
    assert nodes == 3
    assert _compressed_ids(history) == {f"m{i}" for i in range(1, 9)}
    assert all(m.summary_node_id == "old" for m in history if m.id.split("_")[0] in ("m1", "m2"))


@pytest.mark.asyncio
async def test_force_compact_returns_zero_when_nothing_compressible() -> None:
    """Only head + tail turns (or all middle turns compressed): nothing to do, no LLM call."""
    db, llm = make_db(), make_llm()
    assert await HistoryCompressor(db).force_compact_session("s", make_history(4, 100), llm, _force_cfg()) == 0
    history = make_history(6, 100)
    for m in history:
        if m.id.split("_")[0] in ("m1", "m2"):
            m.summary_node_id = "old"
    assert await HistoryCompressor(db).force_compact_session("s", history, llm, _force_cfg()) == 0
    llm.generate.assert_not_called()
    db.insert_summary_node.assert_not_called()


@pytest.mark.asyncio
async def test_force_compact_zero_tail() -> None:
    """protect_tail_turns = 0 compresses everything after the head."""
    cfg = _force_cfg()
    cfg.agent.protect_tail_turns = 0
    db, llm = make_db(), make_llm()
    history = make_history(5, 100)
    assert await HistoryCompressor(db).force_compact_session("s", history, llm, cfg) == 1
    assert _compressed_ids(history) == {"m1", "m2", "m3", "m4"}


@pytest.mark.asyncio
async def test_force_compact_directive_reaches_prompt_sanitised() -> None:
    """The directive is stripped, whitespace-collapsed, kept verbatim with braces, and steers the L0 prompt."""
    db, llm = make_db(), make_llm()
    directive = "  keep only\n\n the   {API} decisions;\tdrop subsystem X  "
    await HistoryCompressor(db).force_compact_session(
        "s", make_history(8, 100), llm, _force_cfg(), directive=directive
    )
    prompt = llm.generate.call_args_list[0][1]["prompt"]
    assert 'Compaction focus requested by the user: "keep only the {API} decisions; drop subsystem X".' in prompt
    assert "never omit dead_ends or pending_state.constraints" in prompt


@pytest.mark.asyncio
async def test_force_compact_directive_capped_and_blank_ignored() -> None:
    """Directives are capped at 500 chars; a blank directive behaves like none."""
    db, llm = make_db(), make_llm()
    await HistoryCompressor(db).force_compact_session(
        "s", make_history(8, 100), llm, _force_cfg(), directive="a" * 2000
    )
    prompt = llm.generate.call_args_list[0][1]["prompt"]
    assert f'"{"a" * 500}"' in prompt and "a" * 501 not in prompt
    db, llm = make_db(), make_llm()
    await HistoryCompressor(db).force_compact_session("s", make_history(8, 100), llm, _force_cfg(), directive=" \n ")
    assert "Compaction focus" not in llm.generate.call_args_list[0][1]["prompt"]


@pytest.mark.asyncio
async def test_force_compact_directive_reaches_consolidation() -> None:
    """Consolidation after a forced compaction gets the ordering-only focus line."""
    db, llm = make_db(), make_llm()
    roots = [
        SummaryNode(
            id=f"n{i}", session_id="s", level=0, summary=f"Timeline:\n- t{i}",
            start_timestamp=1000 + i, end_timestamp=1001 + i, token_count=10, source_token_count=100,
        )
        for i in range(8)
    ]

    async def roots_for(session_id: str, level: int | None = None) -> list[SummaryNode]:
        return roots if level == 0 else []

    db.get_root_summary_nodes.side_effect = roots_for
    await HistoryCompressor(db).force_compact_session(
        "s", make_history(8, 100), llm, _force_cfg(), directive="focus on Y"
    )
    consolidate_prompt = llm.generate.call_args_list[-1][1]["prompt"]
    assert "Summaries to Merge" in consolidate_prompt
    assert 'Compaction focus requested by the user: "focus on Y". Use the focus only to order and emphasise' in (
        consolidate_prompt
    )


@pytest.mark.asyncio
async def test_concurrent_compactions_produce_one_node_set() -> None:
    """Two concurrent compactions of one session (separately loaded histories) summarise the turns once."""
    db = make_db()
    llm = make_llm()

    async def slow_generate(prompt: str) -> LLMResponse:
        await asyncio.sleep(0.01)  # yield so the second compaction runs while the first is mid-summary
        return LLMResponse(content=SUMMARY_JSON)

    llm.generate.side_effect = slow_generate
    compressor = HistoryCompressor(db)
    first, second = make_history(10, 100), make_history(10, 100)
    results = await asyncio.gather(
        compressor.force_compact_session("s", first, llm, _force_cfg()),
        compressor.force_compact_session("s", second, llm, _force_cfg()),
    )
    assert sorted(results) == [0, 1]
    assert db.insert_summary_node.call_count == 1
    # The losing caller's history was brought up to date with the winner's node.
    assert _compressed_ids(second) == _compressed_ids(first) == {f"m{i}" for i in range(1, 7)}


@pytest.mark.asyncio
async def test_auto_and_forced_compaction_serialised() -> None:
    """An auto (chunk) compaction waiting on a forced one does not re-summarise the same turns."""
    db = make_db()
    llm = make_llm()

    async def slow_generate(prompt: str) -> LLMResponse:
        await asyncio.sleep(0.01)
        return LLMResponse(content=SUMMARY_JSON)

    llm.generate.side_effect = slow_generate
    cfg = _force_cfg()
    cfg.agent.base_node_turns = 1
    cfg.agent.base_node_min_tokens = 1
    compressor = HistoryCompressor(db)
    forced, auto = await asyncio.gather(
        compressor.force_compact_session("s", make_history(10, 100), llm, cfg),
        compressor.auto_compact_session("s", make_history(10, 100), llm, cfg),
    )
    assert forced == 1 and auto is False
    assert db.insert_summary_node.call_count == 1


@pytest.mark.asyncio
async def test_consolidate_forest_public_call_takes_lock() -> None:
    """consolidate_forest waits for a running compaction of the same session (lock is shared, no deadlock)."""
    db, llm = make_db(), make_llm()
    lock = compressor_module._get_session_lock("s")
    await lock.acquire()
    task = asyncio.create_task(HistoryCompressor(db).consolidate_forest("s", llm, K=4))
    await asyncio.sleep(0.01)
    assert not task.done()
    lock.release()
    await asyncio.wait_for(task, 1)
    db.get_root_summary_nodes.assert_awaited()



@pytest.mark.asyncio
async def test_known_links_ignored_when_node_no_longer_exists() -> None:
    """Links to summary nodes that were deleted are not applied (turns must not vanish from the history)."""
    db, llm = make_db(), make_llm()
    compressor = HistoryCompressor(db)
    assert await compressor.force_compact_session("s", make_history(8, 100), llm, _force_cfg()) == 1
    # Same session with the forest deleted in the DB: the same turns are compressible again.
    db.get_all_summary_nodes.side_effect = lambda session_id: []
    history = make_history(8, 100)
    assert await compressor.force_compact_session("s", history, llm, _force_cfg()) == 1
    assert _compressed_ids(history) == {"m1", "m2", "m3", "m4"}
