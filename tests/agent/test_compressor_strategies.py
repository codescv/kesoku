"""Tests for compression strategies (legacy chunk vs. watermark) and related helpers."""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from kesoku.agent.compressor import (
    DEFAULT_SUMMARY_MAX_CHARS,
    MAX_L0_SUMMARY_CHARS,
    HistoryCompressor,
    estimate_tokens_calibrated,
    summary_max_chars,
)
from kesoku.agent.llm import LLMResponse
from kesoku.config import AgentConfig, KesokuConfig
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message

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
    """DB mock with no existing summary nodes."""
    db = AsyncMock()
    db.get_root_summary_nodes.return_value = []
    return db


def test_estimate_tokens_calibrated() -> None:
    """Calibrated estimator: ~0.25 per ASCII char and ~0.9 per non-ASCII char."""
    assert estimate_tokens_calibrated("a" * 400) == 100
    assert estimate_tokens_calibrated("あ" * 100) == 90
    assert estimate_tokens_calibrated("") == 0
    assert estimate_tokens_calibrated(None) == 0


def test_summary_max_chars_scales_and_clamps() -> None:
    """Level-0 budget is ~8% of source tokens, clamped to [1200, 3000] chars."""
    assert summary_max_chars(1000) == DEFAULT_SUMMARY_MAX_CHARS
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
