from unittest.mock import AsyncMock, MagicMock

import pytest
from openlcm.core.dag import SummaryNode

from kesoku.agent.lcm_engine import KesokuLCMEngine
from kesoku.agent.llm import BaseLLM
from kesoku.context import KesokuContext


@pytest.mark.asyncio
async def test_filter_already_compacted():
    context = KesokuContext()

    # Mock LLM
    mock_llm = MagicMock(spec=BaseLLM)
    mock_llm.generate = AsyncMock()
    context.get_llm = MagicMock(return_value=mock_llm)

    engine = context.get_lcm_engine("test_session_filter")
    assert isinstance(engine, KesokuLCMEngine)

    # Ingest some messages
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Message 1"},
        {"role": "assistant", "content": "Response 1"},
        {"role": "user", "content": "Message 2"},
    ]

    engine._ingest_messages(messages)

    # Verify they are in DB and get their store_ids
    db_msgs = engine._store.get_session_messages("test_session_filter")
    assert len(db_msgs) == 4

    # Let's say we compacted up to "Response 1"
    # db_msgs[0] is System
    # db_msgs[1] is Message 1
    # db_msgs[2] is Response 1
    engine._last_compacted_store_id = db_msgs[2]["store_id"]

    # Now try to filter
    filtered = engine._filter_already_compacted(messages)

    # Expected: System prompt (kept), and Message 2.
    assert len(filtered) == 2
    assert filtered[0]["content"] == "System prompt"
    assert filtered[1]["content"] == "Message 2"


@pytest.mark.asyncio
async def test_assemble_context_includes_all_remaining():
    context = KesokuContext()
    engine = context.get_lcm_engine("test_session_assemble")

    # Configure fresh_tail_count = 2
    engine._config.fresh_tail_count = 2

    # Add an active node to DAG
    node = SummaryNode(
        session_id="test_session_assemble",
        depth=0,
        summary="Summary of 1-2",
        token_count=10,
        source_token_count=100,
        source_ids=[1, 2],
        source_type="messages",
        created_at=12345,
    )
    engine._dag.add_node(node)

    system_msg = {"role": "system", "content": "System Prompt"}
    remaining_messages = [
        {"role": "user", "content": "Raw 3"},
        {"role": "assistant", "content": "Raw 4"},
        {"role": "user", "content": "Raw 5"},
        {"role": "assistant", "content": "Raw 6"},
    ]

    # Call assemble
    result = engine._assemble_context(system_msg, remaining_messages)

    # Expected to include ALL remaining_messages, not just last 2 (fresh_tail)
    assert len(result) == 7
    assert result[0] == system_msg
    assert "Note: This conversation uses Lossless Context Management" in result[1]["content"]
    assert "Summary of 1-2" in result[1]["content"]
    assert result[2]["content"] == "Understood. I have access to the full conversation history through LCM tools."
    assert result[3]["content"] == "Raw 3"
    assert result[4]["content"] == "Raw 4"
    assert result[5]["content"] == "Raw 5"
    assert result[6]["content"] == "Raw 6"


@pytest.mark.asyncio
async def test_incremental_compress_flow():
    context = KesokuContext()

    # Mock LLM to return dummy summaries
    mock_llm = MagicMock(spec=BaseLLM)
    mock_res = MagicMock()
    mock_res.content = "Summary of chunk"
    mock_llm.generate = AsyncMock(return_value=mock_res)
    mock_llm.context_window_limit = 1000
    context.get_llm = MagicMock(return_value=mock_llm)

    engine = context.get_lcm_engine("test_session_compress", context_length=1000)

    # Configure LCM for quick triggering
    engine._config.leaf_chunk_tokens = 5
    engine._config.fresh_tail_count = 2
    engine._config.context_threshold = 0.1  # Low threshold

    # Turn 1 messages
    t1_messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Message 1"},
        {"role": "assistant", "content": "Response 1"},
        {"role": "user", "content": "Message 2"},
        {"role": "assistant", "content": "Response 2"},
    ]

    # Run compress Turn 1
    # This should trigger compaction because we have 5 messages,
    # and candidate_raw (Message 1, Response 1) will have enough tokens (>5).
    # Wait, candidate_raw is messages[1:3] -> Message 1, Response 1.
    # If they have > 5 tokens, it compacts them.
    compressed_t1 = await engine.compress(t1_messages)

    # Verify compaction happened
    assert engine.compression_count == 1
    # DAG should have 1 node (Summary of Message 1, Response 1)
    nodes = engine._dag.get_session_nodes("test_session_compress")
    assert len(nodes) == 1
    node1 = nodes[0]
    assert node1.depth == 0
    # source_ids should correspond to Message 1 (store_id 2) and Response 1 (store_id 3)
    # (System is 1)
    assert node1.source_ids == [2, 3]

    # Turn 2: We append new messages to the RAW history (as Kesoku does)
    # Raw history now has: System, M1, R1, M2, R2, M3, R3
    t2_messages = list(t1_messages) + [
        {"role": "user", "content": "Message 3"},
        {"role": "assistant", "content": "Response 3"},
    ]

    # Run compress Turn 2
    # If it was NOT incremental, it would re-compact [M1, R1] or [M1..R2].
    # With incremental fix, it should filter out [M1, R1] (already compacted, store_id <= 3).
    # Working messages will be: [System, M2, R2, M3, R3] (after filtering).
    # Candidate raw: [M2, R2] (since fresh tail is [M3, R3]).
    # Tokens of [M2, R2] > 5, so it compacts them -> Node 2 (covers M2, R2).
    # DAG should have 2 nodes, Node 2 should cover [4, 5] (M2 is 4, R2 is 5).
    # It should NOT cover [2, 3] again.
    compressed_t2 = await engine.compress(t2_messages)

    assert engine.compression_count == 2
    nodes = engine._dag.get_session_nodes("test_session_compress")
    assert len(nodes) == 2

    # Node 2 checks
    # Find the new node (it should have later created_at, or higher node_id)
    new_nodes = [n for n in nodes if n.node_id != node1.node_id]
    assert len(new_nodes) == 1
    node2 = new_nodes[0]
    assert node2.source_ids == [4, 5]  # Should cover M2 and R2, not M1/R1
