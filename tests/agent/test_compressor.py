"""Unit tests for HistoryCompressor in kesoku.agent.compressor."""

import os
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from kesoku.agent.compressor import (
    HistoryCompressor,
)
from kesoku.agent.llm import LLMResponse
from kesoku.config import KesokuConfig
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message, SummaryNode


def test_parse_summary_json_valid_markdown():
    """Test parsing a valid JSON block inside markdown fence."""
    raw = """Here is the json:
```json
{
  "timeline": ["2026-01-01 18:00: Started task"],
  "key_decisions_and_changes": null,
  "pitfalls": "None",
  "files": ["/repo/src/main.py"]
}
```"""
    data = HistoryCompressor.parse_summary_json(raw)
    assert data["timeline"] == ["2026-01-01 18:00: Started task"]
    assert data["key_decisions_and_changes"] is None
    assert data["pitfalls"] == "None"
    assert data["files"] == ["/repo/src/main.py"]


def test_parse_summary_json_invalid():
    """Test parsing invalid JSON returns empty dict gracefully."""
    raw = "Not a json response at all"
    data = HistoryCompressor.parse_summary_json(raw)
    assert data == {}


def test_is_outside_staging_dir(tmp_path):
    """Test STAGING_DIR file path checking."""
    staging_dir = str(tmp_path / "staging")
    os.makedirs(staging_dir, exist_ok=True)

    # File inside staging dir
    inside_file = os.path.join(staging_dir, "output.png")
    assert HistoryCompressor.is_outside_staging_dir(inside_file, staging_dir) is False

    # Relative sessions path (e.g. sessions/7c277e98/asuka_bbq_chef_bird_1823.png)
    assert (
        HistoryCompressor.is_outside_staging_dir(
            "sessions/7c277e98/asuka_bbq_chef_bird_1823.png", staging_dir
        )
        is False
    )

    # File containing STAGING_DIR substring
    assert HistoryCompressor.is_outside_staging_dir("$STAGING_DIR/test.log", staging_dir) is False

    # File outside staging dir
    outside_file = "/usr/local/repo/src/main.py"
    assert HistoryCompressor.is_outside_staging_dir(outside_file, staging_dir) is True

    # When staging_dir is None, regular paths are considered outside
    assert HistoryCompressor.is_outside_staging_dir("/any/path/foo.py", None) is True


def test_format_field():
    """Test field formatting for None, empty, strings, and lists."""
    assert HistoryCompressor._format_field(None, default_none=True) == "None"
    assert HistoryCompressor._format_field(None, default_none=False) == ""
    assert HistoryCompressor._format_field("None", default_none=True) == "None"
    assert HistoryCompressor._format_field("[]", default_none=True) == "None"

    items = ["2026-01-01 18:00: Event A", "2026-01-01 18:05: Event B"]
    formatted = HistoryCompressor._format_field(items)
    assert "- 2026-01-01 18:00: Event A" in formatted
    assert "- 2026-01-01 18:05: Event B" in formatted


def test_format_summary_filtering_staging_files(tmp_path):
    """Test complete format_summary pipeline including STAGING_DIR filtering and template assembly."""
    staging_dir = str(tmp_path / "staging")
    os.makedirs(staging_dir, exist_ok=True)

    json_input = f"""```json
{{
  "timeline": ["2026-01-01 18:00: User requested feature X"],
  "key_decisions_and_changes": "None",
  "pitfalls": null,
  "files": [
    "/repo/src/kesoku/agent/compressor.py",
    "{staging_dir}/output.log"
  ]
}}
```"""

    result = HistoryCompressor.format_summary(json_input, staging_dir=staging_dir)

    assert "Timeline:\n- 2026-01-01 18:00: User requested feature X" in result
    assert "Key decisions and changes:\nNone" in result
    assert "Pitfalls:\nNone" in result
    assert "/repo/src/kesoku/agent/compressor.py" in result
    assert "output.log" not in result
    assert result.startswith("Timeline:")


def test_format_summary_all_none():
    """Test format_summary when optional sections are missing or empty."""
    json_input = """{
  "timeline": ["2026-01-01 18:00: Checked system status."],
  "key_decisions_and_changes": [],
  "pitfalls": "null",
  "files": []
}"""
    result = HistoryCompressor.format_summary(json_input)
    assert "Key decisions and changes:\nNone" in result
    assert "Pitfalls:\nNone" in result
    assert "Files:\nNone" in result


@pytest.mark.asyncio
async def test_auto_compact_session(tmp_path):
    """Test auto_compact_session uses updated prompt and formats summary using template."""
    db_mock = AsyncMock()
    llm_mock = AsyncMock()
    llm_mock.estimate_tokens_fallback = MagicMock(return_value=10000)

    json_reply = """{
      "timeline": ["2026-01-01 18:00: Started task"],
      "key_decisions_and_changes": "None",
      "pitfalls": "None",
      "files": ["/repo/file.py"]
    }"""
    llm_mock.generate.return_value = LLMResponse(content=json_reply)

    compressor = HistoryCompressor(db_mock)
    cfg = KesokuConfig()
    cfg.agent.protect_front_turns = 1
    cfg.agent.protect_tail_turns = 1
    cfg.agent.base_node_turns = 2
    cfg.agent.base_node_min_tokens = 1000

    # Build 4 turns (1 front protect, 2 candidates, 1 tail protect)
    messages = []
    for idx in range(4):
        msg = Message(
            id=f"msg_{idx}",
            session_id="sess_1",
            chatbot_id="cb",
            channel_id="ch",
            sender="user",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content=f"Turn {idx}",
            status=MessageStatus.RESPONDED,
            timestamp=time.time() + idx,
        )
        messages.append(msg)

    db_mock.get_root_summary_nodes.return_value = []

    compacted = await compressor.auto_compact_session(
        session_id="sess_1",
        history=messages,
        llm=llm_mock,
        config=cfg,
        staging_dir=str(tmp_path),
    )

    assert compacted is True
    llm_mock.generate.assert_called_once()
    call_prompt = llm_mock.generate.call_args[1]["prompt"]
    assert "2026-01-01 18:00" in call_prompt
    assert "STAGING_DIR" in call_prompt

    db_mock.insert_summary_node.assert_called_once()
    inserted_node: SummaryNode = db_mock.insert_summary_node.call_args[0][0]
    assert "Timeline:\n- 2026-01-01 18:00: Started task" in inserted_node.summary
    assert "Files:\n- /repo/file.py" in inserted_node.summary


@pytest.mark.asyncio
async def test_consolidate_forest():
    """Test consolidate_forest merges nodes and formats JSON output using template."""
    db_mock = AsyncMock()
    llm_mock = AsyncMock()
    llm_mock.estimate_tokens_fallback = MagicMock(return_value=100)

    json_reply = """{
      "timeline": ["2026-01-01 19:00: Merged events"],
      "key_decisions_and_changes": "None",
      "pitfalls": "None",
      "files": ["/repo/merged.py"]
    }"""
    llm_mock.generate.return_value = LLMResponse(content=json_reply)

    compressor = HistoryCompressor(db_mock)

    # 8 Level-0 nodes (with K=4, triggers merge)
    roots = []
    for i in range(8):
        roots.append(
            SummaryNode(
                id=f"node_{i}",
                session_id="sess_1",
                level=0,
                summary=f"Summary {i}",
                start_timestamp=1000 + i,
                end_timestamp=1001 + i,
                token_count=50,
                source_token_count=100,
            )
        )

    # First call returns 8 level-0 nodes; second call for level 1 returns empty
    db_mock.get_root_summary_nodes.side_effect = [roots, []]

    await compressor.consolidate_forest("sess_1", llm_mock, K=4, staging_dir="/staging")

    assert db_mock.insert_summary_node.call_count == 1
    inserted_parent = db_mock.insert_summary_node.call_args_list[0][0][0]
    assert inserted_parent.level == 1
    assert "Timeline:\n- 2026-01-01 19:00: Merged events" in inserted_parent.summary
    assert "/repo/merged.py" in inserted_parent.summary
