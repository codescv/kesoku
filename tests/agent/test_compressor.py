"""Unit tests for HistoryCompressor in kesoku.agent.compressor."""

import os
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from kesoku.agent.compressor import (
    CONSOLIDATE_SUMMARIES_PROMPT,
    SUMMARIZE_TURN_PROMPT,
    HistoryCompressor,
    sanitize_directive,
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
  "timeline": ["2026-01-01 18:00 - 2026-01-01 18:30: Started task"],
  "user_asks": ["Always reply in Chinese across this session"],
  "tools_and_skills": ["run_shell_command: Executed CLI script"],
  "learnings": "None"
}
```"""
    data = HistoryCompressor.parse_summary_json(raw)
    assert data["timeline"] == ["2026-01-01 18:00 - 2026-01-01 18:30: Started task"]
    assert data["user_asks"] == ["Always reply in Chinese across this session"]
    assert data["tools_and_skills"] == ["run_shell_command: Executed CLI script"]
    assert data["learnings"] == "None"


def test_parse_summary_json_invalid():
    """Test parsing invalid JSON returns empty dict gracefully."""
    raw = "Not a json response at all"
    data = HistoryCompressor.parse_summary_json(raw)
    assert data == {}


def test_is_outside_staging_dir(tmp_path):
    """Test STAGING_DIR file path checking."""
    staging_dir = str(tmp_path / "sessions" / "my_sess")
    os.makedirs(staging_dir, exist_ok=True)

    # File inside staging dir
    inside_file = os.path.join(staging_dir, "output.png")
    assert HistoryCompressor.is_outside_staging_dir(inside_file, staging_dir) is False

    # Relative path matching this session folder
    assert (
        HistoryCompressor.is_outside_staging_dir(
            "sessions/my_sess/asuka_bbq_chef_bird_1823.png", staging_dir
        )
        is False
    )
    assert (
        HistoryCompressor.is_outside_staging_dir(
            "my_sess/asuka_bbq_chef_bird_1823.png", staging_dir
        )
        is False
    )

    # File from another session must be considered OUTSIDE (should return True)
    assert (
        HistoryCompressor.is_outside_staging_dir(
            "sessions/other_sess/important_file.txt", staging_dir
        )
        is True
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


def test_format_summary_with_new_structure():
    """Test format_summary renders timeline, tools_and_skills, learnings and drops any user_asks."""
    json_input = """```json
{
  "timeline": ["2026-01-01 18:00 - 2026-01-01 18:30: User requested feature X"],
  "user_asks": [
    "Keep all code snippets compatible with Python 3.12"
  ],
  "tools_and_skills": [
    "run_shell_command (ai-image): Generated illustrations for story scene"
  ],
  "learnings": "Use rg instead of grep for searching."
}
```"""

    result = HistoryCompressor.format_summary(json_input)

    assert "Timeline:\n- 2026-01-01 18:00 - 2026-01-01 18:30: User requested feature X" in result
    assert "User Asks" not in result
    assert "Python 3.12" not in result
    assert "Tools & Skills" not in result
    assert "Learnings:\nUse rg instead of grep for searching." in result
    assert result.startswith("Timeline:")


def test_format_summary_all_none():
    """Test format_summary when optional sections are missing or empty."""
    json_input = """{
  "timeline": ["2026-01-01 18:00: Checked system status."],
  "user_asks": null,
  "tools_and_skills": [],
  "learnings": "null"
}"""
    result = HistoryCompressor.format_summary(json_input)
    assert "Timeline:\n- 2026-01-01 18:00: Checked system status." in result
    assert "User Asks" not in result
    assert "Tools & Skills" not in result
    assert "Learnings:\nNone" in result


@pytest.mark.asyncio
async def test_auto_compact_session(tmp_path):
    """Test auto_compact_session uses updated prompt and formats summary using template."""
    db_mock = AsyncMock()
    llm_mock = AsyncMock()
    llm_mock.estimate_tokens_fallback = MagicMock(return_value=10000)

    json_reply = """{
      "timeline": ["2026-01-01 18:00 - 2026-01-01 18:30: Started task"],
      "user_asks": ["Do not modify production database"],
      "tools_and_skills": ["run_shell_command: Executed CLI script"],
      "learnings": "None"
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
    assert "This conversation segment occurred from" in call_prompt
    assert "Use these exact calendar dates" in call_prompt

    db_mock.insert_summary_node.assert_called_once()
    inserted_node: SummaryNode = db_mock.insert_summary_node.call_args[0][0]
    assert "Timeline:\n- 2026-01-01 18:00 - 2026-01-01 18:30: Started task" in inserted_node.summary
    assert "User Asks" not in inserted_node.summary
    assert "user_asks" not in call_prompt
    assert "Tools & Skills" not in inserted_node.summary


@pytest.mark.asyncio
async def test_consolidate_forest():
    """Test consolidate_forest merges nodes and formats JSON output using template."""
    db_mock = AsyncMock()
    llm_mock = AsyncMock()
    llm_mock.estimate_tokens_fallback = MagicMock(return_value=100)

    json_reply = """{
      "timeline": ["2026-01-01 19:00 - 2026-01-01 19:30: Merged events"],
      "user_asks": ["Do not modify production database"],
      "tools_and_skills": ["run_shell_command: Executed CLI script"],
      "learnings": "None"
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
    assert "Timeline:\n- 2026-01-01 19:00 - 2026-01-01 19:30: Merged events" in inserted_parent.summary
    assert "User Asks" not in inserted_parent.summary
    assert "Tools & Skills" not in inserted_parent.summary

def test_estimate_prepared_turn_tokens():
    """Test estimate_prepared_turn_tokens drops thoughts and truncates long tool outputs."""
    db_mock = AsyncMock()
    llm_mock = MagicMock()
    # Mock fallback token estimation to return len(prompt) // 4
    llm_mock.estimate_tokens_fallback.side_effect = lambda prompt: len(prompt) // 4

    compressor = HistoryCompressor(db_mock)

    turn = [
        Message(
            id="msg_1",
            session_id="s1",
            chatbot_id="cb",
            channel_id="ch",
            sender="user",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="Hello " * 10,  # 60 chars -> 15 tokens
            status=MessageStatus.RESPONDED,
            timestamp=100.0,
        ),
        Message(
            id="msg_2",
            session_id="s1",
            chatbot_id="cb",
            channel_id="ch",
            sender="assistant",
            role=MessageRole.ASSISTANT,
            type=MessageType.THOUGHT,
            content="Thinking deeply... " * 500,  # Should be dropped entirely
            status=MessageStatus.RESPONDED,
            timestamp=101.0,
        ),
        Message(
            id="msg_3",
            session_id="s1",
            chatbot_id="cb",
            channel_id="ch",
            sender="tool",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_RESULT,
            content="x" * 2000,  # Exceeds 500 chars -> truncated to 500 chars -> 125 tokens
            status=MessageStatus.RESPONDED,
            timestamp=102.0,
        ),
    ]

    tokens = compressor.estimate_prepared_turn_tokens(turn, llm_mock, max_historical_tool_chars=500)
    # Expected: 60 // 4 = 15 tokens from msg_1 + 500 // 4 = 125 tokens from truncated msg_3 = 140
    assert tokens == 140


def test_format_summary_does_not_touch_notes(tmp_path):
    """format_summary must never write extracted requirements into $STAGING_DIR/notes.md."""
    staging_dir = str(tmp_path / "sess_notes")
    os.makedirs(staging_dir)
    notes_path = os.path.join(staging_dir, "notes.md")
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("- current rule")
    json_input = """{
      "timeline": ["2026-01-01 18:00: Started chat"],
      "user_asks": ["Outdated rule that must not come back"],
      "tools_and_skills": null,
      "learnings": null
    }"""
    HistoryCompressor.format_summary(json_input, staging_dir=staging_dir)
    with open(notes_path, encoding="utf-8") as f:
        assert f.read() == "- current rule"


def test_strip_legacy_sections():
    """Legacy summaries have their User Asks section removed; other sections are kept intact."""
    legacy = (
        "Timeline:\n- 2026-10-13 10:40 - 2026-10-13 16:56: Scene\n\n"
        "User Asks:\n- 偶数回合必须双图\n- 全员比基尼\n\n"
        "Tools & Skills:\n- run_shell_command: images\n\n"
        "Learnings:\nUse uv."
    )
    stripped = HistoryCompressor.strip_legacy_sections(legacy)
    assert "User Asks" not in stripped
    assert "比基尼" not in stripped
    assert stripped.startswith("Timeline:\n- 2026-10-13 10:40")
    assert "Tools & Skills" not in stripped
    assert stripped.endswith("\n\nLearnings:\nUse uv.")

    # User Asks as the last section, and summaries without it
    tail = "Timeline:\n- a\n\nUser Asks:\nNone"
    assert HistoryCompressor.strip_legacy_sections(tail) == "Timeline:\n- a"
    plain = "Timeline:\n- a\n\nLearnings:\nNone"
    assert HistoryCompressor.strip_legacy_sections(plain) == plain


@pytest.mark.asyncio
async def test_consolidate_strips_legacy_user_asks_from_children():
    """Consolidation input must not contain legacy User Asks from child summaries."""
    db_mock = AsyncMock()
    llm_mock = AsyncMock()
    llm_mock.estimate_tokens_fallback = MagicMock(return_value=100)
    llm_mock.generate.return_value = LLMResponse(
        content='{"timeline": ["x"], "tools_and_skills": null, "learnings": null}'
    )
    roots = [
        SummaryNode(
            id=f"n{i}", session_id="s", level=0,
            summary=f"Timeline:\n- t{i}\n\nUser Asks:\n- stale rule {i}\n\nLearnings:\nNone",
            start_timestamp=1000 + i, end_timestamp=1001 + i, token_count=10, source_token_count=100,
        )
        for i in range(8)
    ]
    db_mock.get_root_summary_nodes.side_effect = [roots, []]
    await HistoryCompressor(db_mock).consolidate_forest("s", llm_mock, K=4)
    prompt = llm_mock.generate.call_args[1]["prompt"]
    assert "stale rule" not in prompt
    assert "- t0" in prompt


def test_estimate_prepared_turn_tokens_truncates_notes_tool_args():
    """Estimation mirrors history truncation of long update_session_notes arguments."""
    compressor = HistoryCompressor(AsyncMock())
    llm_mock = MagicMock()
    llm_mock.estimate_tokens_fallback = MagicMock(side_effect=lambda prompt=None, **_: len(prompt or ""))
    call = Message(
        session_id="s", chatbot_id="cb", channel_id="ch", sender="update_session_notes",
        role=MessageRole.TOOL, type=MessageType.TOOL_CALL, content="Calling tool ...",
        status=MessageStatus.RESPONDED,
        metadata={"tool_name": "update_session_notes", "tool_arguments": {"content": "n" * 5000}},
    )
    assert compressor.estimate_prepared_turn_tokens([call], llm_mock) < 500



# --- P1-1: continuation-style summary schema -----------------------------------------------------------------

GOLDEN_JSON = """{
  "timeline": ["2026-10-19 13:00 - 2026-10-19 13:40: Implemented the validator", "13:40 - 13:55: Ran tests"],
  "dead_ends": ["Cached the clock -> test_clock_skew failed, reverted"],
  "pending_state": {
    "unfinished": ["Add 503 retry to _build_request; next: write the retry test"],
    "constraints": ["Error codes use E_DOM_CORE_NNN", "skip env.py, keep legacy.py"],
    "key_files": ["migration_checklist.md: checklist, 3/5 done"]
  },
  "learnings": "Use uv run pytest -x for fast feedback."
}"""

GOLDEN_SUMMARY = """Timeline:
- 2026-10-19 13:00 - 2026-10-19 13:40: Implemented the validator
- 13:40 - 13:55: Ran tests

Dead Ends:
- Cached the clock -> test_clock_skew failed, reverted

Pending State:
Unfinished:
- Add 503 retry to _build_request; next: write the retry test
Constraints:
- Error codes use E_DOM_CORE_NNN
- skip env.py, keep legacy.py
Key files:
- migration_checklist.md: checklist, 3/5 done

Learnings:
Use uv run pytest -x for fast feedback."""


def test_format_summary_golden() -> None:
    """All four sections render in a stable order and layout."""
    assert HistoryCompressor.format_summary(GOLDEN_JSON) == GOLDEN_SUMMARY


def test_format_summary_null_sections_render_none() -> None:
    """Absent / null dead_ends and pending_state render "None", like Learnings."""
    result = HistoryCompressor.format_summary('{"timeline": ["t"], "dead_ends": null, "learnings": null}')
    assert result == "Timeline:\n- t\n\nDead Ends:\nNone\n\nPending State:\nNone\n\nLearnings:\nNone"


def test_format_pending_state_forms() -> None:
    """pending_state accepts the dict form, partial dicts, strings, lists and nulls."""
    fmt = HistoryCompressor._format_pending_state
    assert fmt(None) == "None"
    assert fmt({}) == "None"
    assert fmt({"unfinished": [], "constraints": None, "key_files": ["null"]}) == "None"
    # Empty groups are omitted; a string group value becomes a single bullet.
    assert fmt({"unfinished": [], "constraints": "keep Y"}) == "Constraints:\n- keep Y"
    # Unknown keys are kept after the known groups.
    assert fmt({"key_files": ["a.py: x"], "open_questions": ["which port?"]}) == (
        "Key files:\n- a.py: x\nOpen questions:\n- which port?"
    )
    # Non-conforming list / string forms are rendered as plain bullets / text.
    assert fmt(["finish X", "keep Y"]) == "- finish X\n- keep Y"
    assert fmt("finish X") == "finish X"
    assert fmt("none") == "None"


def test_format_dead_ends_forms() -> None:
    """dead_ends renders lists as bullets, strings verbatim, empty/null as None."""
    for dead_ends, expected in [
        ('["a -> b", "c -> d"]', "- a -> b\n- c -> d"),
        ('"tried a -> failed"', "tried a -> failed"),
        ("[]", "None"),
        ("null", "None"),
    ]:
        result = HistoryCompressor.format_summary(f'{{"timeline": ["t"], "dead_ends": {dead_ends}}}')
        assert f"Dead Ends:\n{expected}\n\nPending State:" in result


def test_prompts_format_with_and_without_directive() -> None:
    """Both prompts format without KeyError, list the new keys, and keep a {directive} slot."""
    common = {"max_chars": 1500, "date_context": "ctx", "example_start_time": "s", "example_end_time": "e"}
    for template, extra in [
        (SUMMARIZE_TURN_PROMPT, {"segment": "seg {not a field}"}),
        (CONSOLIDATE_SUMMARIES_PROMPT, {"summaries": "sums {x}"}),
    ]:
        for directive in ("", "Compaction focus requested by the user: \"{weird}\"."):
            prompt = template.format(directive=directive, **common, **extra)
            for key in ('"timeline"', '"dead_ends"', '"pending_state"', '"unfinished"', '"constraints"',
                        '"key_files"', '"learnings"'):
                assert key in prompt
            assert directive in prompt


def test_summarize_prompt_guidelines() -> None:
    """The L0 prompt explains dead_ends and pending_state (verbatim rules, no completed work)."""
    assert "so they are not retried later" in SUMMARIZE_TURN_PROMPT
    assert "quote short rules\n     verbatim" in SUMMARIZE_TURN_PROMPT
    assert "Do NOT list completed work here" in SUMMARIZE_TURN_PROMPT


def test_consolidation_prompt_contains_merge_rules() -> None:
    """The consolidation prompt dedupes dead ends, keeps only open items and never drops live constraints."""
    assert "Merge and deduplicate the dead ends" in CONSOLIDATE_SUMMARIES_PROMPT
    assert "still open at the end of the latest summary" in CONSOLIDATE_SUMMARIES_PROMPT
    assert "an item shown as\n   completed in a later summary is dropped" in CONSOLIDATE_SUMMARIES_PROMPT
    assert "a later rule supersedes an earlier conflicting one" in CONSOLIDATE_SUMMARIES_PROMPT
    assert "Never drop a constraint that is still in force, whatever the focus directive says." in (
        CONSOLIDATE_SUMMARIES_PROMPT
    )


def test_strip_legacy_sections_keeps_new_sections() -> None:
    """A legacy Tools & Skills section followed by Dead Ends / Pending State keeps the new sections."""
    legacy = (
        "Timeline:\n- t\n\n"
        "Tools & Skills:\n- run_shell_command: x\n\n"
        "Dead Ends:\n- a -> b\n\n"
        "User Asks:\n- stale\n\n"
        "Pending State:\nConstraints:\n- keep Y\n\n"
        "Learnings:\nNone"
    )
    assert HistoryCompressor.strip_legacy_sections(legacy) == (
        "Timeline:\n- t\n\nDead Ends:\n- a -> b\n\nPending State:\nConstraints:\n- keep Y\n\nLearnings:\nNone"
    )
    # No-op without legacy markers.
    assert HistoryCompressor.strip_legacy_sections(GOLDEN_SUMMARY) == GOLDEN_SUMMARY


def test_sanitize_directive() -> None:
    """Directives are stripped, whitespace-collapsed, capped at 500 chars; blank -> None."""
    assert sanitize_directive(None) is None
    assert sanitize_directive("  \n\t ") is None
    assert sanitize_directive("  keep\n\nonly   {Y} ") == "keep only {Y}"
    assert sanitize_directive("x" * 600) == "x" * 500


@pytest.mark.asyncio
async def test_auto_compact_prompt_has_no_directive_and_new_sections_persist(tmp_path):
    """auto_compact_session passes no directive; dead_ends/pending_state reach the stored node."""
    db_mock = AsyncMock()
    db_mock.get_root_summary_nodes.return_value = []
    llm_mock = AsyncMock()
    llm_mock.estimate_tokens_fallback = MagicMock(return_value=10000)
    llm_mock.generate.return_value = LLMResponse(content=GOLDEN_JSON)
    cfg = KesokuConfig()
    cfg.agent.protect_front_turns = 1
    cfg.agent.protect_tail_turns = 1
    cfg.agent.base_node_turns = 2
    cfg.agent.base_node_min_tokens = 1000
    messages = [
        Message(
            id=f"golden_{idx}", session_id="sess_golden", chatbot_id="cb", channel_id="ch", sender="user",
            role=MessageRole.USER, type=MessageType.TEXT, content=f"Turn {idx}", status=MessageStatus.RESPONDED,
            timestamp=time.time() + idx,
        )
        for idx in range(4)
    ]
    assert await HistoryCompressor(db_mock).auto_compact_session("sess_golden", messages, llm_mock, cfg)
    prompt = llm_mock.generate.call_args[1]["prompt"]
    assert "Compaction focus" not in prompt
    inserted: SummaryNode = db_mock.insert_summary_node.call_args[0][0]
    assert inserted.summary == GOLDEN_SUMMARY
