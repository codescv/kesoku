"""Unit tests for the Chatbot base class and its utilities."""

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kesoku.constants import (
    MessageRole,
    MessageType,
)
from kesoku.db import Message, Session
from kesoku.gateway.chatbot.base import Chatbot, _format_uptime
from kesoku.gateway.gateway import Gateway


def test_format_uptime() -> None:
    """Test formatting various timedeltas with _format_uptime."""
    assert _format_uptime(datetime.timedelta(seconds=45)) == "45s"
    assert _format_uptime(datetime.timedelta(minutes=2, seconds=15)) == "2m 15s"
    assert _format_uptime(datetime.timedelta(hours=5, minutes=0, seconds=8)) == "5h 0m 8s"
    assert _format_uptime(datetime.timedelta(days=1, hours=12, minutes=30, seconds=5)) == "1d 12h 30m 5s"


class MockChatbot(Chatbot):
    """A concrete chatbot adapter for testing base class functionality."""

    async def handle_message(self, message: Message) -> None:
        """Dummy handler for MockChatbot."""
        pass


@pytest.mark.asyncio
async def test_get_session_status_by_channel() -> None:
    """Test that get_session_status_by_channel returns correct metrics and uptime info."""
    mock_gateway = MagicMock(spec=Gateway)
    mock_db = AsyncMock()
    mock_gateway.db = mock_db
    mock_session = Session(id="session123", title="Test Session")
    mock_db.get_session_by_channel = AsyncMock(return_value=mock_session)

    # Mock history with one user message and one assistant message containing turn metrics
    history = [
        Message(
            session_id="session123",
            chatbot_id="mock_bot",
            channel_id="channel123",
            sender="User",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="Hello",
            timestamp=1000.0,
        ),
        Message(
            session_id="session123",
            chatbot_id="mock_bot",
            channel_id="channel123",
            sender="Agent",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content="Hi",
            timestamp=1002.0,
            metadata={
                "turn_metrics": {
                    "context_tokens": 1200,
                    "cached_tokens": 1000,
                    "turn_tool_calls": 3,
                    "turn_tokens": 150,
                    "turn_time": 2.5,
                }
            },
        ),
    ]
    mock_db.get_session_history = AsyncMock(return_value=history)
    mock_db.get_session_turns_count = AsyncMock(return_value=1)

    chatbot = MockChatbot(chatbot_id="mock_bot", gateway=mock_gateway)

    # Freeze time/started time to verify output matches
    fixed_now = datetime.datetime(2026, 5, 26, 12, 0, 0)
    fixed_start = datetime.datetime(2026, 5, 26, 11, 0, 0)

    with (
        patch("datetime.datetime") as mock_datetime,
        patch("kesoku.gateway.chatbot.base.SYSTEM_START_TIME", fixed_start),
    ):
        mock_datetime.now.return_value = fixed_now
        status_str = await chatbot.get_session_status_by_channel("channel123")

    assert "【Current Stats】" in status_str
    assert "⏰ Uptime: 1h 0m 0s (started: 2026-05-26 11:00:00)" in status_str
    assert "⚡ Session: 1 turns (ID: session123)" in status_str
    assert "📖 Context: 1K tokens (Cached: 1K)" in status_str
    assert "  - Tool Calls: 3" in status_str
    assert "  - Tokens: 0K" in status_str
    assert "  - Time: 2.5s" in status_str


def test_format_text_headers_shifting() -> None:
    """Test shifting and clamping of markdown headers via MockChatbot delegation."""
    mock_gateway = MagicMock(spec=Gateway)
    chatbot = MockChatbot(chatbot_id="mock_bot", gateway=mock_gateway)

    # Simple smoke test to verify delegation works
    input_text = "## Header A\n### Header B\n"
    expected = "# Header A\n\n## Header B\n"
    assert chatbot.format_text(input_text) == expected


@pytest.mark.asyncio
async def test_resolve_outbound_path(tmp_path) -> None:
    """Test resolve_outbound_path under fuzzy matching for misspelled absolute paths."""
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()

    # Create test files in staging
    file_in_staging = staging_dir / "cat.png"
    file_in_staging.write_text("staging cat")

    # Create a nested file to test path scoring
    nested_dir = staging_dir / "output"
    nested_dir.mkdir()
    nested_file = nested_dir / "result.png"
    nested_file.write_text("nested result")

    flat_file = staging_dir / "result.png"
    flat_file.write_text("flat result")

    mock_gateway = MagicMock(spec=Gateway)
    chatbot = MockChatbot(chatbot_id="mock_bot", gateway=mock_gateway)

    from kesoku.config import AgentConfig, KesokuConfig, WorkspaceConfig
    mock_cfg = KesokuConfig(
        workspace=WorkspaceConfig(sessions_dir=str(tmp_path / "sessions"), db_path=":memory:"),
        agent=AgentConfig(user_prompts=[]),
        agent_working_dir=str(tmp_path / "awd"),
    )

    with patch("kesoku.gateway.chatbot.base.get_config", return_value=mock_cfg):
        # Mock session retrieval and staging dir resolution
        mock_session = Session(id="session123", title="Test Session", workspace_name="test_ws")
        mock_gateway.db = AsyncMock()
        mock_gateway.db.get_session = AsyncMock(return_value=mock_session)
        chatbot.get_session_staging_dir = MagicMock(return_value=str(staging_dir))

        # 1. Existing absolute path
        abs_path = str(file_in_staging.resolve())
        resolved = await chatbot.resolve_outbound_path(abs_path, "session123")
        assert resolved == abs_path

        # 2. Fuzzy match misspelled filename (catt.png -> cat.png) with absolute path typo
        staging_dir_abs = str(staging_dir.resolve())
        misspelled_abs = staging_dir_abs.replace("staging", "stating") + "/catt.png"
        resolved = await chatbot.resolve_outbound_path(misspelled_abs, "session123")
        assert resolved == str(file_in_staging.resolve())

        # 3. Fuzzy match path with nested files (comprehensive check):
        # R: "stating/output/results.png"
        # C1 (nested): "staging/output/result.png" -> score will be higher
        # C2 (flat): "staging/result.png"
        misspelled_nested = str(nested_file.resolve())
        misspelled_nested = misspelled_nested.replace("staging", "stating").replace("result.png", "results.png")
        resolved = await chatbot.resolve_outbound_path(misspelled_nested, "session123")
        assert resolved == str(nested_file.resolve())

        misspelled_flat = str(flat_file.resolve())
        misspelled_flat = misspelled_flat.replace("staging", "stating").replace("result.png", "results.png")
        resolved = await chatbot.resolve_outbound_path(misspelled_flat, "session123")
        assert resolved == str(flat_file.resolve())

        # 4. Path not found / low confidence (score < PATH_RESOLUTION_CONFIDENCE_THRESHOLD), returns original
        low_confidence_path = staging_dir_abs.replace("staging", "stating") + "/completely_different.png"
        resolved = await chatbot.resolve_outbound_path(low_confidence_path, "session123")
        assert resolved == low_confidence_path

        # 5. Exact match of a relative path under staging_dir
        resolved = await chatbot.resolve_outbound_path("cat.png", "session123")
        assert resolved == str(file_in_staging.resolve())

        # 6. Fuzzy match of a relative path under staging_dir (catt.png -> cat.png)
        resolved = await chatbot.resolve_outbound_path("catt.png", "session123")
        assert resolved == str(file_in_staging.resolve())

        # 7. Exact match of a nested relative path under staging_dir
        resolved = await chatbot.resolve_outbound_path("output/result.png", "session123")
        assert resolved == str(nested_file.resolve())

        # 8. Fuzzy match of a nested relative path under staging_dir (outtput/results.png -> output/result.png)
        resolved = await chatbot.resolve_outbound_path("outtput/results.png", "session123")
        assert resolved == str(nested_file.resolve())
