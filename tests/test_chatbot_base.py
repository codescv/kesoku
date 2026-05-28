"""Unit tests for the Chatbot base class and its utilities."""

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kesoku.constants import MessageRole, MessageType
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
    mock_session = Session(id="session123", title="Test Session")
    mock_gateway.get_session_by_channel = AsyncMock(return_value=mock_session)

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
    mock_gateway.get_session_history = AsyncMock(return_value=history)

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
    """Test shifting and clamping of markdown headers."""
    mock_gateway = MagicMock(spec=Gateway)
    chatbot = MockChatbot(chatbot_id="mock_bot", gateway=mock_gateway)

    # Case 1: Headers start at level 2 -> shifted to level 1, clamp levels > 3
    input_text = (
        "## Header A\n"
        "### Header B\n"
        "#### Header C\n"
        "##### Header D\n"
    )
    expected = (
        "# Header A\n"
        "\n"
        "## Header B\n"
        "\n"
        "### Header C\n"
        "\n"
        "### Header D\n"
    )
    assert chatbot.format_text(input_text) == expected

    # Case 2: Headers already start at level 1 -> no shifting, only clamp levels > 3
    input_text_2 = (
        "# Header A\n"
        "## Header B\n"
        "#### Header C\n"
    )
    expected_2 = (
        "# Header A\n"
        "\n"
        "## Header B\n"
        "\n"
        "### Header C\n"
    )
    assert chatbot.format_text(input_text_2) == expected_2


def test_format_text_consecutive_newlines_and_spacing() -> None:
    """Test collapsing consecutive newlines and ensuring spacing before headers."""
    mock_gateway = MagicMock(spec=Gateway)
    chatbot = MockChatbot(chatbot_id="mock_bot", gateway=mock_gateway)

    input_text = (
        "Some text.\n"
        "## Header A\n"
        "\n"
        "\n"
        "Other text.\n"
    )
    # Note: chatbot.format_text preserves trailing newlines,
    # and ensures exactly one blank line before headings if not at the start.
    expected = (
        "Some text.\n"
        "\n"
        "# Header A\n"
        "\n"
        "Other text.\n"
    )
    assert chatbot.format_text(input_text) == expected


def test_format_text_code_blocks_ignored() -> None:
    """Test that formatting is not applied to content inside code blocks."""
    mock_gateway = MagicMock(spec=Gateway)
    chatbot = MockChatbot(chatbot_id="mock_bot", gateway=mock_gateway)

    input_text = (
        "## Header A\n"
        "```python\n"
        "## This heading inside code block should not be shifted\n"
        "\n"
        "\n"
        "def foo():\n"
        "    pass\n"
        "```\n"
    )
    # Outer heading ## Header A shifts to # Header A.
    # Inner heading and inner blank lines remain intact.
    expected = (
        "# Header A\n"
        "```python\n"
        "## This heading inside code block should not be shifted\n"
        "\n"
        "\n"
        "def foo():\n"
        "    pass\n"
        "```\n"
    )
    assert chatbot.format_text(input_text) == expected


def test_split_text_into_chunks_code_block() -> None:
    """Test splitting text into chunks while preserving and wrapping code blocks."""
    mock_gateway = MagicMock(spec=Gateway)
    chatbot = MockChatbot(chatbot_id="mock_bot", gateway=mock_gateway)

    text = (
        "Some introduction.\n"
        "```python\n"
        "line1\n"
        "line2\n"
        "line3\n"
        "```\n"
        "Conclusion."
    )
    # With max_length set so that it splits in the middle of the python code block:
    # e.g. max_length = 45.
    # Let's verify chunking:
    chunks = chatbot.split_text_into_chunks(text, max_length=45)

    # The first chunk should contain "Some introduction." and the start of code block,
    # and automatically close with "```".
    # The second chunk should automatically open with "```python" and contain the rest.
    assert len(chunks) > 1
    for chunk in chunks:
        # If chunk contains code lines, it must be wrapped in ```
        if "line1" in chunk or "line2" in chunk or "line3" in chunk:
            assert chunk.startswith("```python") or chunk.endswith("```")
