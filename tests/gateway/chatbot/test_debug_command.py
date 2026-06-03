"""Unit tests for the /debug slash command."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kesoku.config import AgentConfig, KesokuConfig, WorkspaceConfig
from kesoku.db import Message, Session
from kesoku.gateway.chatbot.base import Chatbot
from kesoku.gateway.gateway import Gateway


class MockChatbot(Chatbot):
    """A concrete chatbot adapter for testing base class functionality."""

    async def handle_message(self, message: Message) -> None:
        """Dummy handler for MockChatbot."""
        pass


@pytest.mark.asyncio
async def test_debug_command_toggle() -> None:
    """Test that /debug toggles raw_llm_logs and prints staging dir when enabled."""
    mock_gateway = MagicMock(spec=Gateway)
    mock_db = AsyncMock()
    mock_gateway.db = mock_db

    chatbot = MockChatbot(chatbot_id="mock_bot", gateway=mock_gateway)

    mock_cfg = KesokuConfig(
        workspace=WorkspaceConfig(sessions_dir="/tmp/sessions", db_path=":memory:"),
        agent=AgentConfig(raw_llm_logs=False),  # Start with False
    )

    mock_session = Session(id="session123", title="Test Session")
    mock_db.get_session_by_channel = AsyncMock(return_value=mock_session)

    chatbot.get_session_staging_dir = MagicMock(return_value="/tmp/sessions/test_ws")

    replies = []

    async def reply_func(text: str) -> None:
        replies.append(text)

    with patch("kesoku.gateway.chatbot.base.get_config", return_value=mock_cfg):
        # 1. Toggle ON (False -> True)
        await chatbot.commands.execute("debug", reply_func, channel_id="channel123")
        assert mock_cfg.agent.raw_llm_logs is True
        assert len(replies) == 1
        assert "Debug mode enabled" in replies[0]
        assert "Staging dir: `/tmp/sessions/test_ws`" in replies[0]

        # 2. Toggle OFF (True -> False)
        await chatbot.commands.execute("debug", reply_func, channel_id="channel123")
        assert mock_cfg.agent.raw_llm_logs is False
        assert len(replies) == 2
        assert "Debug mode disabled" in replies[1]


@pytest.mark.asyncio
async def test_debug_command_no_session() -> None:
    """Test /debug behavior when no active session is found."""
    mock_gateway = MagicMock(spec=Gateway)
    mock_db = AsyncMock()
    mock_gateway.db = mock_db
    mock_db.get_session_by_channel = AsyncMock(return_value=None)  # No session

    chatbot = MockChatbot(chatbot_id="mock_bot", gateway=mock_gateway)

    mock_cfg = KesokuConfig(
        workspace=WorkspaceConfig(sessions_dir="/tmp/sessions", db_path=":memory:"),
        agent=AgentConfig(raw_llm_logs=False),
    )

    replies = []

    async def reply_func(text: str) -> None:
        replies.append(text)

    with patch("kesoku.gateway.chatbot.base.get_config", return_value=mock_cfg):
        await chatbot.commands.execute("debug", reply_func, channel_id="channel123")
        assert mock_cfg.agent.raw_llm_logs is True
        assert len(replies) == 1
        assert "Debug mode enabled" in replies[0]
        assert "No active session found" in replies[0]
