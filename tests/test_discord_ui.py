"""Unit tests for Kesoku Discord Chatbot UI components."""

from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from kesoku.constants import (
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL,
    ROLE_USER,
    TYPE_TEXT,
    TYPE_THOUGHT,
    TYPE_TOOL_CALL,
)
from kesoku.db import Message
from kesoku.gateway.chatbot.discord_ui import MessageHeaderView
from kesoku.gateway.gateway import Gateway


@pytest.fixture
def mock_gateway() -> MagicMock:
    """Provide a mock Gateway instance."""
    gw = MagicMock(spec=Gateway)
    gw.get_session_history = AsyncMock(return_value=[])
    return gw


def test_message_header_view_init(mock_gateway: MagicMock) -> None:
    """Test that MessageHeaderView initializes successfully with a session ID."""
    view = MessageHeaderView(gateway=mock_gateway, session_id="session_123")
    assert view.gateway == mock_gateway
    assert view.session_id == "session_123"


def test_generate_html_trajectory(mock_gateway: MagicMock) -> None:
    """Test HTML trajectory generation with various message types."""
    view = MessageHeaderView(gateway=mock_gateway, session_id="session_123")

    history = [
        Message(
            id="msg1",
            session_id="session_123",
            chatbot_id="discord",
            channel_id="ch1",
            sender="System",
            role=ROLE_SYSTEM,
            type=TYPE_TEXT,
            content="System prompt instruction",
            timestamp=1716120000.0,
        ),
        Message(
            id="msg2",
            session_id="session_123",
            chatbot_id="discord",
            channel_id="ch1",
            sender="User",
            role=ROLE_USER,
            type=TYPE_TEXT,
            content="Hello agent!",
            timestamp=1716120005.0,
        ),
        Message(
            id="msg3",
            session_id="session_123",
            chatbot_id="discord",
            channel_id="ch1",
            sender="Kesoku",
            role=ROLE_ASSISTANT,
            type=TYPE_THOUGHT,
            content="Thinking process...",
            timestamp=1716120006.0,
        ),
        Message(
            id="msg4",
            session_id="session_123",
            chatbot_id="discord",
            channel_id="ch1",
            sender="Kesoku",
            role=ROLE_ASSISTANT,
            type=TYPE_TEXT,
            content="Hello user, how can I help you today?",
            timestamp=1716120010.0,
        ),
        Message(
            id="msg5",
            session_id="session_123",
            chatbot_id="discord",
            channel_id="ch1",
            sender="Kesoku",
            role=ROLE_TOOL,
            type=TYPE_TOOL_CALL,
            content="Calling shell tool...",
            timestamp=1716120012.0,
            metadata={"tool_name": "shell", "tool_arguments": {"command": "ls"}},
        ),
        Message(
            id="msg6",
            session_id="session_123",
            chatbot_id="discord",
            channel_id="ch1",
            sender="shell",
            role=ROLE_TOOL,
            type=TYPE_TEXT,
            content="file1.txt\nfile2.txt",
            timestamp=1716120015.0,
            metadata={"tool_name": "shell"},
        ),
        Message(
            id="msg7",
            session_id="session_123",
            chatbot_id="discord",
            channel_id="ch1",
            sender="shell",
            role=ROLE_TOOL,
            type=TYPE_TEXT,
            content="permission denied",
            timestamp=1716120018.0,
            metadata={"tool_name": "shell", "tool_error": "Permission denied"},
        ),
    ]

    html_content = view._generate_html_trajectory(history)
    assert "Agent Trajectory Viewer" in html_content
    assert "session_123" in html_content

    # Check roles classes
    assert 'class="entry system"' in html_content
    assert 'class="entry user"' in html_content
    assert 'class="entry thought"' in html_content
    assert 'class="entry assistant"' in html_content
    assert 'class="entry tool-call"' in html_content
    assert 'class="entry tool-success"' in html_content
    assert 'class="entry tool-error"' in html_content

    # Check contents
    assert "System prompt instruction" in html_content
    assert "Hello agent!" in html_content
    assert "Thinking process..." in html_content
    assert "Hello user, how can I help you today?" in html_content
    assert "Calling shell tool..." in html_content
    assert "file1.txt" in html_content
    assert "permission denied" in html_content


@pytest.mark.asyncio
async def test_view_trajectory_callback_success(mock_gateway: MagicMock) -> None:
    """Test successful click of the 'View Trajectory' button."""
    view = MessageHeaderView(gateway=mock_gateway, session_id="session_123")

    mock_interaction = AsyncMock(spec=discord.Interaction)
    mock_interaction.response = MagicMock()
    mock_interaction.response.defer = AsyncMock()
    mock_interaction.followup = AsyncMock()
    mock_button = MagicMock(spec=discord.ui.Button)

    # Mock get_session_history to return dummy messages
    history = [
        Message(
            id="msg1",
            session_id="session_123",
            chatbot_id="discord",
            channel_id="ch1",
            sender="User",
            role=ROLE_USER,
            type=TYPE_TEXT,
            content="Hello",
            timestamp=1716120000.0,
        )
    ]
    mock_gateway.get_session_history.return_value = history

    with patch("discord.File") as mock_file_class:
        mock_file = MagicMock()
        mock_file_class.return_value = mock_file

        await view.view_trajectory.callback(mock_interaction)

        mock_interaction.response.defer.assert_called_once_with(ephemeral=True)
        mock_gateway.get_session_history.assert_called_once_with("session_123", limit=200, order="grouped")
        mock_file_class.assert_called_once()

        # Check followup send was called with the file
        mock_interaction.followup.send.assert_called_once_with(
            content="Here is the complete interactive trace of the conversation turn:",
            file=mock_file,
            ephemeral=True,
        )


@pytest.mark.asyncio
async def test_view_trajectory_callback_failure(mock_gateway: MagicMock) -> None:
    """Test click of the 'View Trajectory' button when gateway history fetch fails."""
    view = MessageHeaderView(gateway=mock_gateway, session_id="session_123")

    mock_interaction = AsyncMock(spec=discord.Interaction)
    mock_interaction.response = MagicMock()
    mock_interaction.response.defer = AsyncMock()
    mock_interaction.followup = AsyncMock()
    mock_button = MagicMock(spec=discord.ui.Button)

    # Mock failure
    mock_gateway.get_session_history.side_effect = Exception("Database error")

    await view.view_trajectory.callback(mock_interaction)

    mock_interaction.response.defer.assert_called_once_with(ephemeral=True)
    mock_interaction.followup.send.assert_called_once_with(
        content="⚠️ Failed to generate trajectory: Database error",
        ephemeral=True,
    )
