"""Unit tests for Kesoku Discord Chatbot UI components."""

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.gateway.chatbot.discord.ui import MessageHeaderView, QuestionView
from kesoku.gateway.gateway import Gateway


@pytest.fixture
def mock_gateway() -> MagicMock:
    """Provide a mock Gateway instance."""
    gw = MagicMock(spec=Gateway)
    db = AsyncMock()
    gw.db = db
    db.get_session_history = AsyncMock(return_value=[])
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
            role=MessageRole.SYSTEM,
            type=MessageType.TEXT,
            content="System prompt instruction",
            timestamp=1716120000.0,
        ),
        Message(
            id="msg2",
            session_id="session_123",
            chatbot_id="discord",
            channel_id="ch1",
            sender="User",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="Hello agent!",
            timestamp=1716120005.0,
        ),
        Message(
            id="msg3",
            session_id="session_123",
            chatbot_id="discord",
            channel_id="ch1",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.THOUGHT,
            content="Thinking process...",
            timestamp=1716120006.0,
        ),
        Message(
            id="msg4",
            session_id="session_123",
            chatbot_id="discord",
            channel_id="ch1",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content="Hello user, how can I help you today?",
            timestamp=1716120010.0,
        ),
        Message(
            id="msg5",
            session_id="session_123",
            chatbot_id="discord",
            channel_id="ch1",
            sender="Kesoku",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_CALL,
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
            role=MessageRole.TOOL,
            type=MessageType.TEXT,
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
            role=MessageRole.TOOL,
            type=MessageType.TEXT,
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
@patch("kesoku.gateway.chatbot.discord.ui.build_history", new_callable=AsyncMock)
async def test_view_trajectory_callback_success(mock_build: AsyncMock, mock_gateway: MagicMock) -> None:
    """Test successful click of the 'View Trajectory' button."""
    mock_gateway.db.get_session = AsyncMock(return_value=None)
    view = MessageHeaderView(gateway=mock_gateway, session_id="session_123")

    mock_interaction = AsyncMock(spec=discord.Interaction)
    mock_interaction.response = MagicMock()
    mock_interaction.response.defer = AsyncMock()
    mock_interaction.followup = AsyncMock()
    mock_button = MagicMock(spec=discord.ui.Button)

    # Mock build_history to return dummy messages
    history = [
        Message(
            id="msg1",
            session_id="session_123",
            chatbot_id="discord",
            channel_id="ch1",
            sender="User",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="Hello",
            timestamp=1716120000.0,
        )
    ]
    mock_build.return_value = history

    with patch("discord.File") as mock_file_class:
        mock_file = MagicMock()
        mock_file_class.return_value = mock_file

        await view.view_trajectory.callback(mock_interaction)

        mock_interaction.response.defer.assert_called_once_with(ephemeral=True)
        mock_build.assert_called_once_with(
            gateway=mock_gateway,
            session_id="session_123",
            order="grouped",
            heal_orphans=False,
        )
        mock_file_class.assert_called_once()

        # Check followup send was called with the file
        mock_interaction.followup.send.assert_called_once_with(
            content="Here is the complete interactive trace of the conversation turn:",
            file=mock_file,
            ephemeral=True,
        )


@pytest.mark.asyncio
@patch("kesoku.gateway.chatbot.discord.ui.build_history", new_callable=AsyncMock)
async def test_view_trajectory_callback_failure(mock_build: AsyncMock, mock_gateway: MagicMock) -> None:
    """Test click of the 'View Trajectory' button when history fetch fails."""
    mock_gateway.db.get_session = AsyncMock(return_value=None)
    view = MessageHeaderView(gateway=mock_gateway, session_id="session_123")

    mock_interaction = AsyncMock(spec=discord.Interaction)
    mock_interaction.response = MagicMock()
    mock_interaction.response.defer = AsyncMock()
    mock_interaction.followup = AsyncMock()
    mock_button = MagicMock(spec=discord.ui.Button)

    # Mock failure
    mock_build.side_effect = Exception("Database error")

    await view.view_trajectory.callback(mock_interaction)

    mock_interaction.response.defer.assert_called_once_with(ephemeral=True)
    mock_interaction.followup.send.assert_called_once_with(
        content="⚠️ Failed to generate trajectory: Database error",
        ephemeral=True,
    )


def test_message_header_view_visibility_channels_vs_threads(mock_gateway: MagicMock) -> None:
    """Test that clear session button is removed inside threads but visible inside channels."""
    # Case 1: inside a thread session
    view_thread = MessageHeaderView(gateway=mock_gateway, session_id="s123", is_thread=True)
    children_ids_thread = [item.custom_id for item in view_thread.children]
    assert "btn_view_trajectory" in children_ids_thread
    assert "btn_stop_turn" in children_ids_thread
    assert "btn_clear_session" not in children_ids_thread

    # Case 2: inside a regular channel session
    view_channel = MessageHeaderView(gateway=mock_gateway, session_id="s123", is_thread=False)
    children_ids_channel = [item.custom_id for item in view_channel.children]
    assert "btn_view_trajectory" in children_ids_channel
    assert "btn_stop_turn" in children_ids_channel
    assert "btn_clear_session" in children_ids_channel


@pytest.mark.asyncio
async def test_stop_turn_callback(mock_gateway: MagicMock) -> None:
    """Test successful click of the 'Stop' button."""
    # Mock active agent and session worker
    mock_worker = MagicMock()
    mock_agent = MagicMock()
    mock_agent.stop_session_worker = AsyncMock()
    mock_agent.workers = {"s123": mock_worker}
    mock_gateway.agent = mock_agent

    # Mock chatbot
    mock_chatbot = MagicMock()
    mock_typing_task = MagicMock()
    mock_chatbot._typing_tasks = {"chan_abc": mock_typing_task}
    mock_msg1 = AsyncMock(spec=discord.Message)
    mock_chatbot._intermediate_messages = {"chan_abc": [mock_msg1]}

    mock_message = AsyncMock(spec=discord.Message)
    mock_chatbot._header_views = {"s123": (mock_message, MagicMock())}
    mock_chatbot._turn_special_items = {"s123": []}
    mock_chatbot._turn_special_msg = {"s123": MagicMock()}

    view = MessageHeaderView(gateway=mock_gateway, session_id="s123", chatbot=mock_chatbot)

    mock_interaction = AsyncMock(spec=discord.Interaction)
    mock_interaction.response = MagicMock()
    mock_interaction.response.defer = AsyncMock()
    mock_interaction.followup = AsyncMock()
    mock_interaction.channel_id = "chan_abc"
    mock_interaction.message = mock_message
    mock_button = MagicMock(spec=discord.ui.Button)

    # Mock DB user message to stop
    mock_gateway.db.get_session_history.return_value = [
        Message(
            id="msg_u1",
            session_id="s123",
            chatbot_id="discord",
            channel_id="chan_abc",
            sender="User",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="Help me",
            status="processing",
        )
    ]

    await view.stop_turn.callback(mock_interaction)

    # Asserts:
    mock_interaction.response.defer.assert_called_once_with(ephemeral=True)
    # Message containing the view should be deleted entirely
    mock_message.delete.assert_called_once()
    mock_message.edit.assert_not_called()
    # stop_session_worker should be called
    mock_agent.stop_session_worker.assert_called_once_with("s123", immediate=True)
    # Message status updated to interrupted
    mock_gateway.db.update_message_status.assert_called_once_with("msg_u1", MessageStatus.INTERRUPTED)
    # Typing task cancelled
    mock_typing_task.cancel.assert_called_once()
    # Intermediate message deleted
    mock_msg1.delete.assert_called_once()
    # Sent feedback followup
    mock_interaction.followup.send.assert_called_once()
    # Turn caches cleared
    assert "s123" not in mock_chatbot._header_views
    assert "s123" not in mock_chatbot._turn_special_items
    assert "s123" not in mock_chatbot._turn_special_msg


@pytest.mark.asyncio
async def test_clear_session_callback(mock_gateway: MagicMock) -> None:
    """Test successful click of the 'Clear Session' button."""
    mock_worker = MagicMock()
    mock_agent = MagicMock()
    mock_agent.stop_session_worker = AsyncMock()
    mock_agent.workers = {"s123": mock_worker}
    mock_gateway.agent = mock_agent
    mock_gateway.delete_session = AsyncMock()

    mock_chatbot = MagicMock()
    mock_typing_task = MagicMock()
    mock_chatbot._typing_tasks = {"chan_abc": mock_typing_task}
    mock_msg1 = AsyncMock(spec=discord.Message)
    mock_chatbot._intermediate_messages = {"chan_abc": [mock_msg1]}
    mock_chatbot._header_views = {"s123": (MagicMock(), MagicMock()), "some_turn": (MagicMock(), MagicMock())}
    mock_chatbot._turn_special_items = {"s123": []}
    mock_chatbot._turn_special_msg = {"s123": MagicMock()}

    view = MessageHeaderView(gateway=mock_gateway, session_id="s123", chatbot=mock_chatbot)

    mock_interaction = AsyncMock(spec=discord.Interaction)
    mock_interaction.response = MagicMock()
    mock_interaction.response.defer = AsyncMock()
    mock_interaction.followup = AsyncMock()
    mock_interaction.channel_id = "chan_abc"
    mock_button = MagicMock(spec=discord.ui.Button)

    await view.clear_session.callback(mock_interaction)

    # Asserts:
    mock_interaction.response.defer.assert_called_once_with(ephemeral=True)
    # stop_session_worker should be called
    mock_agent.stop_session_worker.assert_called_once_with("s123", immediate=True)
    # delete_session called on Gateway
    mock_gateway.delete_session.assert_called_once_with("s123")
    # Typing task cancelled
    mock_typing_task.cancel.assert_called_once()
    # Intermediate message deleted
    mock_msg1.delete.assert_called_once()
    # header_views cleaned up
    assert "s123" not in mock_chatbot._header_views
    assert "some_turn" in mock_chatbot._header_views
    # special items/msg caches cleaned up
    assert "s123" not in mock_chatbot._turn_special_items
    assert "s123" not in mock_chatbot._turn_special_msg
    # Sent feedback followup
    mock_interaction.followup.send.assert_called_once()


@pytest.mark.asyncio
async def test_stop_turn_callback_with_metrics(mock_gateway: MagicMock) -> None:
    """Test click of the 'Stop' button with metrics present in the user message."""
    # Mock active agent and session worker
    mock_worker = MagicMock()
    mock_agent = MagicMock()
    mock_agent.stop_session_worker = AsyncMock()
    mock_agent.workers = {"s123": mock_worker}
    mock_gateway.agent = mock_agent

    # Mock chatbot
    mock_chatbot = MagicMock()
    mock_chatbot._typing_tasks = {}
    mock_chatbot._intermediate_messages = {}
    mock_message = AsyncMock(spec=discord.Message)
    mock_chatbot._header_views = {"s123": (mock_message, MagicMock())}
    mock_chatbot._turn_special_items = {"s123": []}
    mock_chatbot._turn_special_msg = {"s123": MagicMock()}

    view = MessageHeaderView(gateway=mock_gateway, session_id="s123", chatbot=mock_chatbot)

    mock_interaction = AsyncMock(spec=discord.Interaction)
    mock_interaction.response = MagicMock()
    mock_interaction.response.defer = AsyncMock()
    mock_interaction.followup = AsyncMock()
    mock_interaction.channel_id = "chan_abc"
    mock_interaction.message = mock_message
    mock_button = MagicMock(spec=discord.ui.Button)

    # Mock DB user message to stop, with metrics
    mock_gateway.db.get_session_history.return_value = [
        Message(
            id="msg_u1",
            session_id="s123",
            chatbot_id="discord",
            channel_id="chan_abc",
            sender="User",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="Help me",
            status="processing",
            metadata={
                "turn_metrics": {
                    "session_turns": 5,
                    "context_tokens": 8400,
                    "turn_tool_calls": 2,
                    "turn_tokens": 1200,
                    "turn_time": 4.5,
                    "status": "interrupted",
                }
            },
        )
    ]

    await view.stop_turn.callback(mock_interaction)

    # Asserts:
    mock_interaction.response.defer.assert_called_once_with(ephemeral=True)
    # Message containing the view should be deleted entirely
    mock_message.delete.assert_called_once()
    mock_message.edit.assert_not_called()
    # Turn caches cleared
    assert "s123" not in mock_chatbot._header_views
    assert "s123" not in mock_chatbot._turn_special_items
    assert "s123" not in mock_chatbot._turn_special_msg


@pytest.mark.asyncio
async def test_question_view_init_and_callback(mock_gateway: MagicMock) -> None:
    """Test that QuestionView initializes buttons and handles selection callback correctly."""
    mock_chatbot = MagicMock()
    mock_chatbot.chatbot_id = "discord"
    mock_chatbot._typing_tasks = {}
    mock_chatbot._keep_typing = AsyncMock()

    choices = ["Red", "Blue"]
    view = QuestionView(
        gateway=mock_gateway,
        session_id="s123",
        chatbot=mock_chatbot,
        question="What is your favorite color?",
        choices=choices,
    )

    # Asserts on initialization
    assert view.gateway == mock_gateway
    assert view.session_id == "s123"
    assert view.chatbot == mock_chatbot
    assert view.question == "What is your favorite color?"
    assert view.choices == choices
    assert len(view.children) == 2
    assert isinstance(view.children[0], discord.ui.Button)
    assert view.children[0].label == "Red"
    assert view.children[0].custom_id == "btn_q_s123_0_Red"
    assert view.children[1].label == "Blue"
    assert view.children[1].custom_id == "btn_q_s123_1_Blue"

    # Mock interaction
    mock_interaction = AsyncMock(spec=discord.Interaction)
    mock_interaction.response = MagicMock()
    mock_interaction.response.defer = AsyncMock()
    mock_interaction.channel_id = "chan_abc"

    mock_message = AsyncMock(spec=discord.Message)
    mock_interaction.message = mock_message

    mock_channel = AsyncMock(spec=discord.TextChannel)
    mock_response_msg = MagicMock(spec=discord.Message)
    mock_response_msg.id = "res_123"
    mock_response_msg.created_at = datetime.datetime.now(datetime.UTC)
    mock_channel.send.return_value = mock_response_msg
    mock_interaction.channel = mock_channel
    mock_interaction.user = MagicMock()
    mock_interaction.user.id = "user_999"
    mock_interaction.user.display_name = "Alice"

    # Trigger callback of the first button ("Red")
    button_callback = view.children[0].callback
    await button_callback(mock_interaction)

    # Asserts on callback behaviour:
    # 1. Defer called
    mock_interaction.response.defer.assert_called_once()

    # 2. Buttons are all disabled
    assert view.children[0].disabled is True
    assert view.children[1].disabled is True

    # 3. Message edited to show disabled view
    mock_message.edit.assert_called_once_with(view=view)

    # 4. Visual feedback message sent to channel
    mock_channel.send.assert_called_once_with("<@user_999> selected: **Red**")

    # 5. Gateway post called to ingest MessageRole.USER message
    mock_gateway.post.assert_called_once()
    posted_msg = mock_gateway.post.call_args[0][0]
    assert isinstance(posted_msg, Message)
    assert posted_msg.session_id == "s123"
    assert posted_msg.chatbot_id == "discord"
    assert posted_msg.channel_id == "chan_abc"
    assert posted_msg.sender == "Alice"
    assert posted_msg.role == MessageRole.USER
    assert posted_msg.type == MessageType.TEXT
    assert "Red" in posted_msg.content
    assert posted_msg.metadata["discord_message_id"] == "res_123"
    assert posted_msg.metadata["discord_author_id"] == "user_999"

    # 6. Chatbot typing task is started
    assert "chan_abc" in mock_chatbot._typing_tasks


@pytest.mark.asyncio
async def test_question_view_duplicate_prefixes(mock_gateway: MagicMock) -> None:
    """Test that QuestionView generates unique custom_ids even with same prefix choices."""
    mock_chatbot = MagicMock()
    choices = [
        "This is a very long option A",
        "This is a very long option B",
    ]
    view = QuestionView(
        gateway=mock_gateway,
        session_id="s123",
        chatbot=mock_chatbot,
        question="Select one",
        choices=choices,
    )
    assert len(view.children) == 2
    assert view.children[0].custom_id == "btn_q_s123_0_This is a very long "
    assert view.children[1].custom_id == "btn_q_s123_1_This is a very long "
    assert view.children[0].custom_id != view.children[1].custom_id
