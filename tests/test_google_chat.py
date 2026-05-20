"""Unit tests for Kesoku Google Chat chatbot adapter."""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.cloud import pubsub_v1

from kesoku.config import GoogleChatConfig, KesokuConfig
from kesoku.constants import (
    ROLE_ASSISTANT,
    ROLE_USER,
    STATUS_DELIVERED,
    STATUS_PENDING_AGENT,
    TYPE_TEXT,
    TYPE_THOUGHT,
)
from kesoku.db import Message, Session
from kesoku.gateway.chatbot.google_chat import GoogleChatChatbot
from kesoku.gateway.gateway import Gateway


@pytest.fixture
def mock_config() -> KesokuConfig:
    """Provide a mock Kesoku configuration with Google Chat enabled."""
    cfg = KesokuConfig()
    cfg.google_chat = GoogleChatConfig(
        enabled=True,
        chatbot_id="gchat_test",
        project_id="test-project",
        topic_id="test-topic",
        subscription_id="test-sub",
        user_allowlist=["users/allowed_user", "allowed@example.com"],
    )
    return cfg


@pytest.fixture
def mock_gateway() -> MagicMock:
    """Provide a mock Gateway instance."""
    gw = MagicMock(spec=Gateway)
    gw.get_session_by_channel = AsyncMock(return_value=None)
    gw.create_session = AsyncMock(return_value=Session(id="sess123", title="Test Session"))
    gw.update_session_updated_at = AsyncMock()
    gw.post = AsyncMock()
    gw.update_message_status = AsyncMock()
    gw.delete_session = AsyncMock()
    gw.agent = AsyncMock()
    return gw


@pytest.mark.asyncio
async def test_init_disabled_raises_value_error() -> None:
    """Test that initializing GoogleChatChatbot when disabled raises ValueError."""
    cfg = KesokuConfig()
    cfg.google_chat = GoogleChatConfig(enabled=False)
    gw = MagicMock(spec=Gateway)

    with patch("kesoku.gateway.chatbot.google_chat.get_config", return_value=cfg):
        with pytest.raises(ValueError, match="Google Chat chatbot is disabled"):
            GoogleChatChatbot(chatbot_id="gchat", gateway=gw)


@pytest.mark.asyncio
async def test_init_missing_params_raises_value_error() -> None:
    """Test that initializing GoogleChatChatbot with missing params raises ValueError."""
    cfg = KesokuConfig()
    cfg.google_chat = GoogleChatConfig(
        enabled=True, project_id="", topic_id="", subscription_id=""
    )
    gw = MagicMock(spec=Gateway)

    with patch("kesoku.gateway.chatbot.google_chat.get_config", return_value=cfg):
        with pytest.raises(ValueError, match="project_id, topic_id, or subscription_id are not configured"):
            GoogleChatChatbot(chatbot_id="gchat", gateway=gw)


@pytest.mark.asyncio
@patch("google.auth.default")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.build")
async def test_load_credentials_adc(
    mock_build: MagicMock,
    mock_subscriber: MagicMock,
    mock_auth_default: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test loading credentials using Application Default Credentials (ADC)."""
    mock_auth_default.return_value = (MagicMock(), "test-project")

    with patch("kesoku.gateway.chatbot.google_chat.get_config", return_value=mock_config):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)
        assert bot._project_id == "test-project"
        mock_auth_default.assert_called_once()


@pytest.mark.asyncio
@patch("google.auth.load_credentials_from_file")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.build")
async def test_load_credentials_file(
    mock_build: MagicMock,
    mock_subscriber: MagicMock,
    mock_load_file: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test loading credentials from an explicit JSON key file."""
    mock_config.google_chat.credentials_json = "/path/to/key.json"
    mock_load_file.return_value = (MagicMock(), "file-project")

    with patch("kesoku.gateway.chatbot.google_chat.get_config", return_value=mock_config):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)
        assert bot._project_id == "file-project"
        mock_load_file.assert_called_once_with("/path/to/key.json", scopes=[
            "https://www.googleapis.com/auth/pubsub",
            "https://www.googleapis.com/auth/chat.bot",
        ])


@pytest.mark.asyncio
@patch("google.auth.default")
@patch("google.auth.impersonated_credentials.Credentials")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.build")
async def test_load_credentials_impersonation(
    mock_build: MagicMock,
    mock_subscriber: MagicMock,
    mock_impersonate: MagicMock,
    mock_auth_default: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test loading credentials with Service Account Impersonation wrapper."""
    mock_config.google_chat.impersonate_service_account = "target-sa@gcp.com"
    mock_auth_default.return_value = (MagicMock(), None)
    mock_impersonate.return_value = MagicMock()

    with patch("kesoku.gateway.chatbot.google_chat.get_config", return_value=mock_config):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)
        assert bot._project_id == "test-project"  # Falls back to config project_id
        mock_impersonate.assert_called_once()


@pytest.mark.asyncio
@patch("google.auth.default")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.build")
async def test_incoming_message_parsing(
    mock_build: MagicMock,
    mock_subscriber: MagicMock,
    mock_auth_default: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test parsing standard incoming MESSAGE events and resolving sessions."""
    mock_auth_default.return_value = (MagicMock(), "test-project")

    event_payload = {
        "type": "MESSAGE",
        "space": {"name": "spaces/AAA"},
        "message": {
            "text": "Hello Agent!",
            "sender": {
                "displayName": "Test User",
                "name": "users/allowed_user",
                "email": "allowed@example.com",
            },
            "thread": {"name": "spaces/AAA/threads/BBB"},
        },
    }

    with patch("kesoku.gateway.chatbot.google_chat.get_config", return_value=mock_config):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)

        # Inject mock Pub/Sub message
        pubsub_msg = MagicMock(spec=pubsub_v1.subscriber.message.Message)
        pubsub_msg.data = json.dumps(event_payload).encode("utf-8")

        await bot._on_pubsub_message(pubsub_msg)

        # Verify that thread/channel ID "spaces/AAA/threads/BBB" was queried
        mock_gateway.get_session_by_channel.assert_called_once_with("gchat_test", "spaces/AAA/threads/BBB")
        # Verify that a new session was created with the custom thread prompt
        mock_gateway.create_session.assert_called_once()

        # Verify gateway.post was called with the correct user Message details
        mock_gateway.post.assert_called_once()
        posted_msg = mock_gateway.post.call_args[0][0]
        assert posted_msg.role == ROLE_USER
        assert posted_msg.content == "Hello Agent!"
        assert posted_msg.sender == "Test User"
        assert posted_msg.channel_id == "spaces/AAA/threads/BBB"
        assert posted_msg.status == STATUS_PENDING_AGENT


@pytest.mark.asyncio
@patch("google.auth.default")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.build")
async def test_incoming_message_blocked_by_allowlist(
    mock_build: MagicMock,
    mock_subscriber: MagicMock,
    mock_auth_default: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test that messages from unallowed users are ignored."""
    mock_auth_default.return_value = (MagicMock(), "test-project")

    event_payload = {
        "type": "MESSAGE",
        "space": {"name": "spaces/AAA"},
        "message": {
            "text": "Hello Agent!",
            "sender": {
                "displayName": "Hacker User",
                "name": "users/bad_user",
                "email": "bad@example.com",
            },
        },
    }

    with patch("kesoku.gateway.chatbot.google_chat.get_config", return_value=mock_config):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)

        pubsub_msg = MagicMock(spec=pubsub_v1.subscriber.message.Message)
        pubsub_msg.data = json.dumps(event_payload).encode("utf-8")

        await bot._on_pubsub_message(pubsub_msg)

        # Verify message was ignored and not posted to the gateway
        mock_gateway.get_session_by_channel.assert_not_called()
        mock_gateway.post.assert_not_called()


@pytest.mark.asyncio
@patch("google.auth.default")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.build")
async def test_handle_card_interaction_stop(
    mock_build: MagicMock,
    mock_subscriber: MagicMock,
    mock_auth_default: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test that stop_turn card action stops the logical turn in Gateway agent."""
    mock_auth_default.return_value = (MagicMock(), "test-project")

    event_payload = {
        "type": "CARD_CLICKED",
        "action": {
            "actionMethodName": "stop_turn",
            "parameters": [{"key": "session_id", "value": "sess123"}],
        },
    }

    with patch("kesoku.gateway.chatbot.google_chat.get_config", return_value=mock_config):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)

        pubsub_msg = MagicMock(spec=pubsub_v1.subscriber.message.Message)
        pubsub_msg.data = json.dumps(event_payload).encode("utf-8")

        await bot._on_pubsub_message(pubsub_msg)

        # Verify Gateway agent requested stop worker
        mock_gateway.agent.stop_session_worker.assert_called_once_with("sess123")


@pytest.mark.asyncio
@patch("google.auth.default")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.build")
async def test_handle_card_interaction_clear(
    mock_build: MagicMock,
    mock_subscriber: MagicMock,
    mock_auth_default: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test that clear_session card action triggers session deletion."""
    mock_auth_default.return_value = (MagicMock(), "test-project")

    event_payload = {
        "type": "CARD_CLICKED",
        "action": {
            "actionMethodName": "clear_session",
            "parameters": [{"key": "session_id", "value": "sess123"}],
        },
    }

    with patch("kesoku.gateway.chatbot.google_chat.get_config", return_value=mock_config):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)

        pubsub_msg = MagicMock(spec=pubsub_v1.subscriber.message.Message)
        pubsub_msg.data = json.dumps(event_payload).encode("utf-8")

        await bot._on_pubsub_message(pubsub_msg)

        # Verify Gateway triggered delete session
        mock_gateway.delete_session.assert_called_once_with("sess123")


@pytest.mark.asyncio
@patch("google.auth.default")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.build")
async def test_handle_card_interaction_choice(
    mock_build: MagicMock,
    mock_subscriber: MagicMock,
    mock_auth_default: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test multiple-choice card action posts choice back to Gateway as user message."""
    mock_auth_default.return_value = (MagicMock(), "test-project")

    event_payload = {
        "type": "CARD_CLICKED",
        "space": {"name": "spaces/AAA"},
        "message": {"thread": {"name": "spaces/AAA/threads/BBB"}},
        "user": {"displayName": "Test User"},
        "action": {
            "actionMethodName": "submit_choice",
            "parameters": [
                {"key": "session_id", "value": "sess123"},
                {"key": "choice", "value": "Option 2"},
            ],
        },
    }

    with patch("kesoku.gateway.chatbot.google_chat.get_config", return_value=mock_config):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)

        pubsub_msg = MagicMock(spec=pubsub_v1.subscriber.message.Message)
        pubsub_msg.data = json.dumps(event_payload).encode("utf-8")

        await bot._on_pubsub_message(pubsub_msg)

        # Verify choice is posted back to gateway as user message
        mock_gateway.post.assert_called_once()
        posted_msg = mock_gateway.post.call_args[0][0]
        assert posted_msg.role == ROLE_USER
        assert posted_msg.content == "Option 2"
        assert posted_msg.sender == "Test User"
        assert posted_msg.channel_id == "spaces/AAA/threads/BBB"
        assert posted_msg.status == STATUS_PENDING_AGENT


@pytest.mark.asyncio
@patch("google.auth.default")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.build")
async def test_handle_outgoing_message_delivery_foldable_ui(
    mock_build: MagicMock,
    mock_subscriber: MagicMock,
    mock_auth_default: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test that handle_message successfully constructs the foldable UI card, updates it, and finalizes it."""
    mock_auth_default.return_value = (MagicMock(), "test-project")

    # Mock Google Chat client methods
    mock_chat_client = MagicMock()
    mock_build.return_value = mock_chat_client
    mock_messages = MagicMock()
    mock_chat_client.spaces.return_value.messages.return_value = mock_messages

    mock_create = MagicMock()
    mock_messages.create = mock_create
    # The create call returns a message resource name
    mock_create.return_value.execute = MagicMock(side_effect=[
        {"name": "spaces/AAA/messages/foldable_card_123"},
        {"name": "spaces/AAA/messages/final_reply_456"},
    ])

    mock_patch = MagicMock()
    mock_messages.patch = mock_patch
    mock_patch.return_value.execute = MagicMock(return_value={"name": "spaces/AAA/messages/foldable_card_123"})

    with patch("kesoku.gateway.chatbot.google_chat.get_config", return_value=mock_config):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)

        # 1. Post an intermediate thought
        thought_msg = Message(
            id="thought1",
            session_id="sess123",
            chatbot_id="gchat_test",
            channel_id="spaces/AAA/threads/BBB",
            sender="Kesoku",
            role=ROLE_ASSISTANT,
            type=TYPE_THOUGHT,
            content="Let me think about it",
            timestamp=time.time(),
        )
        await bot.handle_message(thought_msg)

        # Verify foldable card was created
        mock_messages.create.assert_called_once()
        create_args = mock_messages.create.call_args[1]
        assert create_args["parent"] == "spaces/AAA"
        assert create_args["messageReplyOption"] == "REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"
        assert create_args["body"]["thread"]["name"] == "spaces/AAA/threads/BBB"
        assert "cardsV2" in create_args["body"]
        card = create_args["body"]["cardsV2"][0]["card"]
        assert card["header"]["title"] == "Kesoku Agent"
        assert card["sections"][0]["header"] == "Thoughts & Tools"
        assert "Thought:" in card["sections"][0]["widgets"][0]["textParagraph"]["text"]
        # Verify Stop Turn button is not present (only thoughts and tools section is present)
        assert len(card["sections"]) == 1

        # 2. Post the final reply
        final_msg = Message(
            id="msg999",
            session_id="sess123",
            chatbot_id="gchat_test",
            channel_id="spaces/AAA/threads/BBB",
            sender="Kesoku",
            role=ROLE_ASSISTANT,
            type=TYPE_TEXT,
            content="Hello world reply",
            timestamp=time.time(),
            metadata={
                "turn_metrics": {
                    "session_turns": 1,
                    "context_tokens": 12000,
                    "turn_tool_calls": 0,
                    "turn_tokens": 500,
                    "turn_time": 2.5,
                }
            }
        )
        await bot.handle_message(final_msg)

        # Verify foldable card was updated (patched) to finished state
        mock_messages.patch.assert_called_once()
        patch_args = mock_messages.patch.call_args[1]
        assert patch_args["name"] == "spaces/AAA/messages/foldable_card_123"
        assert patch_args["updateMask"] == "cardsV2"
        patched_card = patch_args["body"]["cardsV2"][0]["card"]
        # Verify no Stop Turn button is present in the sections anymore (removed)
        assert len(patched_card["sections"]) == 2
        # Verify metrics are displayed in the second section
        metrics_text = patched_card["sections"][1]["widgets"][0]["textParagraph"]["text"]
        assert "Session:" in metrics_text
        assert "Context:" in metrics_text
        assert "Turn:" in metrics_text

        # Verify final text message was created
        assert mock_messages.create.call_count == 2
        final_create_args = mock_messages.create.call_args_list[1][1]
        assert final_create_args["body"]["text"] == "Hello world reply"
        assert final_create_args["body"]["thread"]["name"] == "spaces/AAA/threads/BBB"

        # Verify delivery statuses were updated
        mock_gateway.update_message_status.assert_any_call("thought1", STATUS_DELIVERED)
        mock_gateway.update_message_status.assert_any_call("msg999", STATUS_DELIVERED)


@pytest.mark.asyncio
@patch("google.auth.default")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.build")
async def test_handle_outgoing_message_delivery_question(
    mock_build: MagicMock,
    mock_subscriber: MagicMock,
    mock_auth_default: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test that handle_message successfully constructs interactive multiple-choice cards."""
    mock_auth_default.return_value = (MagicMock(), "test-project")

    mock_chat_client = MagicMock()
    mock_build.return_value = mock_chat_client
    mock_messages = MagicMock()
    mock_chat_client.spaces.return_value.messages.return_value = mock_messages
    mock_create = MagicMock()
    mock_messages.create = mock_create
    mock_create.return_value.execute = MagicMock(return_value={"name": "spaces/AAA/messages/111"})

    with patch("kesoku.gateway.chatbot.google_chat.get_config", return_value=mock_config):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)

        # Message content containing the question-choice syntax block
        content = "[question: Do you want to compile? | Yes, run it | No, abort]"

        out_msg = Message(
            id="msg999",
            session_id="sess123",
            chatbot_id="gchat_test",
            channel_id="spaces/AAA/threads/BBB",
            sender="Kesoku",
            role=ROLE_ASSISTANT,
            type=TYPE_TEXT,
            content=content,
            timestamp=time.time(),
        )

        await bot.handle_message(out_msg)

        # Verify Chat API create call was structured correctly
        mock_messages.create.assert_called_once()
        body_arg = mock_messages.create.call_args[1]["body"]

        # The raw text should be clean (non-whitespace) or omitted since question is structured in card
        assert "cardsV2" in body_arg
        card = body_arg["cardsV2"][0]["card"]
        assert "header" not in card
        text = card["sections"][0]["widgets"][0]["textParagraph"]["text"]
        assert "Do you want to compile?" in text
        assert "- Yes, run it" in text
        assert "- No, abort" in text

        # Verify no buttons section exists
        assert len(card["sections"]) == 1
