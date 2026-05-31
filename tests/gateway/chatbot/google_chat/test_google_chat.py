"""Unit tests for Kesoku Google Chat chatbot adapter."""

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.cloud import pubsub_v1
from googleapiclient.errors import HttpError

from kesoku.config import GoogleChatConfig, KesokuConfig
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message, Session
from kesoku.gateway.chatbot.google_chat import GoogleChatChatbot
from kesoku.gateway.gateway import Gateway


@pytest.fixture
def mock_config(tmp_path: Any) -> KesokuConfig:
    """Provide a mock Kesoku configuration with Google Chat enabled and temporary paths.

    Args:
        tmp_path: Pytest's temporary path fixture.

    Returns:
        A mock KesokuConfig instance.
    """
    cfg = KesokuConfig()
    cfg.workspace.sessions_dir = str(tmp_path / "sessions")
    cfg.workspace.db_path = str(tmp_path / "kesoku.db")
    cfg.workspace.skills_dir = str(tmp_path / "skills")
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

    with patch("kesoku.gateway.chatbot.google_chat.adapter.get_config", return_value=cfg):
        with pytest.raises(ValueError, match="Google Chat chatbot is disabled"):
            GoogleChatChatbot(chatbot_id="gchat", gateway=gw)


@pytest.mark.asyncio
async def test_init_missing_params_raises_value_error() -> None:
    """Test that initializing GoogleChatChatbot with missing params raises ValueError."""
    cfg = KesokuConfig()
    cfg.google_chat = GoogleChatConfig(enabled=True, project_id="", topic_id="", subscription_id="")
    gw = MagicMock(spec=Gateway)

    with patch("kesoku.gateway.chatbot.google_chat.adapter.get_config", return_value=cfg):
        with pytest.raises(ValueError, match="project_id, topic_id, or subscription_id are not configured"):
            GoogleChatChatbot(chatbot_id="gchat", gateway=gw)


@pytest.mark.asyncio
@patch("google.auth.default")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.adapter.build")
async def test_load_credentials_adc(
    mock_build: MagicMock,
    mock_subscriber: MagicMock,
    mock_auth_default: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test loading credentials using Application Default Credentials (ADC)."""
    mock_auth_default.return_value = (MagicMock(), "test-project")

    with patch("kesoku.gateway.chatbot.google_chat.adapter.get_config", return_value=mock_config):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)
        assert bot._project_id == "test-project"
        mock_auth_default.assert_called_once()


@pytest.mark.asyncio
@patch("google.auth.load_credentials_from_file")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.adapter.build")
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

    with patch("kesoku.gateway.chatbot.google_chat.adapter.get_config", return_value=mock_config):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)
        assert bot._project_id == "file-project"
        mock_load_file.assert_called_once_with(
            "/path/to/key.json",
            scopes=[
                "https://www.googleapis.com/auth/pubsub",
                "https://www.googleapis.com/auth/chat.bot",
            ],
        )


@pytest.mark.asyncio
@patch("google.auth.default")
@patch("google.auth.impersonated_credentials.Credentials")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.adapter.build")
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

    with patch("kesoku.gateway.chatbot.google_chat.adapter.get_config", return_value=mock_config):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)
        assert bot._project_id == "test-project"  # Falls back to config project_id
        mock_impersonate.assert_called_once()


@pytest.mark.asyncio
@patch("google.auth.default")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.adapter.build")
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

    with patch("kesoku.gateway.chatbot.google_chat.adapter.get_config", return_value=mock_config):
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
        assert posted_msg.role == MessageRole.USER
        assert posted_msg.content == "Hello Agent!"
        assert posted_msg.sender == "Test User"
        assert posted_msg.channel_id == "spaces/AAA/threads/BBB"
        assert posted_msg.status == MessageStatus.PENDING_AGENT


@pytest.mark.asyncio
@patch("google.auth.default")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.adapter.build")
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

    with patch("kesoku.gateway.chatbot.google_chat.adapter.get_config", return_value=mock_config):
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
@patch("kesoku.gateway.chatbot.google_chat.adapter.build")
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
    mock_create.return_value.execute = MagicMock(
        side_effect=[
            {"name": "spaces/AAA/messages/foldable_card_123"},
            {"name": "spaces/AAA/messages/final_reply_456"},
        ]
    )

    mock_patch = MagicMock()
    mock_messages.patch = mock_patch
    mock_patch.return_value.execute = MagicMock(return_value={"name": "spaces/AAA/messages/foldable_card_123"})

    with patch("kesoku.gateway.chatbot.google_chat.adapter.get_config", return_value=mock_config):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)

        # 1. Post an intermediate thought
        thought_msg = Message(
            id="thought1",
            session_id="sess123",
            chatbot_id="gchat_test",
            channel_id="spaces/AAA/threads/BBB",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.THOUGHT,
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
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
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
            },
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

        # Verify final response card was created
        assert mock_messages.create.call_count == 2
        final_create_args = mock_messages.create.call_args_list[1][1]
        assert final_create_args["body"]["thread"]["name"] == "spaces/AAA/threads/BBB"
        assert "cardsV2" in final_create_args["body"]
        final_card = final_create_args["body"]["cardsV2"][0]["card"]
        widget = final_card["sections"][0]["widgets"][0]["textParagraph"]
        assert widget["text"] == "Hello world reply"
        assert widget["textSyntax"] == "MARKDOWN"

        # Verify delivery statuses were updated
        mock_gateway.update_message_status.assert_any_call("thought1", MessageStatus.DELIVERED)
        mock_gateway.update_message_status.assert_any_call("msg999", MessageStatus.DELIVERED)


@pytest.mark.asyncio
@patch("google.auth.default")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.adapter.build")
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

    with patch("kesoku.gateway.chatbot.google_chat.adapter.get_config", return_value=mock_config):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)

        # Message content containing the question-choice syntax block
        content = "[question: Do you want to compile? | Yes, run it | No, abort]"

        out_msg = Message(
            id="msg999",
            session_id="sess123",
            chatbot_id="gchat_test",
            channel_id="spaces/AAA/threads/BBB",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
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


@pytest.mark.asyncio
@patch("google.auth.default")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.adapter.build")
async def test_load_user_credentials_adc(
    mock_build: MagicMock,
    mock_subscriber: MagicMock,
    mock_auth_default: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test loading user credentials for reactions using ADC with exact user scopes."""
    mock_auth_default.return_value = (MagicMock(), "user-project")

    with patch("kesoku.gateway.chatbot.google_chat.adapter.get_config", return_value=mock_config):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)
        creds, project = bot._load_user_credentials()
        assert project == "user-project"
        mock_auth_default.assert_any_call(
            scopes=[
                "https://www.googleapis.com/auth/chat.messages.reactions.create",
                "https://www.googleapis.com/auth/chat.messages",
            ]
        )


@pytest.mark.asyncio
@patch("google.auth.default")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.adapter.build")
async def test_user_chat_service_initialization(
    mock_build: MagicMock,
    mock_subscriber: MagicMock,
    mock_auth_default: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test that user_chat_service is initialized only when reaction_emoji is configured."""
    mock_auth_default.return_value = (MagicMock(), "test-project")

    # Case 1: reaction_emoji is None -> user_chat_service should be None
    mock_config.google_chat.reaction_emoji = None
    with patch("kesoku.gateway.chatbot.google_chat.adapter.get_config", return_value=mock_config):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)
        assert bot._user_chat_service is None

    mock_build.reset_mock()

    # Case 2: reaction_emoji is set -> user_chat_service should be initialized
    mock_config.google_chat.reaction_emoji = "👀"
    with patch("kesoku.gateway.chatbot.google_chat.adapter.get_config", return_value=mock_config):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)
        assert bot._user_chat_service is not None
        # The build function should have been called twice (once for _chat_service, once for _user_chat_service)
        assert mock_build.call_count == 2


@pytest.mark.asyncio
@patch("google.auth.default")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.adapter.build")
async def test_incoming_message_triggers_reaction(
    mock_build: MagicMock,
    mock_subscriber: MagicMock,
    mock_auth_default: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test that an incoming user message triggers the user emoji reaction."""
    mock_auth_default.return_value = (MagicMock(), "test-project")
    mock_config.google_chat.reaction_emoji = "👀"

    mock_chat_client = MagicMock()
    # Mock two builds: first for self._chat_service, second for self._user_chat_service
    mock_build.side_effect = [MagicMock(), mock_chat_client]

    mock_reactions = MagicMock()
    mock_chat_client.spaces.return_value.messages.return_value.reactions.return_value = mock_reactions
    mock_create = MagicMock()
    mock_reactions.create = mock_create
    mock_create.return_value.execute = MagicMock()

    event_payload = {
        "type": "MESSAGE",
        "space": {"name": "spaces/AAA"},
        "message": {
            "name": "spaces/AAA/messages/msg123",
            "text": "Hi Agent!",
            "sender": {
                "displayName": "Test User",
                "name": "users/allowed_user",
                "email": "allowed@example.com",
            },
            "thread": {"name": "spaces/AAA/threads/BBB"},
        },
    }

    with patch("kesoku.gateway.chatbot.google_chat.adapter.get_config", return_value=mock_config):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)

        pubsub_msg = MagicMock(spec=pubsub_v1.subscriber.message.Message)
        pubsub_msg.data = json.dumps(event_payload).encode("utf-8")

        # Trigger incoming message handling
        await bot._on_pubsub_message(pubsub_msg)

        # Yield control to allow background asyncio task for reaction to run
        await asyncio.sleep(0.1)

        # Verify the create reaction API was called with the correct parent and payload
        mock_create.assert_called_once_with(
            parent="spaces/AAA/messages/msg123",
            body={"emoji": {"unicode": "👀"}},
        )


@pytest.mark.asyncio
@patch("google.auth.default")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.adapter.build")
async def test_incoming_message_no_reaction_if_disabled(
    mock_build: MagicMock,
    mock_subscriber: MagicMock,
    mock_auth_default: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test that no reaction is added when reaction_emoji is not configured."""
    mock_auth_default.return_value = (MagicMock(), "test-project")
    mock_config.google_chat.reaction_emoji = None

    mock_chat_client = MagicMock()
    mock_build.return_value = mock_chat_client

    mock_reactions = MagicMock()
    mock_chat_client.spaces.return_value.messages.return_value.reactions.return_value = mock_reactions
    mock_create = MagicMock()
    mock_reactions.create = mock_create

    event_payload = {
        "type": "MESSAGE",
        "space": {"name": "spaces/AAA"},
        "message": {
            "name": "spaces/AAA/messages/msg123",
            "text": "Hi Agent!",
            "sender": {
                "displayName": "Test User",
                "name": "users/allowed_user",
                "email": "allowed@example.com",
            },
            "thread": {"name": "spaces/AAA/threads/BBB"},
        },
    }

    with patch("kesoku.gateway.chatbot.google_chat.adapter.get_config", return_value=mock_config):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)

        pubsub_msg = MagicMock(spec=pubsub_v1.subscriber.message.Message)
        pubsub_msg.data = json.dumps(event_payload).encode("utf-8")

        await bot._on_pubsub_message(pubsub_msg)
        await asyncio.sleep(0.1)

        # Verify reaction create was never called
        mock_create.assert_not_called()


@pytest.mark.asyncio
@patch("google.auth.default")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.adapter.build")
async def test_add_reaction_toggle_deletion(
    mock_build: MagicMock,
    mock_subscriber: MagicMock,
    mock_auth_default: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test that attempting to react with an already reacted emoji deletes/toggles the reaction."""
    mock_auth_default.return_value = (MagicMock(), "test-project")
    mock_config.google_chat.reaction_emoji = "👀"

    mock_chat_client = MagicMock()
    mock_build.side_effect = [MagicMock(), mock_chat_client]

    mock_reactions = MagicMock()
    mock_chat_client.spaces.return_value.messages.return_value.reactions.return_value = mock_reactions

    # Configure the create and delete mocks
    mock_create = MagicMock()
    mock_reactions.create = mock_create
    mock_create.return_value.execute = MagicMock(return_value={"name": "spaces/AAA/messages/msg123/reactions/XYZ123"})

    mock_delete = MagicMock()
    mock_reactions.delete = mock_delete
    mock_delete.return_value.execute = MagicMock()

    with patch("kesoku.gateway.chatbot.google_chat.adapter.get_config", return_value=mock_config):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)

        message_name = "spaces/AAA/messages/msg123"

        # 1. Add reaction first time -> triggers create
        await bot._add_reaction(message_name, "👀")
        mock_create.assert_called_once()
        assert bot._used_reactions[message_name]["👀"] == "spaces/AAA/messages/msg123/reactions/XYZ123"

        # 2. Add reaction second time -> triggers delete (toggle)
        await bot._add_reaction(message_name, "👀")
        mock_delete.assert_called_once_with(name="spaces/AAA/messages/msg123/reactions/XYZ123")
        # Confirm it was removed from the used reactions map
        assert "👀" not in bot._used_reactions[message_name]


@pytest.mark.asyncio
@patch("google.auth.default")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.adapter.build")
async def test_add_reaction_409_conflict_triggers_list_and_delete(
    mock_build: MagicMock,
    mock_subscriber: MagicMock,
    mock_auth_default: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test that if create raises a duplicate 409 error, the handler lists reactions and deletes the duplicate."""
    mock_auth_default.return_value = (MagicMock(), "test-project")
    mock_config.google_chat.reaction_emoji = "👀"

    mock_chat_client = MagicMock()
    mock_build.side_effect = [MagicMock(), mock_chat_client]

    mock_reactions = MagicMock()
    mock_chat_client.spaces.return_value.messages.return_value.reactions.return_value = mock_reactions

    # Configure the create mock to raise HTTP 409 HttpError
    resp = MagicMock()
    resp.status = 409
    error_payload = b'{"error": {"message": "Already exists"}}'
    http_error = HttpError(resp, error_payload)
    mock_reactions.create.return_value.execute = MagicMock(side_effect=http_error)

    # Configure list and delete mocks
    mock_list = MagicMock()
    mock_reactions.list = mock_list
    mock_list.return_value.execute = MagicMock(
        return_value={
            "reactions": [{"name": "spaces/AAA/messages/msg123/reactions/XYZ999", "emoji": {"unicode": "👀"}}]
        }
    )

    mock_delete = MagicMock()
    mock_reactions.delete = mock_delete
    mock_delete.return_value.execute = MagicMock()

    with patch("kesoku.gateway.chatbot.google_chat.adapter.get_config", return_value=mock_config):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)

        message_name = "spaces/AAA/messages/msg123"

        # Add reaction -> raises 409 -> lists and deletes
        await bot._add_reaction(message_name, "👀")

        mock_reactions.create.assert_called_once()
        mock_list.assert_called_once_with(parent=message_name)
        mock_delete.assert_called_once_with(name="spaces/AAA/messages/msg123/reactions/XYZ999")


def test_parse_emoji_sequence() -> None:
    """Test parse_emoji_sequence correctly segments different emoji sequences."""
    from kesoku.gateway.chatbot.google_chat.adapter import parse_emoji_sequence

    # Case 1: Space separated
    assert parse_emoji_sequence("👀 🛠️ 🚀") == ["👀", "🛠️", "🚀"]
    # Case 2: Comma separated
    assert parse_emoji_sequence("👀,🛠️,🚀") == ["👀", "🛠️", "🚀"]
    # Case 3: Sequence string with standard emojis and Variation Selectors (🛠️ has VS16)
    assert parse_emoji_sequence("👀🛠️🚀") == ["👀", "🛠️", "🚀"]
    # Case 4: Single emoji
    assert parse_emoji_sequence("👍") == ["👍"]
    # Case 5: Empty string
    assert parse_emoji_sequence("") == []


@pytest.mark.asyncio
@patch("google.auth.default")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.adapter.build")
@patch("random.choice")
async def test_incoming_message_triggers_random_reaction(
    mock_random_choice: MagicMock,
    mock_build: MagicMock,
    mock_subscriber: MagicMock,
    mock_auth_default: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test that a random emoji from the configured sequence is chosen and reacted."""
    mock_auth_default.return_value = (MagicMock(), "test-project")
    # Configure a sequence of 3 emojis
    mock_config.google_chat.reaction_emoji = "👀🛠️🚀"

    # Configure random.choice to return specific emojis sequentially for predictability
    mock_random_choice.side_effect = ["👀", "🛠️", "🚀"]

    mock_chat_client = MagicMock()
    mock_build.side_effect = [MagicMock(), mock_chat_client]

    mock_reactions = MagicMock()
    mock_chat_client.spaces.return_value.messages.return_value.reactions.return_value = mock_reactions
    mock_create = MagicMock()
    mock_reactions.create = mock_create
    mock_create.return_value.execute = MagicMock()

    event_payload = {
        "type": "MESSAGE",
        "space": {"name": "spaces/AAA"},
        "message": {
            "name": "spaces/AAA/messages/msg123",
            "text": "Run some tools!",
            "sender": {
                "displayName": "Test User",
                "name": "users/allowed_user",
                "email": "allowed@example.com",
            },
            "thread": {"name": "spaces/AAA/threads/BBB"},
        },
    }

    with patch("kesoku.gateway.chatbot.google_chat.adapter.get_config", return_value=mock_config):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)

        pubsub_msg = MagicMock(spec=pubsub_v1.subscriber.message.Message)
        pubsub_msg.data = json.dumps(event_payload).encode("utf-8")

        # 1. Handle incoming message -> should trigger randomly chosen emoji "👀"
        await bot._on_pubsub_message(pubsub_msg)
        await asyncio.sleep(0.1)

        mock_create.assert_called_once_with(
            parent="spaces/AAA/messages/msg123",
            body={"emoji": {"unicode": "👀"}},
        )
        mock_create.reset_mock()

        # 2. First tool call -> should trigger randomly chosen emoji "🛠️"
        tool_msg_1 = Message(
            id="toolcall1",
            session_id="sess123",
            chatbot_id="gchat_test",
            channel_id="spaces/AAA/threads/BBB",
            sender="Kesoku",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_CALL,
            content="",
            timestamp=time.time(),
            metadata={"tool_name": "search_web"},
        )
        await bot.handle_message(tool_msg_1)
        await asyncio.sleep(0.1)

        mock_create.assert_called_once_with(
            parent="spaces/AAA/messages/msg123",
            body={"emoji": {"unicode": "🛠️"}},
        )
        mock_create.reset_mock()

        # 3. Second tool call -> should trigger randomly chosen emoji "🚀"
        tool_msg_2 = Message(
            id="toolcall2",
            session_id="sess123",
            chatbot_id="gchat_test",
            channel_id="spaces/AAA/threads/BBB",
            sender="Kesoku",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_CALL,
            content="",
            timestamp=time.time(),
            metadata={"tool_name": "run_command"},
        )
        await bot.handle_message(tool_msg_2)
        await asyncio.sleep(0.1)

        mock_create.assert_called_once_with(
            parent="spaces/AAA/messages/msg123",
            body={"emoji": {"unicode": "🚀"}},
        )


@pytest.mark.asyncio
@patch("google.auth.default")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.adapter.build")
async def test_google_chat_trigger_cronjob(
    mock_build: MagicMock,
    mock_subscriber: MagicMock,
    mock_auth_default: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test that Google Chat trigger_cronjob creates session and posts cronjob message correctly."""
    mock_auth_default.return_value = (MagicMock(), "test-project")
    mock_chat_client = MagicMock()
    mock_build.side_effect = [MagicMock(), mock_chat_client]

    with patch("kesoku.gateway.chatbot.google_chat.adapter.get_config", return_value=mock_config):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)

        # Run trigger_cronjob
        await bot.trigger_cronjob(
            channel_id="spaces/AAAA/threads/BBBB",
            prompt_content="Run weekly database index optimize job",
            mention_user_id="user999",
        )

        # Verify session creation and message posting
        mock_gateway.create_session.assert_called_once()
        create_args = mock_gateway.create_session.call_args[1]
        assert "Google Chat Scheduled Job BBBB" in create_args["title"]

        # Verify message posted to gateway contains prompt and mention format
        mock_gateway.post.assert_called_once()
        posted_msg = mock_gateway.post.call_args[0][0]
        assert "<users/user999> Run weekly database index optimize job" in posted_msg.content
        assert posted_msg.metadata.get("is_cronjob") is True


@pytest.mark.asyncio
@patch("google.auth.default")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.adapter.build")
async def test_google_chat_slash_command_intercept(
    mock_build: MagicMock,
    mock_subscriber: MagicMock,
    mock_auth_default: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test that incoming text starting with '/' is intercepted and run as a slash command."""
    mock_auth_default.return_value = (MagicMock(), "test-project")

    mock_chat_client = MagicMock()
    mock_build.return_value = mock_chat_client
    mock_messages = MagicMock()
    mock_chat_client.spaces.return_value.messages.return_value = mock_messages
    mock_create = MagicMock()
    mock_messages.create = mock_create
    mock_create.return_value.execute = MagicMock(return_value={"name": "spaces/AAA/messages/reply123"})

    event_payload = {
        "type": "MESSAGE",
        "space": {"name": "spaces/AAA"},
        "message": {
            "name": "spaces/AAA/messages/msg123",
            "text": "/role xiaozhang",
            "sender": {
                "displayName": "Test User",
                "name": "users/allowed_user",
                "email": "allowed@example.com",
            },
            "thread": {"name": "spaces/AAA/threads/BBB"},
        },
    }

    with patch("kesoku.gateway.chatbot.google_chat.adapter.get_config", return_value=mock_config):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)

        # Spy/mock bot.commands.execute
        bot.commands.execute = AsyncMock()

        pubsub_msg = MagicMock(spec=pubsub_v1.subscriber.message.Message)
        pubsub_msg.data = json.dumps(event_payload).encode("utf-8")

        # Trigger incoming message
        await bot._on_pubsub_message(pubsub_msg)

        # Verify it intercepted and called commands.execute
        bot.commands.execute.assert_called_once()
        args, kwargs = bot.commands.execute.call_args
        assert args[0] == "role"
        assert kwargs["channel_id"] == "spaces/AAA/threads/BBB"
        assert kwargs["role_name"] == "xiaozhang"

        # Verify no session was created and no user message was posted to gateway
        mock_gateway.create_session.assert_not_called()
        mock_gateway.post.assert_not_called()


@pytest.mark.asyncio
@patch("google.auth.default")
@patch("google.cloud.pubsub_v1.SubscriberClient")
@patch("kesoku.gateway.chatbot.google_chat.adapter.build")
async def test_google_chat_slash_command_execution_reply(
    mock_build: MagicMock,
    mock_subscriber: MagicMock,
    mock_auth_default: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test that slash command execution returns the result directly to Google Chat."""
    mock_auth_default.return_value = (MagicMock(), "test-project")

    mock_chat_client = MagicMock()
    mock_build.return_value = mock_chat_client
    mock_messages = MagicMock()
    mock_chat_client.spaces.return_value.messages.return_value = mock_messages
    mock_create = MagicMock()
    mock_messages.create = mock_create
    mock_create.return_value.execute = MagicMock(return_value={"name": "spaces/AAA/messages/reply123"})

    event_payload = {
        "type": "MESSAGE",
        "space": {"name": "spaces/AAA"},
        "message": {
            "name": "spaces/AAA/messages/msg123",
            "text": "/role",  # Just query current role
            "sender": {
                "displayName": "Test User",
                "name": "users/allowed_user",
                "email": "allowed@example.com",
            },
            "thread": {"name": "spaces/AAA/threads/BBB"},
        },
    }

    # Mock gateway role responses
    mock_gateway.get_channel_role_with_inheritance = AsyncMock(return_value="default")
    # Mock list_roles behavior inside update_role_by_channel
    mock_config.workspace.roles_dir = "/mock/roles"

    with (
        patch("kesoku.gateway.chatbot.google_chat.adapter.get_config", return_value=mock_config),
        patch("os.path.exists", return_value=True),
        patch("os.path.isdir", return_value=True),
        patch("os.listdir", return_value=["default", "xiaozhang"]),
    ):
        bot = GoogleChatChatbot(chatbot_id="gchat_test", gateway=mock_gateway)

        pubsub_msg = MagicMock(spec=pubsub_v1.subscriber.message.Message)
        pubsub_msg.data = json.dumps(event_payload).encode("utf-8")

        await bot._on_pubsub_message(pubsub_msg)

        # Yield control
        await asyncio.sleep(0.1)

        # Verify message creation was called to send the role reply
        mock_messages.create.assert_called_once()
        call_kwargs = mock_messages.create.call_args[1]
        assert call_kwargs["parent"] == "spaces/AAA"
        assert call_kwargs["body"]["thread"]["name"] == "spaces/AAA/threads/BBB"
        assert "Active Persona:" in call_kwargs["body"]["text"]
        assert "xiaozhang" in call_kwargs["body"]["text"]
