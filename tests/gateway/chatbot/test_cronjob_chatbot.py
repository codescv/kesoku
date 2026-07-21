"""Unit tests for CronjobChatbot and silent cronjob scheduling."""

import asyncio
import os
import tempfile
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.cron import CronManager
from kesoku.db import Message
from kesoku.gateway.chatbot.cronjob import CronjobChatbot
from kesoku.gateway.gateway import Gateway


@pytest.mark.asyncio
async def test_cronjob_chatbot_handle_message() -> None:
    """Verify that CronjobChatbot silently accepts and marks messages as DELIVERED."""
    mock_gateway = MagicMock(spec=Gateway)
    mock_db = AsyncMock()
    mock_gateway.db = mock_db

    chatbot = CronjobChatbot(chatbot_id="cronjob", gateway=mock_gateway)
    message = Message(
        id="msg123",
        session_id="session123",
        chatbot_id="cronjob",
        channel_id="silent_0",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content="Hello world",
    )

    await chatbot.handle_message(message)
    mock_gateway.db.update_message_status.assert_called_once_with("msg123", MessageStatus.DELIVERED)


@pytest.mark.asyncio
async def test_cronjob_chatbot_trigger_cronjob() -> None:
    """Verify that trigger_cronjob correctly posts PENDING_AGENT message to Gateway with unique channel ID."""
    mock_gateway = MagicMock(spec=Gateway)
    mock_db = AsyncMock()
    mock_gateway.db = mock_db
    mock_session = MagicMock()
    mock_session.id = "sess123"
    mock_db.get_session_by_channel = AsyncMock(return_value=None)
    mock_gateway.create_session = AsyncMock(return_value=mock_session)
    mock_gateway.post = AsyncMock()

    chatbot = CronjobChatbot(chatbot_id="cronjob", gateway=mock_gateway)

    await chatbot.trigger_cronjob(
        channel_id="silent_0",
        prompt_content="Test silent cron content",
    )

    mock_db.get_session_by_channel.assert_called_once()
    called_channel_id = mock_db.get_session_by_channel.call_args[0][1]
    assert called_channel_id.startswith("silent_0_")

    mock_gateway.create_session.assert_called_once_with(
        session_id=None,
        title=ANY,
        custom_prompt=None,
        chatbot_id="cronjob",
        channel_id=called_channel_id,
        role="default",
    )
    mock_gateway.post.assert_called_once()

    posted_msg = mock_gateway.post.call_args[0][0]
    assert posted_msg.chatbot_id == "cronjob"
    assert posted_msg.channel_id == called_channel_id
    assert posted_msg.session_id == "sess123"
    assert posted_msg.role == MessageRole.USER
    assert posted_msg.status == MessageStatus.PENDING_AGENT
    assert "Test silent cron content" in posted_msg.content
    assert posted_msg.metadata.get("is_silent") is True
    assert posted_msg.metadata.get("parent_channel_id") == "silent_0"


@pytest.mark.asyncio
async def test_cron_manager_defaults_to_cronjob_silent():
    """Verify that CronManager automatically assigns channel_id when chatbot_id='cronjob' but channel_id is omitted."""
    mock_bot = MagicMock()
    mock_bot.chatbot_id = "cronjob"
    mock_bot.trigger_cronjob = AsyncMock()

    mock_gateway = MagicMock()
    mock_db = AsyncMock()
    mock_db.get_cronjob_sent_stats_today.return_value = (0, None)
    mock_db.get_last_message_timestamp.return_value = None
    mock_gateway.db = mock_db
    mock_bot.gateway = mock_gateway

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_real = os.path.realpath(tmpdir)
        prompt_file_path = os.path.join(tmpdir_real, "silent_prompt.md")
        with open(prompt_file_path, "w") as f:
            f.write("Silent cron job content")

        job = {
            "schedule": "* * * * *",
            "prompt": "silent_prompt.md",
            "chatbot_id": "cronjob",
            # channel_id is omitted
        }

        from kesoku.cron import _get_silent_channel_id
        expected_channel_id = _get_silent_channel_id(job)
        assert expected_channel_id.startswith("cronjob_silent_")

        manager = CronManager(chatbots=[mock_bot], config_dir=tmpdir_real)
        await manager._check_and_trigger_jobs([job])
        await asyncio.sleep(0.1)

        mock_bot.trigger_cronjob.assert_called_once_with(
            channel_id=expected_channel_id,
            prompt_content="Silent cron job content",
            mention_user_id=None,
            tag=None,
            role=None,
        )


@pytest.mark.asyncio
async def test_cron_manager_silent_with_role():
    """Verify that CronManager passes role and generates correct channel_id when role is specified."""
    mock_bot = MagicMock()
    mock_bot.chatbot_id = "cronjob"
    mock_bot.trigger_cronjob = AsyncMock()

    mock_gateway = MagicMock()
    mock_db = AsyncMock()
    mock_db.get_cronjob_sent_stats_today.return_value = (0, None)
    mock_db.get_last_message_timestamp.return_value = None
    mock_gateway.db = mock_db
    mock_bot.gateway = mock_gateway

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_real = os.path.realpath(tmpdir)
        prompt_file_path = os.path.join(tmpdir_real, "silent_prompt.md")
        with open(prompt_file_path, "w") as f:
            f.write("Silent cron job content")

        job = {
            "schedule": "* * * * *",
            "prompt": "silent_prompt.md",
            "chatbot_id": "cronjob",
            "role": "custom_role",
        }

        from kesoku.cron import _get_silent_channel_id
        expected_channel_id = _get_silent_channel_id(job)

        manager = CronManager(chatbots=[mock_bot], config_dir=tmpdir_real)
        await manager._check_and_trigger_jobs([job])
        await asyncio.sleep(0.1)

        mock_bot.trigger_cronjob.assert_called_once_with(
            channel_id=expected_channel_id,
            prompt_content="Silent cron job content",
            mention_user_id=None,
            tag=None,
            role="custom_role",
        )


@pytest.mark.asyncio
async def test_cron_manager_missing_chatbot_id():
    """Verify that CronManager logs an error and returns early if chatbot_id is missing."""
    mock_bot = MagicMock()
    mock_bot.chatbot_id = "cronjob"
    mock_bot.trigger_cronjob = AsyncMock()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_real = os.path.realpath(tmpdir)
        prompt_file_path = os.path.join(tmpdir_real, "silent_prompt.md")
        with open(prompt_file_path, "w") as f:
            f.write("Silent cron job content")

        job = {
            "schedule": "* * * * *",
            "prompt": "silent_prompt.md",
            # chatbot_id is omitted entirely
        }

        manager = CronManager(chatbots=[mock_bot], config_dir=tmpdir_real)
        await manager._check_and_trigger_jobs([job])
        await asyncio.sleep(0.1)

        mock_bot.trigger_cronjob.assert_not_called()


@pytest.mark.asyncio
async def test_cronjob_chatbot_trigger_cronjob_with_role() -> None:
    """Verify that trigger_cronjob correctly passes role to trigger_cronjob_message."""
    mock_gateway = MagicMock(spec=Gateway)
    mock_db = AsyncMock()
    mock_gateway.db = mock_db
    mock_session = MagicMock()
    mock_session.id = "sess123"
    mock_db.get_session_by_channel = AsyncMock(return_value=None)
    mock_gateway.create_session = AsyncMock(return_value=mock_session)
    mock_gateway.post = AsyncMock()

    chatbot = CronjobChatbot(chatbot_id="cronjob", gateway=mock_gateway)

    await chatbot.trigger_cronjob(
        channel_id="cronjob_silent_12345678",
        prompt_content="Test silent cron content",
        role="special_role",
    )

    mock_gateway.create_session.assert_called_once_with(
        session_id=None,
        title=ANY,
        custom_prompt=None,
        chatbot_id="cronjob",
        channel_id=ANY,
        role="special_role",
    )
