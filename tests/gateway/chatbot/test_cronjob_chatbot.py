"""Unit tests for CronjobChatbot and silent cronjob scheduling."""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

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
    mock_gateway.update_message_status = AsyncMock()

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
    mock_gateway.update_message_status.assert_called_once_with("msg123", MessageStatus.DELIVERED)


@pytest.mark.asyncio
async def test_cronjob_chatbot_trigger_cronjob() -> None:
    """Verify that trigger_cronjob correctly posts PENDING_AGENT message to Gateway."""
    mock_gateway = MagicMock(spec=Gateway)
    mock_session = MagicMock()
    mock_session.id = "sess123"
    mock_gateway.get_session_by_channel = AsyncMock(return_value=mock_session)
    mock_gateway.post = AsyncMock()

    chatbot = CronjobChatbot(chatbot_id="cronjob", gateway=mock_gateway)

    await chatbot.trigger_cronjob(
        channel_id="silent_0",
        prompt_content="Test silent cron content",
    )

    mock_gateway.get_session_by_channel.assert_called_once_with("cronjob", "silent_0")
    mock_gateway.update_session_updated_at.assert_called_once_with("sess123")
    mock_gateway.post.assert_called_once()

    posted_msg = mock_gateway.post.call_args[0][0]
    assert posted_msg.chatbot_id == "cronjob"
    assert posted_msg.channel_id == "silent_0"
    assert posted_msg.session_id == "sess123"
    assert posted_msg.role == MessageRole.USER
    assert posted_msg.status == MessageStatus.PENDING_AGENT
    assert "Test silent cron content" in posted_msg.content
    assert posted_msg.metadata.get("is_silent") is True


@pytest.mark.asyncio
async def test_cron_manager_defaults_to_cronjob_silent():
    """Verify that CronManager automatically assigns channel_id when chatbot_id='cronjob' but channel_id is omitted."""
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
            "chatbot_id": "cronjob",
            # channel_id is omitted
        }

        manager = CronManager(chatbots=[mock_bot], config_dir=tmpdir_real)
        await manager._check_and_trigger_jobs([job])
        await asyncio.sleep(0.1)

        mock_bot.trigger_cronjob.assert_called_once_with(
            channel_id="silent_0",
            prompt_content="Silent cron job content",
            mention_user_id=None,
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
