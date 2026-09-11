"""Tests for the Chatbot gateway subscriber loop's failure isolation.

A chatbot has exactly one outbound delivery consumer, so an exception escaping
handle_message used to permanently (and silently) stop every reply for that bot.
"""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.gateway.chatbot.base import Chatbot
from kesoku.gateway.gateway import Gateway


def _make_message(content: str) -> Message:
    """Build a minimal outbound assistant message.

    Args:
        content: Message body.

    Returns:
        A pending assistant text Message.
    """
    return Message(
        session_id="session123",
        chatbot_id="mock_bot",
        channel_id="channel123",
        sender="Agent",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content=content,
        status=MessageStatus.PENDING,
    )


class FlakyChatbot(Chatbot):
    """Chatbot whose handler raises on messages containing 'boom'."""

    def __init__(self, chatbot_id: str, gateway: Gateway) -> None:
        """Initialize the adapter and its delivery record.

        Args:
            chatbot_id: Unique chatbot identifier.
            gateway: Gateway providing the message stream.
        """
        super().__init__(chatbot_id=chatbot_id, gateway=gateway)
        self.handled: list[str] = []

    async def handle_message(self, message: Message) -> None:
        """Record the message, blowing up on poisoned content.

        Args:
            message: The outbound message being delivered.

        Raises:
            RuntimeError: When the message content contains 'boom'.
        """
        if "boom" in message.content:
            raise RuntimeError("delivery exploded")
        self.handled.append(message.content)


def _gateway_yielding(messages: list[Message]) -> MagicMock:
    """Build a Gateway mock whose listen() yields the given messages then ends.

    Args:
        messages: Messages to emit from the subscriber stream.

    Returns:
        A Gateway mock with a stubbed listen() and async db.
    """
    gateway = MagicMock(spec=Gateway)
    gateway.db = AsyncMock()

    async def _listen(*_args: Any, **_kwargs: Any) -> AsyncGenerator[Message, None]:
        for msg in messages:
            yield msg

    gateway.listen = _listen
    return gateway


@pytest.mark.asyncio
async def test_start_survives_handler_exception() -> None:
    """A failing message must not stop delivery of the messages behind it."""
    messages = [_make_message("first"), _make_message("boom"), _make_message("third")]
    gateway = _gateway_yielding(messages)
    chatbot = FlakyChatbot(chatbot_id="mock_bot", gateway=gateway)

    await chatbot.start()

    assert chatbot.handled == ["first", "third"]
    gateway.db.update_message_status.assert_awaited_once_with(messages[1].id, MessageStatus.ERROR)


@pytest.mark.asyncio
async def test_start_propagates_cancellation() -> None:
    """Cancelling the subscriber must not be swallowed as a delivery failure."""
    gateway = MagicMock(spec=Gateway)
    gateway.db = AsyncMock()

    async def _listen(*_args: Any, **_kwargs: Any) -> AsyncGenerator[Message, None]:
        yield _make_message("first")
        await asyncio.sleep(3600)

    gateway.listen = _listen

    class SlowChatbot(Chatbot):
        """Chatbot whose handler never returns."""

        async def handle_message(self, message: Message) -> None:
            """Block forever so the task can be cancelled mid-delivery.

            Args:
                message: The outbound message being delivered.
            """
            await asyncio.sleep(3600)

    chatbot = SlowChatbot(chatbot_id="mock_bot", gateway=gateway)
    task = asyncio.create_task(chatbot.start())
    await asyncio.sleep(0.05)
    task.cancel()
    await task

    gateway.db.update_message_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_spawn_subscriber_task_logs_crash() -> None:
    """A dead subscriber task must be reported instead of failing silently."""
    gateway = MagicMock(spec=Gateway)
    gateway.db = AsyncMock()

    async def _listen(*_args: Any, **_kwargs: Any) -> AsyncGenerator[Message, None]:
        raise RuntimeError("stream died")
        yield  # pragma: no cover - makes this an async generator

    gateway.listen = _listen
    chatbot = FlakyChatbot(chatbot_id="mock_bot", gateway=gateway)

    with patch("kesoku.gateway.chatbot.base.logger") as mock_logger:
        task = chatbot.spawn_subscriber_task()
        with pytest.raises(RuntimeError):
            await task
        # The done-callback runs on the next loop iteration.
        await asyncio.sleep(0)

    messages = [str(call.args[0]) for call in mock_logger.critical.call_args_list]
    assert any("subscriber task died" in message for message in messages)
