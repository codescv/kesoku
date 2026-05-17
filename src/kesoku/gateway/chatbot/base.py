"""Base class for Kesoku chatbot adapters."""

import asyncio
from abc import ABC, abstractmethod

from kesoku.constants import ROLE_USER, STATUS_COMPLETED, STATUS_PENDING
from kesoku.db import Message
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger

logger = setup_logger(__name__)


class Chatbot(ABC):
    """Abstract base class for chatbot adapters connecting to Kesoku Gateway."""

    def __init__(self, chatbot_id: str, gateway: Gateway) -> None:
        """Initialize the chatbot with a unique identifier and gateway instance.

        Args:
            chatbot_id: Unique identifier for this chatbot instance (e.g., 'console', 'discord_primary').
            gateway: The Kesoku Gateway instance managing routing and persistence.
        """
        self.chatbot_id = chatbot_id
        self.gateway = gateway
        self._listener_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start listening as a decentralized subscriber for model responses.

        Subscribes to gateway messages matching this chatbot_id and routes non-user
        messages to handle_message.
        """
        self._listener_task = asyncio.current_task()
        try:
            async for msg in self.gateway.listen(chatbot_id=self.chatbot_id, status=STATUS_PENDING):
                if msg.role == ROLE_USER:
                    continue
                await self.handle_message(msg)
        except asyncio.CancelledError:
            logger.debug(f"Chatbot '{self.chatbot_id}' listener cancelled.")

    def stop(self) -> None:
        """Stop the subscriber listener task."""
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()

    @abstractmethod
    async def handle_message(self, message: Message) -> None:
        """Process an outgoing message (e.g., tool call, thought, or final assistant text).

        Args:
            message: The Message instance to handle.
        """
        pass
