"""CLI chatbot adapter for Kesoku one-shot chat command."""

import asyncio

from kesoku.constants import ROLE_ASSISTANT, TYPE_TEXT
from kesoku.db import Message
from kesoku.gateway.chatbot.base import Chatbot
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger

logger = setup_logger(__name__)


class CLIChatbot(Chatbot):
    """CLI chatbot adapter for synchronous command execution using pub/sub subscriber pattern."""

    def __init__(self, chatbot_id: str, gateway: Gateway) -> None:
        """Initialize the CLI chatbot.

        Args:
            chatbot_id: Unique identifier (typically 'cli').
            gateway: Reference to the Kesoku Gateway.
        """
        super().__init__(chatbot_id, gateway)
        self.response_event = asyncio.Event()
        self.final_response: str | None = None

    async def handle_message(self, message: Message) -> None:
        """Receive a response from the agent and signal the awaiting CLI command.

        Args:
            message: The outgoing Message instance.
        """
        if message.role != ROLE_ASSISTANT or message.type != TYPE_TEXT:
            # Intermediate tool calls, tool results, or thoughts can be logged or ignored for final CLI completion
            return
        self.final_response = message.content
        self.response_event.set()
        logger.debug(f"CLIChatbot received final response for channel {message.channel_id}")
