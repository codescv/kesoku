"""Cronjob virtual chatbot adapter for Kesoku AI Agent framework.

Acts as a silent sink/receiver for cronjobs that execute without an active user-facing interface.
"""

from typing import Any

from kesoku.constants import MessageStatus
from kesoku.db import Message
from kesoku.gateway.chatbot.base import Chatbot
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger

logger = setup_logger(__name__)


class CronjobChatbot(Chatbot):
    """Virtual chatbot that acts as a silent receiver for automated tasks."""

    def __init__(self, chatbot_id: str, gateway: Gateway) -> None:
        """Initialize the CronjobChatbot.

        Args:
            chatbot_id: Unique identifier for this chatbot instance (e.g., 'cronjob').
            gateway: Gateway instance.
        """
        super().__init__(chatbot_id, gateway)
        logger.info(f"Virtual chatbot '{chatbot_id}' successfully initialized.")

    async def handle_message(self, message: Message) -> None:
        """Automatically accept and drop outgoing messages from this virtual chatbot.

        Args:
            message: Outgoing message from the agent.
        """
        logger.debug(f"CronjobChatbot '{self.chatbot_id}' silently dropping outgoing message {message.id}")
        await self.gateway.db.update_message_status(message.id, MessageStatus.DELIVERED)

    async def trigger_cronjob(
        self,
        channel_id: str,
        prompt_content: str,
        mention_user_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Trigger a scheduled cronjob in the silent virtual session.

        Args:
            channel_id: Virtual channel identifier.
            prompt_content: The prompt message content to run.
            mention_user_id: Unused parameter for virtual chatbot.
            **kwargs: Additional optional arguments.
        """
        tag = kwargs.get("tag")
        await self.trigger_cronjob_message(
            channel_id=channel_id,
            prompt_content=prompt_content,
            sender_name="Cronjob",
            custom_prompt=None,
            metadata={"is_cronjob": True, "is_silent": True},
            tag=tag,
        )
