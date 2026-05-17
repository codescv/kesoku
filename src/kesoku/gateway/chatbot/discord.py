"""Discord chatbot adapter for Kesoku (Stub)."""

from kesoku.db import Message
from kesoku.gateway.chatbot.base import Chatbot
from kesoku.gateway.gateway import Gateway


class DiscordChatbot(Chatbot):
    """Discord chatbot adapter (Planned for future steps)."""

    def __init__(self, chatbot_id: str, gateway: Gateway) -> None:
        """Initialize the Discord chatbot stub."""
        super().__init__(chatbot_id, gateway)
        raise NotImplementedError("Discord chatbot is not implemented in the first step.")

    async def handle_message(self, message: Message) -> None:
        """Process an outgoing message for Discord."""
        pass
