"""CLI chatbot adapter for Kesoku one-shot chat command."""

import asyncio
import time
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from kesoku.constants import (
    ROLE_ASSISTANT,
    ROLE_TOOL,
    STATUS_COMPLETED,
    TYPE_TEXT,
    TYPE_TOOL_CALL,
    TYPE_TOOL_RESULT,
)
from kesoku.db import Message
from kesoku.gateway.chatbot.base import Chatbot
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger

logger = setup_logger(__name__)


class CLIChatbot(Chatbot):
    """CLI chatbot adapter for synchronous command execution using pub/sub subscriber pattern."""

    def __init__(
        self,
        chatbot_id: str,
        gateway: Gateway,
        session_id: str | None = None,
        console: Console | None = None,
    ) -> None:
        """Initialize the CLI chatbot.

        Args:
            chatbot_id: Unique identifier (typically 'cli').
            gateway: Reference to the Kesoku Gateway.
            session_id: Optional internal session ID.
            console: Optional Rich Console instance for rendering tool messages.
        """
        super().__init__(chatbot_id, gateway, session_id=session_id)
        self.console = console
        self.start_time = time.time()
        self.final_response_event = asyncio.Event()
        self.final_response: str | None = None

    async def handle_message(self, message: Message) -> None:
        """Receive messages from the agent and render tools or signal final completion.

        Args:
            message: The outgoing Message instance.
        """
        if message.timestamp < self.start_time:
            return

        if message.role == ROLE_TOOL and self.console:
            if message.type == TYPE_TOOL_CALL:
                self.console.print(
                    Panel(
                        Markdown(message.content),
                        title=f"[bold yellow]🛠️ Tool Call ({message.sender})[/bold yellow]",
                        title_align="left",
                        border_style="yellow",
                    )
                )
            elif message.type == TYPE_TOOL_RESULT:
                self.console.print(
                    Panel(
                        Markdown(message.content),
                        title=f"[bold magenta]📥 Tool Output ({message.sender})[/bold magenta]",
                        title_align="left",
                        border_style="magenta",
                    )
                )
            return

        if message.role != ROLE_ASSISTANT or message.type != TYPE_TEXT:
            # Intermediate thoughts or other messages can be ignored for final CLI completion
            return
        self.final_response = message.content
        await self.gateway.update_message_status(message.id, STATUS_COMPLETED)
        self.final_response_event.set()
        logger.debug(f"CLIChatbot received final response for channel {message.channel_id}")
