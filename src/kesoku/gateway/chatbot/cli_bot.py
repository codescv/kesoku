"""CLI chatbot adapter for Kesoku one-shot chat command."""

import asyncio
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from kesoku.constants import (
    ROLE_ASSISTANT,
    ROLE_TOOL,
    STATUS_DELIVERED,
    TYPE_TEXT,
    TYPE_THOUGHT,
    TYPE_TOOL_CALL,
    TYPE_TOOL_RESULT,
)
from kesoku.db import Message
from kesoku.gateway.chatbot.base import Chatbot, parse_message_content
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

        if message.role == ROLE_ASSISTANT and self.console:
            if message.type == TYPE_THOUGHT:
                self.console.print(
                    Panel(
                        Markdown(message.content),
                        title=f"[bold cyan]💭 Thought ({message.sender})[/bold cyan]",
                        title_align="left",
                        border_style="cyan",
                    )
                )
                return
            elif message.type == TYPE_TEXT:
                self.final_response = message.content
                segments = parse_message_content(message.content)
                for segment in segments:
                    if segment["type"] == "text":
                        text_content = segment["content"]
                        if text_content.strip():
                            self.console.print(
                                Panel(
                                    Markdown(text_content),
                                    title=f"[bold blue]{message.sender}[/bold blue]",
                                    title_align="left",
                                    border_style="blue",
                                )
                            )
                    elif segment["type"] == "file":
                        file_path = segment["path"]
                        self.console.print(
                            Panel(
                                f"📎 [bold cyan]File Attachment:[/bold cyan] [underline]{file_path}[/underline]",
                                title=f"[bold blue]{message.sender} (Attachment)[/bold blue]",
                                title_align="left",
                                border_style="cyan",
                            )
                        )
                    elif segment["type"] == "voice":
                        file_path = segment["path"]
                        self.console.print(
                            Panel(
                                f"🎙️ [bold green]Voice Message:[/bold green] [underline]{file_path}[/underline]",
                                title=f"[bold blue]{message.sender} (Voice)[/bold blue]",
                                title_align="left",
                                border_style="green",
                            )
                        )
                    elif segment["type"] == "question":
                        question_text = segment["question"]
                        choices = segment["choices"]
                        choices_str = " | ".join(choices)
                        self.console.print(
                            Panel(
                                f"{question_text}\n\n[bold cyan]Choices:[/bold cyan] {choices_str}",
                                title=f"[bold blue]{message.sender} (Question)[/bold blue]",
                                title_align="left",
                                border_style="cyan",
                            )
                        )
                await self.gateway.update_message_status(message.id, STATUS_DELIVERED)
                self.final_response_event.set()
                logger.debug(f"CLIChatbot received final response for channel {message.channel_id}")
                return
