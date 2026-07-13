"""One-shot CLI chat runner and session history display utilities for Kesoku."""

import asyncio
import sys
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from kesoku.agent.agent import Agent
from kesoku.agent.history import build_history
from kesoku.agent.prompt import build_sys_prompt
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.gateway.chatbot.cli_bot import CLIChatbot
from kesoku.gateway.gateway import Gateway
from kesoku.logger import console, setup_logger

logger = setup_logger(__name__)


def render_message(console: Console, msg: Message) -> None:
    """Render a single conversational message as a formatted Rich panel.

    Args:
        console: Rich console instance.
        msg: Message instance.
    """
    if msg.role == MessageRole.USER:
        console.print(
            Panel(msg.content, title=f"[bold green]{msg.sender}[/bold green]", title_align="left", border_style="green")
        )
    elif msg.role == MessageRole.TOOL:
        if msg.type == MessageType.TOOL_CALL:
            console.print(
                Panel(
                    Markdown(msg.content),
                    title=f"[bold yellow]🛠️ Tool Call ({msg.sender})[/bold yellow]",
                    title_align="left",
                    border_style="yellow",
                )
            )
        else:
            console.print(
                Panel(
                    Markdown(msg.content),
                    title=f"[bold magenta]📥 Tool Output ({msg.sender})[/bold magenta]",
                    title_align="left",
                    border_style="magenta",
                )
            )
    elif msg.role == MessageRole.ASSISTANT:
        if msg.type == MessageType.THOUGHT:
            console.print(
                Panel(
                    Markdown(msg.content),
                    title=f"[bold cyan]💭 Thought ({msg.sender})[/bold cyan]",
                    title_align="left",
                    border_style="cyan",
                )
            )
        else:
            console.print(
                Panel(
                    Markdown(msg.content),
                    title=f"[bold blue]{msg.sender}[/bold blue]",
                    title_align="left",
                    border_style="blue",
                )
            )
    elif msg.role == MessageRole.SYSTEM:
        console.print(
            Panel(
                Markdown(msg.content),
                title=f"[bold dim]{msg.sender}[/bold dim]",
                title_align="left",
                border_style="dim",
            )
        )


async def run_cli_chat_async(
    message: str | None,
    resume: str | None,
    resume_latest: bool,
) -> None:
    """Asynchronous runner for one-shot CLI chat session.

    Args:
        message: Latest message string.
        resume: Session ID string to resume or None.
        resume_latest: True if resuming latest session.
    """
    gateway = Gateway()

    if not message:
        logger.error("Please provide a message to send.")
        sys.exit(1)

    # Handle session identification for sending a message
    is_resumed = False
    if resume:
        session_id = resume
        session = await gateway.db.get_session(session_id)
        if not session:
            logger.error(f"Session '{session_id}' not found. Use 'kesoku history list' to list available sessions.")
            sys.exit(1)
        await gateway.db.update_session_updated_at(session_id, time.time())
        is_resumed = True
    elif resume_latest:
        latest = await gateway.db.get_latest_session()
        if not latest:
            logger.info("No existing sessions found. Starting a new session.")
            title = message[:40] + ("..." if len(message) > 40 else "")
            sess = await gateway.create_session(title=title)
            session_id = sess.id
        else:
            session_id = latest.id
            logger.info(f"Resuming latest session: '{session_id}' ({latest.title})")
            await gateway.db.update_session_updated_at(session_id, time.time())
            is_resumed = True
    else:
        title = message[:40] + ("..." if len(message) > 40 else "")
        sess = await gateway.create_session(title=title)
        session_id = sess.id
        logger.info(f"Started new session: '{session_id}'")

    cli_bot = CLIChatbot(chatbot_id="cli", gateway=gateway, session_id=session_id, console=console)
    bot_task = asyncio.create_task(cli_bot.start())
    agent = Agent(gateway=gateway)
    agent_task = asyncio.create_task(agent.start())

    order = "grouped"
    history = await build_history(gateway=gateway, session_id=session_id, order=order, heal_orphans=False)
    session = await gateway.db.get_session(session_id)
    if is_resumed and session:
        new_sys_prompt = build_sys_prompt(session=session)
        await gateway.db.update_session_system_prompt(session_id, new_sys_prompt)
        session = await gateway.db.get_session(session_id)

    if is_resumed:
        logger.info(f"Resuming Session '{session_id}' History:")
        if session and session.system_prompt:
            sys_msg = Message(
                session_id=session_id,
                chatbot_id="system",
                channel_id="system",
                sender="System",
                role=MessageRole.SYSTEM,
                type=MessageType.TEXT,
                content=session.system_prompt,
                status=MessageStatus.RESPONDED,
                timestamp=session.created_at - 0.01,
            )
            render_message(console, sys_msg)
        for m in history:
            render_message(console, m)
    else:
        if session and session.system_prompt:
            sys_msg = Message(
                session_id=session_id,
                chatbot_id="system",
                channel_id="system",
                sender="System",
                role=MessageRole.SYSTEM,
                type=MessageType.TEXT,
                content=session.system_prompt,
                status=MessageStatus.RESPONDED,
                timestamp=session.created_at - 0.01,
            )
            render_message(console, sys_msg)

    # Ingest user message
    msg = Message(
        session_id=session_id,
        chatbot_id="cli",
        channel_id="cli_local",
        sender="User",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content=message,
        status=MessageStatus.PENDING_AGENT,
    )
    await gateway.post(msg)

    # Display user prompt panel
    console.print(Panel(message, title="[bold green]You[/bold green]", title_align="left", border_style="green"))

    # Await response with spinner
    try:
        with console.status("[bold cyan]Kesoku Agent is thinking..."):
            await cli_bot.final_response_event.wait()

    finally:
        cli_bot.stop()
        agent.stop()
        await asyncio.gather(bot_task, agent_task, return_exceptions=True)
