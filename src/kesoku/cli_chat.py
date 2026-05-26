"""One-shot CLI chat runner and session history display utilities for Kesoku."""

import asyncio
import sys
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from kesoku.agent.agent import Agent
from kesoku.agent.history import build_clean_history
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.gateway.chatbot.cli_bot import CLIChatbot
from kesoku.gateway.gateway import Gateway
from kesoku.logger import console, setup_logger

logger = setup_logger(__name__)


def _render_message(console: Console, msg: Message) -> None:
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


async def _list_chat_sessions(gateway: Gateway, console: Console) -> None:
    """Retrieve and render a formatted table of all existing chat sessions.

    Args:
        gateway: Gateway instance.
        console: Rich console instance.
    """
    sessions = await gateway.list_sessions()
    if not sessions:
        logger.info("No chat sessions found.")
        return
    table = Table(title="Kesoku Chat Sessions", show_header=True, header_style="bold cyan")
    table.add_column("Session ID", style="bold green")
    table.add_column("Created At", style="dim")
    table.add_column("Last Updated", style="dim")
    table.add_column("Title", style="magenta")
    for s in sessions:
        created = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s.created_at))
        updated = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s.updated_at))
        table.add_row(s.id, created, updated, s.title)
    console.print(table)


async def _show_session_history(gateway: Gateway, console: Console, session_id: str, grouped: bool = False) -> None:
    """Retrieve and display the full formatted conversation history for a session.

    Args:
        gateway: Gateway instance.
        console: Rich console instance.
        session_id: Target session ID.
        grouped: True to sort history by grouping tool call and result together.
    """
    session = await gateway.get_session(session_id)
    if not session:
        logger.error(f"Session '{session_id}' not found.")
        sys.exit(1)
    order = "grouped" if grouped else "phased"
    history = await build_clean_history(gateway=gateway, session_id=session_id, order=order, heal_orphans=False)
    if not history:
        logger.warning(f"Session '{session_id}' has no recorded messages.")
        return
    logger.info(f"Chat History for Session '{session_id}' ({session.title}):")
    for msg in history:
        _render_message(console, msg)


async def run_cli_chat_async(
    message: str | None,
    list_sessions: bool,
    resume: str | None,
    resume_latest: bool,
    show_history: str | None,
    grouped: bool = False,
) -> None:
    """Asynchronous runner for one-shot CLI chat session.

    Args:
        message: Latest message string or None.
        list_sessions: True if listing sessions.
        resume: Session ID string to resume or None.
        resume_latest: True if resuming latest session.
        show_history: Session ID string to view history or None.
        grouped: True to sort history by grouping tool call and result together.
    """
    gateway = Gateway()

    if list_sessions:
        await _list_chat_sessions(gateway, console)
        return

    if show_history:
        await _show_session_history(gateway, console, show_history, grouped=grouped)
        return

    if not message:
        logger.error("Please provide a message, or use -l to list sessions or --show-history to view history.")
        sys.exit(1)

    # Handle session identification for sending a message
    is_resumed = False
    if resume:
        session_id = resume
        session = await gateway.get_session(session_id)
        if not session:
            logger.error(f"Session '{session_id}' not found. Use -l to list available sessions.")
            sys.exit(1)
        await gateway.update_session_updated_at(session_id)
        is_resumed = True
    elif resume_latest:
        latest = await gateway.get_latest_session()
        if not latest:
            logger.info("No existing sessions found. Starting a new session.")
            title = message[:40] + ("..." if len(message) > 40 else "")
            sess = await gateway.create_session(title=title)
            session_id = sess.id
        else:
            session_id = latest.id
            logger.info(f"Resuming latest session: '{session_id}' ({latest.title})")
            await gateway.update_session_updated_at(session_id)
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

    order = "grouped" if grouped else "phased"
    history = await build_clean_history(gateway=gateway, session_id=session_id, order=order, heal_orphans=False)
    if is_resumed:
        logger.info(f"Resuming Session '{session_id}' History:")
        for m in history:
            _render_message(console, m)
    else:
        for m in history:
            if m.role == MessageRole.SYSTEM:
                _render_message(console, m)
                break

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
