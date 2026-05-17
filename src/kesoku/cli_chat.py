"""One-shot CLI chat runner and session history display utilities for Kesoku."""

import asyncio
import sys
import time
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from kesoku.agent.agent import Agent
from kesoku.constants import (
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL,
    ROLE_USER,
    STATUS_PENDING_AGENT,
    TYPE_TEXT,
    TYPE_TOOL_CALL,
)
from kesoku.db import Message
from kesoku.gateway.chatbot.cli_bot import CLIChatbot
from kesoku.gateway.gateway import Gateway


async def _list_chat_sessions(gateway: Gateway, console: Console) -> None:
    """Retrieve and render a formatted table of all existing chat sessions.

    Args:
        gateway: Gateway instance.
        console: Rich console instance.
    """
    sessions = await gateway.list_sessions()
    if not sessions:
        console.print("[yellow]No chat sessions found.[/yellow]")
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


async def _show_session_history(gateway: Gateway, console: Console, session_id: str) -> None:
    """Retrieve and display the full formatted conversation history for a session.

    Args:
        gateway: Gateway instance.
        console: Rich console instance.
        session_id: Target session ID.
    """
    session = await gateway.get_session(session_id)
    if not session:
        console.print(f"[bold red]Error: Session '{session_id}' not found.[/bold red]")
        sys.exit(1)
    history = await gateway.get_session_history(session_id=session_id, limit=100)
    if not history:
        console.print(f"[yellow]Session '{session_id}' has no recorded messages.[/yellow]")
        return
    console.print(f"\n[bold cyan]Chat History for Session '{session_id}' ({session.title})[/bold cyan]\n")
    for msg in history:
        if msg.role == ROLE_USER:
            console.print(
                Panel(msg.content, title=f"[bold green]{msg.sender}[/bold green]", title_align="left", border_style="green")
            )
        elif msg.role == ROLE_TOOL:
            if msg.type == TYPE_TOOL_CALL:
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
        elif msg.role == ROLE_ASSISTANT:
            console.print(
                Panel(
                    Markdown(msg.content),
                    title=f"[bold blue]{msg.sender}[/bold blue]",
                    title_align="left",
                    border_style="blue",
                )
            )
        elif msg.role == ROLE_SYSTEM:
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
    list_sessions: bool,
    resume: str | None,
    resume_latest: bool,
    show_history: str | None,
) -> None:
    """Asynchronous runner for one-shot CLI chat session.

    Args:
        message: Latest message string or None.
        list_sessions: True if listing sessions.
        resume: Session ID string to resume or None.
        resume_latest: True if resuming latest session.
        show_history: Session ID string to view history or None.
    """
    console = Console()
    gateway = Gateway()

    if list_sessions:
        await _list_chat_sessions(gateway, console)
        return

    if show_history:
        await _show_session_history(gateway, console, show_history)
        return

    if not message:
        console.print(
            "[bold red]Error: Please provide a message, or use -l to list sessions "
            "or --show-history to view history.[/bold red]"
        )
        sys.exit(1)

    # Handle session identification for sending a message
    if resume:
        session_id = resume
        session = await gateway.get_session(session_id)
        if not session:
            console.print(
                f"[bold red]Error: Session '{session_id}' not found. Use -l to list available sessions.[/bold red]"
            )
            sys.exit(1)
        await gateway.update_session_updated_at(session_id)
    elif resume_latest:
        latest = await gateway.get_latest_session()
        if not latest:
            console.print("[yellow]No existing sessions found. Starting a new session.[/yellow]")
            title = message[:40] + ("..." if len(message) > 40 else "")
            sess = await gateway.create_session(title=title)
            session_id = sess.id
        else:
            session_id = latest.id
            console.print(f"[dim]Resuming latest session: '{session_id}' ({latest.title})[/dim]")
            await gateway.update_session_updated_at(session_id)
    else:
        title = message[:40] + ("..." if len(message) > 40 else "")
        sess = await gateway.create_session(title=title)
        session_id = sess.id
        console.print(f"[dim]Started new session: '{session_id}'[/dim]")

    cli_bot = CLIChatbot(chatbot_id="cli", gateway=gateway)
    bot_task = asyncio.create_task(cli_bot.start())
    agent = Agent(gateway=gateway)
    agent_task = asyncio.create_task(agent.start())

    # Ingest user message
    msg = Message(
        session_id=session_id,
        chatbot_id="cli",
        channel_id="cli_local",
        sender="User",
        role=ROLE_USER,
        type=TYPE_TEXT,
        content=message,
        status=STATUS_PENDING_AGENT,
    )
    await gateway.post(msg)

    # Display user prompt panel
    console.print(Panel(message, title="[bold green]You[/bold green]", title_align="left", border_style="green"))

    # Await response with spinner
    try:
        with console.status("[bold cyan]Kesoku Agent is thinking..."):
            await cli_bot.response_event.wait()

        if cli_bot.final_response:
            console.print(
                Panel(
                    Markdown(cli_bot.final_response),
                    title="[bold blue]Kesoku Agent[/bold blue]",
                    title_align="left",
                    border_style="blue",
                )
            )
    finally:
        cli_bot.stop()
        agent.stop()
        await asyncio.gather(bot_task, agent_task, return_exceptions=True)
