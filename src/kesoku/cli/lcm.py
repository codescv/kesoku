"""CLI commands to inspect and manage context compression for Kesoku."""

import asyncio
import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kesoku.agent.history import build_history
from kesoku.config import get_config, load_config
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager
from kesoku.gateway.gateway import Gateway

lcm_app = typer.Typer(help="Inspect, audit, and manage context compression forest states.")


@lcm_app.command("status")
def lcm_status(
    session_id: Annotated[
        str | None, typer.Argument(help="Session ID to inspect. Omit to use the latest session.")
    ] = None,
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
) -> None:
    """Show context compression status, active summary nodes, and token savings for a session."""
    load_config(config)
    cfg = get_config()
    db = DatabaseManager(cfg.workspace.db_path)
    db.verify_db()

    console = Console()

    # Resolve session
    if session_id:
        session = db.get_session(session_id)
    else:
        session = db.get_latest_session()

    if not session:
        console.print("[bold red]Error: No session found![/bold red]")
        sys.exit(1)

    session_id = session.id
    console.print(f"\n[bold green]=== Compression Status for Session '{session_id}' ({session.title}) ===[/bold green]")

    try:
        # Retrieve all summary nodes
        all_nodes = db.get_all_summary_nodes(session_id)
        root_nodes = db.get_root_summary_nodes(session_id)

        # Count levels
        levels_map = {}
        for nd in all_nodes:
            levels_map[nd.level] = levels_map.get(nd.level, 0) + 1

        total_source_tokens = sum(nd.source_token_count for nd in all_nodes if nd.level == 0)
        total_compacted_tokens = sum(nd.token_count for nd in root_nodes)
        savings = round(total_source_tokens / total_compacted_tokens, 1) if total_compacted_tokens > 0 else 1.0

        # Get raw history turns count
        context = KesokuContext(config=cfg, db=db)
        gateway = Gateway(context=context)
        history = asyncio.run(build_history(gateway=gateway, session_id=session_id, heal_orphans=False))
        total_raw_turns = len(history) if history else 0

        # Print results using a beautiful table
        table = Table(show_header=False, box=None)
        table.add_row("[bold cyan]Total Raw Turns (History):[/bold cyan]", f"{total_raw_turns} messages")
        table.add_row("[bold cyan]Active root nodes in context:[/bold cyan]", f"{len(root_nodes)} nodes")
        table.add_row("[bold cyan]Total summary nodes in DB:[/bold cyan]", f"{len(all_nodes)} nodes")

        for lvl, count in sorted(levels_map.items()):
            table.add_row(f"[dim]  - Level {lvl} summaries:[/dim]", f"{count} nodes")

        if len(all_nodes) > 0:
            table.add_row(
                "[bold green]Historical Token Savings:[/bold green]",
                f"{total_source_tokens} source tokens compressed to "
                f"{total_compacted_tokens} active tokens ({savings}x savings)",
            )

        console.print(table)
        console.print()

    except Exception as e:
        console.print(f"[bold red]Error retrieving compression status: {e}[/bold red]")
        sys.exit(1)


@lcm_app.command("view")
def lcm_view(
    session_id: Annotated[
        str | None, typer.Argument(help="Session ID to view. Omit to use the latest session.")
    ] = None,
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
) -> None:
    """View the fully assembled context (system instructions, summaries, buffer, tail) as seen by the LLM."""
    load_config(config)
    cfg = get_config()
    db = DatabaseManager(cfg.workspace.db_path)
    db.verify_db()

    console = Console()

    # Resolve session
    if session_id:
        session = db.get_session(session_id)
    else:
        session = db.get_latest_session()

    if not session:
        console.print("[bold red]Error: No session found![/bold red]")
        sys.exit(1)

    session_id = session.id
    console.print(f"\n[bold green]=== Assembled LLM Context for Session '{session_id}' ===[/bold green]\n")

    try:
        context = KesokuContext(config=cfg, db=db)
        gateway = Gateway(context=context)

        # Retrieve raw history
        history = asyncio.run(build_history(gateway=gateway, session_id=session_id, heal_orphans=False))

        # We construct a mock current message to trigger assembly
        # Import turn executor's internal assembly call safely
        from kesoku.agent.llm import get_llm
        from kesoku.constants import MessageRole, MessageStatus, MessageType
        from kesoku.db import Message
        llm = get_llm(provider=cfg.agent.llm, config=cfg)

        mock_msg = Message(
            session_id=session_id,
            chatbot_id="cli",
            channel_id="cli_channel",
            sender="user",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="Context view test",
            status=MessageStatus.PENDING,
        )

        # Build history through turn executor's auto compaction/assembly
        async def assemble():
            # Trigger assembly
            from kesoku.agent.turn_executor import TurnExecutor
            executor = TurnExecutor(gateway=gateway, session_id=session_id)
            assembled_history, _ = await executor._check_and_auto_compact_history(
                history=history,
                system_prompt=session.system_prompt,
                tools_list=[],
                llm=llm,
                cfg=cfg,
                current_msg=mock_msg,
            )
            return assembled_history

        assembled_history = asyncio.run(assemble())

        # Render assembled context
        if session.system_prompt:
            console.print(
                Panel(
                    session.system_prompt.strip(),
                    title="[bold yellow]System Instructions[/bold yellow]",
                    title_align="left",
                    border_style="yellow",
                )
            )

        for msg in assembled_history:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            content = msg.content or ""
            is_scaffold = msg.metadata.get("is_scaffold", False)
            is_ack = msg.metadata.get("is_scaffold_ack", False)

            if is_scaffold:
                console.print(
                    Panel(
                        content.strip(),
                        title="[bold magenta]Compacted Summary Forest Scaffold[/bold magenta]",
                        title_align="left",
                        border_style="magenta",
                    )
                )
            elif is_ack:
                console.print(
                    Panel(
                        content.strip(),
                        title="[bold magenta]Scaffold Acknowledge[/bold magenta]",
                        title_align="left",
                        border_style="magenta",
                    )
                )
            elif role == "user":
                console.print(
                    Panel(
                        content.strip(),
                        title="[bold green]User[/bold green]",
                        title_align="left",
                        border_style="green",
                    )
                )
            elif role == "assistant":
                console.print(
                    Panel(
                        content.strip(),
                        title="[bold cyan]Assistant[/bold cyan]",
                        title_align="left",
                        border_style="cyan",
                    )
                )
            else:
                sender = msg.sender or role.capitalize()
                console.print(
                    Panel(
                        content.strip(),
                        title=f"[bold dim]{sender}[/bold dim]",
                        title_align="left",
                        border_style="dim",
                    )
                )

    except Exception as e:
        console.print(f"[bold red]Error rendering compression active context: {e}[/bold red]")
        sys.exit(1)


@lcm_app.command("compact")
def lcm_compact(
    session_id: Annotated[
        str | None, typer.Argument(help="Session ID to compact. Omit to use the latest session.")
    ] = None,
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
) -> None:
    """Manually trigger context compression for a session immediately."""
    load_config(config)
    cfg = get_config()
    db = DatabaseManager(cfg.workspace.db_path)
    db.verify_db()

    console = Console()

    if session_id:
        session = db.get_session(session_id)
    else:
        session = db.get_latest_session()

    if not session:
        console.print("[bold red]Error: No session found![/bold red]")
        sys.exit(1)

    session_id = session.id
    console.print(f"\n[bold yellow]Initiating Manual Compression for session '{session_id}'...[/bold yellow]")

    context = KesokuContext(config=cfg, db=db)
    gateway = Gateway(context=context)

    try:
        history = asyncio.run(build_history(gateway=gateway, session_id=session_id, heal_orphans=False))
        if not history:
            console.print("[bold red]Error: Active session has no messages to compact.[/bold red]")
            sys.exit(1)

        from kesoku.agent.llm import get_llm
        llm = get_llm(provider=cfg.agent.llm, config=cfg)

        from kesoku.agent.compressor import HistoryCompressor
        compressor = HistoryCompressor(context.db)

        # Run compression
        console.print("[bold cyan]Running turn-based compaction algorithm...[/bold cyan]")

        async def run_compaction():
            return await compressor.auto_compact_session(
                session_id=session_id,
                history=history,
                llm=llm,
                config=cfg,
            )

        compacted = asyncio.run(run_compaction())

        if compacted:
            console.print("[bold green]🔄 Context Compaction completed successfully![/bold green]")
        else:
            console.print("[bold yellow]Compaction bypassed: Session turns do not meet threshold limits.[/bold yellow]")

    except Exception as e:
        console.print(f"[bold red]Compaction failed: {e}[/bold red]")
        sys.exit(1)
