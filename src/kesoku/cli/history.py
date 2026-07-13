"""History and context inspection CLI commands for Kesoku."""

import asyncio
import sys
import time
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from kesoku.agent.history import build_history
from kesoku.cli.chat import render_message
from kesoku.config import get_config, load_config
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import DatabaseManager, Message
from kesoku.logger import setup_logger

logger = setup_logger(__name__)
console = Console()

history_app = typer.Typer(help="Manage and query chat history and context status.")


@history_app.command("list")
def history_list(
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
) -> None:
    """List all chat sessions."""
    cfg = load_config(config)
    db = DatabaseManager(cfg.workspace.db_path)

    async def run():
        from kesoku.context import KesokuContext
        from kesoku.gateway.gateway import Gateway
        gw = Gateway(context=KesokuContext(config=cfg, db=db))

        sessions = await gw.db.list_sessions()
        if not sessions:
            logger.info("No chat sessions found.")
            return
        table = Table(title="Kesoku Chat Sessions", show_header=True, header_style="bold cyan")
        table.add_column("Session ID", style="bold green")
        table.add_column("Created At", style="dim")
        table.add_column("Last Updated", style="dim")
        table.add_column("Title", style="magenta")
        table.add_column("Role", style="yellow")
        for s in sessions:
            created = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s.created_at))
            updated = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s.updated_at))
            table.add_row(s.id, created, updated, s.title, s.role_name or "default")
        console.print(table)

    asyncio.run(run())


@history_app.command("show")
def history_show(
    session_id: Annotated[str, typer.Argument(help="Session ID to display")],
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
    grouped: Annotated[
        bool,
        typer.Option("-g", "--grouped", help="Sort history by grouping tool call and result together"),
    ] = False,
) -> None:
    """Show full chat history of a session."""
    cfg = load_config(config)
    db = DatabaseManager(cfg.workspace.db_path)

    async def run():
        from kesoku.context import KesokuContext
        from kesoku.gateway.gateway import Gateway
        gw = Gateway(context=KesokuContext(config=cfg, db=db))

        session = await gw.db.get_session(session_id)
        if not session:
            logger.error(f"Session '{session_id}' not found.")
            sys.exit(1)
        order = "grouped" if grouped else "phased"
        history = await build_history(gateway=gw, session_id=session_id, order=order, heal_orphans=False)
        if not history:
            logger.warning(f"Session '{session_id}' has no recorded messages.")
            return
        logger.info(f"Chat History for Session '{session_id}' ({session.title}):")

        if session.system_prompt:
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

        for msg in history:
            render_message(console, msg)

    asyncio.run(run())


@history_app.command("search")
def history_search(
    query: Annotated[str, typer.Argument(help="Search query")],
    role: Annotated[str, typer.Option("-r", "--role", help="Role persona scope to search")] = "default",
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
    limit: Annotated[int, typer.Option("-l", "--limit", help="Max results to return")] = 20,
) -> None:
    """Search chat history semantically or with keywords."""
    cfg = load_config(config)
    db = DatabaseManager(cfg.workspace.db_path)

    async def run():
        from kesoku.context import KesokuContext
        from kesoku.gateway.gateway import Gateway
        gw = Gateway(context=KesokuContext(config=cfg, db=db))

        threshold = cfg.agent.search_threshold
        messages = await gw.db.search_role_messages_semantic(
            role,
            query,
            limit=limit,
            threshold=threshold,
        )

        if not messages:
            console.print(f"[yellow]No messages matching '{query}' found for role '{role}'.[/yellow]")
            return

        is_wildcard = not query or query.strip() == "*"
        title = f"Search Results for '{query}' (Role: {role})"
        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("Time", style="dim")
        table.add_column("Session ID", style="bold green")
        table.add_column("Sender", style="magenta")
        if not is_wildcard:
            table.add_column("Score", style="yellow")
        table.add_column("Content")

        from kesoku.utils.text import truncate_middle
        for m in messages:
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m.timestamp))
            score_str = ""
            if "similarity_score" in m.metadata and not is_wildcard:
                score_str = f"{m.metadata['similarity_score']:.4f}"

            # Extract content context if available
            parts = []
            if m.metadata.get("prev_chunk"):
                parts.append(f"... {m.metadata['prev_chunk']}")
            parts.append(m.content)
            if m.metadata.get("post_chunk"):
                parts.append(f"{m.metadata['post_chunk']} ...")
            combined_content = " ".join(parts)
            truncated_content = truncate_middle(combined_content, 150).replace("\n", " ")

            row_data = [time_str, m.session_id, f"{m.sender} ({m.role})"]
            if not is_wildcard:
                row_data.append(score_str)
            row_data.append(truncated_content)
            table.add_row(*row_data)

        console.print(table)

    asyncio.run(run())


@history_app.command("rebuild-index")
def history_rebuild_index(
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
    force: Annotated[
        bool, typer.Option("--force", help="Force rebuild of all embeddings, not just missing ones")
    ] = False,
) -> None:
    """Generate and store embeddings for all unindexed messages."""
    cfg = load_config(config)
    db = DatabaseManager(cfg.workspace.db_path)
    db.verify_db()

    try:
        if force:
            console.print("[bold yellow]Force mode enabled. Clearing all existing message embeddings...[/bold yellow]")
            with db.connection_provider.connection() as conn:
                with conn:
                    conn.execute("DELETE FROM message_chunks")
            console.print("[bold green]Existing message embeddings cleared.[/bold green]")

        unindexed_messages = db.get_unindexed_messages()
        total_msg = len(unindexed_messages)

        if total_msg == 0:
            console.print("[bold green]All messages are already fully indexed![/bold green]")
            return

        console.print(f"Found [bold]{total_msg}[/bold] unindexed messages.")
        console.print("Initializing local embedding model (fastembed)...")
        from kesoku.utils import embedding
        embedding.get_embedding_model()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
        ) as progress:
            task_msg = progress.add_task("[green]Indexing messages...", total=total_msg)
            batch_size = 32
            for i in range(0, total_msg, batch_size):
                batch = unindexed_messages[i : i + batch_size]
                chunks_to_embed = []
                for msg in batch:
                    from kesoku.utils.text import chunk_message_text
                    chunks = chunk_message_text(msg["content"], threshold=80)
                    for idx, chunk_content in enumerate(chunks):
                        chunks_to_embed.append((msg["id"], idx, chunk_content))

                if chunks_to_embed:
                    texts = [c[2] for c in chunks_to_embed]
                    embs = embedding.get_embeddings(texts)

                    chunks_data = []
                    for (msg_id, idx, content), emb in zip(chunks_to_embed, embs):
                        emb_bytes = embedding.vector_to_bytes(emb)
                        chunks_data.append((msg_id, idx, content, emb_bytes))

                    db.save_message_chunks_batch(chunks_data)

                progress.advance(task_msg, advance=len(batch))

        console.print("[bold green]✓ Index rebuilding completed successfully![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error rebuilding index: {e}[/bold red]")
        sys.exit(1)


@history_app.command("status")
def history_status(
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
        from kesoku.context import KesokuContext
        from kesoku.gateway.gateway import Gateway
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


@history_app.command("view")
def history_view(
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
        from kesoku.context import KesokuContext
        from kesoku.gateway.gateway import Gateway
        context = KesokuContext(config=cfg, db=db)
        gateway = Gateway(context=context)

        # Retrieve raw history
        history = asyncio.run(build_history(gateway=gateway, session_id=session_id, heal_orphans=False))

        # We construct a mock current message to trigger assembly
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
            from kesoku.agent.turn_executor import TurnExecutor

            executor = TurnExecutor(gateway=gateway, session_id=session_id, tool_runner=None)
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
