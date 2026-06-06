import asyncio
import os
import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from kesoku.agent.history import build_history, messages_to_openlcm_dicts
from kesoku.config import get_config, load_config
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager
from kesoku.gateway.gateway import Gateway

lcm_app = typer.Typer(help="Inspect, audit, and manage Lossless Context Management (LCM) states.")


@lcm_app.command("status")
def lcm_status(
    session_id: Annotated[
        str | None, typer.Argument(help="Session ID to inspect. Omit to use the latest session.")
    ] = None,
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
) -> None:
    """Show Lossless Context Management (LCM) status, active DAG nodes, and token savings for a session."""
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
    console.print(f"\n[bold green]=== LCM Status for Session '{session_id}' ({session.title}) ===[/bold green]")

    try:
        context = KesokuContext(config=cfg, db=db)
        gateway = Gateway(context=context)
        lcm_engine = context.get_lcm_engine(session_id)
        all_nodes = lcm_engine._dag.get_session_nodes(session_id)

        # Count total compacted nodes & savings
        num_nodes = len(all_nodes)
        total_source_tokens = sum(nd.source_token_count for nd in all_nodes)
        total_compacted_tokens = sum(nd.token_count for nd in all_nodes)
        savings = round(total_source_tokens / total_compacted_tokens, 1) if total_compacted_tokens > 0 else 1

        # Get active history length from raw kesoku.db
        history = asyncio.run(build_history(gateway=gateway, session_id=session_id, heal_orphans=False))
        total_raw_turns = len(history) if history else 0

        # Find fresh tail count
        lcm_input = messages_to_openlcm_dicts(history)
        system_message = None
        remaining_messages = list(lcm_input)
        if remaining_messages and remaining_messages[0].get("role") == "system":
            system_message = remaining_messages.pop(0)
        if not system_message and session.system_prompt:
            system_message = {"role": "system", "content": session.system_prompt}

        assembled = lcm_engine._assemble_context(system_message, remaining_messages)
        from openlcm.core.tokens import count_messages_tokens
        assembled_tokens = count_messages_tokens(assembled)

        # Find number of assistant bubbles in tail
        fresh_msgs = []
        for msg in assembled:
            role = msg.get("role")
            content = msg.get("content") or ""
            if role == "system":
                continue
            elif role == "user" and "[Note: This conversation uses Lossless Context Management" in content:
                continue
            elif role == "assistant" and (
                "I have access to the full conversation history through LCM tools" in content
            ):
                continue
            fresh_msgs.append(msg)

        fresh_turns = len(fresh_msgs)

        # Print results using a beautiful table
        table = Table(show_header=False, box=None)
        table.add_row("[bold cyan]Total Raw Turns (History):[/bold cyan]", f"{total_raw_turns} messages")
        table.add_row("[bold cyan]Compacted Summary Nodes:[/bold cyan]", f"{num_nodes} active nodes")
        if num_nodes > 0:
            savings_str = f"{total_source_tokens} -> {total_compacted_tokens} tokens ({savings}x savings)"
            table.add_row(
                "[bold cyan]Compacted Token Savings:[/bold cyan]",
                f"[bold green]{savings_str}[/bold green]",
            )
        table.add_row("[bold cyan]Fresh Tail Context Size:[/bold cyan]", f"{fresh_turns} messages")
        table.add_row("[bold cyan]Total Active Context Size:[/bold cyan]", f"{assembled_tokens} tokens")

        # Threshold parameters
        lcm_threshold = os.environ.get("LCM_CONTEXT_THRESHOLD") or cfg.agent.compact_history_threshold
        table.add_row("[bold dim]Compaction Threshold:[/bold dim]", f"[dim]{lcm_threshold}[/dim]")

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error retrieving LCM status: {e}[/bold red]")
        sys.exit(1)


@lcm_app.command("view")
def lcm_view(
    session_id: Annotated[
        str | None, typer.Argument(help="Session ID to view. Omit to use the latest session.")
    ] = None,
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
) -> None:
    """View the complete assembled LCM active context (compaction summaries + uncompacted tail) for a session."""
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
    console.print(
        f"\n[bold green]=== LCM Active Context for Session '{session_id}' ({session.title}) ===[/bold green]"
    )

    context = KesokuContext(config=cfg, db=db)
    gateway = Gateway(context=context)
    lcm_engine = context.get_lcm_engine(session_id)

    try:
        all_nodes = lcm_engine._dag.get_session_nodes(session_id)

        # 1. Print System Prompt
        if session.system_prompt:
            console.print(
                Panel(
                    Markdown(session.system_prompt), title="⚙️ System Prompt", title_align="left", border_style="dim"
                )
            )

        # 2. Render Compacted Summary Nodes
        if all_nodes:
            console.print("\n[bold cyan]📦 Compacted Summary Scaffold (DAG Nodes)[/bold cyan]")
            # Render nodes by depth
            from collections import defaultdict

            by_depth = defaultdict(list)
            for node in all_nodes:
                by_depth[node.depth].append(node)

            max_dag_depth = max(by_depth.keys()) if by_depth else -1
            for depth in range(max_dag_depth, -1, -1):
                nodes_at_depth = sorted(by_depth.get(depth, []), key=lambda nd: nd.created_at)
                depth_label = {0: "Recent", 1: "Session Arc", 2: "Durable"}.get(depth, f"Depth-{depth}")
                for node in nodes_at_depth:
                    savings = round(node.source_token_count / node.token_count, 1) if node.token_count > 0 else 1
                    title_str = (
                        f"{depth_label} Node {node.node_id} "
                        f"({node.source_token_count} -> {node.token_count} tokens, {savings}x savings)"
                    )
                    console.print(
                        Panel(
                            Markdown(node.summary),
                            title=f"[bold green]📦 {title_str}[/bold green]",
                            title_align="left",
                            border_style="green",
                        )
                    )

        # 3. Render Uncompacted Fresh Tail Messages
        history = asyncio.run(build_history(gateway=gateway, session_id=session_id, heal_orphans=False))
        if history:
            lcm_input = messages_to_openlcm_dicts(history)
            system_message = None
            remaining_messages = list(lcm_input)
            if remaining_messages and remaining_messages[0].get("role") == "system":
                system_message = remaining_messages.pop(0)
            if not system_message and session.system_prompt:
                system_message = {"role": "system", "content": session.system_prompt}

            assembled = lcm_engine._assemble_context(system_message, remaining_messages)

            fresh_msgs = []
            for msg in assembled:
                role = msg.get("role")
                content = msg.get("content") or ""
                if role == "system":
                    continue
                elif role == "user" and "[Note: This conversation uses Lossless Context Management" in content:
                    continue
                elif role == "assistant" and (
                    "I have access to the full conversation history through LCM tools" in content
                ):
                    continue
                fresh_msgs.append(msg)

            if fresh_msgs:
                import json
                console.print("\n[bold cyan]🧵 Active Fresh Tail (Chronological Messages)[/bold cyan]")
                for msg in fresh_msgs:
                    role = msg.get("role")
                    content = msg.get("content") or ""

                    if role == "user":
                        console.print(
                            Panel(
                                escape(content),
                                title="[bold blue]User[/bold blue]",
                                title_align="left",
                                border_style="blue",
                            )
                        )
                    elif role == "tool":
                        tool_name = msg.get("name") or "unknown_tool"
                        console.print(
                            Panel(
                                escape(content),
                                title=f"[bold magenta]📥 Tool Output ({tool_name})[/bold magenta]",
                                title_align="left",
                                border_style="magenta",
                            )
                        )
                    elif role == "assistant":
                        from rich.console import Group
                        from rich.text import Text

                        renderables = []
                        if content.strip():
                            renderables.append(Markdown(content.strip()))

                        if msg.get("tool_calls"):
                            for tc in msg["tool_calls"]:
                                fn = tc.get("function", {})
                                name = fn.get("name", "unknown_tool")
                                arguments = fn.get("arguments", "")
                                try:
                                    if isinstance(arguments, str):
                                        args_dict = json.loads(arguments)
                                        arguments_pretty = json.dumps(args_dict, indent=2, ensure_ascii=False)
                                    else:
                                        arguments_pretty = json.dumps(arguments, indent=2, ensure_ascii=False)
                                except Exception:
                                    arguments_pretty = str(arguments)

                                tc_text = Text()
                                tc_text.append("\n🔧 Called Tool: ", style="bold yellow")
                                tc_text.append(name, style="bold cyan")
                                tc_text.append("\nArguments:\n", style="dim")
                                tc_text.append(arguments_pretty)
                                renderables.append(tc_text)

                        console.print(
                            Panel(
                                Group(*renderables),
                                title="[bold cyan]Assistant[/bold cyan]",
                                title_align="left",
                                border_style="cyan",
                            )
                        )
                    else:
                        sender = role.capitalize()
                        console.print(
                            Panel(
                                content,
                                title=f"[bold dim]{sender}[/bold dim]",
                                title_align="left",
                                border_style="dim",
                            )
                        )

    except Exception as e:
        console.print(f"[bold red]Error rendering LCM active context: {e}[/bold red]")
        sys.exit(1)


@lcm_app.command("compact")
def lcm_compact(
    session_id: Annotated[
        str | None, typer.Argument(help="Session ID to compact. Omit to use the latest session.")
    ] = None,
    force: Annotated[
        bool, typer.Option("-f", "--force", help="Bypass eligibility checks and force compaction immediately.")
    ] = False,
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
) -> None:
    """Manually trigger Lossless Context Compaction via OpenLCM for a session."""
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
    console.print(f"\n[bold yellow]Initiating Manual LCM Compaction for session '{session_id}'...[/bold yellow]")

    context = KesokuContext(config=cfg, db=db)
    gateway = Gateway(context=context)
    lcm_engine = context.get_lcm_engine(session_id)

    try:
        history = asyncio.run(build_history(gateway=gateway, session_id=session_id, heal_orphans=False))
        if not history:
            console.print("[bold red]Error: Active session has no messages to compact.[/bold red]")
            sys.exit(1)

        lcm_input = messages_to_openlcm_dicts(history)
        if session.system_prompt:
            lcm_input.insert(0, {"role": "system", "content": session.system_prompt})

        if not force:
            eligible, reason = lcm_engine._leaf_compaction_candidate_status(lcm_input)
            if not eligible:
                console.print(f"[bold yellow]Context compaction is not needed right now:[/bold yellow] {reason}.")
                console.print("[dim]Use '--force' to bypass this check and compact immediately.[/dim]")
                return

        # Run compression
        console.print("[bold cyan]Running compaction algorithm via OpenLCM...[/bold cyan]")

        async def run_compression():
            await lcm_engine.compress(lcm_input)

        asyncio.run(run_compression())

        console.print("[bold green]🔄 Lossless Context Compaction completed successfully![/bold green]")

        # Fetch status to show savings
        all_nodes = lcm_engine._dag.get_session_nodes(session_id)
        if all_nodes:
            latest_node = sorted(all_nodes, key=lambda nd: nd.created_at)[-1]
            savings = (
                round(latest_node.source_token_count / latest_node.token_count, 1) if latest_node.token_count > 0 else 1
            )
            node_savings_str = (
                f"{latest_node.source_token_count} -> {latest_node.token_count} tokens ({savings}x savings)"
            )
            console.print(
                f"  - [cyan]Created Node {latest_node.node_id}[/cyan] at depth {latest_node.depth}\n"
                f"  - [bold green]Token Savings: {node_savings_str}[/bold green]"
            )

    except Exception as e:
        console.print(f"[bold red]Compaction failed: {e}[/bold red]")
        sys.exit(1)
