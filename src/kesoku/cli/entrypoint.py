"""Typer CLI interface for Kesoku AI Agent framework.

Provides commands: init and chat (session-based one-shot CLI chat).
"""

import asyncio
import json
import logging
import os
import sys
import time
import tomllib
from typing import Annotated

import tomli_w
import typer
from rich.console import Console

from kesoku.agent.agent import Agent
from kesoku.agent.tools import sanitize_key, validate_key
from kesoku.cli.chat import run_cli_chat_async
from kesoku.cli.context import context_app
from kesoku.cli.service import service_app
from kesoku.config import get_config, init_config, init_roles, init_skills, load_config
from kesoku.cron import CronManager
from kesoku.db import DatabaseManager
from kesoku.gateway.chatbot.cronjob import CronjobChatbot
from kesoku.gateway.gateway import Gateway
from kesoku.logger import configure_logging, setup_logger

# Setup global colored logging configuration
configure_logging(logging.INFO)
logger = setup_logger(__name__)

app = typer.Typer(help="Kesoku AI Agent CLI manager.")

app.add_typer(service_app, name="service")

wechat_app = typer.Typer(help="Manage WeChat chatbot integration.")
app.add_typer(wechat_app, name="wechat")

memory_app = typer.Typer(help="Inspect, audit, and configure agent memories.")
app.add_typer(memory_app, name="memory")

app.add_typer(context_app, name="context")


@memory_app.command("list")
def cli_memory_list(
    category: Annotated[str | None, typer.Argument(help="Memory category (e.g., progress, memo)")] = None,
    role: Annotated[str, typer.Option("-r", "--role", help="Optional roleplay persona scope")] = "default",
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
) -> None:
    """List all active memory keys in the given category, or list all permitted categories if category is omitted."""
    load_config(config)
    cfg = get_config()
    db = DatabaseManager(cfg.workspace.db_path)
    db.verify_db()

    console = Console()

    if not category:
        allowed = db.get_allowed_memory_categories()
        console.print("\n[bold green]=== Permitted & Active Categories ===[/bold green]")
        core = {"progress"}
        for cat in sorted(list(allowed)):
            if cat in core:
                console.print(f"  - [cyan]{cat}[/cyan] [yellow](Standard)[/yellow]")
            else:
                console.print(f"  - [cyan]{cat}[/cyan] [magenta](Custom)[/magenta]")
        return

    memories = db.get_agent_memories(category=category, role=role)
    if not memories:
        typer.echo(f"No memories found in category '{category}' for role scope '{role}'.")
        return

    console.print(f"\n[bold green]=== Memories in '{category}' (scope: {role}) ===[/bold green]")
    for m in memories:
        updated_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m["updated_at"]))
        console.print(
            f"  - key: [cyan]{m['key']}[/cyan] | "
            f'title: "[bold]{m["title"]}[/bold]" | '
            f"scope: [yellow]{m['role']}[/yellow] | "
            f"updated: {updated_str}"
        )


@memory_app.command("view")
def cli_memory_view(
    category: Annotated[str, typer.Argument(help="Memory category (e.g., progress, memo)")],
    key: Annotated[str | None, typer.Argument(help="Unique memory key. Omit to view all category entries.")] = None,
    role: Annotated[str, typer.Option("-r", "--role", help="Optional roleplay persona scope")] = "default",
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
) -> None:
    """Retrieve the details of a specific memory key, or render all entries dynamically."""
    load_config(config)
    cfg = get_config()
    db = DatabaseManager(cfg.workspace.db_path)
    db.verify_db()

    console = Console()

    if key:
        if not validate_key(key):
            typer.echo(
                f"Error: Invalid Key '{key}'.\n"
                "Memory keys must strictly contain only lowercase letters, "
                "underscores, and numbers (regex: ^[a-z0-9_]+$)."
            )
            sys.exit(1)
        sanitized = sanitize_key(key)
        mem = db.get_agent_memory(category=category, key=sanitized, role=role)
        if not mem:
            typer.echo(f"No memory found for category='{category}', key='{sanitized}', role='{role}'.")
            return
        console.print(
            f"\n[bold green]=== Memory: {mem['title']} (key: {mem['key']}, scope: {mem['role']}) ===[/bold green]"
        )
        console.print(mem["content"])
        return

    memories = db.get_agent_memories(category=category, role=role)
    if not memories:
        typer.echo(f"No memories found in category '{category}' for role scope '{role}'.")
        return

    console.print(f"\n[bold green]# Category: {category} (scope: {role})[/bold green]")
    for m in memories:
        console.print(f"\n[bold cyan]## {m['title']} (key: {m['key']}, scope: {m['role']})[/bold cyan]")
        console.print(m["content"].strip())


@memory_app.command("update")
def cli_memory_update(
    category: Annotated[str, typer.Argument(help="Memory category (e.g., progress, memo)")],
    key: Annotated[str, typer.Argument(help="Unique memory key")],
    title: Annotated[str, typer.Argument(help="Human-readable title or label")],
    content: Annotated[str, typer.Argument(help="Markdown or JSON content payload")],
    role: Annotated[str, typer.Option("-r", "--role", help="Optional roleplay persona scope")] = "default",
    create_category: Annotated[
        bool, typer.Option("--create-category", help="Override validation to create a new category")
    ] = False,
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
) -> None:
    """Create or update a memory record atomically."""
    load_config(config)
    cfg = get_config()
    db = DatabaseManager(cfg.workspace.db_path)
    db.verify_db()

    allowed = db.get_allowed_memory_categories()
    if category not in allowed and not create_category:
        typer.echo(
            f"Error: Category '{category}' is not recognized.\n"
            f"Permitted categories: {sorted(list(allowed))}.\n"
            "Use '--create-category' to explicitly force creation of a new category."
        )
        sys.exit(1)

    if not validate_key(key):
        typer.echo(
            f"Error: Invalid Key '{key}'.\n"
            "Memory keys must strictly contain only lowercase letters, underscores, and numbers (regex: ^[a-z0-9_]+$)."
        )
        sys.exit(1)

    sanitized = sanitize_key(key)
    db.upsert_agent_memory(
        category=category,
        key=sanitized,
        title=title,
        content=content,
        role=role,
    )
    Console().print(
        f"\n[bold green]Memory successfully updated![/bold green] "
        f"(key: [cyan]{sanitized}[/cyan], scope: [yellow]{role}[/yellow])"
    )


@memory_app.command("delete")
def cli_memory_delete(
    category: Annotated[str, typer.Argument(help="Memory category (e.g., progress, memo)")],
    key: Annotated[str, typer.Argument(help="Unique memory key")],
    role: Annotated[str, typer.Option("-r", "--role", help="Optional roleplay persona scope")] = "default",
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
) -> None:
    """Delete a specific memory record."""
    load_config(config)
    cfg = get_config()
    db = DatabaseManager(cfg.workspace.db_path)
    db.verify_db()

    if not validate_key(key):
        typer.echo(
            f"Error: Invalid Key '{key}'.\n"
            "Memory keys must strictly contain only lowercase letters, underscores, and numbers (regex: ^[a-z0-9_]+$)."
        )
        sys.exit(1)

    sanitized = sanitize_key(key)
    mem = db.get_agent_memory(category=category, key=sanitized, role=role)
    if not mem:
        typer.echo(f"No memory entry found for category='{category}', key='{sanitized}', role='{role}'.")
        return

    db.delete_agent_memory(category=category, key=sanitized, role=role)
    Console().print(
        f"\n[bold green]Memory successfully deleted![/bold green] "
        f"(category: [cyan]{category}[/cyan], "
        f"key: [cyan]{sanitized}[/cyan], "
        f"scope: [yellow]{role}[/yellow])"
    )


@memory_app.command("export")
def cli_memory_export(
    output_path: Annotated[str, typer.Argument(help="Target filepath to save the exported TOML file")],
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
) -> None:
    """Export all agent memories from SQLite database into a structured TOML file."""
    load_config(config)
    cfg = get_config()
    db = DatabaseManager(cfg.workspace.db_path)
    db.verify_db()

    console = Console()

    try:
        memories = db.get_agent_memories()
        if not memories:
            console.print("[bold yellow]No memories found in database to export.[/bold yellow]")
            return

        # Restructure into a nested dict: role -> category -> key -> data
        export_data = {}
        for m in memories:
            role = m["role"]
            cat = m["category"]
            key = m["key"]

            export_data.setdefault(role, {}).setdefault(cat, {})[key] = {
                "title": m["title"],
                "content": m["content"],
                "updated_at": m["updated_at"],
            }

        # Ensure parent directories exist
        target_path = os.path.abspath(output_path)
        parent_dir = os.path.dirname(target_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Serialize to TOML
        with open(target_path, "wb") as f:
            tomli_w.dump(export_data, f)

        console.print(
            f"\n[bold green]Successfully exported {len(memories)} memory records to TOML![/bold green]\n"
            f"  File: [cyan]{target_path}[/cyan]"
        )
    except Exception as e:
        console.print(f"[bold red]Error exporting memories: {e}[/bold red]")
        sys.exit(1)


@memory_app.command("import")
def cli_memory_import(
    input_path: Annotated[str, typer.Argument(help="Target filepath to read the imported TOML file")],
    create_category: Annotated[
        bool, typer.Option("--create-category", help="Override validation to create new categories if they don't exist")
    ] = False,
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
) -> None:
    """Import agent memories from a structured TOML file into the SQLite database."""
    load_config(config)
    cfg = get_config()
    db = DatabaseManager(cfg.workspace.db_path)
    db.verify_db()

    console = Console()

    if not os.path.exists(input_path):
        console.print(f"[bold red]Error: File not found: {input_path}[/bold red]")
        sys.exit(1)

    try:
        with open(input_path, "rb") as f:
            import_data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        console.print(f"[bold red]Error: Invalid TOML format: {e}[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error reading file: {e}[/bold red]")
        sys.exit(1)

    if not isinstance(import_data, dict):
        console.print("[bold red]Error: Top-level structure of TOML must be a table.[/bold red]")
        sys.exit(1)

    allowed_categories = db.get_allowed_memory_categories()
    valid_records = []

    for role, categories in import_data.items():
        if not isinstance(categories, dict):
            console.print(f"[bold red]Error: Role '{role}' must map to a category table.[/bold red]")
            sys.exit(1)
        for category, keys in categories.items():
            if not isinstance(keys, dict):
                console.print(
                    f"[bold red]Error: Category '{category}' under role '{role}' must map to a key table.[/bold red]"
                )
                sys.exit(1)

            if category not in allowed_categories and not create_category:
                console.print(
                    f"[bold red]Error: Category '{category}' under role '{role}' is not recognized.[/bold red]\n"
                    f"Permitted categories: {sorted(list(allowed_categories))}.\n"
                    "Use '--create-category' to explicitly force importing unknown categories."
                )
                sys.exit(1)

            for key, record in keys.items():
                if not isinstance(record, dict):
                    console.print(
                        f"[bold red]Error: Key '{key}' under category '{category}' "
                        f"and role '{role}' must map to a table.[/bold red]"
                    )
                    sys.exit(1)

                if not validate_key(key):
                    console.print(
                        f"[bold red]Error: Invalid Key '{key}' under category '{category}' "
                        f"and role '{role}'.[/bold red]\n"
                        "Memory keys must strictly contain only lowercase letters, "
                        "underscores, and numbers (regex: ^[a-z0-9_]+$)."
                    )
                    sys.exit(1)

                if "title" not in record or not isinstance(record["title"], str):
                    console.print(
                        f"[bold red]Error: Record '{role}.{category}.{key}' "
                        "is missing 'title' or it is not a string.[/bold red]"
                    )
                    sys.exit(1)
                if "content" not in record or not isinstance(record["content"], str):
                    console.print(
                        f"[bold red]Error: Record '{role}.{category}.{key}' "
                        "is missing 'content' or it is not a string.[/bold red]"
                    )
                    sys.exit(1)

                sanitized = sanitize_key(key)

                valid_records.append(
                    {
                        "role": role,
                        "category": category,
                        "key": sanitized,
                        "title": record["title"],
                        "content": record["content"],
                    }
                )

    try:
        for rec in valid_records:
            db.upsert_agent_memory(
                category=rec["category"],
                key=rec["key"],
                title=rec["title"],
                content=rec["content"],
                role=rec["role"],
            )

        console.print(
            f"\n[bold green]Successfully imported {len(valid_records)} memory records from TOML![/bold green]"
        )
    except Exception as e:
        console.print(f"[bold red]Error importing memories: {e}[/bold red]")
        sys.exit(1)


@wechat_app.command("pair")
def wechat_pair(
    chatbot_id: Annotated[
        str | None, typer.Option("-b", "--chatbot-id", help="Chatbot ID of the WeChat bot to pair")
    ] = None,
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
    timeout: Annotated[int, typer.Option("-t", "--timeout", help="Timeout in seconds for QR scan confirmation")] = 480,
) -> None:
    """Pair WeChat account with Kesoku via terminal barcode QR code."""
    cfg = load_config(config)

    if chatbot_id:
        w_cfg = cfg.get_wechat_config(chatbot_id)
        if not w_cfg:
            logger.error(f"WeChat chatbot ID '{chatbot_id}' not found in configuration.")
            sys.exit(1)
    else:
        if cfg.chatbots.wechat:
            w_cfg = cfg.chatbots.wechat[0]
        else:
            w_cfg = cfg.wechat

    from kesoku.gateway.chatbot.wechat import qr_login

    logger.info("Starting WeChat pairing flow...")
    credentials = asyncio.run(qr_login(timeout_seconds=timeout))
    if not credentials:
        logger.error("WeChat pairing failed or timed out.")
        sys.exit(1)

    # Update config
    w_cfg.enabled = True
    w_cfg.account_id = credentials["account_id"]
    w_cfg.token = credentials["token"]
    w_cfg.base_url = credentials["base_url"]

    from kesoku.config import save_config

    save_config(cfg, config)

    console = Console()
    console.print("\n[bold green]WeChat chatbot paired and enabled successfully![/bold green]")
    console.print(f"  Account ID: [cyan]{credentials['account_id']}[/cyan]")
    console.print("  Token saved to configuration file.")


@wechat_app.command("show-channels")
def wechat_show_channels(
    chatbot_id: Annotated[
        str | None, typer.Option("-b", "--chatbot-id", help="Chatbot ID of the WeChat bot to query")
    ] = None,
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
) -> None:
    """Show all paired channels/chats for the active WeChat account."""
    cfg = load_config(config)

    # Resolve list of configs to query
    target_configs = []
    if chatbot_id:
        w_cfg = cfg.get_wechat_config(chatbot_id)
        if not w_cfg:
            logger.error(f"WeChat chatbot ID '{chatbot_id}' not found in configuration.")
            sys.exit(1)
        target_configs.append(w_cfg)
    else:
        active_wechats = cfg.active_wechats
        if active_wechats:
            target_configs.extend(active_wechats)
        else:
            target_configs.append(cfg.wechat)

    valid_configs = [c for c in target_configs if c.enabled and c.account_id]
    if not valid_configs:
        logger.error("No enabled or paired WeChat chatbot configuration found.")
        sys.exit(1)

    working_dir = cfg.agent_working_dir or os.getcwd()
    tokens_path = os.path.join(working_dir, ".wechat_context_tokens.json")

    console = Console()
    if not os.path.exists(tokens_path):
        console.print("[yellow]No channel tokens found. Send a message to the bot first to register channels.[/yellow]")
        return

    try:
        with open(tokens_path, encoding="utf-8") as f:
            tokens = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load tokens file {tokens_path}: {e}")
        sys.exit(1)

    for w_cfg in valid_configs:
        account_id = w_cfg.account_id
        prefix = f"{account_id}:"

        channels = []
        for k in tokens.keys():
            if k.startswith(prefix):
                channels.append(k[len(prefix) :])

        console.print(
            f"\n[bold green]=== Active WeChat Channels for {w_cfg.chatbot_id} ({account_id}) ===[/bold green]"
        )
        if not channels:
            console.print("  [yellow]No active channels found for this account.[/yellow]")
            continue

        for chan in channels:
            chan_type = "Group" if chan.endswith("@chatroom") else "DM/User"
            console.print(f"  - ID: [cyan]{chan}[/cyan] | Type: [yellow]{chan_type}[/yellow]")


@app.command("init")
def init_cmd(
    workspace_path: Annotated[
        str | None, typer.Option("-w", "--workspace-path", help="Workspace directory to initialize")
    ] = None,
    config: Annotated[str, typer.Option("-c", "--config", help="Path to save/initialize config.toml")] = "config.toml",
    overwrite_config: Annotated[
        bool, typer.Option("--overwrite-config", help="Overwrite existing config file (creates backup)")
    ] = False,
    overwrite_skills: Annotated[
        bool, typer.Option("--overwrite-skills", help="Overwrite existing resource skills")
    ] = False,
    overwrite_roles: Annotated[
        bool, typer.Option("--overwrite-roles", help="Overwrite existing resource roles")
    ] = False,
    overwrite_db: Annotated[
        bool, typer.Option("--overwrite-db", help="Overwrite existing database file (creates backup)")
    ] = False,
) -> None:
    """Initialize Kesoku workspace and generate config.toml from template.

    Args:
        workspace_path: Path to workspace directory to initialize.
        config: Path to save/initialize config.toml.
        overwrite_config: Whether to overwrite existing config.toml.
        overwrite_skills: Whether to overwrite existing skills in skills dir.
        overwrite_roles: Whether to overwrite existing roles in roles dir.
        overwrite_db: Whether to overwrite existing SQLite database file.
    """
    if workspace_path is not None:
        workspace_path = os.path.abspath(workspace_path)
        target_config_path = os.path.join(workspace_path, "config.toml")
    else:
        target_config_path = os.path.abspath(config)
        workspace_path = os.path.dirname(target_config_path)

    logger.info(f"Initializing Kesoku workspace at {workspace_path} (config: {target_config_path})...")
    init_config(target_config_path, overwrite=overwrite_config)

    cfg = load_config(target_config_path)
    DatabaseManager(cfg.workspace.db_path).init_tables(overwrite=overwrite_db)
    init_skills(cfg.workspace.skills_dir, overwrite=overwrite_skills)
    init_roles(cfg.workspace.roles_dir, overwrite=overwrite_roles)
    logger.info(f"Workspace initialized successfully at {workspace_path}")


@app.command("chat")
def chat_cmd(
    message: Annotated[str | None, typer.Argument(help="Message to send to the agent")] = None,
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
    list_sessions: Annotated[
        bool, typer.Option("-l", "--list-sessions", help="List all current chat sessions")
    ] = False,
    resume: Annotated[
        str | None, typer.Option("-r", "--resume", metavar="SESSION_ID", help="Resume a specific session by ID")
    ] = None,
    resume_latest: Annotated[bool, typer.Option("-z", "--resume-latest", help="Resume the latest session")] = False,
    show_history: Annotated[
        str | None,
        typer.Option("-s", "--show-history", metavar="SESSION_ID", help="Show full chat history of a session"),
    ] = None,
    grouped: Annotated[
        bool,
        typer.Option("-g", "--grouped", help="Sort history by grouping tool call and result together"),
    ] = False,
) -> None:
    """Chat with Kesoku Agent in one-shot session mode."""
    load_config(config)
    try:
        asyncio.run(
            run_cli_chat_async(
                message=message,
                list_sessions=list_sessions,
                resume=resume,
                resume_latest=resume_latest,
                show_history=show_history,
                grouped=grouped,
            )
        )
    except KeyboardInterrupt:
        logger.info("Kesoku chat stopped by user.")
    except Exception as e:
        console = Console()
        console.print(f"[bold red]Error during Kesoku chat session: {e}[/bold red]")
        logger.error(f"Error during Kesoku chat session: {e}", exc_info=True)
        sys.exit(1)


@app.command("start")
def start_cmd(
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
) -> None:
    """Start Kesoku background bots and agent dispatcher in indefinite foreground service mode."""
    load_config(config)
    cfg = get_config()

    active_discords = cfg.active_discords
    active_gchats = cfg.active_google_chats
    active_wechats = cfg.active_wechats

    if not active_discords and not active_gchats and not active_wechats:
        console = Console()
        console.print("[bold red]Error: No chatbots are enabled in the configuration.[/bold red]")
        sys.exit(1)

    gateway = Gateway()
    agent = Agent(gateway=gateway)
    bot_tasks = []
    bots = []

    for d_cfg in active_discords:
        discord_token = d_cfg.bot_token or os.environ.get("DISCORD_TOKEN")
        if discord_token:
            from kesoku.gateway.chatbot.discord import DiscordChatbot

            discord_bot = DiscordChatbot(
                chatbot_id=d_cfg.chatbot_id,
                gateway=gateway,
                bot_token=discord_token,
                discord_config=d_cfg,
            )
            bot_tasks.append(discord_bot.start())
            bots.append(discord_bot)
        else:
            console = Console()
            console.print(
                f"[bold red]Error: Discord bot '{d_cfg.chatbot_id}' is enabled "
                "but bot_token is not configured.[/bold red]"
            )
            sys.exit(1)

    for g_cfg in active_gchats:
        from kesoku.gateway.chatbot.google_chat import GoogleChatChatbot

        gchat_bot = GoogleChatChatbot(
            chatbot_id=g_cfg.chatbot_id,
            gateway=gateway,
            google_chat_config=g_cfg,
        )
        bot_tasks.append(gchat_bot.start())
        bots.append(gchat_bot)

    for w_cfg in active_wechats:
        from kesoku.gateway.chatbot.wechat import WechatChatbot

        wechat_bot = WechatChatbot(
            chatbot_id=w_cfg.chatbot_id,
            gateway=gateway,
            wechat_config=w_cfg,
        )
        bot_tasks.append(wechat_bot.start())
        bots.append(wechat_bot)

    cron_manager = None
    config_dir = cfg.agent_working_dir
    if config_dir:
        cron_toml_path = os.path.join(config_dir, "cronjob.toml")
        if os.path.exists(cron_toml_path):
            cronjob_bot = CronjobChatbot(chatbot_id="cronjob", gateway=gateway)
            bot_tasks.append(cronjob_bot.start())
            bots.append(cronjob_bot)

            cron_manager = CronManager(chatbots=bots, config_dir=config_dir)
            gateway.register_cron_manager(cron_manager)
            logger.info(f"Loaded cronjobs configuration from {cron_toml_path}")

    async def _service_runner() -> None:
        agent_task = asyncio.create_task(agent.start())
        chatbot_tasks = [asyncio.create_task(bt) for bt in bot_tasks]
        tasks = [agent_task] + chatbot_tasks

        if cron_manager:
            cron_task = asyncio.create_task(cron_manager.start())
            tasks.append(cron_task)

        logger.info("Kesoku service started in foreground mode. Press Ctrl+C to stop.")
        await asyncio.gather(*tasks)

    try:
        asyncio.run(_service_runner())
    except KeyboardInterrupt:
        logger.info("Kesoku service stopped by user.")


@app.command("index")
def cli_index(
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
    concurrency: Annotated[int, typer.Option("--concurrency", help="Max concurrent embedding requests")] = 10,
) -> None:
    """Rebuild the semantic vector index for memories and messages."""
    import sqlite3

    load_config(config)

    async def _run_indexing():
        from kesoku.context import KesokuContext

        ctx = KesokuContext()
        if not ctx.embedding_store.enabled:
            Console().print(
                "[bold red]Error: Semantic search is disabled. "
                "Check embedding_model configuration or sqlite-vec installation.[/bold red]"
            )
            sys.exit(1)

        console = Console()
        console.print("[yellow]Starting bulk indexing...[/yellow]")

        db = ctx.sync_db

        # 1. Fetch memories
        memories = db.get_agent_memories()
        console.print(f"Found {len(memories)} memories to index.")

        # 2. Fetch messages
        with db.connection_provider.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT m.id, m.content, COALESCE(s.role_name, 'default') AS session_role
                FROM messages m
                JOIN sessions s ON m.session_id = s.id
                WHERE m.role IN ('user', 'assistant')
                  AND m.type = 'text'
            """)
            messages = cursor.fetchall()
        console.print(f"Found {len(messages)} messages to index.")

        total = len(memories) + len(messages)
        if total == 0:
            console.print("[green]Nothing to index.[/green]")
            return

        sem = asyncio.Semaphore(concurrency)
        processed = 0
        failed = 0

        async def embed_item(content_type: str, content_id: str, text: str):
            nonlocal processed, failed
            async with sem:
                try:
                    success = await ctx.embedding_store.embed(
                        content_type=content_type, content_id=content_id, text=text
                    )
                    if success:
                        processed += 1
                    else:
                        failed += 1
                except Exception as ex:
                    logger.error(f"Failed to embed {content_type} {content_id}: {ex}")
                    failed += 1

        tasks = []
        for m in memories:
            content_type = f"kesoku_memory:{m['role']}"
            content_id = f"{m['category']}:{m['key']}"
            text_to_index = f"{m['key']}: {m['title']}\n{m['content']}"
            tasks.append(embed_item(content_type, content_id, text_to_index))

        for m in messages:
            content_type = f"kesoku_message:{m['session_role']}"
            tasks.append(embed_item(content_type, m["id"], m["content"]))

        console.print(f"Indexing {total} items with concurrency={concurrency}...")

        await asyncio.gather(*tasks)

        console.print(f"[bold green]Indexing completed![/bold green] Processed: {processed}, Failed: {failed}")

    asyncio.run(_run_indexing())


if __name__ == "__main__":
    app()
