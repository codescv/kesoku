"""Typer CLI interface for Kesoku AI Agent framework.

Provides commands: init and chat (session-based one-shot CLI chat).
"""

import asyncio
import logging
import os
import re
import sys
from typing import Annotated

import typer
from rich.console import Console

from kesoku.agent.agent import Agent
from kesoku.cli_chat import run_cli_chat_async
from kesoku.cli_service import service_app
from kesoku.config import get_config, init_config, init_skills, load_config
from kesoku.db import DatabaseManager
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


@memory_app.command("list")
def cli_memory_list(
    category: Annotated[
        str | None, typer.Argument(help="Memory category (e.g., progress, user_profile, learnings)")
    ] = None,
    role: Annotated[str, typer.Option("-r", "--role", help="Optional roleplay persona scope")] = "global",
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
) -> None:
    """List all active memory keys in the given category, or list all permitted categories if category is omitted."""
    load_config(config)
    cfg = get_config()
    db = DatabaseManager(cfg.workspace.db_path)
    db.verify_db()

    console = Console()

    if not category:
        from kesoku.agent.tools import get_allowed_categories

        allowed = get_allowed_categories(db)
        console.print("\n[bold green]=== Permitted & Active Categories ===[/bold green]")
        core = {"user_profile", "learnings", "progress"}
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
        import time

        updated_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m["updated_at"]))
        console.print(
            f"  - key: [cyan]{m['key']}[/cyan] | "
            f"title: \"[bold]{m['title']}[/bold]\" | "
            f"scope: [yellow]{m['role']}[/yellow] | "
            f"updated: {updated_str}"
        )


@memory_app.command("view")
def cli_memory_view(
    category: Annotated[str, typer.Argument(help="Memory category (e.g., progress, user_profile, learnings)")],
    key: Annotated[str | None, typer.Argument(help="Unique memory key. Omit to view all category entries.")] = None,
    role: Annotated[str, typer.Option("-r", "--role", help="Optional roleplay persona scope")] = "global",
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
) -> None:
    """Retrieve the details of a specific memory key, or render all entries dynamically."""
    load_config(config)
    cfg = get_config()
    db = DatabaseManager(cfg.workspace.db_path)
    db.verify_db()

    from kesoku.agent.tools import sanitize_key, validate_key

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
    category: Annotated[str, typer.Argument(help="Memory category (e.g., progress, user_profile, learnings)")],
    key: Annotated[str, typer.Argument(help="Unique memory key")],
    title: Annotated[str, typer.Argument(help="Human-readable title or label")],
    content: Annotated[str, typer.Argument(help="Markdown or JSON content payload")],
    role: Annotated[str, typer.Option("-r", "--role", help="Optional roleplay persona scope")] = "global",
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

    from kesoku.agent.tools import get_allowed_categories, sanitize_key, validate_key

    allowed = get_allowed_categories(db)
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
    category: Annotated[str, typer.Argument(help="Memory category (e.g., progress, user_profile, learnings)")],
    key: Annotated[str, typer.Argument(help="Unique memory key")],
    role: Annotated[str, typer.Option("-r", "--role", help="Optional roleplay persona scope")] = "global",
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
) -> None:
    """Delete a specific memory record."""
    load_config(config)
    cfg = get_config()
    db = DatabaseManager(cfg.workspace.db_path)
    db.verify_db()

    from kesoku.agent.tools import sanitize_key, validate_key

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


@wechat_app.command("pair")
def wechat_pair(
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
    timeout: Annotated[int, typer.Option("-t", "--timeout", help="Timeout in seconds for QR scan confirmation")] = 480,
) -> None:
    """Pair WeChat account with Kesoku via terminal barcode QR code."""
    cfg = load_config(config)

    from kesoku.gateway.chatbot.wechat import qr_login

    logger.info("Starting WeChat pairing flow...")
    credentials = asyncio.run(qr_login(timeout_seconds=timeout))
    if not credentials:
        logger.error("WeChat pairing failed or timed out.")
        sys.exit(1)

    # Update config
    cfg.wechat.enabled = True
    cfg.wechat.account_id = credentials["account_id"]
    cfg.wechat.token = credentials["token"]
    cfg.wechat.base_url = credentials["base_url"]

    from kesoku.config import save_config
    save_config(cfg, config)

    console = Console()
    console.print("\n[bold green]WeChat chatbot paired and enabled successfully![/bold green]")
    console.print(f"  Account ID: [cyan]{credentials['account_id']}[/cyan]")
    console.print("  Token saved to configuration file.")


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

    if not cfg.discord.enabled and not cfg.google_chat.enabled and not cfg.wechat.enabled:
        console = Console()
        console.print("[bold red]Error: No chatbots are enabled in the configuration.[/bold red]")
        sys.exit(1)

    gateway = Gateway()
    agent = Agent(gateway=gateway)
    bot_tasks = []
    bots = []

    discord_token = cfg.discord.bot_token or os.environ.get("DISCORD_TOKEN")
    if cfg.discord.enabled and discord_token:
        from kesoku.gateway.chatbot.discord import DiscordChatbot

        discord_bot = DiscordChatbot(chatbot_id=cfg.discord.chatbot_id, gateway=gateway)
        bot_tasks.append(discord_bot.start())
        bots.append(discord_bot)
    elif cfg.discord.enabled and not discord_token:
        console = Console()
        console.print("[bold red]Error: Discord is enabled but bot_token is not configured.[/bold red]")
        sys.exit(1)

    if cfg.google_chat.enabled:
        from kesoku.gateway.chatbot.google_chat import GoogleChatChatbot

        gchat_bot = GoogleChatChatbot(chatbot_id=cfg.google_chat.chatbot_id, gateway=gateway)
        bot_tasks.append(gchat_bot.start())
        bots.append(gchat_bot)

    if cfg.wechat.enabled:
        from kesoku.gateway.chatbot.wechat import WechatChatbot

        wechat_bot = WechatChatbot(chatbot_id=cfg.wechat.chatbot_id, gateway=gateway)
        bot_tasks.append(wechat_bot.start())
        bots.append(wechat_bot)

    cron_manager = None
    config_dir = cfg.agent_working_dir
    if config_dir:
        cron_toml_path = os.path.join(config_dir, "cronjob.toml")
        if os.path.exists(cron_toml_path):
            from kesoku.cron import CronManager

            cron_manager = CronManager(chatbots=bots, config_dir=config_dir)
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

def map_legacy_title_to_key(title: str) -> str:
    """Maps legacy Chinese/Japanese titles to clean, conforming snake_case English keys."""
    mapping = {
        "标准日本语": "standard_japanese",
        "《标准日本语》": "standard_japanese",
        "智能简史": "brief_history_of_intelligence",
        "智能简史 (a brief history of intelligence)": "brief_history_of_intelligence",
        "学习进度与里程碑": "learning_progress_milestones",
        "notes by the user": "user_notes",
        "user profile": "user_profile",
        "my interest": "my_interests",
    }
    normalized = title.lower().strip()
    if normalized in mapping:
        return mapping[normalized]
    for k, v in mapping.items():
        if k in normalized or normalized in k:
            return v
    return title


@app.command("migrate-memory")
def migrate_memory_cmd(
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
    legacy_dir: Annotated[
        str, typer.Option("-l", "--legacy-dir", help="Directory containing legacy memory files")
    ] = "../band/memory",
) -> None:
    """Migrate legacy Markdown memory files (User, Agent, Progress) into SQLite database tables."""
    load_config(config)
    cfg = get_config()
    db = DatabaseManager(cfg.workspace.db_path)
    db.verify_db()

    from kesoku.agent.tools import sanitize_key

    console = Console()
    console.print(f"\n[bold yellow]Starting legacy memory migration from '{legacy_dir}'...[/bold yellow]")

    if not os.path.exists(legacy_dir):
        console.print(f"[bold red]Error: Legacy memory directory '{legacy_dir}' not found.[/bold red]")
        sys.exit(1)

    # 1. Migrate User.md -> user_profile
    user_path = os.path.join(legacy_dir, "User.md")
    if os.path.exists(user_path):
        console.print(f"Migrating '{user_path}'...")
        with open(user_path, encoding="utf-8") as f:
            content = f.read()
        # Split by "# "
        sections = content.split("\n# ")
        for s in sections:
            s = s.strip()
            if not s:
                continue
            lines = s.split("\n")
            title = lines[0].strip("# ").strip()
            body = "\n".join(lines[1:]).strip()
            if not title or not body:
                continue
            sanitized = sanitize_key(map_legacy_title_to_key(title))
            db.upsert_agent_memory(
                category="user_profile",
                key=sanitized,
                title=title,
                content=body,
                role="global",
            )
            console.print(f"  -> Seeded user_profile: [cyan]{sanitized}[/cyan] (\"{title}\")")

    # 2. Migrate Progress.md -> progress
    progress_path = os.path.join(legacy_dir, "Progress.md")
    if os.path.exists(progress_path):
        console.print(f"Migrating '{progress_path}'...")
        with open(progress_path, encoding="utf-8") as f:
            content = f.read()
        # Split by "## "
        sections = content.split("\n## ")
        for s in sections:
            s = s.strip()
            if not s:
                continue
            lines = s.split("\n")
            title = lines[0].strip("# ").strip()
            body = "\n".join(lines[1:]).strip()
            if not title or not body:
                continue
            sanitized = sanitize_key(map_legacy_title_to_key(title))
            db.upsert_agent_memory(
                category="progress",
                key=sanitized,
                title=title,
                content=body,
                role="global",
            )
            console.print(f"  -> Seeded progress: [cyan]{sanitized}[/cyan] (\"{title}\")")

    # 3. Migrate Agent.md -> learnings (bullet rules)
    agent_path = os.path.join(legacy_dir, "Agent.md")
    if os.path.exists(agent_path):
        console.print(f"Migrating '{agent_path}'...")
        with open(agent_path, encoding="utf-8") as f:
            content = f.read()
        # Extract bullet points
        bullets = re.findall(r"^\s*-\s*(.+)$", content, re.MULTILINE)
        for i, b in enumerate(bullets):
            b = b.strip()
            if not b:
                continue
            descriptor = f"lesson_{i+1}"
            if "uv run" in b.lower():
                descriptor = "uv_run_constraint"
            elif "playwright" in b.lower():
                descriptor = "playwright_system_chrome"
            elif "staging directory" in b.lower() or "staging_dir" in b.lower():
                descriptor = "staging_directory_protocol"
            elif "role-playing" in b.lower() or "角色扮演" in b.lower():
                descriptor = "role_playing_skill_activation"
            elif "gemini<3.1" in b.lower():
                descriptor = "deprecated_model_restriction"

            title = descriptor.replace("_", " ").title()
            db.upsert_agent_memory(
                category="learnings",
                key=descriptor,
                title=title,
                content=b,
                role="global",
            )
            console.print(f"  -> Seeded learning: [cyan]{descriptor}[/cyan] (\"{title}\")")

    console.print("\n[bold green]Legacy memory migration completed successfully![/bold green]")


if __name__ == "__main__":
    app()
