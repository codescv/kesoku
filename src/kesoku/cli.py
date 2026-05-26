"""Typer CLI interface for Kesoku AI Agent framework.

Provides commands: init and chat (session-based one-shot CLI chat).
"""

import asyncio
import logging
import os
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


if __name__ == "__main__":
    app()
