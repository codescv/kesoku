"""Typer CLI interface for Kesoku AI Agent framework.

Provides commands: init and chat (session-based one-shot CLI chat).
"""

import asyncio
import json
import logging
import os
import sys
from typing import Annotated

import typer
from rich.console import Console

from kesoku.agent.agent import Agent
from kesoku.cli.chat import run_cli_chat_async
from kesoku.cli.history import history_app
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


app.add_typer(history_app, name="history")


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
    resume: Annotated[
        str | None, typer.Option("-r", "--resume", metavar="SESSION_ID", help="Resume a specific session by ID")
    ] = None,
    resume_latest: Annotated[bool, typer.Option("-z", "--resume-latest", help="Resume the latest session")] = False,
) -> None:
    """Chat with Kesoku Agent in one-shot session mode."""
    load_config(config)
    try:
        asyncio.run(
            run_cli_chat_async(
                message=message,
                resume=resume,
                resume_latest=resume_latest,
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





if __name__ == "__main__":
    app()
