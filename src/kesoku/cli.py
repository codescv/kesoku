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

from kesoku.cli_chat import run_cli_chat_async
from kesoku.config import init_config, load_config
from kesoku.db import DatabaseManager
from kesoku.logger import configure_logging, setup_logger

# Setup global colored logging configuration
configure_logging(logging.INFO)
logger = setup_logger(__name__)

app = typer.Typer(help="Kesoku AI Agent CLI manager.")


@app.callback()
def main_callback(
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
) -> None:
    """Global callback option for configuration file location."""
    load_config(config)


@app.command("init")
def init_cmd(
    workspace_path: Annotated[
        str | None, typer.Option("-w", "--workspace-path", help="Workspace directory to initialize")
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing config file (creates backup)")] = False,
) -> None:
    """Initialize Kesoku workspace and generate config.toml from template."""
    if workspace_path is None:
        workspace_path = "."
    workspace_path = os.path.abspath(workspace_path)
    target_config_path = os.path.join(workspace_path, "config.toml")

    logger.info(f"Initializing Kesoku workspace at {workspace_path} (config: {target_config_path})...")
    init_config(target_config_path, force=force)

    cfg = load_config(target_config_path)
    DatabaseManager(cfg.workspace.db_path).init_tables()
    os.makedirs(cfg.workspace.skills_dir, exist_ok=True)
    logger.info(f"Workspace initialized successfully at {workspace_path}")


@app.command("chat")
def chat_cmd(
    message: Annotated[str | None, typer.Argument(help="Message to send to the agent")] = None,
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
) -> None:
    """Chat with Kesoku Agent in one-shot session mode."""
    try:
        asyncio.run(
            run_cli_chat_async(
                message=message,
                list_sessions=list_sessions,
                resume=resume,
                resume_latest=resume_latest,
                show_history=show_history,
            )
        )
    except KeyboardInterrupt:
        logger.info("Kesoku chat stopped by user.")
    except Exception as e:
        console = Console()
        console.print(f"[bold red]Error during Kesoku chat session: {e}[/bold red]")
        logger.error(f"Error during Kesoku chat session: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    app()
