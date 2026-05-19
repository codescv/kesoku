"""Typer sub-commands for managing Kesoku as a systemd service.

Provides commands: install, uninstall, start, stop, restart.
"""

import os
import shutil
import subprocess
import sys
from typing import Annotated

import typer
from rich.console import Console

from kesoku.config import load_config
from kesoku.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_INHERITED_ENVS = [
    "PATH",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "GOOGLE_API_KEY",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_GENAI_USE_VERTEXAI",
    "DISCORD_TOKEN",
]

# Setup Sub-Typer app for services
service_app = typer.Typer(help="Manage Kesoku AI Agent as a systemd background service.")


def _verify_linux_platform(console: Console) -> None:
    """Verify that the host platform is Linux.

    Args:
        console: Console to output error message.

    Raises:
        typer.Exit: If platform is not Linux.
    """
    if sys.platform != "linux":
        console.print("[bold red]Error: systemd services are only supported on Linux platforms.[/bold red]")
        raise typer.Exit(code=1)


def _get_service_params(user: bool) -> tuple[str, list[str], str]:
    """Determine target path, systemctl reload command, and user flags based on execution level.

    Args:
        user: True for user-level installation, False for system-level.

    Returns:
        A tuple of (service_file_path, reload_cmd_list, user_flag_string)
    """
    if user:
        target_path = os.path.expanduser("~/.config/systemd/user/kesoku.service")
        systemctl_reload = ["systemctl", "--user", "daemon-reload"]
        user_flag = "--user "
    else:
        target_path = "/etc/systemd/system/kesoku.service"
        systemctl_reload = ["sudo", "systemctl", "daemon-reload"]
        user_flag = ""
    return target_path, systemctl_reload, user_flag


@service_app.command("install")
def install_cmd(
    config_path: Annotated[
        str,
        typer.Option(
            "-c",
            "--config",
            help="Path to the config.toml file for the service to run with",
        ),
    ] = "config.toml",
    env: Annotated[
        list[str] | None,
        typer.Option(
            "-e",
            "--env",
            help="Environment variables to set in KEY=VALUE format (can be specified multiple times)",
        ),
    ] = None,
    user: Annotated[
        bool,
        typer.Option(
            "--user/--system",
            help="Install as a user-level systemd service (default) or system-level service",
        ),
    ] = True,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Only print the systemd service file to stdout without installing",
        ),
    ] = False,
) -> None:
    """Install Kesoku as a systemd background service on Linux."""
    console = Console()
    _verify_linux_platform(console)

    # Resolve the absolute path of the configuration file and load it
    config_abs_path = os.path.abspath(config_path)
    if not os.path.exists(config_abs_path):
        console.print(
            f"[bold red]Error: Configuration file not found at '{config_abs_path}'. "
            "Please run 'kesoku init' first.[/bold red]"
        )
        raise typer.Exit(code=1)

    cfg = load_config(config_abs_path)
    working_dir = os.path.dirname(config_abs_path)

    # Detect absolute path of the kesoku executable
    executable_dir = os.path.dirname(sys.executable)
    kesoku_path = os.path.join(executable_dir, "kesoku")
    if not os.path.exists(kesoku_path):
        kesoku_path = shutil.which("kesoku") or "kesoku"
    else:
        kesoku_path = os.path.abspath(kesoku_path)

    # Construct Environment lines for systemd unit file
    env_dict = {}
    for key in DEFAULT_INHERITED_ENVS:
        if key in os.environ:
            env_dict[key] = os.environ[key]

    if env:
        for item in env:
            if "=" not in item:
                logger.warning(f"Skipping invalid environment variable format: '{item}'. Expected KEY=VALUE.")
                continue
            key, val = item.split("=", 1)
            env_dict[key] = val

    # Early validation of Discord chatbot token configuration
    if cfg.discord.enabled:
        discord_token = cfg.discord.bot_token or env_dict.get("DISCORD_TOKEN")
        if not discord_token:
            console.print(
                "[bold red]Error: Discord chatbot is enabled in the configuration, but no bot token "
                "was configured in either config.toml or inherited/passed environment variables. "
                "Please configure the bot token or specify it via '-e DISCORD_TOKEN=value'.[/bold red]"
            )
            raise typer.Exit(code=1)

    env_lines = [f'Environment="{k}={v}"' for k, v in env_dict.items()]
    environment_block = "\n".join(env_lines)

    target_path, systemctl_reload, user_flag = _get_service_params(user)
    wanted_by = "default.target" if user else "multi-user.target"

    # Generate Systemd unit file content with journal logging and improved recovery/lifecycle options
    unit_content = f"""[Unit]
Description=Kesoku AI Agent Service
After=network.target

[Service]
Type=simple
WorkingDirectory={working_dir}
ExecStart={kesoku_path} start -c {config_abs_path}
Restart=always
RestartSec=5
TimeoutStopSec=210
StandardOutput=journal
StandardError=journal
{environment_block}

[Install]
WantedBy={wanted_by}
"""

    if dry_run:
        console.print(f"[bold green]# Generated systemd service unit path: {target_path}[/bold green]")
        console.print(unit_content)
        return

    # Write the service file to the target path
    try:
        if user:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

        with open(target_path, "w") as f:
            f.write(unit_content)
        logger.info(f"Systemd service file written to: {target_path}")

    except PermissionError:
        console.print(f"[bold red]Error: Permission denied when writing to '{target_path}'.[/bold red]")
        if not user:
            console.print(
                "[bold yellow]Hint: Installing as a system service requires root/sudo permissions. "
                "Try running with 'sudo' or use '--user' instead.[/bold yellow]"
            )
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error writing systemd service file: {e}[/bold red]")
        raise typer.Exit(code=1)

    # Reload the systemd daemon configurations
    logger.info(f"Reloading systemd daemon via: {' '.join(systemctl_reload)}...")
    try:
        subprocess.run(systemctl_reload, check=True, capture_output=True, text=True)
        logger.info("Systemd daemon reloaded successfully.")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Could not reload systemd daemon automatically: {e.stderr.strip() or e}")
        console.print(
            "[bold yellow]Warning: Please reload systemd manually by running 'systemctl daemon-reload'.[/bold yellow]"
        )

    # Always automatically enable the service to register boot-time auto-start
    enable_cmd = ["systemctl"] + user_flag.split() + ["enable", "kesoku"]
    if not user:
        enable_cmd = ["sudo"] + enable_cmd

    logger.info(f"Enabling service via: {' '.join(enable_cmd)}...")
    try:
        subprocess.run(enable_cmd, check=True, capture_output=True, text=True)
        logger.info("Service enabled successfully.")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Could not enable service automatically: {e.stderr.strip() or e}")
        console.print(f"[bold yellow]Warning: Please enable the service manually: {' '.join(enable_cmd)}[/bold yellow]")

    console.print("\n[bold green]Kesoku service installed successfully![/bold green]")
    console.print("You can control the service using the following commands:")
    console.print(f"  [bold cyan]kesoku service start {user_flag.strip()}[/bold cyan]   - Start the service")
    console.print(f"  [bold cyan]kesoku service stop {user_flag.strip()}[/bold cyan]    - Stop the service")
    console.print(
        f"  [bold cyan]kesoku service status {user_flag.strip()}[/bold cyan]  - Check service status (via systemctl)"
    )
    console.print(f"  [bold cyan]kesoku service logs {user_flag.strip()}[/bold cyan]    - View service logs")


@service_app.command("uninstall")
def uninstall_cmd(
    user: Annotated[
        bool,
        typer.Option(
            "--user/--system",
            help="Uninstall as a user-level systemd service (default) or system-level service",
        ),
    ] = True,
) -> None:
    """Stop, disable, and uninstall Kesoku systemd service."""
    console = Console()
    _verify_linux_platform(console)

    target_path, systemctl_reload, user_flag = _get_service_params(user)

    # 1. Stop the service
    logger.info("Stopping the background service...")
    stop_cmd_list = ["systemctl"] + user_flag.split() + ["stop", "kesoku"]
    if not user:
        stop_cmd_list = ["sudo"] + stop_cmd_list
    subprocess.run(stop_cmd_list, capture_output=True)

    # 2. Disable the service
    logger.info("Disabling the background service...")
    disable_cmd_list = ["systemctl"] + user_flag.split() + ["disable", "kesoku"]
    if not user:
        disable_cmd_list = ["sudo"] + disable_cmd_list
    subprocess.run(disable_cmd_list, capture_output=True)

    # 3. Remove the service file from disk
    if os.path.exists(target_path):
        try:
            if not user:
                # System-level removal might require sudo privileges if we can't write
                subprocess.run(["sudo", "rm", "-f", target_path], check=True)
            else:
                os.remove(target_path)
            logger.info(f"Removed service file: {target_path}")
        except PermissionError:
            console.print(
                f"[bold red]Error: Permission denied when deleting '{target_path}'. Try running with 'sudo'.[/bold red]"
            )
            raise typer.Exit(code=1)
        except Exception as e:
            console.print(f"[bold red]Error removing service file: {e}[/bold red]")
            raise typer.Exit(code=1)
    else:
        logger.info(f"Service file '{target_path}' did not exist.")

    # 4. Reload the systemd configurations
    logger.info(f"Reloading systemd daemon via: {' '.join(systemctl_reload)}...")
    try:
        subprocess.run(systemctl_reload, check=True, capture_output=True, text=True)
        logger.info("Systemd daemon reloaded successfully.")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Could not reload systemd daemon automatically: {e.stderr.strip() or e}")

    console.print("[bold green]Kesoku service has been uninstalled successfully![/bold green]")


def _run_systemctl_action(action: str, user: bool, console: Console) -> None:
    """Internal helper to run systemctl command for start/stop/restart.

    Args:
        action: The systemctl command verb (e.g., 'start', 'stop', 'restart').
        user: True for user service scope, False for system.
        console: Rich Console instance.
    """
    _verify_linux_platform(console)
    _, _, user_flag = _get_service_params(user)

    cmd_list = ["systemctl"] + user_flag.split() + [action, "kesoku"]
    if not user:
        cmd_list = ["sudo"] + cmd_list

    logger.info(f"Running: {' '.join(cmd_list)}...")
    try:
        subprocess.run(cmd_list, check=True, capture_output=True, text=True)
        console.print(f"[bold green]Successfully executed service {action}![/bold green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error executing service {action}: {e.stderr.strip() or e}[/bold red]")
        raise typer.Exit(code=1)


@service_app.command("start")
def start_cmd(
    user: Annotated[
        bool,
        typer.Option(
            "--user/--system",
            help="Start as a user-level systemd service (default) or system-level service",
        ),
    ] = True,
) -> None:
    """Start the Kesoku background service."""
    console = Console()
    _run_systemctl_action("start", user, console)


@service_app.command("stop")
def stop_cmd(
    user: Annotated[
        bool,
        typer.Option(
            "--user/--system",
            help="Stop as a user-level systemd service (default) or system-level service",
        ),
    ] = True,
) -> None:
    """Stop the Kesoku background service."""
    console = Console()
    _run_systemctl_action("stop", user, console)


@service_app.command("restart")
def restart_cmd(
    user: Annotated[
        bool,
        typer.Option(
            "--user/--system",
            help="Restart as a user-level systemd service (default) or system-level service",
        ),
    ] = True,
) -> None:
    """Restart the Kesoku background service."""
    console = Console()
    _run_systemctl_action("restart", user, console)


@service_app.command("logs")
def logs_cmd(
    user: Annotated[
        bool,
        typer.Option(
            "--user/--system",
            help="Show logs for user-level systemd service (default) or system-level service",
        ),
    ] = True,
    follow: Annotated[
        bool,
        typer.Option(
            "-f",
            "--follow",
            help="Follow log output (like tail -f)",
        ),
    ] = False,
    lines: Annotated[
        int,
        typer.Option(
            "-n",
            "--lines",
            help="Number of journal entries to show",
        ),
    ] = 50,
) -> None:
    """Show logs from journald for the Kesoku background service."""
    console = Console()
    _verify_linux_platform(console)
    _, _, user_flag = _get_service_params(user)

    cmd_list = ["journalctl"]
    if user:
        cmd_list += ["--user"]
    cmd_list += ["-u", "kesoku"]

    if follow:
        cmd_list += ["-f"]
    if lines > 0:
        cmd_list += ["-n", str(lines)]

    logger.info(f"Viewing logs via: {' '.join(cmd_list)}...")
    try:
        subprocess.run(cmd_list, check=True)
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error retrieving logs: {e}[/bold red]")
        raise typer.Exit(code=1)


@service_app.command("status")
def status_cmd(
    user: Annotated[
        bool,
        typer.Option(
            "--user/--system",
            help="Check status of user-level systemd service (default) or system-level service",
        ),
    ] = True,
) -> None:
    """Check the status of the Kesoku background service."""
    console = Console()
    _verify_linux_platform(console)
    _, _, user_flag = _get_service_params(user)

    cmd_list = ["systemctl"] + user_flag.split() + ["status", "kesoku"]
    if not user:
        cmd_list = ["sudo"] + cmd_list

    logger.info(f"Checking status via: {' '.join(cmd_list)}...")
    subprocess.run(cmd_list)
