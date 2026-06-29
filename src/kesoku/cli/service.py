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
service_app = typer.Typer(help="Manage Kesoku AI Agent as a systemd (Linux) or launchd (macOS) background service.")


def _verify_supported_platform(console: Console) -> None:
    """Verify that the host platform is Linux or macOS.

    Args:
        console: Console to output error message.

    Raises:
        typer.Exit: If platform is not supported.
    """
    if sys.platform not in ("linux", "darwin"):
        console.print(
            "[bold red]Error: Background services are only supported on Linux and macOS platforms.[/bold red]"
        )
        raise typer.Exit(code=1)


def _get_service_params(user: bool, name: str | None = None) -> tuple[str, list[str], str, str]:
    """Determine target path, reload command, user flags, and service name.

    Args:
        user: True for user-level installation, False for system-level.
        name: Optional name suffix of the service instance.

    Returns:
        A tuple of (service_file_path, reload_cmd_list, user_flag_string, service_name)
    """
    is_mac = sys.platform == "darwin"

    if is_mac:
        service_name = "com.kesoku.agent" if not name else f"com.kesoku.agent-{name}"
        if user:
            target_path = os.path.expanduser(f"~/Library/LaunchAgents/{service_name}.plist")
            user_flag = "--user "
        else:
            target_path = f"/Library/LaunchDaemons/{service_name}.plist"
            user_flag = ""
        return target_path, [], user_flag, service_name
    else:
        service_name = "kesoku" if not name else f"kesoku-{name}"
        if user:
            target_path = os.path.expanduser(f"~/.config/systemd/user/{service_name}.service")
            systemctl_reload = ["systemctl", "--user", "daemon-reload"]
            user_flag = "--user "
        else:
            target_path = f"/etc/systemd/system/{service_name}.service"
            systemctl_reload = ["sudo", "systemctl", "daemon-reload"]
            user_flag = ""
        return target_path, systemctl_reload, user_flag, service_name


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
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            help="Name/identifier suffix of the service instance (installs kesoku-<name>.service)",
        ),
    ] = None,
) -> None:
    """Install Kesoku as a background service (systemd on Linux, launchd on macOS)."""
    console = Console()
    _verify_supported_platform(console)

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

    # Construct Environment lines
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
    for d_cfg in cfg.active_discords:
        discord_token = d_cfg.bot_token or env_dict.get("DISCORD_TOKEN")
        if not discord_token:
            console.print(
                f"[bold red]Error: Discord chatbot is enabled in the configuration "
                f"(instance: '{d_cfg.chatbot_id}'), but no bot token was configured "
                "in either config.toml or inherited/passed environment variables. "
                "Please configure the bot token or specify it via '-e DISCORD_TOKEN=value'.[/bold red]"
            )
            raise typer.Exit(code=1)

    # Inject service instance metadata environment variables
    env_dict["KESOKU_SERVICE_USER"] = "true" if user else "false"
    if name:
        env_dict["KESOKU_SERVICE_INSTANCE_NAME"] = name

    target_path, reload_cmd, user_flag, service_name = _get_service_params(user, name)

    is_mac = sys.platform == "darwin"

    if is_mac:
        env_dict_plist = "".join(f"        <key>{k}</key>\n        <string>{v}</string>\n" for k, v in env_dict.items())
        log_path = (
            os.path.expanduser(f"~/Library/Logs/Kesoku/{service_name}.log") if user else f"/var/log/{service_name}.log"
        )
        unit_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{service_name}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{kesoku_path}</string>
        <string>start</string>
        <string>-c</string>
        <string>{config_abs_path}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{working_dir}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_path}</string>
    <key>StandardErrorPath</key>
    <string>{log_path}</string>
    <key>EnvironmentVariables</key>
    <dict>
{env_dict_plist}    </dict>
</dict>
</plist>
"""
    else:
        env_lines = [f'Environment="{k}={v}"' for k, v in env_dict.items()]
        environment_block = "\n".join(env_lines)
        wanted_by = "default.target" if user else "multi-user.target"
        description = "Kesoku AI Agent Service"
        if name:
            description += f" ({name})"

        unit_content = f"""[Unit]
Description={description}
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
        type_str = "launchd plist" if is_mac else "systemd service unit"
        console.print(f"[bold green]# Generated {type_str} path: {target_path}[/bold green]")
        console.print(unit_content)
        return

    # Write the service file to the target path
    try:
        if user:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            if is_mac:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)

        with open(target_path, "w") as f:
            f.write(unit_content)
        logger.info(f"Service file written to: {target_path}")

    except PermissionError:
        console.print(f"[bold red]Error: Permission denied when writing to '{target_path}'.[/bold red]")
        if not user:
            console.print(
                "[bold yellow]Hint: Installing as a system service requires root/sudo permissions. "
                "Try running with 'sudo' or use '--user' instead.[/bold yellow]"
            )
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error writing service file: {e}[/bold red]")
        raise typer.Exit(code=1)

    if is_mac:
        # Load plist
        load_cmd = ["launchctl", "load", "-w", target_path]
        if not user:
            load_cmd = ["sudo"] + load_cmd
        logger.info(f"Loading macOS launchd service via: {' '.join(load_cmd)}...")
        try:
            subprocess.run(load_cmd, check=True, capture_output=True, text=True)
            logger.info("Service loaded successfully.")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not load service automatically: {e.stderr.strip() or e}")
            console.print(f"[bold yellow]Warning: Please load the service manually: {' '.join(load_cmd)}[/bold yellow]")
    else:
        # Reload the systemd daemon configurations
        logger.info(f"Reloading systemd daemon via: {' '.join(reload_cmd)}...")
        try:
            subprocess.run(reload_cmd, check=True, capture_output=True, text=True)
            logger.info("Systemd daemon reloaded successfully.")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not reload systemd daemon automatically: {e.stderr.strip() or e}")
            console.print(
                "[bold yellow]Warning: Please reload systemd manually by running "
                "'systemctl daemon-reload'.[/bold yellow]"
            )

        # Always automatically enable the service to register boot-time auto-start
        enable_cmd = ["systemctl"] + user_flag.split() + ["enable", service_name]
        if not user:
            enable_cmd = ["sudo"] + enable_cmd

        logger.info(f"Enabling service via: {' '.join(enable_cmd)}...")
        try:
            subprocess.run(enable_cmd, check=True, capture_output=True, text=True)
            logger.info("Service enabled successfully.")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not enable service automatically: {e.stderr.strip() or e}")
            console.print(
                f"[bold yellow]Warning: Please enable the service manually: {' '.join(enable_cmd)}[/bold yellow]"
            )

    console.print("\n[bold green]Kesoku service installed successfully![/bold green]")
    console.print("You can control the service using the following commands:")
    name_opt = f" --name {name}" if name else ""
    console.print(f"  [bold cyan]kesoku service start {user_flag.strip()}{name_opt}[/bold cyan]   - Start the service")
    console.print(f"  [bold cyan]kesoku service stop {user_flag.strip()}{name_opt}[/bold cyan]    - Stop the service")
    status_label = f"kesoku service status {user_flag.strip()}{name_opt}"
    console.print(f"  [bold cyan]{status_label}[/bold cyan]  - Check service status")
    console.print(f"  [bold cyan]kesoku service logs {user_flag.strip()}{name_opt}[/bold cyan]    - View service logs")


@service_app.command("uninstall")
def uninstall_cmd(
    user: Annotated[
        bool,
        typer.Option(
            "--user/--system",
            help="Uninstall as a user-level service (default) or system-level service",
        ),
    ] = True,
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            help="Name/identifier suffix of the service instance to uninstall",
        ),
    ] = None,
) -> None:
    """Stop, disable, and uninstall Kesoku background service."""
    console = Console()
    _verify_supported_platform(console)

    target_path, reload_cmd, user_flag, service_name = _get_service_params(user, name)

    is_mac = sys.platform == "darwin"

    # 1. Stop and disable the service
    if is_mac:
        logger.info(f"Unloading the background service: {service_name}...")
        unload_cmd_list = ["launchctl", "unload", "-w", target_path]
        if not user:
            unload_cmd_list = ["sudo"] + unload_cmd_list
        subprocess.run(unload_cmd_list, capture_output=True)
    else:
        logger.info(f"Stopping the background service: {service_name}...")
        stop_cmd_list = ["systemctl"] + user_flag.split() + ["stop", service_name]
        if not user:
            stop_cmd_list = ["sudo"] + stop_cmd_list
        subprocess.run(stop_cmd_list, capture_output=True)

        logger.info(f"Disabling the background service: {service_name}...")
        disable_cmd_list = ["systemctl"] + user_flag.split() + ["disable", service_name]
        if not user:
            disable_cmd_list = ["sudo"] + disable_cmd_list
        subprocess.run(disable_cmd_list, capture_output=True)

    # 2. Remove the service file from disk
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

    # 3. Reload daemon configurations (Linux only)
    if not is_mac and reload_cmd:
        logger.info(f"Reloading systemd daemon via: {' '.join(reload_cmd)}...")
        try:
            subprocess.run(reload_cmd, check=True, capture_output=True, text=True)
            logger.info("Systemd daemon reloaded successfully.")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not reload systemd daemon automatically: {e.stderr.strip() or e}")

    console.print("[bold green]Kesoku service has been uninstalled successfully![/bold green]")


def _run_service_action(action: str, user: bool, name: str | None, console: Console) -> None:
    """Internal helper to run command for start/stop/restart.

    Args:
        action: The command verb (e.g., 'start', 'stop', 'restart').
        user: True for user service scope, False for system.
        name: Optional name suffix of the service instance.
        console: Rich Console instance.
    """
    _verify_supported_platform(console)
    target_path, _, user_flag, service_name = _get_service_params(user, name)

    is_mac = sys.platform == "darwin"

    if is_mac:
        if action == "restart":
            # launchd has no restart verb, so we do stop + start
            stop_cmd = ["launchctl", "stop", service_name]
            if not user:
                stop_cmd = ["sudo"] + stop_cmd
            logger.info(f"Running: {' '.join(stop_cmd)}...")
            subprocess.run(stop_cmd, capture_output=True)

            start_cmd = ["launchctl", "start", service_name]
            if not user:
                start_cmd = ["sudo"] + start_cmd
            logger.info(f"Running: {' '.join(start_cmd)}...")
            try:
                subprocess.run(start_cmd, check=True, capture_output=True, text=True)
                console.print(f"[bold green]Successfully executed service restart for {service_name}![/bold green]")
            except subprocess.CalledProcessError as e:
                err_msg = e.stderr.strip() or e
                console.print(f"[bold red]Error executing service restart for {service_name}: {err_msg}[/bold red]")
                raise typer.Exit(code=1)
        else:
            cmd_list = ["launchctl", action, service_name]
            if not user:
                cmd_list = ["sudo"] + cmd_list

            logger.info(f"Running: {' '.join(cmd_list)}...")
            try:
                subprocess.run(cmd_list, check=True, capture_output=True, text=True)
                console.print(f"[bold green]Successfully executed service {action} for {service_name}![/bold green]")
            except subprocess.CalledProcessError as e:
                err_msg = e.stderr.strip() or e
                console.print(f"[bold red]Error executing service {action} for {service_name}: {err_msg}[/bold red]")
                raise typer.Exit(code=1)
    else:
        cmd_list = ["systemctl"] + user_flag.split() + [action, service_name]
        if not user:
            cmd_list = ["sudo"] + cmd_list

        logger.info(f"Running: {' '.join(cmd_list)}...")
        try:
            subprocess.run(cmd_list, check=True, capture_output=True, text=True)
            console.print(f"[bold green]Successfully executed service {action} for {service_name}![/bold green]")
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.strip() or e
            console.print(f"[bold red]Error executing service {action} for {service_name}: {err_msg}[/bold red]")
            raise typer.Exit(code=1)


@service_app.command("start")
def start_cmd(
    user: Annotated[
        bool,
        typer.Option(
            "--user/--system",
            help="Start as a user-level service (default) or system-level service",
        ),
    ] = True,
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            help="Name/identifier suffix of the service instance to start",
        ),
    ] = None,
) -> None:
    """Start the Kesoku background service."""
    console = Console()
    _run_service_action("start", user, name, console)


@service_app.command("stop")
def stop_cmd(
    user: Annotated[
        bool,
        typer.Option(
            "--user/--system",
            help="Stop as a user-level service (default) or system-level service",
        ),
    ] = True,
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            help="Name/identifier suffix of the service instance to stop",
        ),
    ] = None,
) -> None:
    """Stop the Kesoku background service."""
    console = Console()
    _run_service_action("stop", user, name, console)


@service_app.command("restart")
def restart_cmd(
    user: Annotated[
        bool,
        typer.Option(
            "--user/--system",
            help="Restart as a user-level service (default) or system-level service",
        ),
    ] = True,
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            help="Name/identifier suffix of the service instance to restart",
        ),
    ] = None,
) -> None:
    """Restart the Kesoku background service."""
    console = Console()
    _run_service_action("restart", user, name, console)


@service_app.command("logs")
def logs_cmd(
    user: Annotated[
        bool,
        typer.Option(
            "--user/--system",
            help="Show logs for user-level service (default) or system-level service",
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
            help="Number of log entries to show",
        ),
    ] = 50,
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            help="Name/identifier suffix of the service instance to show logs for",
        ),
    ] = None,
) -> None:
    """Show logs for the Kesoku background service."""
    console = Console()
    _verify_supported_platform(console)
    target_path, _, user_flag, service_name = _get_service_params(user, name)

    if sys.platform == "darwin":
        log_path = (
            os.path.expanduser(f"~/Library/Logs/Kesoku/{service_name}.log") if user else f"/var/log/{service_name}.log"
        )
        if not os.path.exists(log_path):
            console.print(
                f"[bold yellow]No log file found at '{log_path}'. The service may not have started yet.[/bold yellow]"
            )
            return

        cmd_list = ["tail"]
        if follow:
            cmd_list += ["-f"]
        if lines > 0:
            cmd_list += ["-n", str(lines)]
        cmd_list += [log_path]

        logger.info(f"Viewing logs via: {' '.join(cmd_list)}...")
        try:
            subprocess.run(cmd_list, check=True)
        except KeyboardInterrupt:
            pass
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]Error retrieving logs: {e}[/bold red]")
            raise typer.Exit(code=1)
    else:
        cmd_list = ["journalctl"]
        if user:
            cmd_list += ["--user"]
        cmd_list += ["-u", service_name]

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
            help="Check status of user-level service (default) or system-level service",
        ),
    ] = True,
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            help="Name/identifier suffix of the service instance to check status of",
        ),
    ] = None,
) -> None:
    """Check the status of the Kesoku background service."""
    console = Console()
    _verify_supported_platform(console)
    target_path, _, user_flag, service_name = _get_service_params(user, name)

    if sys.platform == "darwin":
        cmd_list = ["launchctl", "list", service_name]
        if not user:
            cmd_list = ["sudo"] + cmd_list
        logger.info(f"Checking status via: {' '.join(cmd_list)}...")
        subprocess.run(cmd_list)
    else:
        cmd_list = ["systemctl"] + user_flag.split() + ["status", service_name]
        if not user:
            cmd_list = ["sudo"] + cmd_list

        logger.info(f"Checking status via: {' '.join(cmd_list)}...")
        subprocess.run(cmd_list)
