"""CLI subcommands for the app channel (`kesoku app ...`): pairing, devices and status."""

from __future__ import annotations

import datetime
import os
import time
from typing import Annotated
from urllib.parse import urlparse

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from kesoku.cli.service import resolve_service_config_path
from kesoku.config import KesokuConfig, load_config
from kesoku.db import DatabaseManager
from kesoku.gateway.chatbot.app.pairing import PAIRING_TTL_SECONDS, create_pairing_offer
from kesoku.logger import setup_logger

logger = setup_logger(__name__)
app_app = typer.Typer(help="Manage the app channel: pair devices, list / revoke devices, show status.")

ConfigOpt = Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")]
ServiceNameOpt = Annotated[
    str | None,
    typer.Option(
        "--name",
        "-n",
        help="Service instance suffix (kesoku-<name>.service) or workspace name; defaults to the main kesoku service",
    ),
]
UserScopeOpt = Annotated[
    bool,
    typer.Option(
        "--user/--system",
        help="Resolve service config from user-level (default) or system-level service",
    ),
]


def _load(
    config_path: str,
    name: str | None = None,
    user: bool = True,
) -> tuple[KesokuConfig, DatabaseManager, bool]:
    resolved_path, matched_service = resolve_service_config_path(config_path, name=name, user=user)
    cfg = load_config(resolved_path)
    db = DatabaseManager(cfg.workspace.db_path)
    db._config = cfg
    db.verify_db()
    return cfg, db, matched_service


def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return "-"
    return datetime.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")


def _safe_host(url: str) -> str:
    """Return `host[:port]` of a URL without any userinfo (never prints credentials)."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        port = parsed.port
    except ValueError:
        return ""
    if not host:
        return ""
    return f"{host}:{port}" if port else host


def _print_qr(console: Console, text: str) -> None:
    try:
        import qrcode

        qr = qrcode.QRCode(border=1)
        qr.add_data(text)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except Exception as e:
        console.print(f"[yellow]QR rendering unavailable: {e}[/yellow]")


@app_app.command("pair")
def pair_cmd(
    name: ServiceNameOpt = None,
    device_name: Annotated[
        str, typer.Option("--device-name", "-d", help="Display name of the device being paired")
    ] = "device",
    config: ConfigOpt = "config.toml",
    user: UserScopeOpt = True,
    wait: Annotated[bool, typer.Option("--wait/--no-wait", help="Wait until the code is redeemed")] = True,
    qr: Annotated[
        bool, typer.Option("--qr/--no-qr", help="Print a QR code with a pairing deep link (enabled by default)")
    ] = True,
) -> None:
    """Create a one-time 6-digit pairing code (valid for 2 minutes) and QR code for a new app device."""
    cfg, db, matched_service = _load(config, name=name, user=user)
    effective_device_name = device_name
    if name and not matched_service and device_name == "device":
        effective_device_name = name
    console = Console()
    if not cfg.active_apps:
        console.print(
            "[yellow]" + escape("Warning: no enabled [[chatbots.app]] entry; the running service will not relay "
                                "this code.") + "[/yellow]"
        )
    offer = create_pairing_offer(db, effective_device_name)
    code = offer["code"]
    console.print(f"\n[bold]Pairing code:[/bold] [bold cyan]{code[:3]} {code[3:]}[/bold cyan]  "
                  f"(valid for {PAIRING_TTL_SECONDS // 60} minutes, single use)")
    console.print("Scan the QR code below in the Kesoku mobile app (or enter the 6-digit code manually).")
    if qr:
        app_cfg = cfg.get_app_config()
        relay_base = ""
        if app_cfg and app_cfg.relay_url:
            host = _safe_host(app_cfg.relay_url)
            relay_base = f"https://{host}" if host else ""
        pair_link = f"kesoku://pair?code={code}" + (f"&relay={relay_base}" if relay_base else "")
        _print_qr(console, pair_link)
        console.print(f"[dim]Pairing link: {escape(pair_link)}[/dim]")
    if not wait:
        return
    deadline = float(offer["expires_at"])
    with console.status("Waiting for the device to redeem the code..."):
        while time.time() < deadline:
            row = db.get_app_pairing(offer["code_hash"])
            if row is None and time.time() >= deadline:
                break
            if row is None:
                console.print("[bold red]The code was burned after too many failed attempts. Run "
                              "`kesoku app pair` again.[/bold red]")
                raise typer.Exit(code=1)
            if row.get("redeemed_device_id"):
                console.print(f"[bold green]Paired device {row['redeemed_device_id']} ({row['name']}).[/bold green]")
                return
            time.sleep(1.0)
    console.print("[bold red]The pairing code expired. Run `kesoku app pair` again.[/bold red]")
    raise typer.Exit(code=1)


@app_app.command("devices")
def devices_cmd(
    name: ServiceNameOpt = None,
    config: ConfigOpt = "config.toml",
    user: UserScopeOpt = True,
    all_devices: Annotated[bool, typer.Option("--all", help="Include revoked devices")] = False,
) -> None:
    """List paired app devices."""
    _, db, _ = _load(config, name=name, user=user)
    rows = db.list_app_devices(include_revoked=all_devices)
    console = Console()
    if not rows:
        console.print("No paired devices.")
        return
    table = Table(title="App devices")
    for col in ("device_id", "name", "kind", "platform", "created", "last seen", "revoked"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            r["device_id"],
            r["name"],
            r["kind"],
            r.get("platform") or "-",
            _fmt_ts(r.get("created_at")),
            _fmt_ts(r.get("last_seen_at")),
            _fmt_ts(r.get("revoked_at")),
        )
    console.print(table)


@app_app.command("revoke")
def revoke_cmd(
    device_id: Annotated[str, typer.Argument(help="Device id to revoke")],
    name: ServiceNameOpt = None,
    config: ConfigOpt = "config.toml",
    user: UserScopeOpt = True,
) -> None:
    """Revoke a device (the running service pushes the revocation to the relay within seconds)."""
    _, db, _ = _load(config, name=name, user=user)
    console = Console()
    if db.get_app_device(device_id) is None:
        console.print(f"[bold red]Unknown device '{device_id}'.[/bold red]")
        raise typer.Exit(code=1)
    if db.revoke_app_device(device_id):
        console.print(f"[green]Revoked device {device_id}.[/green]")
    else:
        console.print(f"Device {device_id} was already revoked.")


@app_app.command("status")
def status_cmd(
    name: ServiceNameOpt = None,
    config: ConfigOpt = "config.toml",
    user: UserScopeOpt = True,
) -> None:
    """Show app channel configuration, devices, memory policies and pending proposals (never secrets)."""
    cfg, db, _ = _load(config, name=name, user=user)
    console = Console()
    apps = cfg.chatbots.app
    if not apps:
        console.print(escape("No [[chatbots.app]] configured."))
    for a in apps:
        host = _safe_host(a.relay_url) if a.relay_url else ""
        token_set = bool(os.environ.get(a.engine_token_env))
        console.print(
            f"[bold]{escape(a.chatbot_id)}[/bold] "
            + escape(
                f"enabled={a.enabled} relay={host or '-'} token_env={a.engine_token_env} "
                f"({'set' if token_set else 'missing'}) stream_min_interval={a.stream_min_interval_sec}s "
                f"modes={','.join(a.mode_ids)}"
            )
        )
    devices = db.list_app_devices(include_revoked=True)
    active = [d for d in devices if d.get("revoked_at") is None]
    console.print(f"Devices: {len(active)} active, {len(devices) - len(active)} revoked")
    offer = db.get_active_app_pairing()
    console.print(f"Active pairing offer: {'yes, expires ' + _fmt_ts(offer['expires_at']) if offer else 'none'}")
    prefs = db.list_app_mode_memory_prefs()
    if prefs:
        table = Table(title="Per-mode memory policy")
        table.add_column("mode")
        table.add_column("policy")
        table.add_column("updated")
        for mode_id, row in sorted(prefs.items()):
            table.add_row(mode_id, row["memory_policy"], _fmt_ts(row["updated_at"]))
        console.print(table)
    pending = db.list_memory_proposals(status="pending", limit=500)
    console.print(f"Pending memory proposals: {len(pending)}")
