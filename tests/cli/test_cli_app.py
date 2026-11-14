"""Tests for the `kesoku app` CLI (pairing, devices, revoke, status)."""

from __future__ import annotations

import importlib
import os
import re
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from kesoku.cli import app
from kesoku.db import DatabaseManager
from kesoku.gateway.chatbot.app.pairing import PAIRING_TTL_SECONDS, hash_pairing_code

runner = CliRunner()

# The `kesoku app` CLI module.
cli_app = importlib.import_module("kesoku.cli.app_channel")

TOKEN_ENV = "KESOKU_TEST_RELAY_ENGINE_TOKEN"
FAKE_TOKEN = "fake-relay-token-ZXQ81-do-not-print"
CODE_RE = re.compile(r"Pairing code:\s*(\d{3}) (\d{3})")


def _write_config(tmp_path: Path, *, relay_url: str = "wss://relay.example.test/engine", enabled: bool = True,
                  with_app: bool = True) -> Path:
    """Write a minimal config.toml (optionally with one `[[chatbots.app]]` entry) and init its DB."""
    lines = [
        "[workspace]",
        'db_path = "kesoku.db"',
        'skills_dir = "skills"',
        'roles_dir = "roles"',
        'sessions_dir = "sessions"',
        "",
    ]
    if with_app:
        lines += [
            "[[chatbots.app]]",
            f"enabled = {'true' if enabled else 'false'}",
            'chatbot_id = "app"',
            f'relay_url = "{relay_url}"',
            f'engine_token_env = "{TOKEN_ENV}"',
            "",
        ]
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("\n".join(lines), encoding="utf-8")
    db = DatabaseManager(str(tmp_path / "kesoku.db"))
    db.init_tables()
    db.verify_db()
    return cfg_path


@pytest.fixture(autouse=True)
def _wide_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep Rich from wrapping/truncating output lines and tables."""
    monkeypatch.setenv("COLUMNS", "240")
    monkeypatch.delenv(TOKEN_ENV, raising=False)


@pytest.fixture
def cfg_path(tmp_path: Path) -> Path:
    """A tmp workspace with an enabled `[[chatbots.app]]` entry."""
    return _write_config(tmp_path)


@pytest.fixture
def db(cfg_path: Path) -> DatabaseManager:
    """A second DatabaseManager on the same file, standing in for the running service."""
    manager = DatabaseManager(str(cfg_path.parent / "kesoku.db"))
    manager.verify_db()
    return manager


def _invoke(*args: str) -> Any:
    return runner.invoke(app, ["app", *args])


def _pairing_rows(db_path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM app_pairings").fetchall()]
    finally:
        conn.close()


def _all_text_values(db_path: Path) -> list[str]:
    """Every TEXT/INTEGER cell value of every table, stringified."""
    conn = sqlite3.connect(db_path)
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        values: list[str] = []
        for t in tables:
            if t.startswith("sqlite_"):
                continue
            try:
                rows = conn.execute(f'SELECT * FROM "{t}"').fetchall()  # noqa: S608 - names from sqlite_master
            except sqlite3.DatabaseError:
                continue  # virtual tables (e.g. FTS shadow) may not be directly readable
            for row in rows:
                values.extend(str(v) for v in row if v is not None)
        return values
    finally:
        conn.close()


class _FakeClock:
    """Replacement for the `time` module used by the CLI wait loop."""

    def __init__(self, start: float, on_sleep: Callable[[_FakeClock], None], step: float = 1.0) -> None:
        self.now = start
        self.sleeps = 0
        self._on_sleep = on_sleep
        self._step = step

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        self.now += self._step
        self._on_sleep(self)


def _install_clock(monkeypatch: pytest.MonkeyPatch, on_sleep: Callable[[_FakeClock], None],
                   step: float = 1.0) -> _FakeClock:
    clock = _FakeClock(time.time(), on_sleep, step)
    monkeypatch.setattr(cli_app, "time", clock)
    return clock


# --------------------------------------------------------------------------- registration / help


def test_app_subcommand_is_registered() -> None:
    result = runner.invoke(app, ["app", "--help"])
    assert result.exit_code == 0, result.output
    for cmd in ("pair", "devices", "revoke", "status"):
        assert cmd in result.output


def test_missing_config_fails(tmp_path: Path) -> None:
    result = _invoke("devices", "-c", str(tmp_path / "missing.toml"))
    assert result.exit_code != 0
    assert isinstance(result.exception, FileNotFoundError)


def test_uninitialized_db_fails(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text('[workspace]\ndb_path = "kesoku.db"\n', encoding="utf-8")
    result = _invoke("status", "-c", str(cfg))
    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)


# --------------------------------------------------------------------------- pair --no-wait


def test_pair_no_wait_prints_six_digit_code_and_stores_only_salted_hash(cfg_path: Path) -> None:
    result = _invoke("pair", "--no-wait", "--name", "Test Phone", "-c", str(cfg_path))
    assert result.exit_code == 0, result.output
    m = CODE_RE.search(result.output)
    assert m, result.output
    code = m.group(1) + m.group(2)
    assert len(code) == 6 and code.isdigit()
    assert f"valid for {PAIRING_TTL_SECONDS // 60} minutes" in result.output

    rows = _pairing_rows(cfg_path.parent / "kesoku.db")
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "Test Phone"
    assert row["redeemed_device_id"] is None
    assert re.fullmatch(r"[0-9a-f]{32}", row["salt"])
    assert re.fullmatch(r"[0-9a-f]{64}", row["code_hash"])
    assert row["code_hash"] == hash_pairing_code(code, row["salt"])
    assert row["code_hash"] != hash_pairing_code(code, ""), "hash must be salted"
    assert row["expires_at"] - row["created_at"] == pytest.approx(PAIRING_TTL_SECONDS)

    # The plain code is never stored in any column of any table.
    assert code not in _all_text_values(cfg_path.parent / "kesoku.db")
    assert code not in {str(v) for v in row.values()}


def test_pair_replaces_previous_unredeemed_offer(cfg_path: Path) -> None:
    first = _invoke("pair", "--no-wait", "-c", str(cfg_path))
    second = _invoke("pair", "--no-wait", "-c", str(cfg_path))
    assert first.exit_code == 0 and second.exit_code == 0
    rows = _pairing_rows(cfg_path.parent / "kesoku.db")
    assert len(rows) == 1
    m = CODE_RE.search(second.output)
    assert m
    assert rows[0]["code_hash"] == hash_pairing_code(m.group(1) + m.group(2), rows[0]["salt"])
    assert rows[0]["name"] == "device"


def test_pair_warns_without_enabled_app_entry(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, enabled=False)
    result = _invoke("pair", "--no-wait", "-c", str(cfg))
    assert result.exit_code == 0, result.output
    assert "Warning: no enabled" in result.output
    assert CODE_RE.search(result.output)


# Known src bug: Rich parses "[chatbots.app]" as a markup tag, so "[[chatbots.app]]" renders as "[]".


def test_pair_warning_names_config_section_literally(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, enabled=False)
    result = _invoke("pair", "--no-wait", "-c", str(cfg))
    assert "no enabled [[chatbots.app]] entry" in result.output


def test_pair_qr_deep_link_uses_relay_origin_only(cfg_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []
    monkeypatch.setattr(cli_app, "_print_qr", lambda _console, text: captured.append(text))
    result = _invoke("pair", "--no-wait", "--qr", "-c", str(cfg_path))
    assert result.exit_code == 0, result.output
    m = CODE_RE.search(result.output)
    assert m
    assert captured == [f"kesoku://pair?code={m.group(1)}{m.group(2)}&relay=https://relay.example.test"]


def test_pair_qr_without_relay_url_omits_relay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _write_config(tmp_path, relay_url="")
    captured: list[str] = []
    monkeypatch.setattr(cli_app, "_print_qr", lambda _console, text: captured.append(text))
    result = _invoke("pair", "--no-wait", "--qr", "-c", str(cfg))
    assert result.exit_code == 0, result.output
    assert len(captured) == 1 and "relay=" not in captured[0]


# --------------------------------------------------------------------------- pair (waiting)


def test_pair_wait_reports_redeemed_device(cfg_path: Path, db: DatabaseManager,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    def redeem_on_second_poll(clock: _FakeClock) -> None:
        if clock.sleeps == 2:
            offer = db.get_active_app_pairing(now=clock.now)
            assert offer is not None
            db.upsert_app_device("dev-abc", offer["name"], "phone", "android")
            assert db.mark_app_pairing_redeemed(offer["code_hash"], "dev-abc")

    clock = _install_clock(monkeypatch, redeem_on_second_poll)
    result = _invoke("pair", "--name", "Tablet", "-c", str(cfg_path))
    assert result.exit_code == 0, result.output
    assert "Paired device dev-abc (Tablet)." in result.output
    assert clock.sleeps == 2


def test_pair_wait_reports_burned_code(cfg_path: Path, db: DatabaseManager,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    def burn(clock: _FakeClock) -> None:
        offer = db.get_active_app_pairing(now=clock.now)
        assert offer is not None
        db.delete_app_pairing(offer["code_hash"])

    _install_clock(monkeypatch, burn)
    result = _invoke("pair", "-c", str(cfg_path))
    assert result.exit_code == 1, result.output
    assert "burned after too many failed attempts" in result.output
    assert "Paired device" not in result.output


def test_pair_wait_reports_expiry(cfg_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _install_clock(monkeypatch, lambda _c: None, step=30.0)
    result = _invoke("pair", "-c", str(cfg_path))
    assert result.exit_code == 1, result.output
    assert "pairing code expired" in result.output
    # 120 s TTL polled every 30 fake seconds: the loop stops once the deadline passes.
    assert 3 <= clock.sleeps <= 5
    # The unredeemed offer is left in place for the service to clean up; nothing was redeemed.
    rows = _pairing_rows(cfg_path.parent / "kesoku.db")
    assert len(rows) == 1 and rows[0]["redeemed_device_id"] is None


# --------------------------------------------------------------------------- devices


def test_devices_empty(cfg_path: Path) -> None:
    result = _invoke("devices", "-c", str(cfg_path))
    assert result.exit_code == 0, result.output
    assert "No paired devices." in result.output


def test_devices_lists_active_and_hides_revoked_unless_all(cfg_path: Path, db: DatabaseManager) -> None:
    now = time.time()
    db.upsert_app_device("dev-one", "Laptop", "desktop", "macos", created_at=now - 100)
    db.upsert_app_device("dev-two", "Phone", "phone", None, created_at=now - 50)
    db.touch_app_device("dev-one", now - 10)
    assert db.revoke_app_device("dev-two")

    result = _invoke("devices", "-c", str(cfg_path))
    assert result.exit_code == 0, result.output
    assert "dev-one" in result.output and "Laptop" in result.output and "macos" in result.output
    assert "dev-two" not in result.output

    result_all = _invoke("devices", "--all", "-c", str(cfg_path))
    assert result_all.exit_code == 0, result_all.output
    assert "dev-one" in result_all.output and "dev-two" in result_all.output
    # Missing platform renders as "-".
    phone_line = next(line for line in result_all.output.splitlines() if "dev-two" in line)
    assert "-" in phone_line


# --------------------------------------------------------------------------- revoke


def test_revoke_unknown_device_fails(cfg_path: Path) -> None:
    result = _invoke("revoke", "nope", "-c", str(cfg_path))
    assert result.exit_code == 1
    assert "Unknown device 'nope'" in result.output


def test_revoke_then_revoke_again(cfg_path: Path, db: DatabaseManager) -> None:
    db.upsert_app_device("dev-r", "Watch", "wearable")
    first = _invoke("revoke", "dev-r", "-c", str(cfg_path))
    assert first.exit_code == 0, first.output
    assert "Revoked device dev-r." in first.output
    dev = db.get_app_device("dev-r")
    assert dev is not None and dev["revoked_at"] is not None
    assert db.list_revoked_app_device_ids() == ["dev-r"]

    second = _invoke("revoke", "dev-r", "-c", str(cfg_path))
    assert second.exit_code == 0, second.output
    assert "already revoked" in second.output


# --------------------------------------------------------------------------- status


def test_status_without_app_entry(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, with_app=False)
    result = _invoke("status", "-c", str(cfg))
    assert result.exit_code == 0, result.output
    assert re.search(r"No \[.*\] configured\.", result.output)
    assert "Devices: 0 active, 0 revoked" in result.output
    assert "Active pairing offer: none" in result.output
    assert "Pending memory proposals: 0" in result.output


def test_status_without_app_entry_names_config_section_literally(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, with_app=False)
    result = _invoke("status", "-c", str(cfg))
    assert "No [[chatbots.app]] configured." in result.output


def test_status_reports_missing_token(cfg_path: Path) -> None:
    result = _invoke("status", "-c", str(cfg_path))
    assert result.exit_code == 0, result.output
    assert f"token_env={TOKEN_ENV} (missing)" in result.output


def test_status_never_prints_token_value(cfg_path: Path, db: DatabaseManager,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TOKEN_ENV, FAKE_TOKEN)
    now = time.time()
    db.upsert_app_device("dev-a", "A", "phone")
    db.upsert_app_device("dev-b", "B", "phone")
    db.revoke_app_device("dev-b")
    db.insert_app_pairing("h" * 64, "s" * 32, "pending device", now, now + 60)
    db.set_app_mode_memory_policy("chat", "full")
    db.set_app_mode_memory_policy("practice", "confirm_only")
    db.insert_memory_proposal(
        proposal_id="p1",
        session_id="s1",
        message_id=None,
        role="default",
        source_mode="practice",
        ops=[{"op": "add", "text": "x"}],
        fingerprint="fp1",
        created_at=now,
        expires_at=now + 3600,
    )

    result = _invoke("status", "-c", str(cfg_path))
    assert result.exit_code == 0, result.output
    out = result.output
    assert FAKE_TOKEN not in out
    assert "ZXQ81" not in out
    assert f"token_env={TOKEN_ENV} (set)" in out
    assert "relay=relay.example.test" in out
    assert "enabled=True" in out
    assert "modes=chat" in out
    assert "Devices: 1 active, 1 revoked" in out
    assert "Active pairing offer: yes, expires" in out
    assert "practice" in out and "confirm_only" in out
    assert "Pending memory proposals: 1" in out
    # Salt / hash of the pending offer are never shown either.
    assert "s" * 32 not in out and "h" * 64 not in out


def test_status_never_prints_token_set_via_config_env(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    cfg.write_text(cfg.read_text(encoding="utf-8") + f'\n[env]\n{TOKEN_ENV} = "{FAKE_TOKEN}"\n', encoding="utf-8")
    try:
        result = _invoke("status", "-c", str(cfg))
    finally:
        # load_config injects [env] into os.environ directly; undo it.
        os.environ.pop(TOKEN_ENV, None)
    assert result.exit_code == 0, result.output
    assert FAKE_TOKEN not in result.output
    assert f"token_env={TOKEN_ENV} (set)" in result.output


def test_status_hides_token_in_relay_query_string(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _write_config(tmp_path, relay_url=f"wss://relay.example.test/engine?token={FAKE_TOKEN}")
    monkeypatch.setenv(TOKEN_ENV, FAKE_TOKEN)
    result = _invoke("status", "-c", str(cfg))
    assert result.exit_code == 0, result.output
    assert FAKE_TOKEN not in result.output
    assert "relay=relay.example.test " in result.output




def test_status_hides_credentials_in_relay_userinfo(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, relay_url=f"wss://engine:{FAKE_TOKEN}@relay.example.test/engine")
    result = _invoke("status", "-c", str(cfg))
    assert result.exit_code == 0, result.output
    assert FAKE_TOKEN not in result.output


def test_pair_qr_hides_credentials_in_relay_userinfo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _write_config(tmp_path, relay_url=f"wss://engine:{FAKE_TOKEN}@relay.example.test/engine")
    captured: list[str] = []
    monkeypatch.setattr(cli_app, "_print_qr", lambda _console, text: captured.append(text))
    result = _invoke("pair", "--no-wait", "--qr", "-c", str(cfg))
    assert result.exit_code == 0, result.output
    assert captured and FAKE_TOKEN not in captured[0]


def test_pair_prints_qr_by_default_and_no_qr_disables_it(cfg_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []
    monkeypatch.setattr(cli_app, "_print_qr", lambda _console, text: captured.append(text))
    res_default = _invoke("pair", "--no-wait", "-c", str(cfg_path))
    assert res_default.exit_code == 0, res_default.output
    assert len(captured) == 1
    assert captured[0].startswith("kesoku://pair?code=")

    captured.clear()
    res_no_qr = _invoke("pair", "--no-wait", "--no-qr", "-c", str(cfg_path))
    assert res_no_qr.exit_code == 0, res_no_qr.output
    assert captured == []


def test_app_commands_resolve_config_from_service_without_entering_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kesoku.cli import service as cli_service

    ws_main = tmp_path / "band"
    ws_main.mkdir()
    cfg_main = _write_config(ws_main, relay_url="wss://main.example.test/engine")

    ws_pub = tmp_path / "band-pub"
    ws_pub.mkdir()
    cfg_pub = _write_config(ws_pub, relay_url="wss://pub.example.test/engine")

    svc_dir = tmp_path / "systemd_user"
    svc_dir.mkdir()
    (svc_dir / "kesoku.service").write_text(
        f"[Service]\nWorkingDirectory={ws_main}\nExecStart=/usr/bin/kesoku start -c {cfg_main}\n",
        encoding="utf-8",
    )
    (svc_dir / "kesoku-pub.service").write_text(
        f"[Service]\nWorkingDirectory={ws_pub}\nExecStart=/usr/bin/kesoku start -c {cfg_pub}\n",
        encoding="utf-8",
    )

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    monkeypatch.chdir(outside_dir)

    orig_resolve = cli_service.resolve_service_config_path
    monkeypatch.setattr(
        cli_app,
        "resolve_service_config_path",
        lambda config="config.toml", name=None, user=True: orig_resolve(
            config, name=name, user=user, service_dirs=[str(svc_dir)]
        ),
    )

    # 1. `kesoku app pair --no-wait` outside `band` resolves default `kesoku.service`
    res1 = _invoke("pair", "--no-wait", "--device-name", "iPhone 17")
    assert res1.exit_code == 0, res1.output
    rows_main = _pairing_rows(ws_main / "kesoku.db")
    assert len(rows_main) == 1
    assert rows_main[0]["name"] == "iPhone 17"

    # 2. `kesoku app pair --name pub --no-wait` resolves `kesoku-pub.service`
    res2 = _invoke("pair", "--name", "pub", "--no-wait")
    assert res2.exit_code == 0, res2.output
    rows_pub = _pairing_rows(ws_pub / "kesoku.db")
    assert len(rows_pub) == 1
    assert rows_pub[0]["name"] == "device"

    # 3. `kesoku app status --name band` matches WorkingDirectory basename `band`
    res3 = _invoke("status", "--name", "band")
    assert res3.exit_code == 0, res3.output
    assert "relay=main.example.test" in res3.output
