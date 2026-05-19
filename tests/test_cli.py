"""Unit tests for Kesoku Typer CLI."""

import os
import re
import sqlite3
from typing import Any
from unittest.mock import mock_open, patch

from typer.testing import CliRunner

from kesoku.agent.llm import MockLLM
from kesoku.cli import app

runner = CliRunner()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from the given text.

    Args:
        text: The input string containing potential ANSI escape sequences.

    Returns:
        The input string with all ANSI escape sequences stripped out.
    """
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def test_cli_init(tmp_path: Any) -> None:
    """Test 'kesoku init' subcommand using Typer runner."""
    config_path = tmp_path / "config.toml"
    result = runner.invoke(app, ["init", "-w", str(tmp_path)])
    assert result.exit_code == 0
    assert os.path.exists(config_path)
    assert os.path.exists(tmp_path / "kesoku.db")
    assert os.path.exists(tmp_path / "skills")


def test_cli_help() -> None:
    """Test CLI help option."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "init" in result.stdout
    assert "chat" in result.stdout
    assert "console" not in result.stdout


def test_cli_chat_before_init(tmp_path: Any) -> None:
    """Verify running chat before init fails safely with clear error."""
    config_path = tmp_path / "nonexistent.toml"
    result = runner.invoke(app, ["chat", "-c", str(config_path), "Hello"])
    assert result.exit_code == 1
    assert "Please run 'kesoku init' first" in result.output


def test_cli_init_overwrite_options(tmp_path: Any) -> None:
    """Verify init overwrite options work as expected."""
    config_path = tmp_path / "config.toml"
    db_path = tmp_path / "kesoku.db"
    skills_dir = tmp_path / "skills"

    runner.invoke(app, ["init", "-w", str(tmp_path)])
    assert os.path.exists(config_path)
    assert os.path.exists(db_path)
    assert os.path.exists(skills_dir)

    # 1. Test config overwrite
    with open(config_path, "w") as f:
        f.write("# custom config")

    # Init without overwrite-config should preserve custom config
    runner.invoke(app, ["init", "-w", str(tmp_path)])
    with open(config_path) as f:
        assert "# custom config" in f.read()

    # Init with --overwrite-config should backup and overwrite
    runner.invoke(app, ["init", "-w", str(tmp_path), "--overwrite-config"])
    with open(config_path) as f:
        assert "# custom config" not in f.read()
    config_backups = [f for f in os.listdir(tmp_path) if "config.toml.bak" in f]
    assert len(config_backups) == 1

    # 2. Test db overwrite
    # Create dummy data in a table to verify DB overwrite
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO sessions (id, title, created_at, updated_at) VALUES ('t1', 'title', 1.0, 1.0)")
    conn.commit()
    conn.close()

    # Re-init without --overwrite-db should preserve DB data
    runner.invoke(app, ["init", "-w", str(tmp_path)])
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sessions")
    assert cursor.fetchone()[0] == 1
    conn.close()

    # Re-init with --overwrite-db should backup and clear/re-init DB
    runner.invoke(app, ["init", "-w", str(tmp_path), "--overwrite-db"])
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sessions")
    assert cursor.fetchone()[0] == 0
    conn.close()
    db_backups = [f for f in os.listdir(tmp_path) if "kesoku.db.bak" in f]
    assert len(db_backups) == 1

    # 3. Test skills overwrite
    # Add a dummy skill file in a custom skill folder
    custom_skill_file = skills_dir / "ai-image" / "custom.py"
    os.makedirs(os.path.dirname(custom_skill_file), exist_ok=True)
    with open(custom_skill_file, "w") as f:
        f.write("# custom skill modification")

    # Init without --overwrite-skills should preserve custom skill file
    runner.invoke(app, ["init", "-w", str(tmp_path)])
    assert os.path.exists(custom_skill_file)

    # Init with --overwrite-skills should clean and overwrite
    runner.invoke(app, ["init", "-w", str(tmp_path), "--overwrite-skills"])
    assert not os.path.exists(custom_skill_file)


@patch("kesoku.agent.agent.get_llm", return_value=MockLLM())
def test_cli_chat_workflow(mock_gemini: Any, tmp_path: Any) -> None:
    """Test complete chat session workflow using Typer runner with MockLLM patch."""
    config_path = tmp_path / "config.toml"
    runner.invoke(app, ["init", "-w", str(tmp_path)])

    # 1. Check empty session list
    res_list_empty = runner.invoke(app, ["chat", "-c", str(config_path), "-l"])
    assert res_list_empty.exit_code == 0
    assert "No chat sessions found" in res_list_empty.stdout

    # 2. No args error
    res_no_args = runner.invoke(app, ["chat", "-c", str(config_path)])
    assert res_no_args.exit_code == 1
    assert "Please provide a message" in res_no_args.stdout

    # 3. Start a new chat session with patched backend
    res_chat1 = runner.invoke(app, ["chat", "-c", str(config_path), "Calculate 10 + 20"])
    assert res_chat1.exit_code == 0
    plain_chat1 = strip_ansi(res_chat1.stdout)
    assert "Started new session" in plain_chat1
    assert "You" in plain_chat1
    assert "Kesoku Agent" in plain_chat1

    # Extract session ID from output
    match = re.search(r"Started new session: '([a-f0-9]+)'", plain_chat1)
    assert match is not None
    session_id = match.group(1)

    # 4. Check session list contains the new session
    res_list = runner.invoke(app, ["chat", "-c", str(config_path), "-l"])
    assert res_list.exit_code == 0
    plain_list = strip_ansi(res_list.stdout)
    assert session_id in plain_list
    assert "Calculate 10 + 20" in plain_list

    # 5. Resume specific session
    res_resume = runner.invoke(app, ["chat", "-c", str(config_path), "-r", session_id, "And multiply by 2"])
    assert res_resume.exit_code == 0

    # 6. Resume latest session
    res_latest = runner.invoke(app, ["chat", "-c", str(config_path), "-z", "And add 5"])
    assert res_latest.exit_code == 0
    assert f"Resuming latest session: '{session_id}'" in strip_ansi(res_latest.stdout)

    # 7. Show history (--show-history)
    res_history = runner.invoke(app, ["chat", "-c", str(config_path), "--show-history", session_id])
    assert res_history.exit_code == 0
    plain_history = strip_ansi(res_history.stdout)
    assert f"Chat History for Session '{session_id}'" in plain_history
    assert "Calculate 10 + 20" in plain_history
    assert "And multiply by 2" in plain_history
    assert "And add 5" in plain_history

    # 8. Show history (-s short flag)
    res_history_short = runner.invoke(app, ["chat", "-c", str(config_path), "-s", session_id])
    assert res_history_short.exit_code == 0
    assert f"Chat History for Session '{session_id}'" in strip_ansi(res_history_short.stdout)


def test_cli_service_non_linux() -> None:
    """Verify service command fails on non-Linux systems."""
    with patch("sys.platform", "darwin"):
        result = runner.invoke(app, ["service", "install"])
        assert result.exit_code == 1
        assert "only supported on Linux" in result.stdout


def test_cli_service_dry_run() -> None:
    """Verify service command dry-run generated systemd unit content."""
    with (
        patch("sys.platform", "linux"),
        patch("kesoku.cli.load_config"),
        patch(
            "os.path.abspath",
            side_effect=lambda p: (
                "/mock/workspace/config.toml" if "config.toml" in p else f"/mock/bin/{os.path.basename(p)}"
            ),
        ),
        patch("os.path.exists", return_value=True),
    ):
        # Test basic user dry-run
        result = runner.invoke(app, ["service", "install", "--dry-run"])
        assert result.exit_code == 0
        assert "WorkingDirectory=/mock/workspace" in result.stdout
        assert "ExecStart=" in result.stdout
        assert "/mock/bin/kesoku -c /mock/workspace/config.toml start" in result.stdout
        assert "WantedBy=default.target" in result.stdout

        # Test system dry-run with environment variables and -c flag
        result_system = runner.invoke(
            app,
            [
                "service",
                "install",
                "--dry-run",
                "--system",
                "-c",
                "/mock/workspace/config.toml",
                "-e",
                "GEMINI_API_KEY=secret_key",
                "-e",
                "DISCORD_BOT_TOKEN=discord_token",
            ],
        )
        assert result_system.exit_code == 0
        assert "WorkingDirectory=/mock/workspace" in result_system.stdout
        assert "WantedBy=multi-user.target" in result_system.stdout
        assert 'Environment="GEMINI_API_KEY=secret_key"' in result_system.stdout
        assert 'Environment="DISCORD_BOT_TOKEN=discord_token"' in result_system.stdout


def test_cli_service_install_user() -> None:
    """Verify successful user-level installation of the service."""
    m_open = mock_open()
    original_open = open

    def selective_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if "kesoku.service" in str(file):
            return m_open(file, *args, **kwargs)
        return original_open(file, *args, **kwargs)

    with (
        patch("sys.platform", "linux"),
        patch("kesoku.cli.load_config"),
        patch(
            "os.path.abspath",
            side_effect=lambda p: (
                "/mock/workspace/config.toml" if "config.toml" in p else f"/mock/bin/{os.path.basename(p)}"
            ),
        ),
        patch("os.path.exists", return_value=True),
        patch("os.makedirs") as mock_makedirs,
        patch("builtins.open", side_effect=selective_open),
        patch("subprocess.run") as mock_run,
    ):
        result = runner.invoke(app, ["service", "install", "-c", "config.toml"])
        assert result.exit_code == 0
        assert "service installed successfully" in result.stdout.lower()
        mock_makedirs.assert_called_once()
        m_open.assert_called_once()
        # Verify daemon-reload was run
        mock_run.assert_called_once_with(
            ["systemctl", "--user", "daemon-reload"],
            check=True,
            capture_output=True,
            text=True,
        )


def test_cli_service_permission_error() -> None:
    """Verify service command exits gracefully on write permission issues."""
    original_open = open

    def selective_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if "kesoku.service" in str(file):
            raise PermissionError("Permission Denied")
        return original_open(file, *args, **kwargs)

    with (
        patch("sys.platform", "linux"),
        patch("kesoku.cli.load_config"),
        patch(
            "os.path.abspath",
            side_effect=lambda p: (
                "/mock/workspace/config.toml" if "config.toml" in p else f"/mock/bin/{os.path.basename(p)}"
            ),
        ),
        patch("os.path.exists", return_value=True),
        patch("os.makedirs"),
        patch("builtins.open", side_effect=selective_open),
    ):
        result = runner.invoke(app, ["service", "install", "--system"])
        assert result.exit_code == 1
        assert "Permission denied" in result.stdout


def test_cli_service_uninstall() -> None:
    """Verify successful service uninstallation."""
    with (
        patch("sys.platform", "linux"),
        patch("kesoku.cli.load_config"),
        patch("os.path.exists", return_value=True),
        patch("os.remove") as mock_remove,
        patch("subprocess.run") as mock_run,
    ):
        # Test user uninstall
        result = runner.invoke(app, ["service", "uninstall"])
        assert result.exit_code == 0
        assert "uninstalled successfully" in result.stdout.lower()
        mock_remove.assert_called_once()

        # Verify systemctl stop, disable and daemon-reload were run
        stop_call = mock_run.mock_calls[0]
        disable_call = mock_run.mock_calls[1]
        reload_call = mock_run.mock_calls[2]

        assert "stop" in stop_call[1][0]
        assert "disable" in disable_call[1][0]
        assert "daemon-reload" in reload_call[1][0]


def test_cli_service_start_stop_restart() -> None:
    """Verify start, stop, and restart service wrapper command invocations."""
    with (
        patch("sys.platform", "linux"),
        patch("subprocess.run") as mock_run,
    ):
        # 1. Start User Service
        res_start = runner.invoke(app, ["service", "start"])
        assert res_start.exit_code == 0
        assert "executed service start" in res_start.stdout.lower()
        mock_run.assert_any_call(
            ["systemctl", "--user", "start", "kesoku"],
            check=True,
            capture_output=True,
            text=True,
        )

        # 2. Stop System Service
        res_stop = runner.invoke(app, ["service", "stop", "--system"])
        assert res_stop.exit_code == 0
        assert "executed service stop" in res_stop.stdout.lower()
        mock_run.assert_any_call(
            ["sudo", "systemctl", "stop", "kesoku"],
            check=True,
            capture_output=True,
            text=True,
        )

        # 3. Restart User Service
        res_restart = runner.invoke(app, ["service", "restart"])
        assert res_restart.exit_code == 0
        assert "executed service restart" in res_restart.stdout.lower()
        mock_run.assert_any_call(
            ["systemctl", "--user", "restart", "kesoku"],
            check=True,
            capture_output=True,
            text=True,
        )
