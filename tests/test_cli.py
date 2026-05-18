"""Unit tests for Kesoku Typer CLI."""

import os
import re
import sqlite3
from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner

from kesoku.agent.llm import MockLLM
from kesoku.cli import app

runner = CliRunner()


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
    result = runner.invoke(app, ["-c", str(config_path), "chat", "Hello"])
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
    res_list_empty = runner.invoke(app, ["-c", str(config_path), "chat", "-l"])
    assert res_list_empty.exit_code == 0
    assert "No chat sessions found" in res_list_empty.stdout

    # 2. No args error
    res_no_args = runner.invoke(app, ["-c", str(config_path), "chat"])
    assert res_no_args.exit_code == 1
    assert "Please provide a message" in res_no_args.stdout

    # 3. Start a new chat session with patched backend
    res_chat1 = runner.invoke(app, ["-c", str(config_path), "chat", "Calculate 10 + 20"])
    assert res_chat1.exit_code == 0
    assert "Started new session" in res_chat1.stdout
    assert "You" in res_chat1.stdout
    assert "Kesoku Agent" in res_chat1.stdout

    # Extract session ID from output
    match = re.search(r"Started new session: '([a-f0-9]+)'", res_chat1.stdout)
    assert match is not None
    session_id = match.group(1)

    # 4. Check session list contains the new session
    res_list = runner.invoke(app, ["-c", str(config_path), "chat", "-l"])
    assert res_list.exit_code == 0
    assert session_id in res_list.stdout
    assert "Calculate 10 + 20" in res_list.stdout

    # 5. Resume specific session
    res_resume = runner.invoke(app, ["-c", str(config_path), "chat", "-r", session_id, "And multiply by 2"])
    assert res_resume.exit_code == 0

    # 6. Resume latest session
    res_latest = runner.invoke(app, ["-c", str(config_path), "chat", "-z", "And add 5"])
    assert res_latest.exit_code == 0
    assert f"Resuming latest session: '{session_id}'" in res_latest.stdout

    # 7. Show history (--show-history)
    res_history = runner.invoke(app, ["-c", str(config_path), "chat", "--show-history", session_id])
    assert res_history.exit_code == 0
    assert f"Chat History for Session '{session_id}'" in res_history.stdout
    assert "Calculate 10 + 20" in res_history.stdout
    assert "And multiply by 2" in res_history.stdout
    assert "And add 5" in res_history.stdout

    # 8. Show history (-s short flag)
    res_history_short = runner.invoke(app, ["-c", str(config_path), "chat", "-s", session_id])
    assert res_history_short.exit_code == 0
    assert f"Chat History for Session '{session_id}'" in res_history_short.stdout
