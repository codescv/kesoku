"""Tests for `kesoku history reclassify` and `DatabaseManager.reclassify_message_prefix` (temp databases only)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from kesoku.agent.history import prepare_history_for_llm
from kesoku.cli import app
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import DatabaseManager, Message, Session

runner = CliRunner()

PLAN = "Draft plan: list the options, then pick one."
REPLY = "Option two is the better fit."


@pytest.fixture(autouse=True)
def _wide_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "240")


def _setup(tmp_path: Path) -> tuple[Path, DatabaseManager, Message, Message]:
    """Create a temp config + DB holding one completed turn and a following user message."""
    db_path = tmp_path / "kesoku.db"
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        "\n".join(
            [
                "[workspace]",
                f'db_path = "{db_path}"',
                f'skills_dir = "{tmp_path / "skills"}"',
                f'roles_dir = "{tmp_path / "roles"}"',
                f'sessions_dir = "{tmp_path / "sessions"}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    db = DatabaseManager(str(db_path))
    db.init_tables()
    db.verify_db()
    db.create_session(Session(id="s1", title="Session"))
    user = Message(
        session_id="s1", chatbot_id="cli", channel_id="c", sender="u", role=MessageRole.USER, content="Which option?",
        status=MessageStatus.PROCESSED, timestamp=100.0,
    )
    text = Message(
        session_id="s1", chatbot_id="cli", channel_id="c", sender="Kesoku", role=MessageRole.ASSISTANT,
        type=MessageType.TEXT, content=PLAN + REPLY, status=MessageStatus.DELIVERED, parent_id=user.id,
        timestamp=101.0, metadata={"turn_metrics": {"status": "finished"}},
    )
    nxt = Message(
        session_id="s1", chatbot_id="cli", channel_id="c", sender="u", role=MessageRole.USER, content="Thanks.",
        status=MessageStatus.PENDING_AGENT, timestamp=200.0,
    )
    for m in (user, text, nxt):
        db.save_message(m)
    return cfg_path, db, user, text


def test_reclassify_message_prefix_moves_prefix_into_thought_row(tmp_path: Path) -> None:
    _cfg, db, user, text = _setup(tmp_path)
    updated, thought = db.reclassify_message_prefix(text.id, len(PLAN))

    assert updated.content == REPLY
    stored = db.get_message(text.id)
    assert stored is not None and stored.content == REPLY
    assert stored.metadata["reclassified_prefix_chars"] == len(PLAN)
    assert stored.metadata["turn_metrics"] == {"status": "finished"}

    row = db.get_message(thought.id)
    assert row is not None
    assert row.type == MessageType.THOUGHT and row.role == MessageRole.ASSISTANT
    assert row.content == PLAN.rstrip()
    assert row.metadata == {"final_response": True, "reclassified": "manual"}
    assert row.parent_id == user.id
    assert row.timestamp < stored.timestamp

    # The reclassified thought is kept out of later turns; the reply stays.
    llm_history = prepare_history_for_llm(db.get_session_messages("s1"))
    contents = [m.content for m in llm_history]
    assert PLAN.rstrip() not in contents
    assert REPLY in contents


@pytest.mark.parametrize("n", [0, -1, len(PLAN + REPLY)])
def test_reclassify_message_prefix_rejects_out_of_range(tmp_path: Path, n: int) -> None:
    _cfg, db, _user, text = _setup(tmp_path)
    with pytest.raises(ValueError):
        db.reclassify_message_prefix(text.id, n)
    assert db.get_message(text.id).content == PLAN + REPLY


def test_reclassify_message_prefix_rejects_non_assistant_text(tmp_path: Path) -> None:
    _cfg, db, user, _text = _setup(tmp_path)
    with pytest.raises(ValueError):
        db.reclassify_message_prefix(user.id, 3)
    with pytest.raises(ValueError):
        db.reclassify_message_prefix("missing-id", 3)


def test_cli_reclassify_dry_run_does_not_write(tmp_path: Path) -> None:
    cfg, db, _user, text = _setup(tmp_path)
    res = runner.invoke(
        app, ["history", "reclassify", text.id, "--prefix-chars", str(len(PLAN)), "-c", str(cfg), "--dry-run"]
    )
    assert res.exit_code == 0, res.output
    assert "Dry run" in res.output
    assert db.get_message(text.id).content == PLAN + REPLY


def test_cli_reclassify_requires_confirmation(tmp_path: Path) -> None:
    cfg, db, _user, text = _setup(tmp_path)
    res = runner.invoke(
        app, ["history", "reclassify", text.id, "--prefix-chars", str(len(PLAN)), "-c", str(cfg)], input="n\n"
    )
    assert res.exit_code == 1
    assert db.get_message(text.id).content == PLAN + REPLY


def test_cli_reclassify_applies_with_yes(tmp_path: Path) -> None:
    cfg, db, _user, text = _setup(tmp_path)
    res = runner.invoke(
        app, ["history", "reclassify", text.id, "--prefix-chars", str(len(PLAN)), "-c", str(cfg), "--yes"]
    )
    assert res.exit_code == 0, res.output
    assert db.get_message(text.id).content == REPLY
    thoughts = [m for m in db.get_session_messages("s1") if m.type == MessageType.THOUGHT]
    assert len(thoughts) == 1 and thoughts[0].metadata["reclassified"] == "manual"


def test_cli_reclassify_rejects_bad_input(tmp_path: Path) -> None:
    cfg, db, user, text = _setup(tmp_path)
    bad_n = runner.invoke(app, ["history", "reclassify", text.id, "--prefix-chars", "0", "-c", str(cfg), "--yes"])
    assert bad_n.exit_code == 1
    not_text = runner.invoke(app, ["history", "reclassify", user.id, "--prefix-chars", "2", "-c", str(cfg), "--yes"])
    assert not_text.exit_code == 1
    missing = runner.invoke(app, ["history", "reclassify", "nope", "--prefix-chars", "2", "-c", str(cfg), "--yes"])
    assert missing.exit_code == 1
    assert db.get_message(text.id).content == PLAN + REPLY
