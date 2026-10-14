"""Unit tests for file management tools."""

import os

import pytest

from kesoku.agent.tools.file import update_file
from kesoku.agent.tools.registry import ToolContext
from kesoku.config import get_config


@pytest.fixture
def test_awd(tmp_path):
    """Fixture to set up a temporary agent working directory."""
    cfg = get_config()
    old_awd = cfg.agent_working_dir
    cfg.agent_working_dir = str(tmp_path)
    yield tmp_path
    cfg.agent_working_dir = old_awd


def test_update_file_create_success(test_awd) -> None:
    """Test creating a new file when old_content is None."""
    file_name = "new_file.txt"
    new_content = "Hello World\n"

    res = update_file(file_name=file_name, old_content=None, new_content=new_content)
    assert "Success: Created file" in res

    resolved_path = os.path.join(str(test_awd), file_name)
    assert os.path.exists(resolved_path)
    with open(resolved_path, encoding="utf-8") as f:
        assert f.read() == new_content


def test_update_file_create_fail_old_content_provided(test_awd) -> None:
    """Test that creating a file fails if old_content is provided but file doesn't exist."""
    file_name = "new_file.txt"
    new_content = "Hello World\n"

    res = update_file(file_name=file_name, old_content="something", new_content=new_content)
    assert "Error" in res
    assert "does not exist, but old_content was provided" in res

    resolved_path = os.path.join(str(test_awd), file_name)
    assert not os.path.exists(resolved_path)


def test_update_file_modify_success_single_match(test_awd) -> None:
    """Test modifying a file with a single match of old_content."""
    file_name = "test.txt"
    initial_content = "line 1\nline 2\nline 3\n"
    resolved_path = os.path.join(str(test_awd), file_name)
    with open(resolved_path, "w", encoding="utf-8") as f:
        f.write(initial_content)

    res = update_file(
        file_name=file_name,
        old_content="line 2\n",
        new_content="line 2 modified\n",
    )
    assert "Success: Updated file" in res
    assert "1 replacement(s) made" in res

    with open(resolved_path, encoding="utf-8") as f:
        assert f.read() == "line 1\nline 2 modified\nline 3\n"


def test_update_file_modify_success_double_match(test_awd) -> None:
    """Test modifying a file with exactly two matches of old_content."""
    file_name = "test.txt"
    initial_content = "target\nline 1\ntarget\nline 2\n"
    resolved_path = os.path.join(str(test_awd), file_name)
    with open(resolved_path, "w", encoding="utf-8") as f:
        f.write(initial_content)

    res = update_file(
        file_name=file_name,
        old_content="target\n",
        new_content="replacement\n",
    )
    assert "Success: Updated file" in res
    assert "2 replacement(s) made" in res

    with open(resolved_path, encoding="utf-8") as f:
        assert f.read() == "replacement\nline 1\nreplacement\nline 2\n"


def test_update_file_modify_fail_no_match(test_awd) -> None:
    """Test that modifying fails if old_content is not found."""
    file_name = "test.txt"
    initial_content = "line 1\nline 2\n"
    resolved_path = os.path.join(str(test_awd), file_name)
    with open(resolved_path, "w", encoding="utf-8") as f:
        f.write(initial_content)

    res = update_file(
        file_name=file_name,
        old_content="nonexistent",
        new_content="replacement",
    )
    assert "Error: old_content not found" in res

    # Content should not change
    with open(resolved_path, encoding="utf-8") as f:
        assert f.read() == initial_content


def test_update_file_modify_fail_too_many_matches(test_awd) -> None:
    """Test that modifying fails if old_content matches more than 2 places."""
    file_name = "test.txt"
    initial_content = "target\ntarget\ntarget\n"
    resolved_path = os.path.join(str(test_awd), file_name)
    with open(resolved_path, "w", encoding="utf-8") as f:
        f.write(initial_content)

    res = update_file(
        file_name=file_name,
        old_content="target\n",
        new_content="replacement\n",
    )
    assert "Error: old_content matches 3 places" in res
    assert "max allowed is 2" in res

    # Content should not change
    with open(resolved_path, encoding="utf-8") as f:
        assert f.read() == initial_content


def test_update_file_modify_fail_old_content_none(test_awd) -> None:
    """Test that modifying fails if old_content is None but file exists."""
    file_name = "test.txt"
    initial_content = "line 1\n"
    resolved_path = os.path.join(str(test_awd), file_name)
    with open(resolved_path, "w", encoding="utf-8") as f:
        f.write(initial_content)

    res = update_file(
        file_name=file_name,
        old_content=None,
        new_content="new content",
    )
    assert "Error" in res
    assert "exists, but old_content is None" in res

    # Content should not change
    with open(resolved_path, encoding="utf-8") as f:
        assert f.read() == initial_content


@pytest.fixture
def staging_cfg(tmp_path):
    """AWD + sessions_dir under tmp_path, restored afterwards."""
    cfg = get_config()
    old_awd, old_sessions = cfg.agent_working_dir, cfg.workspace.sessions_dir
    cfg.agent_working_dir = str(tmp_path / "awd")
    cfg.workspace.sessions_dir = str(tmp_path / "sessions")
    os.makedirs(cfg.agent_working_dir, exist_ok=True)
    yield tmp_path
    cfg.agent_working_dir, cfg.workspace.sessions_dir = old_awd, old_sessions


@pytest.mark.parametrize("path", ["$STAGING_DIR/notes.md", "${STAGING_DIR}/notes.md"])
def test_update_file_expands_staging_dir_for_session_notes(staging_cfg, path: str) -> None:
    """The literal $STAGING_DIR path from the prompt lands in the session staging dir, not under AWD."""
    ctx = ToolContext(session_id="sess_x", session_workspace="sess_x")
    res = update_file(path, None, "- rule A\n- scene: park", context=ctx)
    assert res.startswith("Success"), res
    notes = staging_cfg / "sessions" / "sess_x" / "notes.md"
    assert notes.read_text(encoding="utf-8") == "- rule A\n- scene: park"
    assert not (staging_cfg / "awd" / "$STAGING_DIR").exists()
    assert not (staging_cfg / "awd" / "${STAGING_DIR}").exists()

    # Incremental edit: only the changed line is replaced, the other rule survives.
    res2 = update_file(path, "- scene: park", "- scene: museum", context=ctx)
    assert res2.startswith("Success"), res2
    assert notes.read_text(encoding="utf-8") == "- rule A\n- scene: museum"


def test_update_file_notes_edit_requires_existing_old_content(staging_cfg) -> None:
    """Editing notes with stale old_content fails loudly instead of silently rewriting the file."""
    ctx = ToolContext(session_id="sess_y", session_workspace="sess_y")
    update_file("$STAGING_DIR/notes.md", None, "- rule A\n- rule B", context=ctx)
    res = update_file("$STAGING_DIR/notes.md", "- rule Z", "- rule C", context=ctx)
    assert "not found" in res
    notes = staging_cfg / "sessions" / "sess_y" / "notes.md"
    assert notes.read_text(encoding="utf-8") == "- rule A\n- rule B"


def test_update_file_notes_create_twice_rejected(staging_cfg) -> None:
    """old_content=None on an existing notes.md is rejected (no accidental full overwrite)."""
    ctx = ToolContext(session_id="sess_z", session_workspace="sess_z")
    update_file("$STAGING_DIR/notes.md", None, "- rule A", context=ctx)
    res = update_file("$STAGING_DIR/notes.md", None, "- replaced everything", context=ctx)
    assert res.startswith("Error")
    notes = staging_cfg / "sessions" / "sess_z" / "notes.md"
    assert notes.read_text(encoding="utf-8") == "- rule A"
