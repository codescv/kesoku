"""Unit tests for file management tools."""

import os

import pytest

from kesoku.agent.tools.file import update_file
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
