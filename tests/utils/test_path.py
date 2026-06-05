"""Unit tests for the PathResolver utility."""

import os
from unittest.mock import patch

from kesoku.config import KesokuConfig, WorkspaceConfig
from kesoku.utils.path import PathResolver


def test_path_resolver_resolve_absolute(tmp_path) -> None:
    """Verify that absolute paths are resolved as-is."""
    abs_path = str(tmp_path / "some_file.txt")
    resolved = PathResolver.resolve(abs_path)
    assert resolved == os.path.abspath(abs_path)


def test_path_resolver_resolve_relative(tmp_path) -> None:
    """Verify that relative paths are resolved relative to the AWD."""
    cfg = KesokuConfig(workspace=WorkspaceConfig())
    cfg.agent_working_dir = str(tmp_path / "awd")

    with patch("kesoku.utils.path.get_config", return_value=cfg):
        resolved = PathResolver.resolve("nested/file.txt")
        expected = os.path.abspath(os.path.join(cfg.agent_working_dir, "nested/file.txt"))
        assert resolved == expected


def test_path_resolver_resolve_empty(tmp_path) -> None:
    """Verify that empty paths resolve to the AWD root."""
    cfg = KesokuConfig(workspace=WorkspaceConfig())
    cfg.agent_working_dir = str(tmp_path / "awd")

    with patch("kesoku.utils.path.get_config", return_value=cfg):
        assert PathResolver.resolve(None) == os.path.abspath(cfg.agent_working_dir)
        assert PathResolver.resolve("") == os.path.abspath(cfg.agent_working_dir)


def test_path_resolver_session_staging_dir(tmp_path) -> None:
    """Verify that get_session_staging_dir creates and returns the directory under sessions_dir."""
    cfg = KesokuConfig(workspace=WorkspaceConfig(sessions_dir="my_sessions"))
    cfg.agent_working_dir = str(tmp_path / "awd")

    with patch("kesoku.utils.path.get_config", return_value=cfg):
        staging = PathResolver.get_session_staging_dir("session_123")
        expected_base = os.path.abspath(os.path.join(cfg.agent_working_dir, "my_sessions"))
        expected_dir = os.path.join(expected_base, "session_123")
        assert staging == os.path.abspath(expected_dir)
        assert os.path.exists(staging)
