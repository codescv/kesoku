"""Shared fixtures and configuration for Kesoku unit tests."""

from collections.abc import Generator
from typing import Any

import pytest

import kesoku.config


@pytest.fixture(autouse=True)
def setup_test_config(tmp_path: Any) -> Generator[None, None, None]:
    """Automatically load a default mock configuration with safe temporary paths before every test.

    Args:
        tmp_path: Pytest's temporary path fixture.

    Yields:
        None
    """
    original_config = kesoku.config._global_config
    cfg = kesoku.config.KesokuConfig()
    cfg.workspace.sessions_dir = str(tmp_path / "sessions")
    cfg.workspace.db_path = str(tmp_path / "kesoku.db")
    cfg.workspace.skills_dir = str(tmp_path / "skills")
    kesoku.config._global_config = cfg
    yield
    kesoku.config._global_config = original_config
