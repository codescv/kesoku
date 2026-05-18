"""Shared fixtures and configuration for Kesoku unit tests."""

from collections.abc import Generator
from unittest.mock import patch
import pytest
import kesoku.config


@pytest.fixture(autouse=True)
def setup_test_config() -> Generator[None, None, None]:
    """Automatically load a default mock configuration before every test.

    Yields:
        None
    """
    original_config = kesoku.config._global_config
    cfg = kesoku.config.KesokuConfig()
    cfg.resolve_paths("test_workspace/config.toml")
    kesoku.config._global_config = cfg
    yield
    kesoku.config._global_config = original_config
