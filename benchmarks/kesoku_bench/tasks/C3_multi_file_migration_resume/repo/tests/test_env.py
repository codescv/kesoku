"""Unit tests for cfgkit.env."""

from __future__ import annotations

from cfgkit.env import from_env, to_env
from cfgkit.legacy import OldConfig


def test_env_roundtrip() -> None:
    """from_env and to_env round-trip environment variables."""
    env_map = {
        "CFGKIT_SERVICE": "worker",
        "CFGKIT_VERSION": "2.1",
        "CFGKIT_HOST": "0.0.0.0",
        "CFGKIT_PORT": "9091",
        "CFGKIT_DEBUG": "true",
        "CFGKIT_TIMEOUT_MS": "3000",
        "CFGKIT_TAGS": "batch,gpu",
    }
    cfg = from_env(env_map)
    assert isinstance(cfg, OldConfig)
    assert cfg.service == "worker"
    assert cfg.port == 9091
    assert cfg.debug is True
    assert cfg.tags == ["batch", "gpu"]

    exported = to_env(cfg)
    assert exported["CFGKIT_SERVICE"] == "worker"
    assert exported["CFGKIT_PORT"] == "9091"
    assert exported["CFGKIT_TAGS"] == "batch,gpu"
