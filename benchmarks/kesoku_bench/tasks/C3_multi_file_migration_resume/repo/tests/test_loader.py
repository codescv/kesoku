"""Unit tests for cfgkit.loader (accepts both OldConfig and ConfigV2)."""

from __future__ import annotations

import pytest
from cfgkit.legacy import OldConfig
from cfgkit.loader import load_config, load_json, validate_config
from cfgkit.v2 import ConfigV2


def test_load_config_and_json() -> None:
    """load_config and load_json return a valid config object with expected fields."""
    cfg = load_config({"service": "payments", "port": 8443, "debug": True})
    assert isinstance(cfg, (OldConfig, ConfigV2))
    assert cfg.service == "payments"
    assert cfg.port == 8443
    assert cfg.debug is True

    from_json = load_json('{"service": "auth", "timeout_ms": 1500}')
    assert isinstance(from_json, (OldConfig, ConfigV2))
    assert from_json.service == "auth"
    assert from_json.timeout_ms == 1500


@pytest.mark.parametrize("model_cls", [OldConfig, ConfigV2])
def test_validate_config_accepts_both_models(model_cls: type[OldConfig | ConfigV2]) -> None:
    """validate_config works on both OldConfig and ConfigV2."""
    valid = model_cls(service="api", port=8080, timeout_ms=1000)
    assert validate_config(valid) == []

    invalid = model_cls(service="", port=70000, timeout_ms=0)
    errs = validate_config(invalid)
    assert len(errs) == 3


def test_load_config_strict_rejects_unknown_keys() -> None:
    """Strict mode raises ValueError on unknown keys."""
    with pytest.raises(ValueError, match="Unknown configuration keys"):
        load_config({"service": "api", "bogus_key": 123}, strict=True)
