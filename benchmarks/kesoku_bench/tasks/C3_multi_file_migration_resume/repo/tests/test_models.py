"""Unit tests for OldConfig and ConfigV2 models."""

from __future__ import annotations

from cfgkit.legacy import LEGACY_KEY_ORDER, OldConfig
from cfgkit.v2 import ALPHABETICAL_KEYS, ConfigV2


def test_old_config_key_order() -> None:
    """OldConfig.to_dict preserves LEGACY_KEY_ORDER."""
    cfg = OldConfig(service="billing", port=9000, tags=["prod"])
    dumped = cfg.to_dict()
    assert tuple(dumped.keys()) == LEGACY_KEY_ORDER
    assert dumped["service"] == "billing"
    assert dumped["port"] == 9000


def test_config_v2_alphabetical_key_order_and_updates() -> None:
    """ConfigV2.to_dict returns keys alphabetically and with_updates returns a new copy."""
    cfg = ConfigV2(service="billing", port=9000, tags=["prod"])
    dumped = cfg.to_dict()
    assert tuple(dumped.keys()) == ALPHABETICAL_KEYS
    updated = cfg.with_updates(port=9090, debug=True)
    assert cfg.port == 9000
    assert updated.port == 9090
    assert updated.debug is True
