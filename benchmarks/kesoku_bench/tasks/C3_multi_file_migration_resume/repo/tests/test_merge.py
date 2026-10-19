"""Unit tests for cfgkit.merge (accepts both OldConfig and ConfigV2)."""

from __future__ import annotations

import pytest
from cfgkit.legacy import OldConfig
from cfgkit.merge import merge_chain, merge_configs
from cfgkit.v2 import ConfigV2


@pytest.mark.parametrize("model_cls", [OldConfig, ConfigV2])
def test_merge_configs_tags_and_limits(model_cls: type[OldConfig | ConfigV2]) -> None:
    """merge_configs unions tags in order and merges limits dictionary."""
    base = model_cls(
        service="core",
        port=8000,
        tags=["us-east", "v1"],
        limits={"rps": 100, "burst": 200},
    )
    merged = merge_configs(
        base,
        {"port": 8081, "tags": ["v1", "canary"], "limits": {"burst": 500, "conns": 50}},
    )
    assert isinstance(merged, (OldConfig, ConfigV2))
    assert merged.port == 8081
    assert merged.tags == ["us-east", "v1", "canary"]
    assert merged.limits == {"rps": 100, "burst": 500, "conns": 50}


def test_merge_chain_sequence() -> None:
    """merge_chain folds multiple configs or mappings."""
    result = merge_chain(
        [
            {"service": "search", "port": 7000, "tags": ["base"]},
            {"port": 7001, "tags": ["stage"]},
            {"debug": True},
        ]
    )
    assert isinstance(result, (OldConfig, ConfigV2))
    assert result.service == "search"
    assert result.port == 7001
    assert result.debug is True
    assert result.tags == ["base", "stage"]
