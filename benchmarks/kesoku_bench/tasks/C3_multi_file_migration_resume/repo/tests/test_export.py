"""Unit tests for cfgkit.export."""

from __future__ import annotations

import json

import pytest
from cfgkit.export import dump, dump_dict, dump_json, dump_key_value
from cfgkit.legacy import OldConfig
from cfgkit.v2 import ConfigV2


@pytest.mark.parametrize("model_cls", [OldConfig, ConfigV2])
def test_export_formats(model_cls: type[OldConfig | ConfigV2]) -> None:
    """dump, dump_dict, dump_json, and dump_key_value serialize all config fields."""
    cfg = model_cls(
        service="gateway",
        version="1.4",
        host="10.0.0.1",
        port=443,
        debug=False,
        timeout_ms=2500,
        tags=["edge"],
        limits={"conns": 1024},
    )
    as_dict = dump_dict(cfg)
    assert as_dict["service"] == "gateway"
    assert as_dict["port"] == 443

    raw_json = dump(cfg)
    assert raw_json == dump_json(cfg)
    parsed = json.loads(raw_json)
    assert parsed["host"] == "10.0.0.1"
    assert parsed["limits"] == {"conns": 1024}

    kv = dump_key_value(cfg)
    assert 'service="gateway"' in kv
    assert "port=443" in kv
