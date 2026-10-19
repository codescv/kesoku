"""Configuration export and serialization consumer module (migrated to ConfigV2)."""

from __future__ import annotations

import json
from typing import Any

from cfgkit.v2 import ConfigV2

EXPORT_KEY_ORDER: tuple[str, ...] = (
    "service",
    "version",
    "host",
    "port",
    "debug",
    "timeout_ms",
    "tags",
    "limits",
)


def dump_dict(cfg: ConfigV2) -> dict[str, Any]:
    """Return an ordered dictionary representation of ``cfg`` in legacy key order."""
    raw = cfg.to_dict()
    return {key: raw[key] for key in EXPORT_KEY_ORDER if key in raw}


def dump(cfg: ConfigV2, *, indent: int | None = 2) -> str:
    """Serialize ``cfg`` to a formatted JSON string preserving legacy key order."""
    return json.dumps(dump_dict(cfg), indent=indent)


def dump_json(cfg: ConfigV2, *, indent: int | None = 2) -> str:
    """Alias for ``dump``."""
    return dump(cfg, indent=indent)


def dump_key_value(cfg: ConfigV2) -> str:
    """Format ``cfg`` as newline-delimited ``key=json_value`` pairs in legacy key order."""
    ordered = dump_dict(cfg)
    return "\n".join(f"{k}={json.dumps(v)}" for k, v in ordered.items())
