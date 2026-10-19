"""Configuration export and serialization consumer module."""

from __future__ import annotations

import json
from typing import Any

from cfgkit.legacy import OldConfig


def dump_dict(cfg: OldConfig) -> dict[str, Any]:
    """Return an ordered dictionary representation of ``cfg``."""
    return cfg.to_dict()


def dump(cfg: OldConfig, *, indent: int | None = 2) -> str:
    """Serialize ``cfg`` to a formatted JSON string."""
    return json.dumps(dump_dict(cfg), indent=indent)


def dump_json(cfg: OldConfig, *, indent: int | None = 2) -> str:
    """Alias for ``dump``."""
    return dump(cfg, indent=indent)


def dump_key_value(cfg: OldConfig) -> str:
    """Format ``cfg`` as newline-delimited ``key=json_value`` pairs."""
    ordered = dump_dict(cfg)
    return "\n".join(f"{k}={json.dumps(v)}" for k, v in ordered.items())
