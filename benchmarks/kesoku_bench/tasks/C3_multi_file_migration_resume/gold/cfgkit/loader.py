"""Configuration loader and validator consumer module (migrated to ConfigV2)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from cfgkit.v2 import ConfigV2

KNOWN_FIELDS: frozenset[str] = frozenset(
    {"service", "version", "host", "port", "debug", "timeout_ms", "tags", "limits"}
)


def validate_config(cfg: ConfigV2) -> list[str]:
    """Return a list of validation error strings for ``cfg`` (empty if valid)."""
    errors: list[str] = []
    if not cfg.service or not cfg.service.strip():
        errors.append("service must be non-empty")
    if not (1 <= cfg.port <= 65535):
        errors.append(f"port out of range: {cfg.port}")
    if cfg.timeout_ms <= 0:
        errors.append(f"timeout_ms must be positive: {cfg.timeout_ms}")
    return errors


def load_config(raw: Mapping[str, Any], *, strict: bool = False) -> ConfigV2:
    """Load configuration from ``raw`` mapping and validate it."""
    if strict:
        unknown = sorted(set(raw.keys()) - KNOWN_FIELDS)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(unknown)}")
    cfg = ConfigV2.from_dict(raw)
    errors = validate_config(cfg)
    if errors:
        raise ValueError("; ".join(errors))
    return cfg


def load_json(text: str, *, strict: bool = False) -> ConfigV2:
    """Parse JSON string ``text`` and return a validated ``ConfigV2`` object."""
    parsed = json.loads(text)
    if not isinstance(parsed, Mapping):
        raise ValueError("Configuration JSON root must be an object")
    return load_config(parsed, strict=strict)
