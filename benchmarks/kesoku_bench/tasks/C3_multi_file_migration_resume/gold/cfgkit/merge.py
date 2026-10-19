"""Configuration merging consumer module (migrated to ConfigV2)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from cfgkit.v2 import ConfigV2


def merge_configs(
    base: ConfigV2,
    override: Mapping[str, Any] | ConfigV2,
) -> ConfigV2:
    """Merge ``override`` on top of ``base`` with unioned tags and merged limits."""
    merged = base.to_dict()
    override_dict = (
        override.to_dict()
        if hasattr(override, "to_dict") and callable(override.to_dict)
        else dict(override)
    )

    for key, val in override_dict.items():
        if key == "tags":
            seen: list[str] = list(merged.get("tags", []))
            for tag in val or []:
                tag_str = str(tag)
                if tag_str not in seen:
                    seen.append(tag_str)
            merged["tags"] = seen
        elif key == "limits":
            combined_limits = dict(merged.get("limits", {}))
            for lk, lv in (val or {}).items():
                combined_limits[str(lk)] = int(lv)
            merged["limits"] = combined_limits
        elif val is not None and key in merged:
            merged[key] = val

    return ConfigV2.from_dict(merged)


def merge_chain(configs: Iterable[ConfigV2 | Mapping[str, Any]]) -> ConfigV2:
    """Fold a sequence of configurations or mappings left-to-right into a ``ConfigV2``."""
    items = list(configs)
    if not items:
        return ConfigV2()
    first = items[0]
    current = (
        first.copy()
        if isinstance(first, ConfigV2)
        else ConfigV2.from_dict(
            first.to_dict()
            if hasattr(first, "to_dict") and callable(first.to_dict)
            else dict(first)
        )
    )
    for nxt in items[1:]:
        current = merge_configs(current, nxt)
    return current
