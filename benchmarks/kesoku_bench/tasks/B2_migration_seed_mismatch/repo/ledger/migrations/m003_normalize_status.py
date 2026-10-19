"""Migration m003: convert boolean `active` flag into string `status`."""

from __future__ import annotations

import copy
from typing import Any

NAME = "m003_normalize_status"
VERSION = 3


def up(state: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Upgrade from v2 to v3 by replacing `active` bool with `status` string."""
    out: list[dict[str, Any]] = []
    for item in state:
        row = copy.deepcopy(item)
        active = bool(row.pop("active"))
        row["status"] = "ACTIVE" if active else "SUSPENDED"
        row["schema_version"] = 3
        out.append(row)
    return out


def down(state: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Downgrade from v3 to v2 by converting `status` back to boolean `active`."""
    out: list[dict[str, Any]] = []
    for item in state:
        row = copy.deepcopy(item)
        status = str(row.pop("status"))
        row["active"] = status == "ACTIVE"
        row["schema_version"] = 2
        out.append(row)
    return out
