"""Migration m006: normalize empty surnames to NULL (`None`) and add audit flag."""

from __future__ import annotations

import copy
from typing import Any

NAME = "m006_nullable_last_name_audit"
VERSION = 6


def up(state: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Upgrade from v5 to v6 by setting empty `last_name` to None and adding `last_name_missing`."""
    out: list[dict[str, Any]] = []
    for item in state:
        row = copy.deepcopy(item)
        prev_last = row.get("last_name")
        row["_m006_prev_last"] = prev_last
        if prev_last is None or prev_last == "":
            row["last_name"] = None
            row["last_name_missing"] = True
        else:
            row["last_name_missing"] = False
        row["schema_version"] = 6
        out.append(row)
    return out


def down(state: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Downgrade from v6 to v5 by restoring `last_name` and removing `last_name_missing`."""
    out: list[dict[str, Any]] = []
    for item in state:
        row = copy.deepcopy(item)
        if "_m006_prev_last" in row:
            row["last_name"] = row.pop("_m006_prev_last")
        elif row.get("last_name") is None:
            row["last_name"] = ""
        row.pop("last_name_missing", None)
        row["schema_version"] = 5
        out.append(row)
    return out
