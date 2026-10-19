"""Migration m005: split `full_name` into `first_name` and `last_name`."""

from __future__ import annotations

import copy
from typing import Any

NAME = "m005_split_name"
VERSION = 5


def up(state: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Upgrade from v4 to v5 by splitting `full_name` on the first space."""
    out: list[dict[str, Any]] = []
    for item in state:
        row = copy.deepcopy(item)
        full_name = str(row.pop("full_name"))
        if " " in full_name:
            first, rest = full_name.split(" ", 1)
            row["first_name"] = first
            row["last_name"] = rest.lstrip(" ")
        else:
            row["first_name"] = full_name
            row["last_name"] = ""
        row["schema_version"] = 5
        out.append(row)
    return out


def down(state: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Downgrade from v5 to v4 by joining `first_name` and `last_name` into `full_name`."""
    out: list[dict[str, Any]] = []
    for item in state:
        row = copy.deepcopy(item)
        first = str(row.pop("first_name"))
        last = str(row.pop("last_name") or "")
        row["full_name"] = f"{first} {last}"
        row["schema_version"] = 4
        out.append(row)
    return out
