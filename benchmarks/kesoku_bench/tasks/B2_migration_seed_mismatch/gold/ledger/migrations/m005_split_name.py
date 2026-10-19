"""Migration m005: split `full_name` into `first_name` and `last_name`."""

from __future__ import annotations

import copy
from typing import Any

NAME = "m005_split_name"
VERSION = 5


def up(state: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Upgrade from v4 to v5 by splitting `full_name` on the first space and recording separator."""
    out: list[dict[str, Any]] = []
    for item in state:
        row = copy.deepcopy(item)
        full_name = str(row.pop("full_name"))
        if " " in full_name:
            first, rest = full_name.split(" ", 1)
            stripped_rest = rest.lstrip(" ")
            sep_len = 1 + (len(rest) - len(stripped_rest))
            row["first_name"] = first
            row["last_name"] = stripped_rest
            row["_m005_sep"] = " " * sep_len
        else:
            row["first_name"] = full_name
            row["last_name"] = ""
            row["_m005_sep"] = ""
        row["schema_version"] = 5
        out.append(row)
    return out


def down(state: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Downgrade from v5 to v4 by reconstructing `full_name` losslessly."""
    out: list[dict[str, Any]] = []
    for item in state:
        row = copy.deepcopy(item)
        first = str(row.pop("first_name"))
        last = str(row.pop("last_name") or "")
        sep = row.pop("_m005_sep", None)
        if sep is not None:
            row["full_name"] = f"{first}{sep}{last}"
        elif last:
            row["full_name"] = f"{first} {last}"
        else:
            row["full_name"] = first
        row["schema_version"] = 4
        out.append(row)
    return out
