"""Migration m001: normalize account_id prefix and tag schema_version."""

from __future__ import annotations

import copy
from typing import Any

NAME = "m001_init_accounts"
VERSION = 1


def up(state: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Upgrade from v0 to v1 by adding `acct-` prefix if absent and `schema_version`."""
    out: list[dict[str, Any]] = []
    for item in state:
        row = copy.deepcopy(item)
        raw_id = str(row["account_id"])
        row["_m001_raw_id"] = raw_id
        if not raw_id.startswith("acct-"):
            row["account_id"] = f"acct-{raw_id}"
        row["schema_version"] = 1
        out.append(row)
    return out


def down(state: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Downgrade from v1 to v0 by restoring original `account_id` and removing `schema_version`."""
    out: list[dict[str, Any]] = []
    for item in state:
        row = copy.deepcopy(item)
        raw_id = row.pop("_m001_raw_id", None)
        if raw_id is not None:
            row["account_id"] = raw_id
        elif str(row.get("account_id", "")).startswith("acct-"):
            row["account_id"] = str(row["account_id"])[5:]
        row.pop("schema_version", None)
        out.append(row)
    return out
