"""Migration m004: convert decimal `amount` string into integer `amount_cents`."""

from __future__ import annotations

import copy
from typing import Any

NAME = "m004_split_amount_cents"
VERSION = 4


def up(state: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Upgrade from v3 to v4 by converting decimal `amount` into `amount_cents`."""
    out: list[dict[str, Any]] = []
    for item in state:
        row = copy.deepcopy(item)
        amount_str = str(row.pop("amount"))
        negative = amount_str.startswith("-")
        clean = amount_str[1:] if negative else amount_str
        whole_str, frac_str = clean.split(".", 1) if "." in clean else (clean, "00")
        cents = int(whole_str) * 100 + int(frac_str.ljust(2, "0")[:2])
        row["amount_cents"] = -cents if negative else cents
        row["schema_version"] = 4
        out.append(row)
    return out


def down(state: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Downgrade from v4 to v3 by formatting `amount_cents` back into decimal `amount`."""
    out: list[dict[str, Any]] = []
    for item in state:
        row = copy.deepcopy(item)
        cents = int(row.pop("amount_cents"))
        sign = "-" if cents < 0 else ""
        abs_cents = abs(cents)
        row["amount"] = f"{sign}{abs_cents // 100}.{abs_cents % 100:02d}"
        row["schema_version"] = 3
        out.append(row)
    return out
