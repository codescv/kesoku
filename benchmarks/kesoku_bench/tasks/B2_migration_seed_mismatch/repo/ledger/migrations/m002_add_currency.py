"""Migration m002: split `balance_str` into `currency` and `amount`."""

from __future__ import annotations

import copy
from typing import Any

NAME = "m002_add_currency"
VERSION = 2


def up(state: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Upgrade from v1 to v2 by splitting `balance_str` into `currency` and `amount`."""
    out: list[dict[str, Any]] = []
    for item in state:
        row = copy.deepcopy(item)
        balance_str = str(row.pop("balance_str"))
        currency, amount = balance_str.split(" ", 1)
        row["currency"] = currency
        row["amount"] = amount
        row["schema_version"] = 2
        out.append(row)
    return out


def down(state: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Downgrade from v2 to v1 by joining `currency` and `amount` into `balance_str`."""
    out: list[dict[str, Any]] = []
    for item in state:
        row = copy.deepcopy(item)
        currency = row.pop("currency")
        amount = row.pop("amount")
        row["balance_str"] = f"{currency} {amount}"
        row["schema_version"] = 1
        out.append(row)
    return out
