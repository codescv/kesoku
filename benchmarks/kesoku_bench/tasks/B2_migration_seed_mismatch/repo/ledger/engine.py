"""Migration registry and state transition engine."""

from __future__ import annotations

import copy
from types import ModuleType
from typing import Any

from ledger.migrations import (
    m001_init_accounts,
    m002_add_currency,
    m003_normalize_status,
    m004_split_amount_cents,
    m005_split_name,
    m006_nullable_last_name_audit,
)

RowDict = dict[str, Any]
TableState = list[RowDict]

MIGRATIONS: list[ModuleType] = [
    m001_init_accounts,
    m002_add_currency,
    m003_normalize_status,
    m004_split_amount_cents,
    m005_split_name,
    m006_nullable_last_name_audit,
]


class MigrationEngine:
    """Applies forward (`up`) and reverse (`down`) migrations across versions `0..6`."""

    def __init__(self, migrations: list[ModuleType] | None = None) -> None:
        """Initialize the engine with an ordered list of migration modules.

        Args:
            migrations: Optional custom list of migration modules.
        """
        self._migrations = list(migrations) if migrations is not None else list(MIGRATIONS)

    @property
    def max_version(self) -> int:
        """Return the highest schema version supported by the registered migrations."""
        return len(self._migrations)

    def get_migration(self, version: int) -> ModuleType:
        """Return the migration module for 1-based `version`.

        Args:
            version: 1-based migration index (`1..max_version`).

        Returns:
            Corresponding migration module.
        """
        if not 1 <= version <= len(self._migrations):
            raise ValueError(f"Unknown migration version {version}")
        return self._migrations[version - 1]

    def step_up(self, state: TableState, from_version: int) -> TableState:
        """Apply a single `up` migration from `from_version` to `from_version + 1`.

        Args:
            state: Current table rows at `from_version`.
            from_version: Current version (`0..max_version - 1`).

        Returns:
            New table rows at `from_version + 1`.
        """
        mod = self.get_migration(from_version + 1)
        return mod.up(copy.deepcopy(state))

    def step_down(self, state: TableState, from_version: int) -> TableState:
        """Apply a single `down` migration from `from_version` to `from_version - 1`.

        Args:
            state: Current table rows at `from_version`.
            from_version: Current version (`1..max_version`).

        Returns:
            New table rows at `from_version - 1`.
        """
        mod = self.get_migration(from_version)
        return mod.down(copy.deepcopy(state))

    def migrate(self, state: TableState, current_version: int, target_version: int) -> TableState:
        """Migrate `state` from `current_version` to `target_version`.

        Args:
            state: Input table rows.
            current_version: Starting schema version (`0..max_version`).
            target_version: Destination schema version (`0..max_version`).

        Returns:
            Migrated table rows at `target_version`.
        """
        rows = copy.deepcopy(state)
        ver = current_version
        while ver < target_version:
            rows = self.step_up(rows, ver)
            ver += 1
        while ver > target_version:
            rows = self.step_down(rows, ver)
            ver -= 1
        return rows
