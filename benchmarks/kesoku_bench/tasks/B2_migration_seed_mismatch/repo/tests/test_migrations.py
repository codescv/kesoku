"""Visible unit tests for reversible migrations m001 through m006."""

from __future__ import annotations

import copy
from typing import Any

from ledger.engine import MigrationEngine
from ledger.migrations import (
    m001_init_accounts,
    m002_add_currency,
    m003_normalize_status,
    m004_split_amount_cents,
    m005_split_name,
    m006_nullable_last_name_audit,
)


def _sample_v0_rows() -> list[dict[str, Any]]:
    """Return standard v0 rows with single-space two-token names."""
    return [
        {"account_id": "001", "full_name": "Alice Smith", "balance_str": "USD 125.50", "active": True},
        {"account_id": "002", "full_name": "Bob Jones", "balance_str": "EUR 40.00", "active": False},
    ]


def test_migrations_m001_to_m004_roundtrip() -> None:
    """Verify lossless up/down round-trips for migrations m001 through m004."""
    s0 = _sample_v0_rows()
    s1 = m001_init_accounts.up(s0)
    assert s1[0]["account_id"] == "acct-001"
    assert m001_init_accounts.down(s1) == s0

    s2 = m002_add_currency.up(s1)
    assert s2[0]["currency"] == "USD"
    assert s2[0]["amount"] == "125.50"
    assert m002_add_currency.down(s2) == s1

    s3 = m003_normalize_status.up(s2)
    assert s3[0]["status"] == "ACTIVE"
    assert s3[1]["status"] == "SUSPENDED"
    assert m003_normalize_status.down(s3) == s2

    s4 = m004_split_amount_cents.up(s3)
    assert s4[0]["amount_cents"] == 12550
    assert m004_split_amount_cents.down(s4) == s3


def test_migration_m005_standard_names() -> None:
    """Verify m005_split_name on standard two-word names separated by a single space."""
    engine = MigrationEngine()
    s4 = engine.migrate(_sample_v0_rows(), 0, 4)
    s5 = m005_split_name.up(s4)
    assert s5[0]["first_name"] == "Alice"
    assert s5[0]["last_name"] == "Smith"
    assert "full_name" not in s5[0]
    assert m005_split_name.down(s5) == s4


def test_migration_m006_audit_flag() -> None:
    """Verify m006_nullable_last_name_audit sets last_name_missing and round-trips."""
    s5 = [
        {"account_id": "acct-001", "first_name": "Alice", "last_name": "Smith", "schema_version": 5},
        {"account_id": "acct-007", "first_name": "Prince", "last_name": "", "schema_version": 5},
    ]
    s6 = m006_nullable_last_name_audit.up(s5)
    assert s6[0]["last_name"] == "Smith"
    assert s6[0]["last_name_missing"] is False
    assert s6[1]["last_name"] is None
    assert s6[1]["last_name_missing"] is True
    assert m006_nullable_last_name_audit.down(s6) == s5


def test_engine_apply_chain() -> None:
    """Verify full chain migration 0 -> 6 -> 0 on standard two-word names."""
    engine = MigrationEngine()
    s0 = _sample_v0_rows()
    s6 = engine.migrate(s0, 0, 6)
    assert all(r["schema_version"] == 6 for r in s6)
    restored = engine.migrate(copy.deepcopy(s6), 6, 0)
    assert restored == s0
