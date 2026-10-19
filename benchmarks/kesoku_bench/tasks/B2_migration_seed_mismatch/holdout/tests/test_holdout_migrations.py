"""Held-out evaluation tests for B2_migration_seed_mismatch."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from ledger.engine import MigrationEngine
from ledger.migrations import m005_split_name


def _make_v4_rows(names: list[str]) -> list[dict[str, Any]]:
    """Construct schema_version=4 rows for the given `full_name` strings."""
    return [
        {
            "account_id": f"acct-{idx:03d}",
            "full_name": name,
            "currency": "USD",
            "amount_cents": 10000 + idx * 100,
            "status": "ACTIVE",
            "schema_version": 4,
        }
        for idx, name in enumerate(names, start=1)
    ]


def test_fuzz_seed_20251019_steps_17() -> None:
    """Verify that `scripts/fuzz_migrations.py --seed 20251019 --steps 17` passes with exit code 0."""
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "scripts/fuzz_migrations.py", "--seed", "20251019", "--steps", "17"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"Fuzzer failed with exit code {proc.returncode}:\n{combined[:1000]}"
    assert "PASSED (1 of 1 runs)" in combined
    assert "FAILED" not in combined


def test_roundtrip_double_spaces() -> None:
    """Verify m005 and full chain preserve multiple internal spaces in full_name."""
    names = ["Mary  Ann Lee", "A   B", "First  Middle   Last"]
    s4 = _make_v4_rows(names)
    s5 = m005_split_name.up(s4)
    for row in s5:
        assert "first_name" in row
        assert "last_name" in row
        assert "full_name" not in row
    assert m005_split_name.down(s5) == s4

    engine = MigrationEngine()
    s6 = engine.migrate(s4, 4, 6)
    assert engine.migrate(s6, 6, 4) == s4


def test_roundtrip_mononym_and_empty_surname() -> None:
    """Verify m005 and full chain preserve single-word mononyms without adding spaces."""
    names = ["Prince", "Madonna", "Cher", ""]
    s4 = _make_v4_rows(names)
    s5 = m005_split_name.up(s4)
    assert s5[0]["first_name"] == "Prince"
    assert m005_split_name.down(s5) == s4

    engine = MigrationEngine()
    s6 = engine.migrate(s4, 4, 6)
    assert engine.migrate(s6, 6, 4) == s4


def test_roundtrip_trailing_and_leading_spaces() -> None:
    """Verify m005 and full chain preserve trailing and leading spaces in full_name."""
    names = ["Prince ", "Jean-Luc  ", "Alice Smith ", "  Leading Name"]
    s4 = _make_v4_rows(names)
    s5 = m005_split_name.up(s4)
    assert m005_split_name.down(s5) == s4

    engine = MigrationEngine()
    s6 = engine.migrate(s4, 4, 6)
    assert engine.migrate(s6, 6, 4) == s4
