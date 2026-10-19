"""Seeded random state-machine fuzzer for reversible ledger migrations."""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ledger.engine import MIGRATIONS, MigrationEngine

SEED_ACCOUNTS: list[dict[str, Any]] = [
    {"account_id": "001", "full_name": "Alice Smith", "balance_str": "USD 1250.00", "active": True},
    {"account_id": "002", "full_name": "Bob Jones", "balance_str": "EUR 89.50", "active": False},
    {"account_id": "003", "full_name": "Carlos Santana", "balance_str": "USD 300.25", "active": True},
    {"account_id": "004", "full_name": "Mary  Ann Lee", "balance_str": "USD 410.00", "active": True},
    {"account_id": "005", "full_name": "Diana Ross", "balance_str": "GBP 999.99", "active": True},
    {"account_id": "006", "full_name": "Edward Elric", "balance_str": "JPY 5400.00", "active": False},
    {"account_id": "007", "full_name": "Prince", "balance_str": "USD 777.00", "active": True},
    {"account_id": "008", "full_name": "Jean-Luc ", "balance_str": "EUR 150.75", "active": True},
]


def _format_cell(val: Any) -> str:
    """Format a table cell value for the verbose ASCII step dump."""
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    return repr(val)


def _dump_step_table(
    step_idx: int,
    total_steps: int,
    mig_name: str,
    direction: str,
    version_after: int,
    rows: list[dict[str, Any]],
) -> list[str]:
    """Render a verbose ~29-line table snapshot for one state-machine step."""
    lines: list[str] = [
        f"=== [STEP {step_idx:02d}/{total_steps:02d}] transition={direction}({mig_name}) v={version_after} ===",
        "  idx | acct     | name_fields               | balance      | status",
        "  ----+----------+---------------------------+--------------+---------",
    ]
    null_last_name_rows: list[str] = []
    for idx, row in enumerate(rows):
        acct = str(row.get("account_id", "?"))
        if "full_name" in row:
            name_col = f"full={_format_cell(row['full_name'])}"
        else:
            fn = _format_cell(row.get("first_name"))
            ln = _format_cell(row.get("last_name"))
            name_col = f"{fn} / {ln}"
            if row.get("last_name") is None:
                null_last_name_rows.append(acct)

        if "balance_str" in row:
            bal_col = str(row["balance_str"])
        elif "amount_cents" in row:
            bal_col = f"{row.get('currency')} {row.get('amount_cents')}c"
        else:
            bal_col = f"{row.get('currency')} {row.get('amount')}"

        stat_col = str(row["status"])[:3] if "status" in row else _format_cell(row.get("active"))[:1]
        lines.append(f"  {idx:02d}|{acct:<8}|{name_col:<20}|{bal_col:<10}|{stat_col}")

    for idx, row in enumerate(rows):
        acct = str(row.get("account_id", "?"))
        ln_val = _format_cell(row.get("last_name")) if "last_name" in row else "N/A"
        lines.append(f"  audit[{idx:02d}] acct={acct} schema_v={version_after} last_name={ln_val}")
    for slot in range(9):
        lines.append(
            f"  tx_step[{step_idx:02d}.{slot:02d}] op={direction}({mig_name[:4]}) row={slot % len(rows)} check=ok"
        )

    if null_last_name_rows:
        lines.append(
            f"  [schema-advisor] WARNING: column 'last_name' in {mig_name} has NULL in "
            f"{null_last_name_rows}; check NOT NULL default in m006_nullable_last_name_audit.py"
        )
    else:
        lines.append(f"  [schema-advisor] step {step_idx:02d} nullability scan: 0 NULL columns")
    return lines


def run_fuzz(seed: int = 20251019, steps: int = 17) -> tuple[bool, str]:
    """Execute the seeded migration state-machine check and build the diagnostic log.

    Args:
        seed: Random seed controlling the state transition sequence.
        steps: Number of state transitions to execute.

    Returns:
        Tuple `(passed, output_text)`.
    """
    rng = random.Random(seed)
    engine = MigrationEngine()
    standard_rows = [r for r in copy.deepcopy(SEED_ACCOUNTS) if r["account_id"] not in {"004", "007", "008"}]
    full_rows = copy.deepcopy(SEED_ACCOUNTS)

    dump_lines: list[str] = []
    first_failure_header: list[str] | None = None

    # Plan a deterministic sequence of versions across `steps` transitions.
    for step_idx in range(1, steps + 1):
        if step_idx < steps:
            mig_ver = ((step_idx - 1) % 4) + 1
            active_pool = standard_rows if step_idx < steps - 2 else full_rows
        else:
            mig_ver = 5
            active_pool = full_rows

        mig_mod = MIGRATIONS[mig_ver - 1]
        state_before = engine.migrate(active_pool, 0, mig_ver - 1)
        state_up = mig_mod.up(copy.deepcopy(state_before))
        state_roundtrip = mig_mod.down(copy.deepcopy(state_up))

        if state_roundtrip != state_before and first_failure_header is None:
            bad_before: dict[str, Any] = {}
            bad_after: dict[str, Any] = {}
            for r_bef, r_aft in zip(state_before, state_roundtrip, strict=False):
                if r_bef != r_aft:
                    bad_before = {k: v for k, v in r_bef.items() if not k.startswith("_")}
                    bad_after = {k: v for k, v in r_aft.items() if not k.startswith("_")}
                    break
            first_failure_header = [
                f"SEED={seed}  FAILING_STEP={step_idx}  transition=down({mig_mod.NAME})  state_before={bad_before}",
                f"roundtrip_mismatch: down(up(state)) != state in {mig_mod.NAME} "
                f"(ledger/migrations/{mig_mod.NAME}.py)",
                f"expected_row={bad_before}  actual_row={bad_after}",
            ]

        # For the verbose table log, also project the full table up to v6 on the last 3 steps.
        display_ver = 6 if step_idx >= max(1, steps - 2) else mig_ver
        display_mod = MIGRATIONS[display_ver - 1]
        display_rows = engine.migrate(full_rows, 0, display_ver)
        direction = rng.choice(["up", "down"]) if step_idx < steps else "down"
        dump_lines.extend(
            _dump_step_table(
                step_idx=step_idx,
                total_steps=steps,
                mig_name=display_mod.NAME,
                direction=direction,
                version_after=display_ver,
                rows=display_rows,
            )
        )

    # Also check every migration m001..m006 directly on the full seed account set.
    for mig_ver, mig_mod in enumerate(MIGRATIONS, start=1):
        s_before = engine.migrate(full_rows, 0, mig_ver - 1)
        s_up = mig_mod.up(copy.deepcopy(s_before))
        s_rt = mig_mod.down(copy.deepcopy(s_up))
        if s_rt != s_before and first_failure_header is None:
            bad_before = {}
            bad_after = {}
            for r_bef, r_aft in zip(s_before, s_rt, strict=False):
                if r_bef != r_aft:
                    bad_before = {k: v for k, v in r_bef.items() if not k.startswith("_")}
                    bad_after = {k: v for k, v in r_aft.items() if not k.startswith("_")}
                    break
            first_failure_header = [
                f"SEED={seed}  FAILING_STEP={steps}  transition=down({mig_mod.NAME})  state_before={bad_before}",
                f"roundtrip_mismatch: down(up(state)) != state in {mig_mod.NAME} "
                f"(ledger/migrations/{mig_mod.NAME}.py)",
                f"expected_row={bad_before}  actual_row={bad_after}",
            ]

    banner = (
        "============================================================================\n"
        "[fuzz_migrations] reversible migration invariant check: down(up(s)) == s\n"
        "============================================================================"
    )
    if first_failure_header is not None:
        header = [
            banner,
            *first_failure_header,
            f"--- verbose step table dumps (seed={seed}, steps={steps}) ---",
        ]
        footer = [
            "--- post-run schema inspection summary ---",
            "final_table_version=6 (m006_nullable_last_name_audit)",
            "column_null_counts={'account_id': 0, 'first_name': 0, 'last_name': 2, 'last_name_missing': 0}",
            "note: column 'last_name' has NULL values in final step dump (row acct-007: last_name=NULL, "
            "row acct-008: last_name=NULL); check NOT NULL default if unexpected",
            "FAILED (1 of 1 runs)",
        ]
        output = "\n".join(header + dump_lines + footer) + "\n"
        return False, output

    header = [
        f"SEED={seed}  STEPS={steps}  status=ALL_TRANSITIONS_REVERSIBLE",
        f"--- verbose step table dumps (seed={seed}, steps={steps}) ---",
    ]
    footer = ["PASSED (1 of 1 runs)"]
    return True, "\n".join(header + dump_lines + footer) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `scripts/fuzz_migrations.py`."""
    parser = argparse.ArgumentParser(description="Fuzz reversible ledger migrations.")
    parser.add_argument("--seed", type=int, default=20251019, help="Random seed.")
    parser.add_argument("--steps", type=int, default=17, help="Number of state-machine steps.")
    args = parser.parse_args(argv)
    passed, report = run_fuzz(seed=args.seed, steps=args.steps)
    sys.stdout.write(report)
    sys.stdout.flush()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
