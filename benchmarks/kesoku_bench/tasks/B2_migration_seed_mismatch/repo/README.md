# `ledger` — Reversible Account Schema Migrations

`ledger` manages bidirectional schema transformations (`m001` through `m006`) over in-memory account records. Every migration implements `up(state)` and `down(state)` and is verified via stateful property testing.

## Usage

Run the unit test suite:

```bash
uv run pytest
```

Run the seeded state-machine migration fuzzer:

```bash
uv run python scripts/fuzz_migrations.py
```

See `MIGRATIONS.md` for the schema invariants required of every migration module.
