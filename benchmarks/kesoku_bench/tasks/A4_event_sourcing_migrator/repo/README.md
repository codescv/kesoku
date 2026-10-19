# evsrc

`evsrc` is an in-memory event-sourcing library supporting multi-version event and snapshot schema evolution (`v1` -> `v2` -> `v3`), snapshot persistence, and deterministic `Account` state replay.

## Schema Versions

- **v1**: `balance` (`int`, default `0`)
- **v2**: `balance` (`int`), `currency` (`str`, default `"USD"`)
- **v3**: `balance` (`int`), `currency` (`str`), `tags` (`list[str]`, default `[]`)

## Modules

- `evsrc/events.py`: Event envelope (`EventEnvelope`), `Account` aggregate state, schema field definitions (`VERSION_SCHEMA_FIELDS`), and domain exceptions (`SequenceGap`, `CycleError`).
- `evsrc/registry.py`: Schema migration graph registry (`MigratorRegistry`, `default_registry`, `register_upcaster`).
- `evsrc/upcasters.py`: Built-in schema migration functions registered on `default_registry`.
- `evsrc/store.py`: Append-only `EventStore` and versioned `Snapshot` serialization.
- `evsrc/replay.py`: Event stream folding into `Account` state (`replay_account`, `replay_events`).

## Running Tests

```bash
uv run pytest
```
