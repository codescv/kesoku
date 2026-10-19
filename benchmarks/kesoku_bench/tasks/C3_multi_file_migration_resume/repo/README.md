# cfgkit

Configuration management library providing loading, merging, environment variable binding, and structured export.

## Modules

- `cfgkit.legacy`: Legacy `OldConfig` model (`to_dict()` preserves legacy field order).
- `cfgkit.v2`: Modern immutable `ConfigV2` model (`to_dict()` returns keys in alphabetical order).
- `cfgkit.loader`: Load and validate configurations from dictionaries and JSON strings.
- `cfgkit.merge`: Merge base and override configurations or chains of configurations.
- `cfgkit.env`: Parse and format `CFGKIT_*` environment variable mappings.
- `cfgkit.export`: Export configurations to ordered dicts, JSON strings (`dump`), and `key=value` lines.
- `cfgkit.lint`: Static analysis and migration status checker (`uv run python -m cfgkit.lint`).

See `MIGRATION.md` for the consumer migration guide.
