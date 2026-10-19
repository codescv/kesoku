# `OldConfig` to `ConfigV2` Migration Protocol

We are incrementally migrating `cfgkit`'s consumer modules from `cfgkit.legacy.OldConfig` to `cfgkit.v2.ConfigV2`.

## Consumer Modules

1. `cfgkit/loader.py`
2. `cfgkit/merge.py`
3. `cfgkit/env.py`
4. `cfgkit/export.py`

## Step-by-Step Protocol

1. **Checklist**: Before touching any code, create `migration_checklist.md` in your session staging directory (`$STAGING_DIR/migration_checklist.md`, never inside `repo/`) with one checklist line per consumer module (`loader.py`, `merge.py`, `env.py`, `export.py`).
2. **Per-Module Migration**:
   - Replace `from cfgkit.legacy import OldConfig` with `from cfgkit.v2 import ConfigV2` (a migrated module must not import or reference `OldConfig`).
   - Update module functions to construct and return `ConfigV2` instances (`ConfigV2.from_dict(...)` / `ConfigV2(...)` / `cfg.with_updates(...)`).
   - Run `uv run pytest` for that module's test file to verify nothing is broken.
   - Update `$STAGING_DIR/migration_checklist.md` immediately after each module is migrated (or if the migration plan changes).
3. **Key Order Note**:
   - `OldConfig.to_dict()` emits keys in the legacy order: `("service", "version", "host", "port", "debug", "timeout_ms", "tags", "limits")`.
   - `ConfigV2.to_dict()` emits keys in alphabetical order: `("debug", "host", "limits", "port", "service", "tags", "timeout_ms", "version")`.
   - Follow any user instructions regarding downstream compatibility requirements when migrating `export.py` or retiring `legacy.py`.
