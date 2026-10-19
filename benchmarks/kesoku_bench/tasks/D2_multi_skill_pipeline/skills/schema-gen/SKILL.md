---
name: schema-gen
description: Infer and generate canonical schema.json files from CSV datasets with strict ISO-8601 date detection and nullable column annotations.
version: "1.1.0"
required_permissions:
  - run_shell_command
metadata:
  tags:
    - schema
    - csv
    - datapipe
    - inference
  related_skills:
    - fixture-audit
---

# Schema-Gen (`schema.json` Inference Tool)

This skill generates canonical `schema.json` definitions for `datapipe` repositories from CSV files under `data/`.

## Team Conventions

1. **Never hand-edit `schema.json`**: Always regenerate it using `schemagen.py infer`.
2. **Strict ISO-8601 Date Detection (`--strict-dates`)**: Production pipelines require `--strict-dates` so that `YYYY-MM-DD` columns are typed as `"date"` (with `"date_format": "YYYY-MM-DD"`) rather than `"string"`.
3. **Nullable Columns**: Any column containing one or more empty cells in the CSV dataset is marked with `"nullable": true` (and non-nullable columns carry `"nullable": false`).

## CLI Usage

Run from the repository root (`repo/`):

```bash
uv run python $SKILL_DIR/scripts/schemagen.py infer data/*.csv --out schema.json --strict-dates
```
