---
name: sql-explain
description: "Run EXPLAIN on a SQL query against a SQLite or Postgres DSN and highlight full scans and missing indexes."
version: "0.7.0"
---

# sql-explain

Explains a query plan in plain words: full table scans, missing indexes, nested-loop joins over large
tables, and a suggested index.

## Usage

```bash
uv run python $SKILL_DIR/scripts/explain.py --dsn sqlite:///app.db "SELECT * FROM orders WHERE user_id = 7"
```

- `--analyze` runs `EXPLAIN ANALYZE` (Postgres only; executes the query).
- Never point `--analyze` at production write queries.

Output: the raw plan followed by a short list of findings.
