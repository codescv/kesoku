---
name: csv-profiler
description: "Print column statistics (min/max/mean, distinct counts, top values) for a CSV file."
version: "0.9.2"
---

# csv-profiler

Quick exploratory statistics for a CSV file: row count, per-column min/max/mean for numeric columns,
distinct counts and the top 5 values for text columns.

## Usage

```bash
uv run python $SKILL_DIR/scripts/profile_csv.py data.csv --top 5
```

- `--delimiter ';'` for semicolon files.
- `--sample 10000` profiles only the first N rows of very large files.

Read-only: it never modifies the input file. Output is plain text for humans.
