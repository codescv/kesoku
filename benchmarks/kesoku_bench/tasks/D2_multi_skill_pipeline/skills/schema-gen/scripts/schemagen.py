"""CLI tool to infer canonical datapipe schema.json from CSV files."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import re
from pathlib import Path
from typing import Any

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
INT_RE = re.compile(r"^[+-]?\d+$")
FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d+|\d+)$")


def infer_column_type(values: list[str], *, strict_dates: bool) -> str:
    """Infer the canonical column type from non-empty string values.

    Args:
        values: Non-empty cell strings for a column.
        strict_dates: Whether to classify YYYY-MM-DD values as 'date'.

    Returns:
        One of 'integer', 'float', 'date', or 'string'.
    """
    if not values:
        return "string"
    if strict_dates and all(DATE_RE.match(v) for v in values):
        return "date"
    if all(INT_RE.match(v) for v in values):
        return "integer"
    if all(FLOAT_RE.match(v) for v in values) and any("." in v for v in values):
        return "float"
    return "string"


def infer_table_schema(csv_path: Path, *, strict_dates: bool) -> dict[str, Any]:
    """Infer column definitions for a single CSV table.

    Args:
        csv_path: Path to the CSV file.
        strict_dates: Whether strict ISO-8601 date inference is enabled.

    Returns:
        Dictionary describing the table's columns.
    """
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV file is empty: {csv_path}")

    headers = rows[0]
    data_rows = rows[1:]
    columns: list[dict[str, Any]] = []

    for col_idx, col_name in enumerate(headers):
        raw_cells = [r[col_idx].strip() if col_idx < len(r) else "" for r in data_rows]
        nullable = any(cell == "" for cell in raw_cells)
        non_empty = [cell for cell in raw_cells if cell != ""]
        col_type = infer_column_type(non_empty, strict_dates=strict_dates)
        col_spec: dict[str, Any] = {
            "name": col_name,
            "position": col_idx,
            "type": col_type,
            "nullable": nullable,
        }
        if col_type == "date":
            col_spec["date_format"] = "YYYY-MM-DD"
        columns.append(col_spec)

    return {
        "source_file": f"data/{csv_path.name}",
        "columns": columns,
    }


def compute_schema_digest(tables: dict[str, Any], *, strict_dates: bool) -> str:
    """Compute a deterministic SHA-256 fingerprint for the inferred schema."""
    canonical = json.dumps(
        {"version": "2.0", "strict_dates": strict_dates, "tables": tables},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def resolve_csv_inputs(inputs: list[str]) -> list[Path]:
    """Expand directories and glob patterns into sorted CSV file paths, excluding quarantine."""
    resolved: dict[str, Path] = {}
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            candidates = sorted(p.glob("*.csv"))
        elif any(ch in item for ch in ("*", "?", "[")):
            candidates = [Path(m) for m in sorted(glob.glob(item))]
        else:
            candidates = [p]

        for c in candidates:
            if c.is_file() and c.suffix == ".csv" and "quarantine" not in c.parts:
                resolved[c.stem] = c

    return [resolved[k] for k in sorted(resolved)]


def build_schema(csv_paths: list[Path], *, strict_dates: bool) -> dict[str, Any]:
    """Build the full schema.json dictionary for a list of CSV files."""
    tables: dict[str, Any] = {}
    for path in sorted(csv_paths, key=lambda p: p.stem):
        tables[path.stem] = infer_table_schema(path, strict_dates=strict_dates)

    return {
        "schema_version": "2.0",
        "generator": "schema-gen/1.1.0",
        "strict_dates": strict_dates,
        "schema_digest": compute_schema_digest(tables, strict_dates=strict_dates),
        "tables": tables,
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point for schemagen CLI."""
    parser = argparse.ArgumentParser(description="Infer canonical schema.json from CSV files")
    sub = parser.add_subparsers(dest="subcommand")

    p_infer = sub.add_parser("infer", help="Infer schema from CSV files")
    p_infer.add_argument("inputs", nargs="+", help="CSV files or directory")
    p_infer.add_argument("-o", "--out", "--output", dest="out", default="schema.json", help="Output schema JSON path")
    p_infer.add_argument("--strict-dates", action="store_true", help="Type YYYY-MM-DD columns as 'date'")

    args = parser.parse_args(argv)
    if args.subcommand != "infer":
        parser.print_help()
        return 1

    csv_files = resolve_csv_inputs(args.inputs)
    if not csv_files:
        raise SystemExit("No CSV files found for inference")

    schema = build_schema(csv_files, strict_dates=bool(args.strict_dates))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"Generated schema ({len(csv_files)} tables, strict_dates={bool(args.strict_dates)}) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
