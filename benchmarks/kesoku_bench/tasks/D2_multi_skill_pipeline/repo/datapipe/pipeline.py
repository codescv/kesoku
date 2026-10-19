"""Nightly ingestion validator for CSV schemas and audit rules."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from datapipe.exceptions import AuditViolation, SchemaMismatch

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
INT_RE = re.compile(r"^[+-]?\d+$")
FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d+|\d+)$")


def infer_column_type(values: list[str], *, strict_dates: bool = True) -> str:
    """Infer the expected column type from non-empty string values.

    Args:
        values: Non-empty string cell values for a column.
        strict_dates: Whether YYYY-MM-DD strings are typed as 'date'.

    Returns:
        Inferred type string ('integer', 'float', 'date', or 'string').
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


def _parse_scalar(val: str) -> Any:
    """Parse a scalar or inline list value from rules.yaml."""
    val = val.strip()
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip("'\"") for item in inner.split(",")]
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return val


def load_rules_yaml(path: Path) -> dict[str, Any]:
    """Parse rules.yaml into a dictionary of table audit rules.

    Args:
        path: Path to rules.yaml.

    Returns:
        Dictionary containing quarantine_dir and tables specifications.
    """
    result: dict[str, Any] = {"quarantine_dir": "data/quarantine", "tables": {}}
    current_table: str | None = None
    current_rule: dict[str, Any] | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if "#" in raw_line:
            raw_line = raw_line.split("#", 1)[0]
        stripped = raw_line.strip()
        if not stripped:
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))

        if indent == 0 and ":" in stripped:
            key, val = stripped.split(":", 1)
            key = key.strip()
            val = val.strip()
            if key != "tables" and val:
                result[key] = _parse_scalar(val)
        elif indent == 2 and stripped.endswith(":"):
            current_table = stripped[:-1].strip()
            result["tables"][current_table] = {"rules": []}
            current_rule = None
        elif indent == 4 and current_table is not None and ":" in stripped:
            key, val = stripped.split(":", 1)
            key = key.strip()
            val = val.strip()
            if key == "rules":
                current_rule = None
            elif val:
                result["tables"][current_table][key] = _parse_scalar(val)
        elif indent == 6 and current_table is not None:
            if stripped.startswith("- "):
                current_rule = {}
                result["tables"][current_table]["rules"].append(current_rule)
                rest = stripped[2:].strip()
                if ":" in rest:
                    k, v = rest.split(":", 1)
                    current_rule[k.strip()] = _parse_scalar(v)
        elif indent == 8 and current_rule is not None and ":" in stripped:
            k, v = stripped.split(":", 1)
            current_rule[k.strip()] = _parse_scalar(v)

    return result


def evaluate_row(row_dict: dict[str, str], rules: list[dict[str, Any]]) -> list[str]:
    """Validate a single row dictionary against table rules.

    Args:
        row_dict: Column-name to cell-string mapping.
        rules: List of rule dicts.

    Returns:
        List of violation descriptions (empty if valid).
    """
    violations: list[str] = []
    for rule in rules:
        col = str(rule["column"])
        check = str(rule["check"])
        raw_val = row_dict.get(col, "").strip()

        if check == "contains_at":
            if "@" not in raw_val or raw_val.startswith("@") or raw_val.endswith("@"):
                violations.append(f"{col}='{raw_val}' is not a valid email")
        elif check == "float_range":
            try:
                fval = float(raw_val)
            except ValueError:
                violations.append(f"{col}='{raw_val}' is not a float")
                continue
            lo = float(rule.get("min", float("-inf")))
            hi = float(rule.get("max", float("inf")))
            if not (lo <= fval <= hi):
                violations.append(f"{col}={fval} outside [{lo}, {hi}]")
        elif check == "int_range":
            try:
                ival = int(raw_val)
            except ValueError:
                violations.append(f"{col}='{raw_val}' is not an integer")
                continue
            lo_i = int(rule.get("min", -10**9))
            hi_i = int(rule.get("max", 10**9))
            if not (lo_i <= ival <= hi_i):
                violations.append(f"{col}={ival} outside [{lo_i}, {hi_i}]")
        elif check == "allowed_values":
            allowed = [str(x) for x in rule.get("values", [])]
            if raw_val not in allowed:
                violations.append(f"{col}='{raw_val}' not in {allowed}")
    return violations


def validate_schema(repo_root: Path) -> None:
    """Validate schema.json against CSV files in repo_root/data.

    Args:
        repo_root: Repository root directory.

    Raises:
        SchemaMismatch: If schema.json is stale or does not follow team conventions.
    """
    schema_path = repo_root / "schema.json"
    if not schema_path.is_file():
        raise SchemaMismatch("schema.json is missing")

    doc = json.loads(schema_path.read_text(encoding="utf-8"))
    if doc.get("strict_dates") is not True:
        raise SchemaMismatch("schema.json must be generated with strict_dates=true (--strict-dates)")

    tables_doc = doc.get("tables", {})
    csv_files = sorted((repo_root / "data").glob("*.csv"))
    if not csv_files:
        raise SchemaMismatch("No CSV files found in data/")

    for csv_path in csv_files:
        tname = csv_path.stem
        if tname not in tables_doc:
            raise SchemaMismatch(f"Table '{tname}' missing from schema.json")

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        headers = rows[0]
        data_rows = rows[1:]

        declared_cols = tables_doc[tname].get("columns", [])
        if len(declared_cols) != len(headers):
            raise SchemaMismatch(
                f"Table '{tname}' column count mismatch: schema.json has {len(declared_cols)}, "
                f"{csv_path.name} has {len(headers)}"
            )

        for idx, col_name in enumerate(headers):
            col_spec = declared_cols[idx]
            if col_spec.get("name") != col_name or col_spec.get("position") != idx:
                raise SchemaMismatch(f"Table '{tname}' column #{idx} mismatch: expected '{col_name}'")

            raw_cells = [r[idx].strip() if idx < len(r) else "" for r in data_rows]
            expected_nullable = any(cell == "" for cell in raw_cells)
            if "nullable" not in col_spec or col_spec.get("nullable") is not expected_nullable:
                raise SchemaMismatch(
                    f"Table '{tname}.{col_name}' nullable flag mismatch: "
                    f"expected {expected_nullable}, got {col_spec.get('nullable')}"
                )

            non_empty = [cell for cell in raw_cells if cell != ""]
            expected_type = infer_column_type(non_empty, strict_dates=True)
            if col_spec.get("type") != expected_type:
                raise SchemaMismatch(
                    f"Table '{tname}.{col_name}' type mismatch: "
                    f"expected '{expected_type}', got '{col_spec.get('type')}'"
                )
            if expected_type == "date" and col_spec.get("date_format") != "YYYY-MM-DD":
                raise SchemaMismatch(f"Table '{tname}.{col_name}' missing date_format='YYYY-MM-DD'")


def validate_audit(repo_root: Path) -> dict[str, int]:
    """Validate all CSV tables against rules.yaml and verify quarantine row conservation.

    Args:
        repo_root: Repository root directory.

    Returns:
        Mapping of table name to valid row count.

    Raises:
        AuditViolation: If violating rows remain in data/*.csv or rows were deleted instead of quarantined.
    """
    rules_path = repo_root / "rules.yaml"
    config = load_rules_yaml(rules_path)
    quarantine_dir = repo_root / str(config.get("quarantine_dir", "data/quarantine"))
    counts: dict[str, int] = {}
    errors: list[str] = []

    for table_name, table_spec in sorted(config.get("tables", {}).items()):
        csv_path = repo_root / str(table_spec["file"])
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        headers = rows[0]
        data_rows = rows[1:]
        rules = list(table_spec.get("rules", []))

        for row_num, row in enumerate(data_rows, start=2):
            row_dict = {headers[i]: (row[i] if i < len(row) else "") for i in range(len(headers))}
            reasons = evaluate_row(row_dict, rules)
            if reasons:
                errors.append(f"{csv_path.name}:{row_num} ({row[0]}): {'; '.join(reasons)}")

        q_path = quarantine_dir / csv_path.name
        q_rows: list[list[str]] = []
        if q_path.is_file():
            with q_path.open("r", encoding="utf-8", newline="") as f:
                q_all = list(csv.reader(f))
                if q_all:
                    if q_all[0] != headers:
                        errors.append(f"Quarantine file {q_path} has mismatched CSV header")
                    q_rows = q_all[1:]

        expected_total = int(table_spec.get("expected_total_rows", len(data_rows)))
        actual_total = len(data_rows) + len(q_rows)
        if actual_total != expected_total:
            errors.append(
                f"{csv_path.name}: row conservation failed (clean={len(data_rows)} + "
                f"quarantined={len(q_rows)} != expected={expected_total}); "
                "violating rows must be quarantined to data/quarantine/, never deleted"
            )
        counts[table_name] = len(data_rows)

    if errors:
        raise AuditViolation("Audit checks failed:\n  - " + "\n  - ".join(errors))

    return counts


def run(repo_root: Path | str | None = None) -> dict[str, Any]:
    """Execute the full DataPipe schema + audit verification pipeline.

    Args:
        repo_root: Optional path to repository root. Defaults to parent of datapipe package.

    Returns:
        Summary dictionary with status and clean row counts per table.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent.parent
    validate_schema(root)
    table_counts = validate_audit(root)
    return {"status": "ok", "tables": table_counts}


def main() -> int:
    """CLI entry point for `python -m datapipe.pipeline`."""
    result = run()
    print(f"Pipeline succeeded: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
