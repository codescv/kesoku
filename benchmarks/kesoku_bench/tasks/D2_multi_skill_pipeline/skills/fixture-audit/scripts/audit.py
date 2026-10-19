"""CLI tool to audit CSV datasets against rules.yaml and quarantine violating rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


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
    """Parse the declarative rules.yaml file using pure stdlib.

    Args:
        path: Path to rules.yaml.

    Returns:
        Parsed dictionary with `quarantine_dir` and `tables`.
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
    """Check a single CSV row against a table's validation rules.

    Args:
        row_dict: Mapping of column name to raw string cell value.
        rules: List of rule specifications for the table.

    Returns:
        List of violation reason strings (empty if the row is valid).
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


def audit_table(
    repo_root: Path,
    table_spec: dict[str, Any],
    quarantine_dir: Path,
    *,
    check_only: bool = False,
) -> tuple[int, int]:
    """Audit a single CSV table and optionally move violating rows to quarantine.

    Args:
        repo_root: Root directory of the repository.
        table_spec: Table entry from rules.yaml.
        quarantine_dir: Directory where quarantine CSV files are stored.
        check_only: If True, do not modify files.

    Returns:
        Tuple of (clean_row_count, newly_quarantined_count).
    """
    csv_path = repo_root / str(table_spec["file"])
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        return 0, 0

    headers = rows[0]
    data_rows = rows[1:]
    rules = list(table_spec.get("rules", []))

    clean_rows: list[list[str]] = []
    bad_rows: list[list[str]] = []

    for row in data_rows:
        row_dict = {headers[i]: (row[i] if i < len(row) else "") for i in range(len(headers))}
        reasons = evaluate_row(row_dict, rules)
        if reasons:
            bad_rows.append(row)
        else:
            clean_rows.append(row)

    if check_only:
        return len(clean_rows), len(bad_rows)

    q_file = quarantine_dir / csv_path.name
    existing_q_rows: list[list[str]] = []
    if q_file.is_file():
        with q_file.open("r", encoding="utf-8", newline="") as f:
            q_all = list(csv.reader(f))
            if len(q_all) > 1:
                existing_q_rows = q_all[1:]

    combined_q_rows = list(existing_q_rows)
    seen = {tuple(r) for r in combined_q_rows}
    for r in bad_rows:
        if tuple(r) not in seen:
            combined_q_rows.append(r)
            seen.add(tuple(r))

    if combined_q_rows:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        with q_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(headers)
            writer.writerows(combined_q_rows)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(clean_rows)

    return len(clean_rows), len(combined_q_rows)


def main(argv: list[str] | None = None) -> int:
    """Entry point for audit.py CLI."""
    parser = argparse.ArgumentParser(description="Audit CSV fixtures and quarantine violating rows")
    parser.add_argument("--rules", default="rules.yaml", help="Path to rules.yaml")
    parser.add_argument("--check-only", action="store_true", help="Report violations without modifying CSVs")
    args = parser.parse_args(argv)

    rules_path = Path(args.rules).resolve()
    repo_root = rules_path.parent
    config = load_rules_yaml(rules_path)
    quarantine_dir = repo_root / str(config.get("quarantine_dir", "data/quarantine"))

    total_bad = 0
    for table_name, table_spec in sorted(config.get("tables", {}).items()):
        clean_cnt, bad_cnt = audit_table(
            repo_root,
            table_spec,
            quarantine_dir,
            check_only=bool(args.check_only),
        )
        total_bad += bad_cnt
        action = "found" if args.check_only else "quarantined"
        print(f"[{table_name}] clean={clean_cnt}, {action}={bad_cnt} -> {quarantine_dir / f'{table_name}.csv'}")

    if args.check_only and total_bad > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
