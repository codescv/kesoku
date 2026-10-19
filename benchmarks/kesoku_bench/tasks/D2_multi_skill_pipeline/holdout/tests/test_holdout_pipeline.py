"""Holdout FAIL_TO_PASS tests for D2_multi_skill_pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from datapipe.pipeline import run

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_SCHEMA: dict[str, Any] = {
    "schema_version": "2.0",
    "generator": "schema-gen/1.1.0",
    "strict_dates": True,
    "schema_digest": "005144b11f93ec59",
    "tables": {
        "customers": {
            "source_file": "data/customers.csv",
            "columns": [
                {"name": "customer_id", "position": 0, "type": "integer", "nullable": False},
                {"name": "email", "position": 1, "type": "string", "nullable": False},
                {"name": "country", "position": 2, "type": "string", "nullable": False},
                {
                    "name": "signup_date",
                    "position": 3,
                    "type": "date",
                    "nullable": False,
                    "date_format": "YYYY-MM-DD",
                },
                {"name": "credit_limit", "position": 4, "type": "float", "nullable": False},
                {"name": "loyalty_tier", "position": 5, "type": "string", "nullable": False},
                {"name": "notes", "position": 6, "type": "string", "nullable": True},
            ],
        },
        "orders": {
            "source_file": "data/orders.csv",
            "columns": [
                {"name": "order_id", "position": 0, "type": "integer", "nullable": False},
                {"name": "customer_id", "position": 1, "type": "integer", "nullable": False},
                {
                    "name": "order_date",
                    "position": 2,
                    "type": "date",
                    "nullable": False,
                    "date_format": "YYYY-MM-DD",
                },
                {
                    "name": "shipped_at",
                    "position": 3,
                    "type": "date",
                    "nullable": True,
                    "date_format": "YYYY-MM-DD",
                },
                {"name": "status", "position": 4, "type": "string", "nullable": False},
                {"name": "amount", "position": 5, "type": "float", "nullable": False},
                {"name": "discount_pct", "position": 6, "type": "integer", "nullable": False},
                {"name": "coupon_code", "position": 7, "type": "string", "nullable": True},
            ],
        },
    },
}

EXPECTED_CUSTOMERS_HEADER = [
    "customer_id",
    "email",
    "country",
    "signup_date",
    "credit_limit",
    "loyalty_tier",
    "notes",
]
EXPECTED_CUSTOMERS_CLEAN = [
    ["1001", "alice.chen@example.com", "US", "2025-01-10", "5000.00", "GOLD", "enterprise account"],
    ["1002", "bob.smith@example.com", "CA", "2025-01-12", "2500.00", "STANDARD", ""],
    ["1003", "carol.jones@example.co.uk", "GB", "2025-01-18", "12000.00", "PLATINUM", "priority support"],
    ["1004", "david.kim@example.com", "US", "2025-01-25", "1500.00", "STANDARD", ""],
    ["1005", "eva.mueller@example.de", "DE", "2025-02-02", "8000.00", "SILVER", "annual billing"],
    ["1007", "frank.wright@example.com", "US", "2025-02-19", "3500.00", "SILVER", ""],
    ["1008", "grace.hopper@example.com", "US", "2025-02-22", "25000.00", "PLATINUM", "gov contract"],
    ["1009", "hiro.tanaka@example.jp", "JP", "2025-02-27", "6000.00", "GOLD", ""],
    ["1010", "irene.adler@example.co.uk", "GB", "2025-03-01", "9500.00", "GOLD", "verified partner"],
]
EXPECTED_CUSTOMERS_QUARANTINE = [
    ["1006", "diana.prince@example.org", "US", "2025-02-14", "-250.00", "SILVER", "chargeback pending"],
]

EXPECTED_ORDERS_HEADER = [
    "order_id",
    "customer_id",
    "order_date",
    "shipped_at",
    "status",
    "amount",
    "discount_pct",
    "coupon_code",
]
EXPECTED_ORDERS_CLEAN = [
    ["5001", "1001", "2025-03-01", "2025-03-02", "DELIVERED", "420.00", "10", "WELCOME10"],
    ["5002", "1002", "2025-03-02", "", "PENDING", "85.50", "0", ""],
    ["5003", "1003", "2025-03-03", "2025-03-04", "SHIPPED", "1250.00", "15", "PLAT15"],
    ["5005", "1004", "2025-03-05", "2025-03-07", "DELIVERED", "62.25", "0", ""],
    ["5006", "1005", "2025-03-06", "", "PAID", "310.00", "5", "SPRING5"],
    ["5007", "1007", "2025-03-08", "2025-03-09", "DELIVERED", "145.00", "0", ""],
    ["5008", "1008", "2025-03-09", "2025-03-10", "SHIPPED", "4800.00", "20", "GOV20"],
    ["5010", "1009", "2025-03-12", "2025-03-13", "DELIVERED", "720.00", "10", "WELCOME10"],
    ["5011", "1010", "2025-03-14", "", "PAID", "530.00", "0", ""],
    ["5012", "1001", "2025-03-15", "2025-03-16", "REFUNDED", "95.00", "0", ""],
]
EXPECTED_ORDERS_QUARANTINE = [
    ["5004", "1003", "2025-03-04", "2025-03-06", "SHIPPED", "189.50", "125", "VIP125"],
    ["5009", "1007", "2025-03-11", "", "UNKNOWN_STATE", "64.00", "0", ""],
]


def _read_csv(path: Path) -> list[list[str]]:
    """Read all rows from a CSV file."""
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def test_pipeline_run_succeeds() -> None:
    """Verify datapipe.pipeline.run() completes without SchemaMismatch or AuditViolation."""
    result = run(REPO_ROOT)
    assert result == {"status": "ok", "tables": {"customers": 9, "orders": 10}}


def test_schema_matches_reference_inference() -> None:
    """Verify schema.json equals the reference inference from schema-gen (--strict-dates, nullable)."""
    schema_path = REPO_ROOT / "schema.json"
    assert schema_path.is_file()
    actual = json.loads(schema_path.read_text(encoding="utf-8"))
    assert actual == EXPECTED_SCHEMA


def test_quarantine_files_contain_exact_violating_rows() -> None:
    """Verify data/quarantine/*.csv exist and contain the exact 3 violating rows unmodified."""
    q_cust = REPO_ROOT / "data" / "quarantine" / "customers.csv"
    q_ord = REPO_ROOT / "data" / "quarantine" / "orders.csv"
    assert q_cust.is_file(), "Missing data/quarantine/customers.csv"
    assert q_ord.is_file(), "Missing data/quarantine/orders.csv"

    cust_rows = _read_csv(q_cust)
    ord_rows = _read_csv(q_ord)
    assert cust_rows[0] == EXPECTED_CUSTOMERS_HEADER
    assert cust_rows[1:] == EXPECTED_CUSTOMERS_QUARANTINE
    assert ord_rows[0] == EXPECTED_ORDERS_HEADER
    assert ord_rows[1:] == EXPECTED_ORDERS_QUARANTINE


def test_source_csvs_cleaned_and_row_counts_conserved() -> None:
    """Verify source CSVs contain the remaining valid rows unchanged and row counts add up."""
    cust_rows = _read_csv(REPO_ROOT / "data" / "customers.csv")
    ord_rows = _read_csv(REPO_ROOT / "data" / "orders.csv")
    assert cust_rows[0] == EXPECTED_CUSTOMERS_HEADER
    assert cust_rows[1:] == EXPECTED_CUSTOMERS_CLEAN
    assert ord_rows[0] == EXPECTED_ORDERS_HEADER
    assert ord_rows[1:] == EXPECTED_ORDERS_CLEAN

    q_cust_rows = _read_csv(REPO_ROOT / "data" / "quarantine" / "customers.csv")[1:]
    q_ord_rows = _read_csv(REPO_ROOT / "data" / "quarantine" / "orders.csv")[1:]
    assert len(cust_rows[1:]) + len(q_cust_rows) == 10
    assert len(ord_rows[1:]) + len(q_ord_rows) == 12
