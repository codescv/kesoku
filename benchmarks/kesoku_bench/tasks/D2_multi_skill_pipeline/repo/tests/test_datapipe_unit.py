"""Visible unit tests for DataPipe type inference and rule evaluation helpers."""

from __future__ import annotations

from datapipe.pipeline import evaluate_row, infer_column_type


def test_infer_column_type_primitives() -> None:
    """Verify integer, float, date, and string inference."""
    assert infer_column_type(["10", "20", "-5"], strict_dates=True) == "integer"
    assert infer_column_type(["10.5", "20.0", "0.25"], strict_dates=True) == "float"
    assert infer_column_type(["2025-01-01", "2025-12-31"], strict_dates=True) == "date"
    assert infer_column_type(["2025-01-01", "2025-12-31"], strict_dates=False) == "string"


def test_evaluate_row_rules() -> None:
    """Verify declarative rule checks on synthetic rows."""
    rules = [
        {"column": "email", "check": "contains_at"},
        {"column": "amount", "check": "float_range", "min": 0.0, "max": 100.0},
        {"column": "status", "check": "allowed_values", "values": ["OK", "PENDING"]},
    ]
    valid = {"email": "user@example.com", "amount": "45.0", "status": "OK"}
    assert evaluate_row(valid, rules) == []

    invalid = {"email": "bad-email", "amount": "-10.0", "status": "BOGUS"}
    reasons = evaluate_row(invalid, rules)
    assert len(reasons) == 3
