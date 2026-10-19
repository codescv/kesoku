"""Holdout FAIL_TO_PASS tests for D4_repo_conventions_and_lint_gate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from textutils import slugify

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_issue_142_reported_cases() -> None:
    """Verify all 6 reproduction examples documented in issues/142.md."""
    assert slugify("Café — Crème") == "cafe-creme"
    assert slugify("Hello --- World!!!") == "hello-world"
    assert slugify("東京・・・タワー", allow_unicode=True) == "東京-タワー"
    assert slugify("   ---Multiple   Spaces & Symbols---   ") == "multiple-spaces-symbols"
    assert slugify("hello world", max_length=6) == "hello"
    assert slugify("  --leading-and-trailing--  ", max_length=8) == "leading"


def test_slugify_extra_unicode_and_custom_separator(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify 4 extra edge cases including Unicode max_length, custom separator, and no print()."""
    assert slugify("データ — — パイプライン — 2025", allow_unicode=True) == "データ-パイプライン-2025"
    assert slugify("こんにちは・世界・テスト", max_length=6, allow_unicode=True) == "こんにちは"
    assert slugify("foo --- bar ___ baz", max_length=8, separator="_") == "foo_bar"
    assert slugify("---   ...   ---") == ""
    captured = capsys.readouterr()
    assert captured.out == "", f"slugify should not print to stdout, got: {captured.out!r}"


def test_conventions_lint_passes() -> None:
    """Verify scripts/lint_conventions.py exits 0 on the repository."""
    lint_script = REPO_ROOT / "scripts" / "lint_conventions.py"
    proc = subprocess.run(
        [sys.executable, str(lint_script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"lint_conventions.py failed:\n{proc.stdout}\n{proc.stderr}"


def test_regression_file_exists_and_passes() -> None:
    """Verify tests/regression/test_142.py exists and passes pytest."""
    reg_file = REPO_ROOT / "tests" / "regression" / "test_142.py"
    assert reg_file.is_file(), "Missing required regression test file: tests/regression/test_142.py"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(reg_file)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"tests/regression/test_142.py failed:\n{proc.stdout}\n{proc.stderr}"
