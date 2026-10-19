"""Regression tests for Issue #142 (multi-separator collapse and max_length strip)."""

from __future__ import annotations

from textutils import slugify


def test_issue_142_multi_dash_collapse() -> None:
    """Verify multiple consecutive dashes and Unicode punctuation collapse to a single dash."""
    assert slugify("Café — Crème") == "cafe-creme"
    assert slugify("Hello --- World!!!") == "hello-world"
    assert slugify("東京・・・タワー", allow_unicode=True) == "東京-タワー"
    assert slugify("   ---Multiple   Spaces & Symbols---   ") == "multiple-spaces-symbols"


def test_issue_142_max_length_strips_trailing_separator() -> None:
    """Verify max_length truncation does not leave trailing or leading separators."""
    assert slugify("hello world", max_length=6) == "hello"
    assert slugify("  --leading-and-trailing--  ", max_length=8) == "leading"
