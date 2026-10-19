"""Visible unit tests for basic slugify and truncate_words functionality."""

from __future__ import annotations

from textutils import slugify, truncate_words


def test_slugify_simple_ascii() -> None:
    """Verify basic ASCII words and single spaces."""
    assert slugify("Hello World") == "hello-world"
    assert slugify("Python 3.12 Release") == "python-3-12-release"


def test_slugify_accents_stripped() -> None:
    """Verify single-word Latin accents are transliterated in ASCII mode."""
    assert slugify("Résumé") == "resume"
    assert slugify("Naive") == "naive"


def test_truncate_words_basic() -> None:
    """Verify word-boundary truncation."""
    assert truncate_words("one two three four", 2) == "one two..."
    assert truncate_words("short text", 5) == "short text"
