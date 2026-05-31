"""Unit tests for the text and markdown processing utility functions."""

from kesoku.utils.text import format_text, split_text_into_chunks


def test_format_text_headers_shifting() -> None:
    """Test shifting and clamping of markdown headers."""
    # Case 1: Headers start at level 2 -> shifted to level 1, clamp levels > 3
    input_text = "## Header A\n### Header B\n#### Header C\n##### Header D\n"
    expected = "# Header A\n\n## Header B\n\n### Header C\n\n### Header D\n"
    assert format_text(input_text) == expected

    # Case 2: Headers already start at level 1 -> no shifting, only clamp levels > 3
    input_text_2 = "# Header A\n## Header B\n#### Header C\n"
    expected_2 = "# Header A\n\n## Header B\n\n### Header C\n"
    assert format_text(input_text_2) == expected_2


def test_format_text_consecutive_newlines_and_spacing() -> None:
    """Test collapsing consecutive newlines and ensuring spacing before headers."""
    input_text = "Some text.\n## Header A\n\n\nOther text.\n"
    # Note: format_text preserves trailing newlines,
    # and ensures exactly one blank line before headings if not at the start.
    expected = "Some text.\n\n# Header A\n\nOther text.\n"
    assert format_text(input_text) == expected


def test_format_text_code_blocks_ignored() -> None:
    """Test that formatting is not applied to content inside code blocks."""
    input_text = (
        "## Header A\n"
        "```python\n"
        "## This heading inside code block should not be shifted\n"
        "\n"
        "\n"
        "def foo():\n"
        "    pass\n"
        "```\n"
    )
    # Outer heading ## Header A shifts to # Header A.
    # Inner heading and inner blank lines remain intact.
    expected = (
        "# Header A\n"
        "```python\n"
        "## This heading inside code block should not be shifted\n"
        "\n"
        "\n"
        "def foo():\n"
        "    pass\n"
        "```\n"
    )
    assert format_text(input_text) == expected


def test_split_text_into_chunks_code_block() -> None:
    """Test splitting text into chunks while preserving and wrapping code blocks."""
    text = "Some introduction.\n```python\nline1\nline2\nline3\n```\nConclusion."
    # With max_length set so that it splits in the middle of the python code block:
    # e.g. max_length = 45.
    chunks = split_text_into_chunks(text, max_length=45)

    # The first chunk should contain "Some introduction." and the start of code block,
    # and automatically close with "```".
    # The second chunk should automatically open with "```python" and contain the rest.
    assert len(chunks) > 1
    for chunk in chunks:
        # If chunk contains code lines, it must be wrapped in ```
        if "line1" in chunk or "line2" in chunk or "line3" in chunk:
            assert chunk.startswith("```python") or chunk.endswith("```")
