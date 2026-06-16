"""Unit tests for the text and markdown processing utility functions."""

from kesoku.utils.text import extract_grep_snippet, format_text, split_text_into_chunks, truncate_middle


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


def test_clean_latex() -> None:
    """Test cleaning of LaTeX formulas to readable unicode/plain text."""
    from kesoku.utils.text import clean_latex

    # Test inline math $...$
    assert clean_latex(r"Let $x = \alpha + \beta^2$.") == "Let x = α + β²."

    # Test inline math with braces and subscripts
    assert clean_latex(r"We have $x_{i} = y_{j+1}$.") == "We have xᵢ = yⱼ₊₁."


    # Test fractions
    assert clean_latex(r"Ratio is $\frac{a}{b}$.") == "Ratio is (a)/(b)."

    # Test square roots
    assert clean_latex(r"Value is $\sqrt{x} + \sqrt[3]{y}$.") == "Value is √(x) + ³√(y)."

    # Test block math $$...$$
    block_input = "Formula:\n$$\nE = mc^2\n$$"
    expected_block = "Formula:\n\n> E = mc²\n"
    assert clean_latex(block_input) == expected_block

    # Test code block preservation
    code_input = "Math $x^2$ and code:\n```python\nx = y ** 2\n```\nMore math $y^3$."
    expected_code = "Math x² and code:\n```python\nx = y ** 2\n```\nMore math y³."
    assert clean_latex(code_input) == expected_code


def test_truncate_middle() -> None:
    """Test truncate_middle text helper."""
    # Under limit
    assert truncate_middle("Hello World", max_len=20) == "Hello World"
    # Over limit, requires middle truncation
    text = "Hello\nWorld\nLonger\nContent"
    res = truncate_middle(text, max_len=15, placeholder="...")
    assert "Hello" in res
    assert "Content" in res


def test_extract_grep_snippet() -> None:
    """Test extract_grep_snippet text helper."""
    # Case 1: Query in the middle
    text = "This is a very long string that contains the keyword in the middle of it."
    query = "keyword"
    res = extract_grep_snippet(text, query, window=10)
    assert res == "...tains the keyword in the mi..."

    # Case 2: Query near the start
    text = "Keyword is at the start of this string."
    query = "keyword"
    res = extract_grep_snippet(text, query, window=10)
    assert res == "Keyword is at the..."

    # Case 3: Query near the end
    text = "This string ends with the keyword"
    query = "keyword"
    res = extract_grep_snippet(text, query, window=10)
    assert res == "...with the keyword"

    # Case 4: Query not found (fallback to start)
    text = "This string does not contain it."
    query = "missing"
    res = extract_grep_snippet(text, query, window=10)
    assert res == "This string does not..."

    # Case 5: Short text
    text = "Short"
    query = "Short"
    res = extract_grep_snippet(text, query, window=10)
    assert res == "Short"

    # Case 6: Case insensitivity
    text = "Contains KEYWORD here"
    query = "keyword"
    res = extract_grep_snippet(text, query, window=5)
    assert res == "...ains KEYWORD here"



