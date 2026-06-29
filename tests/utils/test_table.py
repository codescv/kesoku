"""Unit tests for markdown table parser and image rendering utilities."""

from kesoku.utils.table import parse_markdown_tables, render_table_to_image


def test_parse_single_markdown_table() -> None:
    """Test parsing a single simple markdown table."""
    text = "Here is a table:\n| Col 1 | Col 2 |\n| :--- | ---: |\n| Val A | Val B |\n| Val C | Val D |\nDone."
    tables = parse_markdown_tables(text)
    assert len(tables) == 1

    table = tables[0]
    assert table.headers == ["Col 1", "Col 2"]
    assert table.alignments == ["left", "right"]
    assert table.rows == [["Val A", "Val B"], ["Val C", "Val D"]]
    assert "| Col 1 | Col 2 |" in table.raw_text
    assert "| Val C | Val D |" in table.raw_text


def test_parse_escaped_pipes() -> None:
    """Test parsing a table with escaped pipes in cells."""
    text = "| Header 1 | Header 2 |\n|---|---|\n| Escaped \\| pipe | Normal |\n"
    tables = parse_markdown_tables(text)
    assert len(tables) == 1
    assert tables[0].rows[0] == ["Escaped | pipe", "Normal"]


def test_parse_multiple_tables() -> None:
    """Test parsing multiple non-contiguous tables in the same text."""
    text = (
        "Table 1:\n"
        "| T1 H1 |\n"
        "|---|\n"
        "| T1 V1 |\n\n"
        "Intermediary text.\n\n"
        "Table 2:\n"
        "| T2 H1 | T2 H2 |\n"
        "|:---:|---|\n"
        "| T2 V1 | T2 V2 |\n"
    )
    tables = parse_markdown_tables(text)
    assert len(tables) == 2

    assert tables[0].headers == ["T1 H1"]
    assert tables[0].rows == [["T1 V1"]]

    assert tables[1].headers == ["T2 H1", "T2 H2"]
    assert tables[1].alignments == ["center", "left"]
    assert tables[1].rows == [["T2 V1", "T2 V2"]]


def test_render_table_to_image() -> None:
    """Test rendering a parsed table into PNG bytes."""
    headers = ["Language", "Text"]
    alignments = ["left", "center"]
    rows = [["Chinese", "你好"], ["Japanese", "こんにちは"], ["English", "Hello"]]

    png_bytes = render_table_to_image(
        headers=headers,
        alignments=alignments,
        rows=rows,
    )
    assert isinstance(png_bytes, bytes)
    # Magic bytes check for PNG format
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
