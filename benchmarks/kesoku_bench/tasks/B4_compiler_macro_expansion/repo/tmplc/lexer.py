"""Line-oriented lexer for `.tpl` templates and `_macros.tpl` definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceLine:
    """A single line from a template source file with 1-based line and column metadata."""

    source_path: str
    line_no: int
    raw_text: str
    stripped: str
    indent_col: int


def lex_lines(source_text: str, source_path: str = "<memory>") -> list[SourceLine]:
    """Tokenize `source_text` into a list of `SourceLine` objects.

    Args:
        source_text: Raw UTF-8 template source string.
        source_path: Relative file path used in diagnostics.

    Returns:
        List of SourceLine entries (one per source line).
    """
    result: list[SourceLine] = []
    for idx, raw in enumerate(source_text.splitlines(), start=1):
        stripped = raw.strip()
        indent_col = (len(raw) - len(raw.lstrip(" "))) + 1
        result.append(
            SourceLine(
                source_path=source_path,
                line_no=idx,
                raw_text=raw,
                stripped=stripped,
                indent_col=indent_col,
            )
        )
    return result
