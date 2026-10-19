"""Parser for macro definitions (`_macros.tpl`) and template modules (`.tpl`)."""

from __future__ import annotations

from dataclasses import dataclass

from tmplc.lexer import lex_lines


@dataclass(frozen=True)
class MacroParam:
    """Parameter declaration on a macro signature."""

    name: str
    type_name: str
    default_raw: str | None
    defined_at: str


@dataclass(frozen=True)
class MacroDef:
    """Parsed macro declaration with parameters and body template."""

    name: str
    params: list[MacroParam]
    body_template: str
    defined_at: str


@dataclass(frozen=True)
class MacroCallNode:
    """AST node representing a `@macro_name(k=v, ...)` invocation in a template."""

    macro_name: str
    raw_args: dict[str, str]
    source_path: str
    line_no: int
    col_no: int


@dataclass(frozen=True)
class TextNode:
    """AST node representing a static HTML/text line in a template."""

    text: str
    source_path: str
    line_no: int


@dataclass(frozen=True)
class TemplateAST:
    """Parsed AST for a single `.tpl` file."""

    source_path: str
    nodes: list[TextNode | MacroCallNode]


def _split_top_level_commas(arg_str: str) -> list[str]:
    """Split a comma-separated parameter or argument list respecting quotes."""
    parts: list[str] = []
    current: list[str] = []
    in_quotes = False
    quote_char = ""
    for ch in arg_str:
        if in_quotes:
            current.append(ch)
            if ch == quote_char:
                in_quotes = False
        elif ch in ('"', "'"):
            in_quotes = True
            quote_char = ch
            current.append(ch)
        elif ch == ",":
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def parse_macros(source_text: str, source_path: str = "templates/_macros.tpl") -> dict[str, MacroDef]:
    """Parse macro declarations from a `_macros.tpl` source string.

    Syntax:
        macro <name>(<param>: <Type> = <default>, ...):
            <body line>
        endmacro

    Args:
        source_text: Raw content of `_macros.tpl`.
        source_path: Relative path used for `defined_at` locations.

    Returns:
        Mapping from macro name to `MacroDef`.
    """
    lines = lex_lines(source_text, source_path=source_path)
    macros: dict[str, MacroDef] = {}
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if not line.stripped or line.stripped.startswith("#"):
            idx += 1
            continue
        if line.stripped.startswith("macro ") and line.stripped.endswith(":"):
            header = line.stripped[len("macro ") : -1].strip()
            open_paren = header.index("(")
            close_paren = header.rindex(")")
            macro_name = header[:open_paren].strip()
            params_str = header[open_paren + 1 : close_paren].strip()
            loc = f"{source_path}:{line.line_no}"

            params: list[MacroParam] = []
            if params_str:
                for item in _split_top_level_commas(params_str):
                    if "=" in item:
                        decl_part, default_raw = item.split("=", 1)
                        default_val: str | None = default_raw.strip()
                    else:
                        decl_part = item
                        default_val = None
                    p_name, p_type = decl_part.split(":", 1)
                    params.append(
                        MacroParam(
                            name=p_name.strip(),
                            type_name=p_type.strip(),
                            default_raw=default_val,
                            defined_at=loc,
                        )
                    )

            body_lines: list[str] = []
            idx += 1
            while idx < len(lines) and lines[idx].stripped != "endmacro":
                body_lines.append(lines[idx].stripped)
                idx += 1
            macros[macro_name] = MacroDef(
                name=macro_name,
                params=params,
                body_template="\n".join(body_lines),
                defined_at=loc,
            )
        idx += 1
    return macros


def parse_template(source_text: str, source_path: str) -> TemplateAST:
    """Parse a `.tpl` template source string into a `TemplateAST`.

    Lines whose first non-space character is `@` are parsed as macro calls
    `@name(arg1=val1, ...)`. Comment lines starting with `#` are ignored.

    Args:
        source_text: Raw `.tpl` file contents.
        source_path: Relative file path used in diagnostics.

    Returns:
        Parsed TemplateAST.
    """
    nodes: list[TextNode | MacroCallNode] = []
    for line in lex_lines(source_text, source_path=source_path):
        if not line.stripped or line.stripped.startswith("#"):
            continue
        if line.stripped.startswith("@"):
            call_expr = line.stripped[1:].strip()
            open_paren = call_expr.index("(")
            close_paren = call_expr.rindex(")")
            macro_name = call_expr[:open_paren].strip()
            args_str = call_expr[open_paren + 1 : close_paren].strip()
            raw_args: dict[str, str] = {}
            if args_str:
                for part in _split_top_level_commas(args_str):
                    k, v = part.split("=", 1)
                    raw_args[k.strip()] = v.strip()
            nodes.append(
                MacroCallNode(
                    macro_name=macro_name,
                    raw_args=raw_args,
                    source_path=source_path,
                    line_no=line.line_no,
                    col_no=line.indent_col,
                )
            )
        else:
            nodes.append(TextNode(text=line.raw_text, source_path=source_path, line_no=line.line_no))
    return TemplateAST(source_path=source_path, nodes=nodes)
