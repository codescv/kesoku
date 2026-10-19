"""Visible unit tests for tmplc lexing, parsing, coercion, and explicit-arg expansion."""

from __future__ import annotations

from pathlib import Path

import pytest
from tmplc.emit import emit_html
from tmplc.expand import expand_template
from tmplc.parser import MacroCallNode, parse_macros, parse_template
from tmplc.typecheck import TypeMismatchError, check_expanded_ir, coerce_literal


def test_lexer_and_parser() -> None:
    """Verify parsing of macro declarations and template call nodes."""
    macro_src = (
        "macro chip(text: Str = \"tag\", count: Int = 1):\n"
        "    <span data-count=\"{count}\">{text}</span>\n"
        "endmacro\n"
    )
    macros = parse_macros(macro_src, source_path="templates/_macros.tpl")
    assert "chip" in macros
    m = macros["chip"]
    assert len(m.params) == 2
    assert m.params[0].name == "text"
    assert m.params[0].type_name == "Str"
    assert m.params[1].name == "count"
    assert m.params[1].type_name == "Int"
    assert m.params[1].default_raw == "1"

    tpl_src = "<div>\n    @chip(text=\"ok\", count=5)\n</div>\n"
    ast = parse_template(tpl_src, source_path="templates/sample.tpl")
    assert len(ast.nodes) == 3
    call_node = ast.nodes[1]
    assert isinstance(call_node, MacroCallNode)
    assert call_node.macro_name == "chip"
    assert call_node.line_no == 2
    assert call_node.col_no == 5


def test_coerce_literal() -> None:
    """Verify coerce_literal converts raw literal tokens into TypedValue instances."""
    v_int = coerce_literal("42", "Int", is_default=True)
    assert v_int.type_name == "Int"
    assert v_int.value == 42
    assert v_int.is_default is True

    v_bool = coerce_literal("true", "Bool")
    assert v_bool.type_name == "Bool"
    assert v_bool.value is True

    v_float = coerce_literal("3.5", "Float")
    assert v_float.type_name == "Float"
    assert v_float.value == 3.5

    v_str = coerce_literal('"Dashboard"', "Str")
    assert v_str.type_name == "Str"
    assert v_str.value == "Dashboard"

    with pytest.raises(TypeMismatchError):
        coerce_literal('"42"', "Int")


def test_expand_explicit_macro_args() -> None:
    """Verify expanding and emitting a template when all macro args are passed explicitly."""
    macro_src = (
        "macro grid(cols: Int = 3, gap: Int = 12, title: Str = \"Dashboard\"):\n"
        "    <section class=\"grid-{cols}-{gap}\">{title}</section>\n"
        "endmacro\n"
    )
    macros = parse_macros(macro_src)
    ast = parse_template('@grid(cols=4, gap=8, title="Overview")\n', source_path="templates/page.tpl")
    ir = expand_template(ast, macros)
    check_expanded_ir(ir)
    html = emit_html(ir)
    assert '<section class="grid-4-8">Overview</section>' in html


def test_build_explicit_templates() -> None:
    """Verify that templates with explicit macro arguments compile cleanly."""
    repo_root = Path(__file__).resolve().parents[1]
    tdir = repo_root / "templates"
    macros = parse_macros((tdir / "_macros.tpl").read_text(encoding="utf-8"))
    for name in ("cards.tpl", "footer.tpl", "header.tpl", "sidebar.tpl"):
        ast = parse_template((tdir / name).read_text(encoding="utf-8"), source_path=f"templates/{name}")
        ir = expand_template(ast, macros)
        check_expanded_ir(ir)
        rendered = emit_html(ir)
        assert rendered.strip() != ""
