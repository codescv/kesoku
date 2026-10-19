"""Typed template and macro compiler package."""

from tmplc.cli import build_directory
from tmplc.expand import bind_defaults, expand_template
from tmplc.parser import MacroDef, MacroParam, parse_macros, parse_template
from tmplc.typecheck import TypedValue, TypeMismatchError, check_expanded_ir, coerce_literal

__all__ = [
    "MacroDef",
    "MacroParam",
    "TypeMismatchError",
    "TypedValue",
    "bind_defaults",
    "build_directory",
    "check_expanded_ir",
    "coerce_literal",
    "expand_template",
    "parse_macros",
    "parse_template",
]
