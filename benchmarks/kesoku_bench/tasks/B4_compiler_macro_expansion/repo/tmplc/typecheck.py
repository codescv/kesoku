"""Type checker and literal coercion utilities for `tmplc`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tmplc.expand import ExpandedIR


@dataclass(frozen=True)
class TypedValue:
    """A typed literal value bound to a macro parameter."""

    type_name: str
    value: Any
    is_default: bool = False
    defined_at: str | None = None

    def format_ir(self) -> str:
        """Render the value in IR syntax (`"3"` for Str, `3` for Int, etc.)."""
        if self.type_name == "Str":
            return f'"{self.value}"'
        if self.type_name == "Bool":
            return "true" if self.value else "false"
        return repr(self.value)


class TypeMismatchError(Exception):
    """Raised when a bound macro argument does not match its declared parameter type."""


def infer_raw_literal(raw: str) -> TypedValue:
    """Parse an explicit call-site literal token into a `TypedValue`.

    Args:
        raw: Raw argument token from a `@macro(...)` call site.

    Returns:
        TypedValue with inferred type (`Str`, `Bool`, `Int`, or `Float`).
    """
    token = raw.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ('"', "'"):
        return TypedValue(type_name="Str", value=token[1:-1])
    if token.lower() in ("true", "false"):
        return TypedValue(type_name="Bool", value=token.lower() == "true")
    try:
        return TypedValue(type_name="Int", value=int(token))
    except ValueError:
        pass
    try:
        return TypedValue(type_name="Float", value=float(token))
    except ValueError:
        return TypedValue(type_name="Str", value=token)


def coerce_literal(
    raw: str,
    target_type: str,
    *,
    is_default: bool = False,
    defined_at: str | None = None,
) -> TypedValue:
    """Coerce a raw literal string to `target_type` (`Int`, `Float`, `Bool`, `Str`).

    Args:
        raw: Raw source token string (e.g. `"3"`, `"true"`, `'"Dashboard"'`).
        target_type: Declared parameter type name (`"Int"`, `"Float"`, `"Bool"`, `"Str"`).
        is_default: Whether the value comes from a macro parameter default.
        defined_at: Optional `path:line` location where the default was declared.

    Returns:
        TypedValue matching `target_type`.

    Raises:
        TypeMismatchError: If `raw` cannot be coerced to `target_type`.
    """
    token = raw.strip()
    if target_type == "Int":
        if len(token) >= 2 and token[0] == token[-1] and token[0] in ('"', "'"):
            raise TypeMismatchError(f"Cannot coerce quoted string {token} to Int")
        try:
            return TypedValue(
                type_name="Int",
                value=int(token),
                is_default=is_default,
                defined_at=defined_at,
            )
        except ValueError as exc:
            raise TypeMismatchError(f"Cannot coerce {token!r} to Int") from exc

    if target_type == "Float":
        try:
            return TypedValue(
                type_name="Float",
                value=float(token),
                is_default=is_default,
                defined_at=defined_at,
            )
        except ValueError as exc:
            raise TypeMismatchError(f"Cannot coerce {token!r} to Float") from exc

    if target_type == "Bool":
        lowered = token.lower()
        if lowered not in ("true", "false"):
            raise TypeMismatchError(f"Cannot coerce {token!r} to Bool")
        return TypedValue(
            type_name="Bool",
            value=lowered == "true",
            is_default=is_default,
            defined_at=defined_at,
        )

    if target_type == "Str":
        if len(token) >= 2 and token[0] == token[-1] and token[0] in ('"', "'"):
            unquoted = token[1:-1]
        else:
            unquoted = token
        return TypedValue(
            type_name="Str",
            value=unquoted,
            is_default=is_default,
            defined_at=defined_at,
        )

    raise TypeMismatchError(f"Unsupported type {target_type!r}")


def check_expanded_ir(ir: ExpandedIR) -> None:
    """Validate that all bound macro arguments in `ir` match their declared parameter types.

    Args:
        ir: ExpandedIR produced by `expand_template`.

    Raises:
        TypeMismatchError: On the first argument whose bound `TypedValue.type_name`
            differs from the parameter's declared `type_name`.
    """
    for node in ir.nodes:
        if node.kind != "macro_call" or node.macro_def is None:
            continue
        for param in node.macro_def.params:
            bound_val = node.bound_args.get(param.name)
            if bound_val is None:
                raise TypeMismatchError(
                    f"{node.source_path}:{node.line_no}:{node.col_no}: "
                    f"missing required argument '{param.name}' in macro '{node.macro_name}'"
                )
            if bound_val.type_name != param.type_name:
                origin_detail = (
                    f" (default for arg '{param.name}' bound at {bound_val.defined_at})"
                    if bound_val.is_default and bound_val.defined_at
                    else ""
                )
                raise TypeMismatchError(
                    f"{node.source_path}:{node.line_no}:{node.col_no}: "
                    f"type mismatch in macro '{node.macro_name}': "
                    f"expected {param.type_name}, got {bound_val.type_name}{origin_detail}"
                )
