"""Macro expansion and default argument binding for `tmplc`."""

from __future__ import annotations

from dataclasses import dataclass, field

from tmplc.parser import MacroCallNode, MacroDef, TemplateAST, TextNode
from tmplc.typecheck import TypedValue, infer_raw_literal


@dataclass(frozen=True)
class IRNode:
    """A single node in the Expanded Intermediate Representation."""

    kind: str
    source_path: str
    line_no: int
    col_no: int = 1
    text: str = ""
    macro_name: str = ""
    macro_def: MacroDef | None = None
    bound_args: dict[str, TypedValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ExpandedIR:
    """Expanded intermediate representation for a single template module."""

    source_path: str
    nodes: list[IRNode]


def bind_defaults(
    macro: MacroDef,
    explicit_args: dict[str, TypedValue],
) -> dict[str, TypedValue]:
    """Bind omitted macro parameters to their declared default values.

    Args:
        macro: Target MacroDef containing parameter declarations.
        explicit_args: Arguments explicitly supplied at the call site.

    Returns:
        Complete mapping from parameter name to bound `TypedValue`.

    Raises:
        KeyError: If a required parameter with no default is omitted.
    """
    bound: dict[str, TypedValue] = dict(explicit_args)
    for param in macro.params:
        if param.name in bound:
            continue
        if param.default_raw is None:
            raise KeyError(
                f"Macro '{macro.name}' missing required argument '{param.name}' (declared at {param.defined_at})"
            )
        raw_val = param.default_raw.strip()
        if len(raw_val) >= 2 and raw_val[0] == raw_val[-1] and raw_val[0] in ('"', "'"):
            raw_val = raw_val[1:-1]
        bound[param.name] = TypedValue(
            type_name="Str",
            value=raw_val,
            is_default=True,
            defined_at=param.defined_at,
        )
    return bound


def expand_template(ast: TemplateAST, macros: dict[str, MacroDef]) -> ExpandedIR:
    """Expand all macro call nodes in `ast` against `macros`.

    Args:
        ast: Parsed TemplateAST.
        macros: Registered MacroDef dictionary from `_macros.tpl`.

    Returns:
        ExpandedIR containing text nodes and bound macro call nodes.

    Raises:
        KeyError: If a template references an unknown macro or omits a required argument.
    """
    ir_nodes: list[IRNode] = []
    for node in ast.nodes:
        if isinstance(node, TextNode):
            ir_nodes.append(
                IRNode(
                    kind="text",
                    source_path=node.source_path,
                    line_no=node.line_no,
                    col_no=1,
                    text=node.text,
                )
            )
        elif isinstance(node, MacroCallNode):
            if node.macro_name not in macros:
                raise KeyError(
                    f"{node.source_path}:{node.line_no}:{node.col_no}: unknown macro '{node.macro_name}'"
                )
            macro_def = macros[node.macro_name]
            explicit_typed = {k: infer_raw_literal(v) for k, v in node.raw_args.items()}
            bound = bind_defaults(macro_def, explicit_typed)
            ir_nodes.append(
                IRNode(
                    kind="macro_call",
                    source_path=node.source_path,
                    line_no=node.line_no,
                    col_no=node.col_no,
                    macro_name=node.macro_name,
                    macro_def=macro_def,
                    bound_args=bound,
                )
            )
    return ExpandedIR(source_path=ast.source_path, nodes=ir_nodes)
