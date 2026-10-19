"""HTML code emitter and verbose IR dump formatter for `tmplc`."""

from __future__ import annotations

from tmplc.expand import ExpandedIR, IRNode


def emit_html(ir: ExpandedIR) -> str:
    """Render a typechecked `ExpandedIR` into an HTML string.

    Args:
        ir: Validated ExpandedIR module.

    Returns:
        Rendered HTML output string.
    """
    out_lines: list[str] = []
    for node in ir.nodes:
        if node.kind == "text":
            out_lines.append(node.text)
        elif node.kind == "macro_call" and node.macro_def is not None:
            rendered = node.macro_def.body_template
            for k, typed_val in node.bound_args.items():
                val_str = ("true" if typed_val.value else "false") if typed_val.type_name == "Bool" else str(
                    typed_val.value
                )
                rendered = rendered.replace("{" + k + "}", val_str)
            out_lines.append(rendered)
    return "\n".join(out_lines) + "\n"


def _format_ir_node(module_idx: int, node_idx: int, node: IRNode) -> list[str]:
    """Format a single IRNode into detailed diagnostic lines for bug reports."""
    loc = f"{node.source_path}:{node.line_no}:{node.col_no}"
    if node.kind == "text":
        preview = node.text.strip()[:64]
        return [
            f"  [IR#{module_idx:02d}.{node_idx:03d}] kind=TextFragment loc={loc} "
            f"len={len(node.text)} text={preview!r}",
            f"    attr.scope='static_dom'  attr.escape_mode='raw'  attr.source_line={node.line_no}",
        ]

    if node.macro_def is not None:
        ordered_keys = [p.name for p in node.macro_def.params if p.name in node.bound_args]
    else:
        ordered_keys = list(node.bound_args.keys())
    arg_pairs = ", ".join(f"{k}={node.bound_args[k].format_ir()}" for k in ordered_keys)
    call_repr = f"{node.macro_name}({arg_pairs})"
    lines = [
        f"  [IR#{module_idx:02d}.{node_idx:03d}] kind=MacroExpansion loc={loc} call={call_repr}",
    ]
    if node.macro_def is not None:
        for param in node.macro_def.params:
            bval = node.bound_args.get(param.name)
            if bval is not None:
                status = "OK" if bval.type_name == param.type_name else "TYPE_MISMATCH"
                lines.append(
                    f"    arg[{param.name}]: declared={param.type_name} actual={bval.type_name} "
                    f"val={bval.format_ir()} status={status} call_expr={call_repr}"
                )
    return lines


def format_ir_dump(modules: list[ExpandedIR], failing_ir: ExpandedIR) -> str:
    """Format a ~25K-char expanded IR dump across compiled modules and `failing_ir`.

    Args:
        modules: Previously expanded IR modules in the build session.
        failing_ir: The ExpandedIR module where typechecking failed.

    Returns:
        Multi-line IR diagnostic dump string.
    """
    lines: list[str] = ["— expanded IR dump (for bug reports) —"]
    all_irs = [*modules, failing_ir]
    for mod_idx, ir in enumerate(all_irs, start=1):
        lines.append(f"=== Module[{mod_idx:02d}] source={ir.source_path} (nodes={len(ir.nodes)}) ===")
        for node_idx, node in enumerate(ir.nodes, start=1):
            lines.extend(_format_ir_node(mod_idx, node_idx, node))
        for pass_no in range(1, 5):
            lines.append(
                f"  [pass_trace:{ir.source_path}:stage_{pass_no:02d}] "
                f"symbol_table_entries={len(ir.nodes) * 3 + pass_no} "
                f"const_pool_slots={pass_no * 4} verification_state='LOWERED_IR_FRAME'"
            )

    # Emit detailed call-site tail for the failing module at the end of the dump.
    lines.append(f"=== Failing Call-Site Lowering Frame ({failing_ir.source_path}) ===")
    for node in failing_ir.nodes:
        if node.kind == "macro_call":
            if node.macro_def is not None:
                ordered_keys = [p.name for p in node.macro_def.params if p.name in node.bound_args]
            else:
                ordered_keys = list(node.bound_args.keys())
            arg_pairs = ", ".join(f"{k}={node.bound_args[k].format_ir()}" for k in ordered_keys)
            loc = f"{node.source_path}:{node.line_no}:{node.col_no}"
            lines.append(
                f"  lowered_call @ {loc} -> {node.macro_name}({arg_pairs}) "
                f"[inspect call site in {node.source_path} line {node.line_no}]"
            )
    return "\n".join(lines)
