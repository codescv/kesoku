"""CLI driver for building a directory of `.tpl` templates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tmplc.emit import emit_html, format_ir_dump
from tmplc.expand import ExpandedIR, expand_template
from tmplc.parser import parse_macros, parse_template
from tmplc.typecheck import TypeMismatchError, check_expanded_ir

COMPILE_ORDER: list[str] = [
    "cards.tpl",
    "footer.tpl",
    "header.tpl",
    "sidebar.tpl",
    "layout.tpl",
    "report.tpl",
]


def build_directory(template_dir: str | Path) -> tuple[int, str, dict[str, str]]:
    """Compile all `.tpl` templates in `template_dir`.

    Args:
        template_dir: Directory containing `_macros.tpl` and template files.

    Returns:
        Tuple of `(exit_code, stdout_log, rendered_outputs_by_filename)`.
    """
    tdir = Path(template_dir)
    macros_path = tdir / "_macros.tpl"
    rel_macros = f"{tdir.name}/_macros.tpl"
    macros = parse_macros(macros_path.read_text(encoding="utf-8"), source_path=rel_macros)

    available = [name for name in COMPILE_ORDER if (tdir / name).is_file()]
    for extra in sorted(p.name for p in tdir.glob("*.tpl") if p.name != "_macros.tpl"):
        if extra not in available:
            available.append(extra)

    total = len(available)
    log_lines: list[str] = [
        f"[tmplc] loading macro definitions from {rel_macros} ({len(macros)} macros)"
    ]
    completed_irs: list[ExpandedIR] = []
    rendered: dict[str, str] = {}

    for idx, filename in enumerate(available, start=1):
        rel_path = f"{tdir.name}/{filename}"
        src_text = (tdir / filename).read_text(encoding="utf-8")
        log_lines.append(f"[tmplc] [{idx}/{total}] parsing {rel_path} ... ok")
        ast = parse_template(src_text, source_path=rel_path)
        ir = expand_template(ast, macros)

        try:
            check_expanded_ir(ir)
        except TypeMismatchError as exc:
            log_lines.append(
                f"[tmplc] [{idx}/{total}] expanding {rel_path} ... ok ({len(ir.nodes)} nodes)"
            )
            log_lines.append(f"error: {exc}")
            log_lines.append(format_ir_dump(completed_irs, ir))
            log_lines.append("build failed (1 error)")
            return 1, "\n".join(log_lines) + "\n", rendered

        log_lines.append(
            f"[tmplc] [{idx}/{total}] expanding & typechecking {rel_path} ... ok ({len(ir.nodes)} nodes)"
        )
        completed_irs.append(ir)
        rendered[filename] = emit_html(ir)

    log_lines.append(f"build succeeded ({total} templates)")
    return 0, "\n".join(log_lines) + "\n", rendered


def main(argv: list[str] | None = None) -> int:
    """Entry point for `python -m tmplc build <template_dir>`."""
    parser = argparse.ArgumentParser(prog="tmplc", description="Compile typed templates.")
    sub = parser.add_subparsers(dest="command", required=True)
    build_p = sub.add_parser("build", help="Compile all templates in a directory.")
    build_p.add_argument("template_dir", type=Path, help="Path to the templates directory.")
    args = parser.parse_args(argv)

    code, log_output, _ = build_directory(args.template_dir)
    sys.stdout.write(log_output)
    sys.stdout.flush()
    return code
