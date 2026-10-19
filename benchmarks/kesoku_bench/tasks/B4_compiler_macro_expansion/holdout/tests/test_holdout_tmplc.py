"""Held-out evaluation tests for B4_compiler_macro_expansion."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tmplc.cli import build_directory
from tmplc.expand import bind_defaults
from tmplc.parser import parse_macros
from tmplc.typecheck import TypedValue


def test_bind_defaults_coerces_types() -> None:
    """Verify expand.bind_defaults coerces Int, Bool, Float, and Str default literals."""
    macro_src = (
        "macro widget(cols: Int = 3, compact: Bool = true, ratio: Float = 1.5, title: Str = \"Main\"):\n"
        "    <div data-cols=\"{cols}\" data-compact=\"{compact}\" data-ratio=\"{ratio}\">{title}</div>\n"
        "endmacro\n"
    )
    macros = parse_macros(macro_src, source_path="templates/_macros.tpl")
    bound = bind_defaults(macros["widget"], explicit_args={})

    assert bound["cols"] == TypedValue(
        type_name="Int",
        value=3,
        is_default=True,
        defined_at="templates/_macros.tpl:1",
    )
    assert isinstance(bound["cols"].value, int)

    assert bound["compact"] == TypedValue(
        type_name="Bool",
        value=True,
        is_default=True,
        defined_at="templates/_macros.tpl:1",
    )
    assert isinstance(bound["compact"].value, bool)

    assert bound["ratio"] == TypedValue(
        type_name="Float",
        value=1.5,
        is_default=True,
        defined_at="templates/_macros.tpl:1",
    )
    assert isinstance(bound["ratio"].value, float)

    assert bound["title"] == TypedValue(
        type_name="Str",
        value="Main",
        is_default=True,
        defined_at="templates/_macros.tpl:1",
    )


def test_build_all_templates() -> None:
    """Verify all 6 templates (including layout.tpl and report.tpl) compile with macro defaults."""
    repo_root = Path(__file__).resolve().parents[1]
    tdir = repo_root / "templates"

    # Guard against decoy fix of hard-coding cols=3 at the call site in layout.tpl / report.tpl.
    layout_src = (tdir / "layout.tpl").read_text(encoding="utf-8")
    report_src = (tdir / "report.tpl").read_text(encoding="utf-8")
    assert '@grid(gap=16, title="Main")' in layout_src
    assert '@grid(title="ReportSummaryGrid")' in report_src
    assert "@paginate()" in report_src

    code, log_output, rendered = build_directory(tdir)
    assert code == 0, f"build_directory failed with code {code}:\n{log_output[:1000]}"
    assert "build succeeded (6 templates)" in log_output
    assert set(rendered.keys()) == {
        "cards.tpl",
        "footer.tpl",
        "header.tpl",
        "sidebar.tpl",
        "layout.tpl",
        "report.tpl",
    }
    assert 'class="grid grid-cols-3 gap-16" data-title="Main"' in rendered["layout.tpl"]
    assert 'class="grid grid-cols-3 gap-12" data-title="ReportSummaryGrid"' in rendered["report.tpl"]
    assert 'class="pager" data-page-size="25" data-compact="false"' in rendered["report.tpl"]


def test_cli_build_templates() -> None:
    """Verify `python -m tmplc build templates/` exits 0 and reports 6 compiled templates."""
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "tmplc", "build", "templates/"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"CLI exited with {proc.returncode}:\n{combined[:1000]}"
    assert "build succeeded (6 templates)" in combined
    assert "error:" not in combined
