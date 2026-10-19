"""Static inspection and module complexity report for svc subsystems."""

from __future__ import annotations

import ast
from pathlib import Path


def run_lint(pkg_dir: Path | None = None) -> str:
    """Inspect all Python modules under svc/ and emit a detailed diagnostic dump."""
    if pkg_dir is None:
        pkg_dir = Path(__file__).resolve().parent

    lines: list[str] = [
        "========================================================================",
        "SVC SUBSYSTEM STATIC AUDIT & METRICS REPORT",
        "========================================================================",
    ]

    for py_path in sorted(pkg_dir.rglob("*.py")):
        rel = py_path.relative_to(pkg_dir.parent)
        src = py_path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(rel))
        funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        lines.append(
            f"\n[MODULE] {rel} | bytes={len(src)} lines={len(src.splitlines())} "
            f"classes={classes} functions={funcs}"
        )
        for idx, raw_line in enumerate(src.splitlines(), start=1):
            stripped = raw_line.strip()
            if stripped and not stripped.startswith("#"):
                lines.append(f"  {rel}:{idx:03d} | {stripped[:72]}")

    lines.append("\n========================================================================")
    lines.append("SVC STATIC AUDIT FINISHED: 0 syntax errors.")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point for ``python -m svc.lint``."""
    print(run_lint())


if __name__ == "__main__":
    main()
