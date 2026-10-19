"""AST-based repository convention linter enforcing AGENTS.md rules 1-3."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def check_library_file(py_path: Path, repo_root: Path) -> list[str]:
    """Check a library module under textutils/ for Google docstrings and no print() calls.

    Args:
        py_path: Path to the Python module.
        repo_root: Root directory of the repository.

    Returns:
        List of lint error strings.
    """
    rel = py_path.relative_to(repo_root)
    source = py_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(rel))
    errors: list[str] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            doc = ast.get_docstring(node) or ""
            has_params = bool(node.args.posonlyargs or node.args.args or node.args.kwonlyargs)
            if has_params and "Args:" not in doc:
                errors.append(
                    f"{rel}:{node.lineno}: public function '{node.name}' docstring missing 'Args:' section"
                )
            if "Returns:" not in doc:
                errors.append(
                    f"{rel}:{node.lineno}: public function '{node.name}' docstring missing 'Returns:' section"
                )

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                errors.append(
                    f"{rel}:{node.lineno}: forbidden print() call in library code (use logging instead)"
                )
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "builtins"
                and node.func.attr == "print"
            ):
                errors.append(
                    f"{rel}:{node.lineno}: forbidden builtins.print() call in library code (use logging instead)"
                )

    return errors


def check_issue_regression_tests(repo_root: Path) -> list[str]:
    """Verify every issue under issues/<id>.md has a corresponding tests/regression/test_<id>.py.

    Args:
        repo_root: Root directory of the repository.

    Returns:
        List of lint error strings.
    """
    issues_dir = repo_root / "issues"
    errors: list[str] = []
    if not issues_dir.is_dir():
        return errors

    for issue_file in sorted(issues_dir.glob("*.md")):
        issue_id = issue_file.stem
        reg_file = repo_root / "tests" / "regression" / f"test_{issue_id}.py"
        if not reg_file.is_file():
            errors.append(
                f"Missing regression test file 'tests/regression/test_{issue_id}.py' for issue #{issue_id}"
            )
            continue

        tree = ast.parse(reg_file.read_text(encoding="utf-8"), filename=str(reg_file))
        test_funcs = [
            n.name
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")
        ]
        if not test_funcs:
            errors.append(
                f"tests/regression/test_{issue_id}.py must define at least one test_* function"
            )

    return errors


def lint_repo(repo_root: Path | None = None) -> list[str]:
    """Run all AGENTS.md AST convention checks across the repository.

    Args:
        repo_root: Optional root directory path. Defaults to parent of scripts/.

    Returns:
        List of all detected convention violations.
    """
    root = repo_root if repo_root is not None else Path(__file__).resolve().parent.parent
    errors: list[str] = []

    pkg_dir = root / "textutils"
    for py_path in sorted(pkg_dir.rglob("*.py")):
        errors.extend(check_library_file(py_path, root))

    errors.extend(check_issue_regression_tests(root))
    return errors


def main() -> int:
    """CLI entry point for scripts/lint_conventions.py."""
    errors = lint_repo()
    if errors:
        for err in errors:
            print(f"[CONVENTION ERROR] {err}", file=sys.stderr)
        return 1
    print("All repository conventions satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
