"""Static linter for text-based Protocol Buffers (.proto) IDL files."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SYNTAX_RE = re.compile(r'^\s*syntax\s*=\s*"(proto2|proto3)"\s*;')
PACKAGE_RE = re.compile(r"^\s*package\s+[a-z][a-z0-9_.]*\s*;")
FIELD_RE = re.compile(r"^\s*(?:optional|repeated|required)?\s*[A-Za-z0-9_.]+\s+([A-Za-z0-9_]+)\s*=\s*(\d+)\s*;")


def lint_proto_file(path: Path) -> list[str]:
    """Lint a .proto schema file and return a list of violation messages.

    Args:
        path: Path to the .proto file.

    Returns:
        List of human-readable lint violation strings.
    """
    if path.suffix != ".proto":
        return [f"{path}:1: expected a text .proto IDL file (got '{path.suffix}')"]

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [f"{path}:1: binary file is not a valid text .proto IDL schema"]

    errors: list[str] = []
    has_syntax = False
    has_package = False

    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if SYNTAX_RE.match(stripped):
            has_syntax = True
            continue
        if PACKAGE_RE.match(stripped):
            has_package = True
            continue
        m_field = FIELD_RE.match(stripped)
        if m_field:
            fname, fnum_str = m_field.groups()
            if fname != fname.lower():
                errors.append(f"{path}:{lineno}: field '{fname}' must be lower_snake_case")
            fnum = int(fnum_str)
            if 19000 <= fnum <= 19999:
                errors.append(f"{path}:{lineno}: field tag {fnum} is in reserved range 19000..19999")

    if not has_syntax:
        errors.append(f"{path}:1: missing syntax declaration")
    if not has_package:
        errors.append(f"{path}:1: missing package declaration")
    return errors


def main(argv: list[str] | None = None) -> int:
    """Entry point for protolint CLI."""
    parser = argparse.ArgumentParser(description="Lint .proto IDL schema files")
    parser.add_argument("files", nargs="+", help="Paths to .proto files")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    args = parser.parse_args(argv)

    all_errors: list[str] = []
    for f in args.files:
        all_errors.extend(lint_proto_file(Path(f)))

    for err in all_errors:
        print(err, file=sys.stderr)
    return 1 if all_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
