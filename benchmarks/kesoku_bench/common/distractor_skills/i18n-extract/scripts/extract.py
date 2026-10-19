"""Tiny gettext string extractor for the i18n-extract skill."""

import argparse
import re
from pathlib import Path

_CALL = re.compile(r"""\b_\(\s*["']([^"']+)["']\s*\)""")


def main() -> int:
    """Print msgid entries for every _("...") call found under the given paths."""
    parser = argparse.ArgumentParser(prog="extract")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", default="messages.pot")
    args = parser.parse_args()
    found: set[str] = set()
    for root in args.paths:
        for path in Path(root).rglob("*.py"):
            found.update(_CALL.findall(path.read_text(encoding="utf-8", errors="ignore")))
    for msgid in sorted(found):
        print(f'msgid "{msgid}"\nmsgstr ""\n')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
