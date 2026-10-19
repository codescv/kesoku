"""Offline command line tools: ``uv run python -m netdiag.cli <command> ...``.

Commands:
    stats <dumpfile> [--json]   summarise a saved packet dump
    replay <dumpfile>   show what a replay of the dump would send (dry run)

The live session diagnostic is a separate entry point
(``uv run python -m netdiag.diagnose``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import replay
from .dump import DumpFormatError
from .stats import compute_stats, format_stats


def _cmd_stats(args: argparse.Namespace) -> int:
    """Print the summary statistics of a dump file (text or JSON)."""
    stats = compute_stats(args.dumpfile)
    print(json.dumps(stats) if args.json else format_stats(stats))
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    """Print the dry-run replay plan of a dump file."""
    p = replay.plan(args.dumpfile)
    print(p.hello)
    print(f"packets={p.packets} frames={p.frames} bytes={p.bytes}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the CLI."""
    parser = argparse.ArgumentParser(prog="netdiag", description="Offline netdiag tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    stats = sub.add_parser("stats", help="summarise a saved packet dump")
    stats.add_argument("dumpfile", type=Path)
    stats.add_argument("--json", action="store_true", help="print the statistics as a JSON object")
    stats.set_defaults(func=_cmd_stats)

    rep = sub.add_parser("replay", help="dry-run a replay of a saved packet dump")
    rep.add_argument("dumpfile", type=Path)
    rep.set_defaults(func=_cmd_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run the selected command.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, DumpFormatError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
