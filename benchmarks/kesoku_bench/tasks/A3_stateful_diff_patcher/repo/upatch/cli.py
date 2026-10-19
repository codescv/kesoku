"""Command-line interface for the upatch unified diff applicator."""

import argparse
import sys
from pathlib import Path

from upatch.apply import ApplyOptions, PatchApplyError, apply_patch_text
from upatch.io import read_patch_file, read_target_file, write_target_file
from upatch.parse import PatchParseError


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser for `upatch`."""
    parser = argparse.ArgumentParser(
        prog="upatch",
        description="Stateful unified diff parser and patch applicator.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    apply_cmd = subparsers.add_parser(
        "apply",
        help="Apply a unified diff patch file to a target file.",
    )
    apply_cmd.add_argument("patch_file", type=Path, help="Path to unified diff (.patch) file")
    apply_cmd.add_argument("target_file", type=Path, help="Path to file to be patched")
    apply_cmd.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write patched output to this file instead of stdout",
    )
    apply_cmd.add_argument(
        "-i",
        "--in-place",
        action="store_true",
        help="Modify target_file in place",
    )
    apply_cmd.add_argument(
        "--allow-offset",
        action="store_true",
        help="Allow scanning for hunks at shifted line offsets",
    )
    apply_cmd.add_argument(
        "--max-offset",
        type=int,
        default=200,
        help="Maximum offset distance in lines when --allow-offset is enabled",
    )
    apply_cmd.add_argument(
        "--fuzz",
        type=int,
        default=0,
        help="Maximum number of outer context lines to ignore (0..2)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `upatch` CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "apply":
        try:
            patch_text = read_patch_file(args.patch_file)
            target_text = read_target_file(args.target_file)
            opts = ApplyOptions(
                allow_offset=bool(args.allow_offset),
                max_offset=int(args.max_offset),
                max_fuzz=int(args.fuzz),
            )
            patched_text = apply_patch_text(target_text, patch_text, options=opts)
        except (PatchParseError, PatchApplyError, OSError) as exc:
            sys.stderr.write(f"upatch error: {exc}\n")
            return 1

        if args.output is not None:
            write_target_file(args.output, patched_text)
        elif args.in_place:
            write_target_file(args.target_file, patched_text)
        else:
            sys.stdout.buffer.write(patched_text.encode("utf-8"))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
