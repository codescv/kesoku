"""Dummy Slack export digester for the slack-digest skill."""

import argparse
import json


def main() -> int:
    """Count messages in a Slack export file and print a stub digest."""
    parser = argparse.ArgumentParser(prog="digest")
    parser.add_argument("export")
    parser.add_argument("--max-items", type=int, default=10)
    parser.add_argument("--include-bots", action="store_true")
    args = parser.parse_args()
    with open(args.export, encoding="utf-8") as fh:
        messages = json.load(fh)
    print(f"# Digest ({len(messages)} messages, showing up to {args.max_items} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
