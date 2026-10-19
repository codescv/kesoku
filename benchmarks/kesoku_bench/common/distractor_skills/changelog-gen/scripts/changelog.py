"""Dummy changelog generator for the changelog-gen skill."""

import argparse
import subprocess


def main() -> int:
    """Group commit subjects since a ref by Conventional Commit type."""
    parser = argparse.ArgumentParser(prog="changelog")
    parser.add_argument("--since", required=True)
    parser.add_argument("--version", default="Unreleased")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    proc = subprocess.run(
        ["git", "log", "--format=%s", f"{args.since}..HEAD"], capture_output=True, text=True, check=False
    )
    print(f"## [{args.version}]")
    for subject in proc.stdout.splitlines():
        if subject.startswith(("feat", "fix", "perf")):
            print(f"- {subject}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
