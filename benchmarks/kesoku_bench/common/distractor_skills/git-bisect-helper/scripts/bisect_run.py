"""Dummy git bisect driver for the git-bisect-helper skill."""

import argparse


def main() -> int:
    """Print the git bisect commands that would be executed."""
    parser = argparse.ArgumentParser(prog="bisect_run")
    parser.add_argument("--good", required=True)
    parser.add_argument("--bad", default="HEAD")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    print(f"git bisect start {args.bad} {args.good}")
    print("git bisect run " + " ".join(c for c in args.command if c != "--"))
    print("git bisect reset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
