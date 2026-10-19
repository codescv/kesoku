"""Dummy flame graph profiler for the perf-flamegraph skill."""

import argparse


def main() -> int:
    """Print what would be profiled and where the SVG would go."""
    parser = argparse.ArgumentParser(prog="flame")
    parser.add_argument("--out", default="profile.svg")
    parser.add_argument("--min-pct", type=float, default=0.5)
    parser.add_argument("--collapsed", default=None)
    parser.add_argument("target", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    target = " ".join(t for t in args.target if t != "--")
    print(f"would profile `python {target}` -> {args.out} (min {args.min_pct}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
