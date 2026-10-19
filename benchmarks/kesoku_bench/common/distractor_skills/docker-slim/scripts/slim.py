"""Dummy Dockerfile analyser for the docker-slim skill."""

import argparse


def main() -> int:
    """Count RUN layers in a Dockerfile and print a stub report."""
    parser = argparse.ArgumentParser(prog="slim")
    parser.add_argument("dockerfile")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--rewrite", default=None)
    args = parser.parse_args()
    with open(args.dockerfile, encoding="utf-8") as fh:
        runs = sum(1 for line in fh if line.lstrip().upper().startswith("RUN"))
    print(f"{args.dockerfile}: {runs} RUN layer(s); consider combining them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
