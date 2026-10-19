"""Tiny CSV profiler for the csv-profiler skill."""

import argparse
import csv
from collections import Counter


def main() -> int:
    """Print row count and the most common values per column."""
    parser = argparse.ArgumentParser(prog="profile_csv")
    parser.add_argument("path")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--delimiter", default=",")
    args = parser.parse_args()
    with open(args.path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter=args.delimiter))
    print(f"rows: {len(rows)}")
    for col in rows[0].keys() if rows else []:
        common = Counter(r[col] for r in rows).most_common(args.top)
        print(f"{col}: {common}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
