"""Minimal EXPLAIN wrapper for SQLite (sql-explain skill)."""

import argparse
import sqlite3


def main() -> int:
    """Print the EXPLAIN QUERY PLAN rows for a query on a SQLite database."""
    parser = argparse.ArgumentParser(prog="explain")
    parser.add_argument("--dsn", default="sqlite:///:memory:")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("query")
    args = parser.parse_args()
    path = args.dsn.removeprefix("sqlite:///")
    with sqlite3.connect(path) as conn:
        try:
            for row in conn.execute("EXPLAIN QUERY PLAN " + args.query):
                print(row)
        except sqlite3.Error as exc:
            print(f"error: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
