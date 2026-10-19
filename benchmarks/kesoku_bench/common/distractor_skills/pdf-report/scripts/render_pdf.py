"""Dummy PDF renderer for the pdf-report skill."""

import argparse


def main() -> int:
    """Parse arguments and pretend to render the report."""
    parser = argparse.ArgumentParser(prog="render_pdf")
    parser.add_argument("source")
    parser.add_argument("--out", default="report.pdf")
    parser.add_argument("--title", default="Report")
    parser.add_argument("--toc", action="store_true")
    parser.add_argument("--landscape", action="store_true")
    args = parser.parse_args()
    print(f"would render {args.source} -> {args.out} (title={args.title!r}, toc={args.toc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
