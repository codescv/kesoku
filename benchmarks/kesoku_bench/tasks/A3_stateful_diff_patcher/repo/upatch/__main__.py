"""Allow running upatch via `python -m upatch`."""

from upatch.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
