"""Command line entry point: ``uv run python -m dagrun.cli <pipeline.py>``.

The pipeline file must define ``build() -> dagrun.Scheduler``. The CLI runs it,
streaming the event log, and exits with 0 (all nodes succeeded), 1 (some node
failed or was skipped) or 2 (``--timeout`` expired).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from .scheduler import RunResult, Scheduler


def load_pipeline(path: Path) -> ModuleType:
    """Import a pipeline file as a module.

    Args:
        path: Path of the Python file defining ``build()``.

    Returns:
        The imported module.
    """
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "build", None)):
        raise SystemExit(f"{path} does not define build()")
    return module


async def _run(scheduler: Scheduler, limit_s: float | None) -> RunResult:
    """Run the scheduler, optionally bounded by ``limit_s`` seconds."""
    if limit_s is None:
        return await scheduler.run()
    return await asyncio.wait_for(scheduler.run(), limit_s)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the pipeline and print a summary.

    Args:
        argv: Command line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(prog="dagrun", description="Run a dagrun pipeline file.")
    parser.add_argument("pipeline", type=Path, help="Python file defining build() -> Scheduler")
    parser.add_argument("--timeout", type=float, default=None, help="give up after this many seconds")
    args = parser.parse_args(argv)

    scheduler = load_pipeline(args.pipeline).build()
    scheduler.events.subscribe(lambda e: print(f"  {e.kind:<9} {e.node:<10} {e.data}", flush=True))
    try:
        result = asyncio.run(_run(scheduler, args.timeout))
    except TimeoutError:
        print(f"timeout: run did not finish within {args.timeout}s", file=sys.stderr)
        return 2
    print(result.summary())
    for name, node in result.nodes.items():
        extra = f" cause={node.cause}" if node.cause else ""
        print(f"  {name}: {node.state.value} attempts={node.attempts}{extra}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
