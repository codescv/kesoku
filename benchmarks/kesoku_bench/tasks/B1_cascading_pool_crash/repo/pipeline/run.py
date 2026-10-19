"""Coordinator entry point for running the parallel batch pipeline."""

from __future__ import annotations

import argparse
import queue
import sys
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from pipeline.config import PipelineConfig, load_pipeline_config
from pipeline.partition import ChunkSpec, plan
from pipeline.sink import RecordSink
from pipeline.worker import WorkerResult, run_worker


def execute_pipeline(cfg: PipelineConfig) -> tuple[int, str]:
    """Run the batch pipeline with a ThreadPoolExecutor and return `(exit_code, report)`.

    Args:
        cfg: Pipeline configuration.

    Returns:
        Tuple of `(exit_code, full_log_output)`. Exit code is 0 iff all expected
        rows were partitioned, processed, and written without error.
    """
    lines: list[str] = [
        "=" * 80,
        f"[pipeline] starting batch coordinator (batch_size={cfg.batch_size}, "
        f"chunk_size={cfg.chunk_size}, workers={cfg.num_workers})",
        "=" * 80,
    ]

    work_queue: queue.Queue[ChunkSpec | None] = queue.Queue(maxsize=max(cfg.queue_size, cfg.num_workers * 2))
    sink = RecordSink(output_path=cfg.output_path, expected_rows=cfg.batch_size)
    planner_error: BaseException | None = None

    with ThreadPoolExecutor(max_workers=cfg.num_workers, thread_name_prefix="pipeline-worker") as pool:
        futures: list[Future[WorkerResult]] = [
            pool.submit(run_worker, worker_id, work_queue, sink, cfg.seed)
            for worker_id in range(cfg.num_workers)
        ]

        try:
            chunks = plan(cfg.batch_size, cfg.chunk_size)
            for chunk in chunks:
                work_queue.put(chunk)
            for _ in range(cfg.num_workers):
                work_queue.put(None)
        except BaseException as exc:  # noqa: BLE001
            planner_error = exc
            repo_prefix = str(Path(__file__).resolve().parents[1]) + "/"
            tb_str = (
                "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                .strip()
                .replace(repo_prefix, "")
            )
            lines.append("[Worker-0 / PartitionDispatcher] fatal error while planning batch partitions:")
            lines.append(tb_str)
            lines.append("[coordinator] aborting sink channel and draining worker pool...")
            sink.abort_pipe()

        results: list[WorkerResult] = [fut.result() for fut in futures]

    results.sort(key=lambda r: r.worker_id)
    crashed_workers = [r for r in results if not r.ok]
    for res in crashed_workers:
        lines.append(res.diagnostic_dump)

    if planner_error is not None or crashed_workers:
        crash_count = len(crashed_workers)
        lines.append(
            f"Summary: pipeline FAILED: {crash_count} worker(s) crashed (BrokenPipeError in sink.py)"
        )
        return 1, "\n".join(lines) + "\n"

    rows_written = sink.finalize_and_write()
    if rows_written != cfg.batch_size:
        lines.append(
            f"Summary: pipeline FAILED: expected {cfg.batch_size} rows, wrote {rows_written} rows"
        )
        return 1, "\n".join(lines) + "\n"

    total_chunks = sum(r.chunks_processed for r in results)
    lines.append(
        f"Summary: pipeline SUCCEEDED: {rows_written} rows processed across {total_chunks} chunks"
    )
    return 0, "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `python -m pipeline.run`.

    Args:
        argv: Optional command-line argument list.

    Returns:
        Process exit code (0 on success, 1 on failure).
    """
    parser = argparse.ArgumentParser(description="Run the batch processing pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("pipeline.toml"),
        help="Path to the pipeline TOML config file.",
    )
    args = parser.parse_args(argv)
    cfg = load_pipeline_config(args.config)
    code, report = execute_pipeline(cfg)
    sys.stdout.write(report)
    sys.stdout.flush()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
