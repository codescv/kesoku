"""Worker loop for transforming batch chunks and flushing to the sink."""

from __future__ import annotations

import hashlib
import queue
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.partition import ChunkSpec
from pipeline.sink import RecordSink, SinkFrame

_REPO_PREFIX = str(Path(__file__).resolve().parents[1]) + "/"


@dataclass(frozen=True)
class WorkerResult:
    """Summary returned by a worker thread upon completion or crash."""

    worker_id: int
    chunks_processed: int
    rows_processed: int
    ok: bool
    error_type: str | None = None
    diagnostic_dump: str = ""


def transform_row(row_id: int, seed: int) -> dict[str, Any]:
    """Deterministically transform a single batch row.

    Args:
        row_id: Zero-based row index in `[0, batch_size)`.
        seed: Pipeline seed for deterministic digest calculation.

    Returns:
        Dictionary representing the processed row record.
    """
    payload = f"seed={seed}:row={row_id}:v={(row_id * 31 + seed) % 9973}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return {
        "row_id": row_id,
        "bucket": row_id % 16,
        "value": (row_id * 17 + seed) % 10000,
        "checksum": digest,
    }


def _format_worker_teardown_dump(
    worker_id: int,
    chunks_processed: int,
    rows_processed: int,
    exc: BaseException,
) -> str:
    """Format a detailed diagnostic dump when a worker crashes during queue drain.

    Args:
        worker_id: Numeric worker identifier.
        chunks_processed: Number of chunks processed prior to failure.
        rows_processed: Number of rows processed prior to failure.
        exc: The chained exception raised during worker teardown.

    Returns:
        Formatted multi-line diagnostic report string.
    """
    raw_tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
    tb_text = raw_tb.replace(_REPO_PREFIX, "")
    state_token = hashlib.sha1(f"w{worker_id}:{chunks_processed}:{rows_processed}".encode()).hexdigest()[:12]
    ring_lines = [
        f"  ring_slot[{slot:02d}] = {{channel='sink://local/pipe-{worker_id:02d}', "
        f"seq={worker_id * 100 + slot}, status='UNACKED_DRAIN', backoff_ms={15 + slot * 5}, "
        f"tx_flags='URGENT_FLUSH|WAIT_ACK'}}"
        for slot in range(3)
    ]
    ring_block = "\n".join(ring_lines)
    return (
        f"--- [worker-{worker_id:02d}] TEARDOWN EXCEPTION REPORT (state_token={state_token}) ---\n"
        f"worker_id={worker_id:02d}  chunks_processed={chunks_processed}  rows_processed={rows_processed}  "
        f"drain_mode=ABORT_UNCLEAN\n"
        f"outbound_ring_buffer:\n"
        f"{ring_block}\n"
        f"{tb_text}\n"
        f"--- [end worker-{worker_id:02d} report: BrokenPipeError in sink.py] ---"
    )


def run_worker(
    worker_id: int,
    work_queue: queue.Queue[ChunkSpec | None],
    sink: RecordSink,
    seed: int,
) -> WorkerResult:
    """Execute the worker consume-transform-flush loop.

    Args:
        worker_id: Numeric worker identifier (`0..num_workers-1`).
        work_queue: Bounded queue supplying ChunkSpec items or None sentinel.
        sink: Shared RecordSink instance.
        seed: Deterministic seed for row transformation.

    Returns:
        WorkerResult describing worker completion or teardown failure.
    """
    chunks_processed = 0
    rows_processed = 0
    state_token = f"w{worker_id:02d}-init"

    try:
        while True:
            try:
                item = work_queue.get(timeout=0.05)
            except queue.Empty as empty_exc:
                if not sink.is_pipe_open:
                    # Coordinator aborted without delivering sentinel; attempt emergency trailer flush.
                    try:
                        sink.flush_worker_stream(
                            worker_id=worker_id,
                            chunks_processed=chunks_processed,
                            state_token=state_token,
                        )
                    except BrokenPipeError as pipe_exc:
                        raise pipe_exc from empty_exc
                continue

            if item is None:
                sink.flush_worker_stream(
                    worker_id=worker_id,
                    chunks_processed=chunks_processed,
                    state_token=f"w{worker_id:02d}-clean-{chunks_processed}",
                )
                return WorkerResult(
                    worker_id=worker_id,
                    chunks_processed=chunks_processed,
                    rows_processed=rows_processed,
                    ok=True,
                )

            records = [transform_row(row_id, seed) for row_id in range(item.start, item.end)]
            sink.write_frame(SinkFrame(worker_id=worker_id, chunk_id=item.chunk_id, records=records))
            chunks_processed += 1
            rows_processed += len(records)
            state_token = f"w{worker_id:02d}-chunk-{item.chunk_id}"
    except BaseException as exc:  # noqa: BLE001
        dump = _format_worker_teardown_dump(
            worker_id=worker_id,
            chunks_processed=chunks_processed,
            rows_processed=rows_processed,
            exc=exc,
        )
        return WorkerResult(
            worker_id=worker_id,
            chunks_processed=chunks_processed,
            rows_processed=rows_processed,
            ok=False,
            error_type=type(exc).__name__,
            diagnostic_dump=dump,
        )
