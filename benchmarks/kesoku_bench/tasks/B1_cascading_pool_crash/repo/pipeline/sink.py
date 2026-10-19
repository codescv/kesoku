"""Buffered record sink for collecting and persisting worker chunk outputs."""

from __future__ import annotations

import errno
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SinkFrame:
    """A single chunk frame submitted by a worker to the sink."""

    worker_id: int
    chunk_id: int
    records: list[dict[str, Any]]


class RecordSink:
    """Thread-safe record sink that collects chunk frames and writes sorted JSONL."""

    def __init__(self, output_path: str | Path, expected_rows: int) -> None:
        """Initialize the sink for a target output path and expected row count.

        Args:
            output_path: Destination JSONL path.
            expected_rows: Total number of rows expected across all chunks.
        """
        self._output_path = Path(output_path)
        self._expected_rows = expected_rows
        self._lock = threading.Lock()
        self._frames: dict[int, list[dict[str, Any]]] = {}
        self._worker_heartbeats: dict[int, int] = {}
        self._pipe_open = True
        self._aborted = False

    @property
    def is_pipe_open(self) -> bool:
        """Return whether the downstream sink channel is open."""
        return self._pipe_open and not self._aborted

    def abort_pipe(self) -> None:
        """Close the sink pipe early when the upstream coordinator aborts."""
        with self._lock:
            self._aborted = True
            self._pipe_open = False

    def write_frame(self, frame: SinkFrame) -> None:
        """Record a processed chunk frame from a worker.

        Args:
            frame: The processed SinkFrame containing records for one chunk.

        Raises:
            BrokenPipeError: If the sink channel was closed or aborted.
        """
        with self._lock:
            if not self._pipe_open or self._aborted:
                raise BrokenPipeError(
                    errno.EPIPE,
                    f"Sink channel closed while worker-{frame.worker_id:02d} was writing chunk {frame.chunk_id}",
                )
            self._frames[frame.chunk_id] = list(frame.records)
            self._worker_heartbeats[frame.worker_id] = self._worker_heartbeats.get(frame.worker_id, 0) + len(
                frame.records
            )

    def flush_worker_stream(self, worker_id: int, chunks_processed: int, state_token: str) -> None:
        """Flush a worker's stream trailer to the downstream sink channel.

        Args:
            worker_id: Numeric worker identifier (`0..num_workers-1`).
            chunks_processed: Count of chunks completed by this worker.
            state_token: Diagnostic state token for the worker drain phase.

        Raises:
            BrokenPipeError: If the downstream sink pipe was closed due to abort.
        """
        with self._lock:
            if not self._pipe_open or self._aborted:
                raise BrokenPipeError(
                    errno.EPIPE,
                    f"Broken pipe in RecordSink.flush_worker_stream for worker-{worker_id:02d} "
                    f"(chunks_processed={chunks_processed}, token={state_token}): "
                    f"downstream collector closed descriptor before trailer ACK",
                )
            self._worker_heartbeats.setdefault(worker_id, 0)

    def finalize_and_write(self) -> int:
        """Sort all collected records by `row_id` and write them as JSONL.

        Returns:
            Total number of rows written to `output_path`.

        Raises:
            BrokenPipeError: If the sink was aborted before finalization.
        """
        with self._lock:
            if self._aborted:
                raise BrokenPipeError(errno.EPIPE, "Cannot finalize aborted RecordSink")
            self._pipe_open = False
            all_records: list[dict[str, Any]] = []
            for chunk_id in sorted(self._frames):
                all_records.extend(self._frames[chunk_id])
            all_records.sort(key=lambda r: int(r["row_id"]))

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._output_path.open("w", encoding="utf-8") as fh:
            for record in all_records:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        return len(all_records)
