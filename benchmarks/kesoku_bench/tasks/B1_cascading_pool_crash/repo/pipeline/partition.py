"""Batch partitioning utilities for splitting row ranges into worker chunks.

The planner divides `[0, batch_size)` into contiguous `ChunkSpec` slices of at
most `chunk_size` rows. When `batch_size` is not divisible by `chunk_size`, the
final chunk is a ragged tail containing the remaining `batch_size % chunk_size`
rows so that every row in `[0, batch_size)` is covered exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkSpec:
    """Descriptor for a contiguous slice of batch rows `[start, end)`."""

    chunk_id: int
    start: int
    end: int

    @property
    def size(self) -> int:
        """Return the number of rows in this chunk."""
        return self.end - self.start


def plan(batch_size: int, chunk_size: int) -> list[ChunkSpec]:
    """Partition `[0, batch_size)` into contiguous chunks of up to `chunk_size` rows.

    When `batch_size` is not an exact multiple of `chunk_size`, the final chunk
    is smaller (`batch_size % chunk_size` rows). If `batch_size == 0`, returns
    an empty list.

    Args:
        batch_size: Total number of rows in the batch (must be >= 0).
        chunk_size: Maximum number of rows per chunk (must be > 0).

    Returns:
        Ordered list of ChunkSpec instances covering `[0, batch_size)`.

    Raises:
        ValueError: If `batch_size < 0` or `chunk_size <= 0`.
    """
    if batch_size < 0:
        raise ValueError(f"batch_size must be non-negative (got {batch_size})")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive (got {chunk_size})")
    if batch_size == 0:
        return []

    if batch_size % chunk_size != 0:
        raise ValueError(
            f"chunk_size must divide batch_size evenly (batch_size={batch_size}, chunk_size={chunk_size})"
        )

    num_chunks = batch_size // chunk_size
    chunks: list[ChunkSpec] = []
    for idx in range(num_chunks):
        start = idx * chunk_size
        end = start + chunk_size
        chunks.append(ChunkSpec(chunk_id=idx, start=start, end=end))
    return chunks
