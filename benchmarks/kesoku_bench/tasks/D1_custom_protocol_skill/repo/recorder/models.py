"""In-memory data models and aggregation helpers for decoded telemetry samples."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TelemetrySample:
    """Represents a decoded telemetry sample from a session capture."""

    seq_no: int
    timestamp_s: int
    reading_tenths: int
    status_code: int
    channel: str


@dataclass(frozen=True)
class SessionStats:
    """Aggregated summary statistics across a telemetry session."""

    sample_count: int
    min_reading_c: float
    max_reading_c: float
    mean_reading_c: float
    contiguous_seq: bool


def summarize_samples(samples: list[TelemetrySample]) -> SessionStats:
    """Compute summary statistics over an ordered list of telemetry samples.

    Args:
        samples: List of decoded TelemetrySample instances.

    Returns:
        Aggregated SessionStats for the given samples.

    Raises:
        ValueError: If samples is empty.
    """
    if not samples:
        raise ValueError("Cannot summarize an empty sample list")

    readings_c = [s.reading_tenths / 10.0 for s in samples]
    expected_seqs = list(range(samples[0].seq_no, samples[0].seq_no + len(samples)))
    actual_seqs = [s.seq_no for s in samples]

    return SessionStats(
        sample_count=len(samples),
        min_reading_c=min(readings_c),
        max_reading_c=max(readings_c),
        mean_reading_c=round(sum(readings_c) / len(readings_c), 3),
        contiguous_seq=actual_seqs == expected_seqs,
    )
