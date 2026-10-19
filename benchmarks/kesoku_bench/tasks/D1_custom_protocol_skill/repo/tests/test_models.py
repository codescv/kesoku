"""Visible unit tests for the recorder aggregation models."""

from __future__ import annotations

import pytest
from recorder.models import TelemetrySample, summarize_samples


def test_summarize_contiguous_samples() -> None:
    """Verify summary statistics and contiguous sequence detection."""
    samples = [
        TelemetrySample(seq_no=4101, timestamp_s=1700000010, reading_tenths=220, status_code=0, channel="ch=1"),
        TelemetrySample(seq_no=4102, timestamp_s=1700000020, reading_tenths=235, status_code=0, channel="ch=2"),
        TelemetrySample(seq_no=4103, timestamp_s=1700000030, reading_tenths=250, status_code=0, channel="ch=3"),
    ]
    stats = summarize_samples(samples)
    assert stats.sample_count == 3
    assert stats.min_reading_c == 22.0
    assert stats.max_reading_c == 25.0
    assert stats.mean_reading_c == 23.5
    assert stats.contiguous_seq is True


def test_summarize_detects_sequence_gap() -> None:
    """Verify non-contiguous sequence numbers are flagged."""
    samples = [
        TelemetrySample(seq_no=4101, timestamp_s=1700000010, reading_tenths=210, status_code=0, channel="ch=1"),
        TelemetrySample(seq_no=4103, timestamp_s=1700000030, reading_tenths=230, status_code=0, channel="ch=3"),
    ]
    stats = summarize_samples(samples)
    assert stats.contiguous_seq is False


def test_summarize_empty_raises() -> None:
    """Verify empty sample list raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        summarize_samples([])
