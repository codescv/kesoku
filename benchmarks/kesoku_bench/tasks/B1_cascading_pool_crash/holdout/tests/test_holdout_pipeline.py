"""Held-out evaluation tests for B1_cascading_pool_crash."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pipeline.config import PipelineConfig
from pipeline.partition import ChunkSpec, plan
from pipeline.run import execute_pipeline


def test_partition_ragged_tail() -> None:
    """Verify that partition.plan creates a ragged final chunk when batch_size % chunk_size != 0."""
    chunks_1000 = plan(1000, 64)
    assert len(chunks_1000) == 16
    assert chunks_1000[0] == ChunkSpec(chunk_id=0, start=0, end=64)
    assert chunks_1000[14] == ChunkSpec(chunk_id=14, start=896, end=960)
    assert chunks_1000[15] == ChunkSpec(chunk_id=15, start=960, end=1000)
    assert chunks_1000[15].size == 40
    assert sum(c.size for c in chunks_1000) == 1000

    chunks_10 = plan(10, 3)
    assert [(c.chunk_id, c.start, c.end, c.size) for c in chunks_10] == [
        (0, 0, 3, 3),
        (1, 3, 6, 3),
        (2, 6, 9, 3),
        (3, 9, 10, 1),
    ]

    chunks_small = plan(5, 8)
    assert chunks_small == [ChunkSpec(chunk_id=0, start=0, end=5)]

    with pytest.raises(ValueError):
        plan(-1, 64)
    with pytest.raises(ValueError):
        plan(100, 0)


def test_pipeline_completes_1000_rows(tmp_path: Path) -> None:
    """Verify that the 1000-row batch pipeline completes with exit code 0 and writes all 1000 rows."""
    out_file = tmp_path / "processed_1000.jsonl"
    cfg = PipelineConfig(
        batch_size=1000,
        chunk_size=64,
        num_workers=16,
        queue_size=32,
        output_path=str(out_file),
        seed=42,
    )
    code, report = execute_pipeline(cfg)
    assert code == 0, f"Expected exit code 0, got {code}:\n{report[:1000]}"
    assert out_file.is_file(), "Output JSONL file was not created"

    rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1000
    assert [r["row_id"] for r in rows] == list(range(1000))
    assert rows[0]["value"] == 42
    assert rows[999]["row_id"] == 999


def test_no_worker_crash_messages() -> None:
    """Verify running `python -m pipeline.run --config pipeline.toml` exits 0 with no crash traces."""
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "pipeline.run", "--config", "pipeline.toml"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"CLI exited with {proc.returncode}:\n{combined[:1000]}"
    assert "BrokenPipeError" not in combined
    assert "FAILED" not in combined
    assert "SUCCEEDED" in combined

    out_path = repo_root / "output" / "processed.jsonl"
    assert out_path.is_file()
    lines = [line for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1000
