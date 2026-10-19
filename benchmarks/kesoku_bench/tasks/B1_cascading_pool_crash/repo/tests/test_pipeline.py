"""Visible unit tests for pipeline configuration, even partitioning, and sink output."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.config import PipelineConfig, load_pipeline_config
from pipeline.partition import ChunkSpec, plan
from pipeline.run import execute_pipeline
from pipeline.sink import RecordSink, SinkFrame


def test_config_load() -> None:
    """Verify that pipeline.toml loads with the expected default parameters."""
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_pipeline_config(repo_root / "pipeline.toml")
    assert cfg.batch_size == 1000
    assert cfg.chunk_size == 64
    assert cfg.num_workers == 16
    assert cfg.output_path == "output/processed.jsonl"


def test_partition_even_division() -> None:
    """Verify partitioning when batch_size is an exact multiple of chunk_size."""
    chunks = plan(256, 64)
    assert len(chunks) == 4
    assert chunks[0] == ChunkSpec(chunk_id=0, start=0, end=64)
    assert chunks[1] == ChunkSpec(chunk_id=1, start=64, end=128)
    assert chunks[2] == ChunkSpec(chunk_id=2, start=128, end=192)
    assert chunks[3] == ChunkSpec(chunk_id=3, start=192, end=256)
    assert sum(c.size for c in chunks) == 256
    assert plan(0, 64) == []


def test_sink_writes_ordered_records(tmp_path: Path) -> None:
    """Verify that RecordSink sorts out-of-order chunk frames by row_id."""
    out_file = tmp_path / "records.jsonl"
    sink = RecordSink(output_path=out_file, expected_rows=4)
    sink.write_frame(
        SinkFrame(
            worker_id=1,
            chunk_id=1,
            records=[{"row_id": 2, "val": "c"}, {"row_id": 3, "val": "d"}],
        )
    )
    sink.write_frame(
        SinkFrame(
            worker_id=0,
            chunk_id=0,
            records=[{"row_id": 0, "val": "a"}, {"row_id": 1, "val": "b"}],
        )
    )
    count = sink.finalize_and_write()
    assert count == 4
    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines()]
    assert [r["row_id"] for r in lines] == [0, 1, 2, 3]


def test_pipeline_small_even_batch(tmp_path: Path) -> None:
    """Verify end-to-end execution on a small batch whose size divides evenly."""
    out_file = tmp_path / "even_out.jsonl"
    cfg = PipelineConfig(
        batch_size=128,
        chunk_size=32,
        num_workers=4,
        queue_size=16,
        output_path=str(out_file),
        seed=7,
    )
    code, report = execute_pipeline(cfg)
    assert code == 0
    assert "Summary: pipeline SUCCEEDED: 128 rows processed across 4 chunks" in report
    rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 128
    assert [r["row_id"] for r in rows] == list(range(128))
