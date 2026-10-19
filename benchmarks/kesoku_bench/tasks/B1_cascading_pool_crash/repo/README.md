# `pipeline` — Parallel Batch Record Processing Pipeline

`pipeline` is a lightweight multi-worker batch processing library that partitions an input batch into contiguous chunks, processes them concurrently across a `ThreadPoolExecutor`, and writes deterministic JSONL records through a buffered `RecordSink`.

## Architecture

- `pipeline.config`: Loads `PipelineConfig` from a TOML file (`batch_size`, `chunk_size`, `num_workers`, `queue_size`, `output_path`, `seed`).
- `pipeline.partition`: Computes contiguous `ChunkSpec(chunk_id, start, end)` slices covering `[0, batch_size)`. Each chunk contains at most `chunk_size` rows; when `batch_size` is not an exact multiple of `chunk_size`, the final chunk is a ragged tail containing the remaining `batch_size % chunk_size` rows.
- `pipeline.worker`: Worker execution loop pulling `ChunkSpec` items from a bounded queue, transforming each row deterministically, and flushing completed chunk frames to the sink.
- `pipeline.sink`: Thread-safe `RecordSink` that validates channel state, collects processed records, and flushes them in row order to the configured output file.
- `pipeline.run`: CLI coordinator (`uv run python -m pipeline.run --config pipeline.toml`).

## Running

```bash
uv run python -m pipeline.run --config pipeline.toml
uv run pytest
```
