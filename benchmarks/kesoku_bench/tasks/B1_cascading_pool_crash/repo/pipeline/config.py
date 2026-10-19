"""Configuration loader for the batch pipeline."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration parameters for a batch pipeline run."""

    batch_size: int = 1000
    chunk_size: int = 64
    num_workers: int = 16
    queue_size: int = 32
    output_path: str = "output/processed.jsonl"
    seed: int = 42


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load and validate a PipelineConfig from a TOML file.

    Args:
        path: Path to the TOML configuration file.

    Returns:
        Parsed PipelineConfig instance.
    """
    cfg_path = Path(path)
    with cfg_path.open("rb") as fh:
        raw = tomllib.load(fh)
    section = raw.get("pipeline", raw)
    return PipelineConfig(
        batch_size=int(section.get("batch_size", 1000)),
        chunk_size=int(section.get("chunk_size", 64)),
        num_workers=int(section.get("num_workers", 16)),
        queue_size=int(section.get("queue_size", 32)),
        output_path=str(section.get("output_path", "output/processed.jsonl")),
        seed=int(section.get("seed", 42)),
    )
