"""Thumbnail geometry, aspect-ratio bucketing, and cache-key normalization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class AspectBucket(StrEnum):
    """Canonical aspect-ratio buckets for thumbnail cache key normalization."""

    SQUARE = "1:1"
    LANDSCAPE_4_3 = "4:3"
    LANDSCAPE_16_9 = "16:9"
    PORTRAIT_3_4 = "3:4"
    ULTRA_WIDE = "21:9"


@dataclass(frozen=True)
class Dimensions:
    """Pixel dimensions of a source image or target thumbnail."""

    width: int
    height: int

    @property
    def aspect_ratio(self) -> float:
        """Return width / height ratio."""
        return self.width / self.height


def classify_aspect(width: int, height: int) -> AspectBucket:
    """Classify ``(width, height)`` into the nearest ``AspectBucket``."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    ratio = width / height
    candidates: list[tuple[float, AspectBucket]] = [
        (1.0, AspectBucket.SQUARE),
        (4.0 / 3.0, AspectBucket.LANDSCAPE_4_3),
        (16.0 / 9.0, AspectBucket.LANDSCAPE_16_9),
        (3.0 / 4.0, AspectBucket.PORTRAIT_3_4),
        (21.0 / 9.0, AspectBucket.ULTRA_WIDE),
    ]
    best_bucket = AspectBucket.SQUARE
    best_dist = float("inf")
    for target, bucket in candidates:
        dist = abs(math.log(ratio / target))
        if dist < best_dist:
            best_dist = dist
            best_bucket = bucket
    return best_bucket


def compute_thumbnail_box(
    src_width: int,
    src_height: int,
    max_width: int,
    max_height: int,
    *,
    allow_upscale: bool = False,
) -> Dimensions:
    """Compute fit-within-box thumbnail dimensions preserving aspect ratio."""
    if min(src_width, src_height, max_width, max_height) <= 0:
        raise ValueError("All dimensions must be positive")
    scale = min(max_width / src_width, max_height / src_height)
    if not allow_upscale:
        scale = min(1.0, scale)
    out_w = max(1, int(round(src_width * scale)))
    out_h = max(1, int(round(src_height * scale)))
    return Dimensions(width=out_w, height=out_h)


def format_cache_key(asset_id: str, width: int, height: int, fmt: str = "webp") -> str:
    """Build a canonical cache key incorporating the aspect bucket."""
    bucket = classify_aspect(width, height)
    return f"thumb:{asset_id}:{width}x{height}:{bucket.value}:{fmt.lower()}"
