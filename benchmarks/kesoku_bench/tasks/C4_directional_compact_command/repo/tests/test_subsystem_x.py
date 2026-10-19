"""Visible unit tests for Subsystem X (ThumbnailLRUCache and resizer)."""

from __future__ import annotations

from svc.subsystem_x.cache import ThumbnailLRUCache
from svc.subsystem_x.resizer import AspectBucket, classify_aspect, compute_thumbnail_box, format_cache_key


def test_thumbnail_lru_byte_and_count_eviction() -> None:
    """Cache evicts oldest unpinned entries to stay within max_items and max_bytes."""
    now = 100.0
    cache = ThumbnailLRUCache(max_items=3, max_bytes=10, default_ttl=60.0, clock=lambda: now)

    assert cache.put("k1", b"aaaa") is True
    assert cache.put("k2", b"bbbb") is True
    # Promote k1 so k2 becomes the LRU victim when k3 overflows 10 bytes.
    assert cache.get("k1") == b"aaaa"
    assert cache.put("k3", b"cccc") is True

    assert cache.get("k2") is None
    assert cache.get("k1") == b"aaaa"
    assert cache.get("k3") == b"cccc"
    assert cache.stats().evictions_lru == 1


def test_pinned_entries_and_ttl_expiry() -> None:
    """Pinned entries survive LRU eviction but expire once TTL passes."""
    now = 50.0
    cache = ThumbnailLRUCache(max_items=2, max_bytes=8, default_ttl=10.0, clock=lambda: now)

    assert cache.put("pinned-1", b"1234", pinned=True) is True
    assert cache.put("normal-1", b"5678") is True
    # Adding normal-2 must evict normal-1 rather than pinned-1.
    assert cache.put("normal-2", b"abcd") is True
    assert cache.get("pinned-1") == b"1234"
    assert cache.get("normal-1") is None

    # Advance past TTL -> pinned-1 expires.
    now = 65.0
    assert cache.get("pinned-1") is None
    assert cache.stats().evictions_expired >= 1


def test_aspect_bucketing_and_box_calculation() -> None:
    """Aspect classifier and fit-box calculator handle standard geometries."""
    assert classify_aspect(1920, 1080) == AspectBucket.LANDSCAPE_16_9
    assert classify_aspect(500, 500) == AspectBucket.SQUARE
    box = compute_thumbnail_box(1600, 800, 200, 200)
    assert box.width == 200
    assert box.height == 100
    assert format_cache_key("img9", 800, 800) == "thumb:img9:800x800:1:1:webp"
