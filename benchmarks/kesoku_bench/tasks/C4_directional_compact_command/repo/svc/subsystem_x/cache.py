"""Thumbnail LRU cache with dual item-count and byte-budget eviction."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """Stored thumbnail payload and metadata inside ``ThumbnailLRUCache``."""

    key: str
    payload: bytes
    width: int
    height: int
    created_at: float
    expires_at: float | None
    pinned: bool = False

    @property
    def size_bytes(self) -> int:
        """Return the payload size in bytes."""
        return len(self.payload)


@dataclass(frozen=True)
class CacheStats:
    """Snapshot of thumbnail LRU cache counters."""

    hits: int
    misses: int
    evictions_lru: int
    evictions_expired: int
    rejected_oversize: int
    current_items: int
    current_bytes: int
    pinned_bytes: int


class ThumbnailLRUCache:
    """Byte-budgeted and TTL-aware LRU cache for rendered image thumbnails.

    Eviction rules and edge cases:
    1. Entries with ``size_bytes > max_bytes`` are rejected immediately without
       evicting existing entries (increments ``rejected_oversize``).
    2. Pinned entries (``pinned=True``) cannot be evicted by LRU pressure, though
       they still expire when their TTL elapses.
    3. Before inserting or updating a key, expired entries along the LRU tail are
       purged first, followed by unpinned LRU entries until both ``max_items``
       and ``max_bytes`` constraints can accommodate the new entry.
    4. If pinned entries consume so much of ``max_bytes`` that evicting all
       unpinned entries still leaves insufficient space, ``put()`` returns ``False``
       and leaves the remaining pinned entries untouched.
    """

    def __init__(
        self,
        max_items: int = 256,
        max_bytes: int = 4 * 1024 * 1024,
        default_ttl: float | None = 600.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Initialize the thumbnail LRU cache."""
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_items: int = int(max_items)
        self.max_bytes: int = int(max_bytes)
        self.default_ttl: float | None = default_ttl
        self._clock: Callable[[], float] = clock if clock is not None else (lambda: 0.0)
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._current_bytes: int = 0
        self._hits: int = 0
        self._misses: int = 0
        self._evictions_lru: int = 0
        self._evictions_expired: int = 0
        self._rejected_oversize: int = 0

    def _is_expired(self, entry: CacheEntry, now: float) -> bool:
        """Return True if ``entry`` has passed its expiration timestamp."""
        return entry.expires_at is not None and now >= entry.expires_at

    def _remove_key(self, key: str) -> CacheEntry | None:
        """Remove ``key`` and update byte accounting."""
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._current_bytes -= entry.size_bytes
        return entry

    def purge_expired(self, now: float | None = None) -> int:
        """Remove all expired entries (including pinned ones) and return count removed."""
        current = self._clock() if now is None else float(now)
        expired_keys = [
            k for k, entry in self._entries.items() if self._is_expired(entry, current)
        ]
        for key in expired_keys:
            self._remove_key(key)
            self._evictions_expired += 1
        return len(expired_keys)

    def get(self, key: str) -> bytes | None:
        """Fetch thumbnail payload for ``key`` and promote it to MRU position."""
        now = self._clock()
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None
        if self._is_expired(entry, now):
            self._remove_key(key)
            self._evictions_expired += 1
            self._misses += 1
            return None
        self._entries.move_to_end(key, last=True)
        self._hits += 1
        return entry.payload

    def get_entry(self, key: str) -> CacheEntry | None:
        """Fetch full ``CacheEntry`` metadata for ``key`` and promote it to MRU."""
        now = self._clock()
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None
        if self._is_expired(entry, now):
            self._remove_key(key)
            self._evictions_expired += 1
            self._misses += 1
            return None
        self._entries.move_to_end(key, last=True)
        self._hits += 1
        return entry

    def put(
        self,
        key: str,
        payload: bytes,
        width: int = 128,
        height: int = 128,
        *,
        ttl: float | None = None,
        pinned: bool = False,
    ) -> bool:
        """Insert or replace a thumbnail entry while enforcing item and byte caps."""
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")

        size = len(payload)
        if size > self.max_bytes:
            self._rejected_oversize += 1
            return False

        now = self._clock()
        self.purge_expired(now)

        # If replacing an existing entry, temporarily account for its removal.
        existing = self._entries.get(key)
        existing_bytes = existing.size_bytes if existing is not None else 0
        existing_item_delta = 0 if existing is not None else 1

        # Evict unpinned LRU entries in oldest-first order until space is available.
        victim_keys = [
            k for k, e in self._entries.items() if k != key and not e.pinned
        ]
        for vkey in victim_keys:
            projected_bytes = self._current_bytes - existing_bytes + size
            projected_items = len(self._entries) + existing_item_delta
            if projected_bytes <= self.max_bytes and projected_items <= self.max_items:
                break
            self._remove_key(vkey)
            self._evictions_lru += 1

        # Final capacity check (may still fail if pinned entries occupy the budget).
        if (
            self._current_bytes - existing_bytes + size > self.max_bytes
            or len(self._entries) + existing_item_delta > self.max_items
        ):
            return False

        if existing is not None:
            self._remove_key(key)

        effective_ttl = self.default_ttl if ttl is None else ttl
        expires_at = (
            now + float(effective_ttl)
            if effective_ttl is not None and effective_ttl > 0
            else None
        )
        new_entry = CacheEntry(
            key=key,
            payload=payload,
            width=int(width),
            height=int(height),
            created_at=now,
            expires_at=expires_at,
            pinned=bool(pinned),
        )
        self._entries[key] = new_entry
        self._current_bytes += size
        return True

    def pin(self, key: str) -> bool:
        """Mark an existing active key as pinned so LRU pressure skips it."""
        entry = self._entries.get(key)
        if entry is None or self._is_expired(entry, self._clock()):
            return False
        entry.pinned = True
        return True

    def unpin(self, key: str) -> bool:
        """Unpin an existing key so it becomes eligible for LRU eviction."""
        entry = self._entries.get(key)
        if entry is None:
            return False
        entry.pinned = False
        return True

    def invalidate_prefix(self, prefix: str) -> int:
        """Remove all cached thumbnails whose key starts with ``prefix``."""
        matching = [k for k in self._entries if k.startswith(prefix)]
        for k in matching:
            self._remove_key(k)
        return len(matching)

    def stats(self) -> CacheStats:
        """Return a snapshot of cache counters and memory usage."""
        pinned_bytes = sum(e.size_bytes for e in self._entries.values() if e.pinned)
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            evictions_lru=self._evictions_lru,
            evictions_expired=self._evictions_expired,
            rejected_oversize=self._rejected_oversize,
            current_items=len(self._entries),
            current_bytes=self._current_bytes,
            pinned_bytes=pinned_bytes,
        )
