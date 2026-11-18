"""SQLite connection manager and context provider for Kesoku.

Connections handed out by :meth:`ConnectionProvider.connection` are cached per thread and per provider
(i.e. per ``db_path``), so the hot path no longer pays ``sqlite3.connect`` + schema parsing + PRAGMA setup on
every query. ``PRAGMA journal_mode=WAL`` is a persistent database property, so it runs once per provider rather
than on every new connection.

Semantics preserved from the previous "open, use, close" implementation:

* Uncommitted work is discarded when the ``with`` block exits (the cached connection is rolled back if it is
  still inside a transaction), just as ``close()`` used to discard it.
* Per-connection state that callers mutate (``isolation_level``, ``row_factory``) is restored on exit.
* Re-entrant use in the same thread (a nested ``connection()`` while the cached one is checked out, or two
  coroutines interleaving on the event-loop thread) gets a fresh, private connection that is closed on exit,
  so an inner block can never commit or roll back the outer block's transaction.
"""

import logging
import sqlite3
import threading
import weakref
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_DEFAULT_ISOLATION_LEVEL = ""


class _ConnHolder:
    """Weak-referenceable owner of one cached connection (`sqlite3.Connection` itself is not weakref-able).

    The only strong reference lives in the owning thread's slot, so when that thread exits the holder (and its
    connection) is freed instead of leaking in the provider's registry.
    """

    __slots__ = ("conn", "in_use", "__weakref__")

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        # Checked out by its thread; guarded by the provider lock so `close_all()` never closes it mid-use.
        self.in_use = False


class _ThreadSlot(threading.local):
    """Per-thread cache slot holding this thread's reusable connection for one provider."""

    def __init__(self) -> None:
        self.holder: _ConnHolder | None = None
        self.generation: int = -1
        self.in_use: bool = False


class ConnectionProvider:
    """Manages SQLite database connections with proper configuration."""

    def __init__(self, db_path: str) -> None:
        """Initialize connection provider with SQLite file path.

        Args:
            db_path: Absolute or relative path to SQLite database file.
        """
        self.db_path = db_path
        self._slot = _ThreadSlot()
        self._lock = threading.Lock()
        # Every live cached connection across all threads, so `close_all()` can close them from any thread.
        self._cached: weakref.WeakSet[_ConnHolder] = weakref.WeakSet()
        # Bumped by `close_all()`; thread slots holding an older generation reopen on next use.
        self._generation = 0
        self._wal_applied = False

    def _apply_wal_once(self, conn: sqlite3.Connection) -> None:
        """Switch the database to WAL mode once per provider (it is persisted in the file header)."""
        if self._wal_applied:
            return
        with self._lock:
            if self._wal_applied:
                return
            # Enable WAL (Write-Ahead Logging) mode for high concurrent read/write
            conn.execute("PRAGMA journal_mode=WAL;")
            self._wal_applied = True

    def _open(self, *, check_same_thread: bool = True) -> sqlite3.Connection:
        """Open and configure a brand-new SQLite connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=check_same_thread)
        try:
            conn.row_factory = sqlite3.Row
            self._apply_wal_once(conn)
            # Use NORMAL synchronous mode for faster WAL writes while maintaining app-level crash safety
            conn.execute("PRAGMA synchronous=NORMAL;")
            # Enforce foreign key constraints
            conn.execute("PRAGMA foreign_keys=ON;")
            # Set busy timeout to 5000ms to avoid database locked exceptions
            conn.execute("PRAGMA busy_timeout=5000;")
        except BaseException:
            conn.close()
            raise
        return conn

    def get_raw_connection(self) -> sqlite3.Connection:
        """Obtain a raw, configured, *uncached* SQLite database connection.

        The caller owns the returned connection and is responsible for closing it.

        Returns:
            A configured sqlite3.Connection instance.
        """
        return self._open()

    def _checkout_cached(self) -> _ConnHolder:
        """Mark this thread's cached connection in use, (re)opening it when missing or invalidated."""
        slot = self._slot
        with self._lock:
            holder = slot.holder
            if holder is not None and slot.generation == self._generation:
                holder.in_use = True
                return holder
        # Cached connections may be closed by `close_all()` from another thread, so they are opened with
        # `check_same_thread=False`; thread affinity for actual use is guaranteed by the thread-local slot.
        # (Opened outside the lock: `_open` may take it for the one-time WAL switch.)
        holder = _ConnHolder(self._open(check_same_thread=False))
        with self._lock:
            holder.in_use = True
            self._cached.add(holder)
            generation = self._generation
        slot.holder = holder
        slot.generation = generation
        return holder

    @staticmethod
    def _reset(conn: sqlite3.Connection) -> bool:
        """Discard uncommitted work and restore mutated connection state. Returns False if unusable."""
        try:
            if conn.in_transaction:
                conn.rollback()
            if conn.isolation_level != _DEFAULT_ISOLATION_LEVEL:
                conn.isolation_level = _DEFAULT_ISOLATION_LEVEL
            if conn.row_factory is not sqlite3.Row:
                conn.row_factory = sqlite3.Row
            return True
        except sqlite3.Error as e:
            logger.warning(f"Discarding unusable cached SQLite connection: {e}")
            return False

    def _discard_current(self) -> None:
        """Close this thread's cached connection and forget it."""
        holder = self._slot.holder
        self._slot.holder = None
        if holder is None:
            return
        with self._lock:
            self._cached.discard(holder)
        try:
            holder.conn.close()
        except sqlite3.Error:
            pass

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager yielding a configured connection; uncommitted work is discarded on exit."""
        slot = self._slot
        if slot.in_use:
            # Re-entrant use in this thread: never share the outer block's connection/transaction.
            conn = self._open()
            try:
                yield conn
            finally:
                conn.close()
            return

        holder = self._checkout_cached()
        slot.in_use = True
        try:
            yield holder.conn
        finally:
            slot.in_use = False
            # Reset while still marked in use, so a concurrent `close_all()` cannot close it underneath us.
            healthy = self._reset(holder.conn)
            with self._lock:
                holder.in_use = False
                stale = slot.generation != self._generation
            if stale or not healthy:
                # `close_all()` ran while we held it (it skipped busy connections), or it is broken: close it now.
                self._discard_current()

    def close(self) -> None:
        """Close this thread's cached connection (if any)."""
        if not self._slot.in_use:
            self._discard_current()

    def close_all(self) -> None:
        """Close every idle cached connection across all threads and reset one-time setup.

        Safe to call from any thread. Connections currently checked out by another thread are not closed under
        it; they are invalidated (generation bump) and closed by their owner when its ``with`` block exits.
        Threads that use the provider afterwards transparently reopen a fresh connection. Call this before
        deleting or replacing the database file.
        """
        with self._lock:
            idle = [h for h in self._cached if not h.in_use]
            self._cached.clear()
            self._generation += 1
            self._wal_applied = False
        for holder in idle:
            try:
                holder.conn.close()
            except sqlite3.Error:
                pass
        if not self._slot.in_use:
            self._slot.holder = None

