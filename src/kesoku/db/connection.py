"""SQLite connection manager and context provider for Kesoku."""

import sqlite3
from contextlib import contextmanager


class ConnectionProvider:
    """Manages SQLite database connections with proper configuration."""

    def __init__(self, db_path: str) -> None:
        """Initialize connection provider with SQLite file path.

        Args:
            db_path: Absolute or relative path to SQLite database file.
        """
        self.db_path = db_path

    def get_raw_connection(self) -> sqlite3.Connection:
        """Obtain a raw, configured SQLite database connection.

        Returns:
            A configured sqlite3.Connection instance.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # Enable WAL (Write-Ahead Logging) mode for high concurrent read/write
        conn.execute("PRAGMA journal_mode=WAL;")

        # Use NORMAL synchronous mode for faster WAL writes while maintaining app-level crash safety
        conn.execute("PRAGMA synchronous=NORMAL;")

        # Enforce foreign key constraints
        conn.execute("PRAGMA foreign_keys=ON;")

        # Set busy timeout to 5000ms to avoid database locked exceptions
        conn.execute("PRAGMA busy_timeout=5000;")

        return conn

    @contextmanager
    def connection(self):
        """Context manager to automatically open and close a database connection."""
        conn = self.get_raw_connection()
        try:
            yield conn
        finally:
            conn.close()
