import asyncio
import datetime
import json
import logging
import os
import shutil
import sqlite3
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal

import numpy as np

from kesoku.agent.history_sorter import sort_session_messages
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db.connection import ConnectionProvider
from kesoku.db.models import Message, Session, SummaryNode
from kesoku.utils.text import fts_tokenize

logger = logging.getLogger(__name__)

# Number of candidate chunks scored per batch during semantic search. Bounds how many
# embedding BLOBs are resident at once, keeping peak memory flat regardless of corpus size.
SEARCH_SCORE_BATCH_SIZE = 2048


def _batch_semantic_scores(
    rows: list[sqlite3.Row],
    query_vec: np.ndarray | None,
    query_norm: float,
) -> list[float]:
    """Cosine-similarity a batch of candidate chunk rows against the query vector.

    Stacks the raw embedding BLOBs into a single float32 matrix and scores them with one
    matmul. This avoids materializing a 384-element Python float list per row, which
    dominated both allocation and CPU time when scanning a large corpus.

    Args:
        rows: Candidate rows exposing a ``chunk_embedding`` BLOB column.
        query_vec: The query embedding, or None when unavailable or degenerate.
        query_norm: Precomputed L2 norm of ``query_vec``.

    Returns:
        One cosine similarity per input row, in row order (0.0 where it cannot be computed).
    """
    scores = [0.0] * len(rows)
    if query_vec is None:
        return scores

    # Only stack embeddings whose byte length matches the query dimensionality; a mismatch
    # means the row was indexed with a different model and cannot be compared.
    expected_bytes = query_vec.nbytes
    indices: list[int] = []
    blobs: list[bytes] = []
    for i, row in enumerate(rows):
        blob = row["chunk_embedding"]
        if blob and len(blob) == expected_bytes:
            indices.append(i)
            blobs.append(blob)
    if not indices:
        return scores

    matrix = np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(len(indices), query_vec.size)
    norms = np.linalg.norm(matrix, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.where(norms > 0.0, (matrix @ query_vec) / (norms * query_norm), 0.0)
    for i, sim in zip(indices, sims.tolist(), strict=True):
        scores[i] = float(sim)
    return scores


class DatabaseManager:
    """Encapsulates all SQLite database schema and CRUD operations for Kesoku."""

    def __init__(self, db_path: str) -> None:
        """Initialize DatabaseManager with SQLite file path.

        Args:
            db_path: Absolute or relative filesystem path to SQLite database file.
        """
        self.db_path = db_path
        self.connection_provider = ConnectionProvider(db_path)
        self._config: Any = None
        self.digests_available: bool = True
        self.fts_available: bool = True

    def _get_chatbot_default_role(self, chatbot_id: str) -> str:
        """Resolve per-chatbot default_role from bound KesokuConfig or global config."""
        cfg = self._config
        if cfg is None:
            from kesoku.config import get_config

            cfg = get_config()
        return str(cfg.get_chatbot_default_role(chatbot_id))

    def _get_connection(self) -> sqlite3.Connection:
        """Expose connection for backward-compatible test mock manipulations."""
        return self.connection_provider.get_raw_connection()

    def _row_to_message(self, row: sqlite3.Row) -> Message:
        """Helper to convert a sqlite Row to a Message Pydantic model."""
        return Message(
            id=row["id"],
            session_id=row["session_id"],
            chatbot_id=row["chatbot_id"],
            channel_id=row["channel_id"],
            sender=row["sender"],
            role=row["role"],
            type=row["type"],
            content=row["content"],
            metadata=json.loads(row["metadata"]),
            timestamp=row["timestamp"],
            status=row["status"],
            parent_id=row["parent_id"],
            summary_node_id=row["summary_node_id"] if "summary_node_id" in row.keys() else None,
            embedding=row["embedding"] if "embedding" in row.keys() else None,
        )

    def _row_to_session(self, row: sqlite3.Row) -> Session:
        """Helper to convert a sqlite Row to a Session Pydantic model."""
        return Session(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            system_prompt=row["system_prompt"],
            role_name=row["role_name"] if "role_name" in row.keys() and row["role_name"] else "default",
        )

    def _ensure_migrations(self, conn: sqlite3.Connection) -> None:
        """Ensure all required columns exist in messages table."""
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(messages)")
                columns = [row["name"] for row in cursor.fetchall()]
                if "role" not in columns:
                    conn.execute("ALTER TABLE messages ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
                if "type" not in columns:
                    conn.execute("ALTER TABLE messages ADD COLUMN type TEXT NOT NULL DEFAULT 'text'")
                if "parent_id" not in columns:
                    conn.execute("ALTER TABLE messages ADD COLUMN parent_id TEXT")
                if "embedding" not in columns:
                    conn.execute("ALTER TABLE messages ADD COLUMN embedding BLOB")
                cursor.execute("PRAGMA table_info(sessions)")
                sess_columns = [row["name"] for row in cursor.fetchall()]
                if "role_name" not in sess_columns:
                    conn.execute("ALTER TABLE sessions ADD COLUMN role_name TEXT")
                # Clean up redundant single-column session index since idx_messages_session_timestamp supersedes it
                conn.execute("DROP INDEX IF EXISTS idx_messages_session")
                conn.execute("DROP TABLE IF EXISTS message_chunks;")
                # Ensure memory_files and memory_chunks tables exist
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        role TEXT NOT NULL,
                        rel_path TEXT NOT NULL,
                        description TEXT NOT NULL DEFAULT '',
                        content_hash TEXT NOT NULL,
                        mtime REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        UNIQUE(role, rel_path)
                    );
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memory_files_role ON memory_files(role);"
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_id INTEGER NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        FOREIGN KEY (file_id) REFERENCES memory_files(id) ON DELETE CASCADE
                    );
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memory_chunks_file_id ON memory_chunks(file_id);"
                )
                # Ensure channel_sessions table exists
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS channel_sessions (
                        chatbot_id TEXT NOT NULL,
                        channel_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        PRIMARY KEY (chatbot_id, channel_id),
                        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                    );
                    """
                )
                # Ensure channel_roles table exists
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS channel_roles (
                        chatbot_id TEXT NOT NULL,
                        channel_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        PRIMARY KEY (chatbot_id, channel_id)
                    );
                    """
                )
                conn.execute("DROP TABLE IF EXISTS agent_memories;")
                # Ensure ui_prompts table exists (interactive UI specs, keyed by rendered message)
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ui_prompts (
                        session_id TEXT NOT NULL,
                        message_id TEXT NOT NULL,
                        spec_json TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        PRIMARY KEY (session_id, message_id)
                    );
                    """
                )
                # Ensure summary_nodes table exists
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS summary_nodes (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        level INTEGER NOT NULL,
                        summary TEXT NOT NULL,
                        start_timestamp REAL NOT NULL,
                        end_timestamp REAL NOT NULL,
                        token_count INTEGER NOT NULL,
                        source_token_count INTEGER NOT NULL,
                        parent_id TEXT,
                        created_at REAL NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                        FOREIGN KEY (parent_id) REFERENCES summary_nodes(id) ON DELETE SET NULL
                    );
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_summary_nodes_session_parent "
                    "ON summary_nodes(session_id, parent_id);"
                )
                # Ensure summary_node_id exists in messages table
                if "summary_node_id" not in columns:
                    conn.execute(
                        "ALTER TABLE messages ADD COLUMN summary_node_id TEXT "
                        "REFERENCES summary_nodes(id) ON DELETE SET NULL;"
                    )
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_summary_node ON messages(summary_node_id);")
        except Exception as e:
            logger.error(f"Failed to apply database schema migrations: {e}")
            raise RuntimeError(f"Database schema migration error: {e}") from e

        self._apply_memory_v2_v3_migrations(conn)

    def _apply_memory_v2_v3_migrations(self, conn: sqlite3.Connection) -> None:
        """Apply idempotent v2 (digests/cursors/visibility) and v3 (FTS5) migrations safely."""
        try:
            user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        except Exception as e:
            logger.warning(f"Failed reading PRAGMA user_version: {e}")
            user_version = 0

        # --- Migration v2 (P0): memory_files columns, session_digests, extraction cursors/log, drop legacy table ---
        v2_ok = True
        try:
            mf_cols = {row["name"] for row in conn.execute("PRAGMA table_info(memory_files)").fetchall()}
            with conn:
                if "visibility" not in mf_cols:
                    conn.execute("ALTER TABLE memory_files ADD COLUMN visibility TEXT NOT NULL DEFAULT 'DEFAULT'")
                if "size_bytes" not in mf_cols:
                    conn.execute("ALTER TABLE memory_files ADD COLUMN size_bytes INTEGER NOT NULL DEFAULT 0")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_files_role_vis ON memory_files(role, visibility)")
        except Exception as e:
            v2_ok = False
            logger.error(f"Migration v2 (memory_files columns) failed: {e}")

        v2_statements = (
            (
                "session_digests",
                """
                CREATE TABLE IF NOT EXISTS session_digests (
                    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    chatbot_id TEXT,
                    channel_id TEXT,
                    title TEXT NOT NULL DEFAULT '',
                    started_at REAL NOT NULL,
                    last_active_at REAL NOT NULL,
                    turn_count INTEGER NOT NULL DEFAULT 0,
                    last_user_msgs TEXT NOT NULL DEFAULT '[]',
                    context_policy TEXT NOT NULL DEFAULT 'full',
                    summary TEXT NOT NULL DEFAULT '',
                    open_threads TEXT NOT NULL DEFAULT '[]',
                    summary_upto_ts REAL,
                    embedding BLOB,
                    updated_at REAL NOT NULL
                )
                """,
            ),
            (
                "idx_session_digests_role_active",
                "CREATE INDEX IF NOT EXISTS idx_session_digests_role_active "
                "ON session_digests(role, last_active_at DESC)",
            ),
            (
                "memory_extraction_cursors",
                """
                CREATE TABLE IF NOT EXISTS memory_extraction_cursors (
                    session_id TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    last_extracted_msg_ts REAL NOT NULL DEFAULT 0,
                    last_extracted_msg_id TEXT NOT NULL DEFAULT '',
                    pending_user_turns INTEGER NOT NULL DEFAULT 0,
                    pending_chars INTEGER NOT NULL DEFAULT 0,
                    last_run_at REAL,
                    last_reason TEXT,
                    tokens_spent INTEGER NOT NULL DEFAULT 0,
                    failures INTEGER NOT NULL DEFAULT 0
                )
                """,
            ),
            (
                "memory_extraction_log",
                """
                CREATE TABLE IF NOT EXISTS memory_extraction_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT,
                    session_id TEXT,
                    reason TEXT,
                    started_at REAL,
                    finished_at REAL,
                    status TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    ops_applied INTEGER,
                    error TEXT
                )
                """,
            ),
            (
                "drop_cross_session_contexts",
                "DROP TABLE IF EXISTS cross_session_contexts",
            ),
        )
        for label, stmt in v2_statements:
            try:
                with conn:
                    conn.execute(stmt)
            except Exception as e:
                v2_ok = False
                if label in ("session_digests", "idx_session_digests_role_active"):
                    self.digests_available = False
                logger.error(f"Migration v2 ({label}) failed; degrading gracefully: {e}")

        if v2_ok and user_version < 2:
            try:
                with conn:
                    conn.execute("PRAGMA user_version = 2")
                user_version = 2
            except Exception as e:
                logger.warning(f"Failed setting PRAGMA user_version = 2: {e}")

        # --- Migration v3 (P1): regular FTS5 tables + backfill state ---
        v3_ok = True
        v3_statements = (
            (
                "messages_fts",
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    message_id UNINDEXED,
                    session_id,
                    persona_role,
                    context_policy,
                    speaker_role UNINDEXED,
                    ts UNINDEXED,
                    raw_excerpt UNINDEXED,
                    text,
                    tokenize='unicode61'
                )
                """,
            ),
            (
                "session_digests_fts",
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS session_digests_fts USING fts5(
                    session_id UNINDEXED,
                    role,
                    text,
                    tokenize='unicode61'
                )
                """,
            ),
            (
                "fts_backfill_state",
                """
                CREATE TABLE IF NOT EXISTS fts_backfill_state (
                    name TEXT PRIMARY KEY,
                    last_ts REAL NOT NULL DEFAULT 0,
                    last_id TEXT NOT NULL DEFAULT '',
                    done INTEGER NOT NULL DEFAULT 0
                )
                """,
            ),
        )
        for label, stmt in v3_statements:
            try:
                with conn:
                    conn.execute(stmt)
            except Exception as e:
                v3_ok = False
                self.fts_available = False
                logger.error(f"Migration v3 ({label}) failed; disabling FTS search gracefully: {e}")

        if v2_ok and v3_ok and user_version < 3:
            try:
                with conn:
                    conn.execute("PRAGMA user_version = 3")
            except Exception as e:
                logger.warning(f"Failed setting PRAGMA user_version = 3: {e}")

    def verify_db(self) -> None:
        """Verify that the database file exists, is non-empty, and contains required tables."""
        if not os.path.exists(self.db_path) or os.path.getsize(self.db_path) == 0:
            raise RuntimeError(
                f"Database file '{self.db_path}' does not exist or is empty. "
                "Please run 'kesoku init' first to initialize the workspace."
            )
        with self.connection_provider.connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
                if not cursor.fetchone():
                    raise RuntimeError(
                        f"Database at '{self.db_path}' is missing required tables. Please run 'kesoku init' first."
                    )
                self._ensure_migrations(conn)
            except sqlite3.DatabaseError as e:
                raise RuntimeError(
                    f"Database at '{self.db_path}' is invalid or corrupt. Please run 'kesoku init' first."
                ) from e

    def init_tables(self, overwrite: bool = False) -> None:
        """Initialize SQLite database tables and indices.

        Args:
            overwrite: Whether to overwrite the existing database file (creates backup).
        """
        if overwrite and os.path.exists(self.db_path):
            backup_path = f"{self.db_path}.bak.{int(time.time())}"
            try:
                shutil.copy(self.db_path, backup_path)
                logger.info(f"Created backup of existing database at {backup_path}")
                os.remove(self.db_path)
                logger.info(f"Removed existing database file {self.db_path}")
            except Exception as e:
                logger.error(f"Failed to backup/overwrite existing database {self.db_path}: {e}")
                raise

        with self.connection_provider.connection() as conn:
            try:
                with conn:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS summary_nodes (
                            id TEXT PRIMARY KEY,
                            session_id TEXT NOT NULL,
                            level INTEGER NOT NULL,
                            summary TEXT NOT NULL,
                            start_timestamp REAL NOT NULL,
                            end_timestamp REAL NOT NULL,
                            token_count INTEGER NOT NULL,
                            source_token_count INTEGER NOT NULL,
                            parent_id TEXT,
                            created_at REAL NOT NULL,
                            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                            FOREIGN KEY (parent_id) REFERENCES summary_nodes(id) ON DELETE SET NULL
                        );
                        """
                    )
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS messages (
                            id TEXT PRIMARY KEY,
                            session_id TEXT NOT NULL,
                            chatbot_id TEXT NOT NULL,
                            channel_id TEXT NOT NULL,
                            sender TEXT NOT NULL,
                            role TEXT NOT NULL DEFAULT 'user',
                            type TEXT NOT NULL DEFAULT 'text',
                            content TEXT NOT NULL,
                            metadata TEXT NOT NULL,
                            timestamp REAL NOT NULL,
                            status TEXT NOT NULL,
                            parent_id TEXT,
                            summary_node_id TEXT,
                            embedding BLOB,
                            FOREIGN KEY (summary_node_id) REFERENCES summary_nodes(id) ON DELETE SET NULL
                        );
                        """
                    )
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS memory_files (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            role TEXT NOT NULL,
                            rel_path TEXT NOT NULL,
                            description TEXT NOT NULL DEFAULT '',
                            content_hash TEXT NOT NULL,
                            mtime REAL NOT NULL,
                            updated_at REAL NOT NULL,
                            UNIQUE(role, rel_path)
                        );
                        """
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_memory_files_role ON memory_files(role);"
                    )
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS memory_chunks (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            file_id INTEGER NOT NULL,
                            chunk_index INTEGER NOT NULL,
                            content TEXT NOT NULL,
                            embedding BLOB NOT NULL,
                            FOREIGN KEY (file_id) REFERENCES memory_files(id) ON DELETE CASCADE
                        );
                        """
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_memory_chunks_file_id ON memory_chunks(file_id);"
                    )
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS sessions (
                            id TEXT PRIMARY KEY,
                            title TEXT NOT NULL,
                            created_at REAL NOT NULL,
                            updated_at REAL NOT NULL,
                            system_prompt TEXT NOT NULL DEFAULT '',
                            role_name TEXT
                        );
                        """
                    )
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS channel_sessions (
                            chatbot_id TEXT NOT NULL,
                            channel_id TEXT NOT NULL,
                            session_id TEXT NOT NULL,
                            PRIMARY KEY (chatbot_id, channel_id),
                            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                        );
                        """
                    )
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS channel_roles (
                            chatbot_id TEXT NOT NULL,
                            channel_id TEXT NOT NULL,
                            role TEXT NOT NULL,
                            PRIMARY KEY (chatbot_id, channel_id)
                        );
                        """
                    )
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS ui_prompts (
                            session_id TEXT NOT NULL,
                            message_id TEXT NOT NULL,
                            spec_json TEXT NOT NULL,
                            created_at REAL NOT NULL,
                            PRIMARY KEY (session_id, message_id)
                        );
                        """
                    )
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_status ON messages(status);")
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_messages_session_timestamp ON messages(session_id, timestamp);"
                    )
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(chatbot_id, channel_id);")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);")
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_summary_nodes_session_parent "
                        "ON summary_nodes(session_id, parent_id);"
                    )
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_summary_node ON messages(summary_node_id);")
                self._ensure_migrations(conn)
                logger.info(f"Database schema initialized successfully at {self.db_path}")
            except Exception as e:
                logger.error(f"Failed to initialize database schema: {e}")
                raise

    # Session CRUD
    def create_session(self, session: Session) -> None:
        """Persist a new chat session record.

        Args:
            session: The Session object to store.
        """
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO sessions
                    (id, title, created_at, updated_at, system_prompt, role_name)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session.id,
                        session.title,
                        session.created_at,
                        session.updated_at,
                        session.system_prompt,
                        session.role_name,
                    ),
                )

    def get_session(self, session_id: str) -> Session | None:
        """Retrieve a chat session record by ID.

        Args:
            session_id: Session ID to query.

        Returns:
            The Session object if found, None otherwise.
        """
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_session(row)
            return None

    def update_session_updated_at(self, session_id: str, updated_at: float) -> None:
        """Update the updated_at timestamp for a session.

        Args:
            session_id: Target session ID.
            updated_at: New timestamp float.
        """
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (updated_at, session_id))

    def _refresh_digest_fts_in_tx(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        role: str,
        title: str,
        summary: str,
        last_user_msgs_json: str,
    ) -> None:
        """Refresh a single session's entry inside `session_digests_fts` within an open transaction."""
        if not self.fts_available:
            return
        try:
            msgs_list = json.loads(last_user_msgs_json) if last_user_msgs_json else []
            if not isinstance(msgs_list, list):
                msgs_list = []
        except Exception:
            msgs_list = []
        combined = "\n".join(part for part in [title, summary, *(str(m) for m in msgs_list)] if part)
        tokenized = fts_tokenize(combined, for_query=False)
        try:
            conn.execute("DELETE FROM session_digests_fts WHERE session_id = ?", (session_id,))
            if tokenized:
                conn.execute(
                    "INSERT INTO session_digests_fts (session_id, role, text) VALUES (?, ?, ?)",
                    (session_id, role or "default", tokenized),
                )
        except Exception as e:
            logger.warning(f"Failed updating session_digests_fts for session '{session_id}': {e}")

    def update_session_title(self, session_id: str, title: str) -> None:
        """Update the title of a session, keeping `session_digests` and `session_digests_fts` synchronized.

        Args:
            session_id: Session ID to update.
            title: New title.
        """
        now = time.time()
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                    (title, now, session_id),
                )
                if self.digests_available:
                    try:
                        cursor = conn.execute(
                            "SELECT role, summary, last_user_msgs FROM session_digests WHERE session_id = ?",
                            (session_id,),
                        )
                        d_row = cursor.fetchone()
                        if d_row is not None:
                            conn.execute(
                                "UPDATE session_digests SET title = ?, updated_at = ? WHERE session_id = ?",
                                (title, now, session_id),
                            )
                            self._refresh_digest_fts_in_tx(
                                conn,
                                session_id=session_id,
                                role=str(d_row["role"]),
                                title=title,
                                summary=str(d_row["summary"] or ""),
                                last_user_msgs_json=str(d_row["last_user_msgs"] or "[]"),
                            )
                    except Exception as e:
                        logger.warning(f"Failed syncing session_digests title for '{session_id}': {e}")



    def list_sessions(self) -> list[Session]:
        """List all chat sessions ordered by most recently updated.

        Returns:
            List of Session objects.
        """
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions ORDER BY updated_at DESC")
            rows = cursor.fetchall()
            return [self._row_to_session(row) for row in rows]

    def get_latest_session(self) -> Session | None:
        """Retrieve the most recently updated chat session.

        Returns:
            The Session object if available, None otherwise.
        """
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions ORDER BY updated_at DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                return self._row_to_session(row)
            return None

    def get_session_by_channel(self, chatbot_id: str, channel_id: str) -> Session | None:
        """Retrieve the chat session associated with a specific chatbot channel.

        Args:
            chatbot_id: Unique identifier of the chatbot.
            channel_id: Channel or room identifier.

        Returns:
            The Session object if found, None otherwise.
        """
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT s.* FROM sessions s
                JOIN channel_sessions cs ON s.id = cs.session_id
                WHERE cs.chatbot_id = ? AND cs.channel_id = ?
                """,
                (chatbot_id, channel_id),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_session(row)
            return None

    def get_last_session_by_channel_and_role(self, chatbot_id: str, channel_id: str, role: str) -> Session | None:
        """Retrieve the most recent session for a chatbot channel under a specific role.

        Args:
            chatbot_id: Unique identifier of the chatbot.
            channel_id: Channel or room identifier.
            role: The role name.

        Returns:
            The Session object if found, None otherwise.
        """
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT s.* FROM sessions s
                JOIN messages m ON s.id = m.session_id
                WHERE m.chatbot_id = ? AND m.channel_id = ? AND COALESCE(s.role_name, 'default') = ?
                ORDER BY m.timestamp DESC
                LIMIT 1
                """,
                (chatbot_id, channel_id, role),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_session(row)
            return None

    def set_active_session_for_channel(self, chatbot_id: str, channel_id: str, session_id: str) -> None:
        """Bind a session as the active session for a chatbot channel (UPSERT).

        Args:
            chatbot_id: Unique chatbot platform identifier.
            channel_id: External channel or thread identifier.
            session_id: Unique session identifier to bind.
        """
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO channel_sessions (chatbot_id, channel_id, session_id)
                    VALUES (?, ?, ?)
                    ON CONFLICT(chatbot_id, channel_id) DO UPDATE SET session_id=excluded.session_id
                    """,
                    (chatbot_id, channel_id, session_id),
                )
                cursor = conn.cursor()
                cursor.execute("SELECT role_name FROM sessions WHERE id = ?", (session_id,))
                row = cursor.fetchone()
                current_sess_role = row["role_name"] if row else None

                resolved_role = self.get_channel_role_with_inheritance(chatbot_id, channel_id, session_id)

                should_update = False
                if not current_sess_role:
                    should_update = True
                elif current_sess_role == "default" and resolved_role != "default":
                    should_update = True

                if should_update:
                    conn.execute(
                        """
                        UPDATE sessions
                        SET role_name = ?
                        WHERE id = ?
                        """,
                        (resolved_role, session_id),
                    )
                    logger.info(
                        f"Explicitly bound channel '{chatbot_id}:{channel_id}' "
                        f"to active session '{session_id}' (role updated to: {resolved_role})"
                    )
                else:
                    logger.info(
                        f"Explicitly bound channel '{chatbot_id}:{channel_id}' "
                        f"to active session '{session_id}' (kept role: {current_sess_role})"
                    )

    def delete_session(self, session_id: str) -> None:
        """Delete a session and all its associated messages, digests, cursors, and FTS rows.

        Args:
            session_id: The unique session identifier to delete.
        """
        with self.connection_provider.connection() as conn:
            with conn:
                if self.fts_available:
                    try:
                        safe_sid = session_id.replace('"', '""')
                        conn.execute(
                            "DELETE FROM messages_fts WHERE messages_fts MATCH ?",
                            (f'session_id:"{safe_sid}"',),
                        )
                        conn.execute("DELETE FROM session_digests_fts WHERE session_id = ?", (session_id,))
                    except Exception as e:
                        logger.warning(f"Failed deleting FTS rows for session '{session_id}': {e}")
                if self.digests_available:
                    try:
                        conn.execute("DELETE FROM session_digests WHERE session_id = ?", (session_id,))
                        conn.execute("DELETE FROM memory_extraction_cursors WHERE session_id = ?", (session_id,))
                    except Exception as e:
                        logger.warning(f"Failed deleting digest/cursor rows for session '{session_id}': {e}")
                # Delete all messages belonging to this session
                conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                # Delete interactive UI prompt specs of this session
                conn.execute("DELETE FROM ui_prompts WHERE session_id = ?", (session_id,))
                # Delete the session itself
                conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    def delete_messages_by_ids(self, message_ids: list[str]) -> None:
        """Delete specific messages from the database by their IDs.

        Args:
            message_ids: List of message ID strings to delete.
        """
        if not message_ids:
            return
        with self.connection_provider.connection() as conn:
            with conn:
                placeholders = ",".join("?" for _ in message_ids)
                if self.fts_available:
                    try:
                        conn.execute(
                            f"DELETE FROM messages_fts WHERE message_id IN ({placeholders})",
                            tuple(message_ids),
                        )
                    except Exception as e:
                        logger.warning(f"Failed deleting messages_fts rows by id: {e}")
                cursor = conn.execute(
                    f"DELETE FROM messages WHERE id IN ({placeholders})",
                    tuple(message_ids),
                )
                logger.info(f"Database deleted {cursor.rowcount} messages out of {len(message_ids)} requested IDs.")

    def delete_messages_before_timestamp(
        self, session_id: str, timestamp: float, exclude_ids: list[str] | None = None
    ) -> int:
        """Delete all messages in a session that are older than a specific timestamp.

        Args:
            session_id: The session identifier.
            timestamp: Cutoff unix timestamp. Messages older than this are deleted.
            exclude_ids: Optional list of message IDs to protect from deletion.

        Returns:
            The number of messages deleted.
        """
        with self.connection_provider.connection() as conn:
            with conn:
                if self.fts_available:
                    try:
                        safe_sid = session_id.replace('"', '""')
                        fts_q = "DELETE FROM messages_fts WHERE messages_fts MATCH ? AND CAST(ts AS REAL) < ?"
                        fts_params: list[Any] = [f'session_id:"{safe_sid}"', timestamp]
                        if exclude_ids:
                            excl_ph = ",".join("?" for _ in exclude_ids)
                            fts_q += f" AND message_id NOT IN ({excl_ph})"
                            fts_params.extend(exclude_ids)
                        conn.execute(fts_q, tuple(fts_params))
                    except Exception as e:
                        logger.warning(f"Failed deleting older messages_fts rows in session '{session_id}': {e}")
                query = "DELETE FROM messages WHERE session_id = ? AND timestamp < ?"
                params = [session_id, timestamp]
                if exclude_ids:
                    placeholders = ",".join("?" for _ in exclude_ids)
                    query += f" AND id NOT IN ({placeholders})"
                    params.extend(exclude_ids)
                cursor = conn.execute(query, tuple(params))
                deleted_count = cursor.rowcount
                logger.info(
                    f"Database deleted {deleted_count} messages older than {timestamp} "
                    f"(excluding {len(exclude_ids) if exclude_ids else 0} IDs) in session {session_id}."
                )
                return deleted_count

    # Message CRUD
    def save_message(self, msg: Message) -> None:
        """Persist a new conversational message record.

        Args:
            msg: The Message object to store.
        """
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO messages
                    (id, session_id, chatbot_id, channel_id, sender, role, type,
                     content, metadata, timestamp, status, parent_id, summary_node_id, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        msg.id,
                        msg.session_id,
                        msg.chatbot_id,
                        msg.channel_id,
                        msg.sender,
                        msg.role,
                        msg.type,
                        msg.content,
                        json.dumps(msg.metadata),
                        msg.timestamp,
                        msg.status,
                        msg.parent_id,
                        msg.summary_node_id,
                    ),
                )

    def update_message_status(self, message_id: str, status: str) -> None:
        """Update the operational lifecycle status of a message.

        Args:
            message_id: Target message ID.
            status: New status string.
        """
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute("UPDATE messages SET status = ? WHERE id = ?", (status, message_id))

    def claim_message(self, message_id: str, new_status: str, expected_statuses: list[str]) -> bool:
        """Atomically update message status only if it is currently in one of expected_statuses.

        Args:
            message_id: Target message ID.
            new_status: The new status to set.
            expected_statuses: The list of valid current statuses.

        Returns:
            True if exactly one message was updated, False otherwise.
        """
        with self.connection_provider.connection() as conn:
            with conn:
                placeholders = ", ".join("?" for _ in expected_statuses)
                cursor = conn.execute(
                    f"UPDATE messages SET status = ? WHERE id = ? AND status IN ({placeholders})",
                    [new_status, message_id, *expected_statuses],
                )
                return cursor.rowcount == 1

    def update_message_metadata(self, message_id: str, metadata: dict[str, Any]) -> None:
        """Update the metadata dictionary of a message.

        Args:
            message_id: Target message ID.
            metadata: Dictionary representing the new metadata.
        """
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute("UPDATE messages SET metadata = ? WHERE id = ?", (json.dumps(metadata), message_id))

    def update_session_system_prompt(self, session_id: str, content: str) -> None:
        """Update the main system prompt content for an existing session.

        Args:
            session_id: Target session ID.
            content: New system prompt content string.
        """
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute(
                    "UPDATE sessions SET system_prompt = ? WHERE id = ?",
                    (content, session_id),
                )

    def get_session_messages(self, session_id: str) -> list[Message]:
        """Retrieve all raw persisted messages for a session in chronological turn order."""
        return self.get_session_history(session_id, limit=0)

    def get_session_history(
        self,
        session_id: str,
        limit: int = 20,
        order: Literal["phased", "grouped"] = "phased",
        active_only: bool = False,
    ) -> list[Message]:
        """Retrieve historical messages for a specific session ordered by logical conversational turn.

        Args:
            session_id: Session ID to query.
            limit: Max messages count (applied after filtering and sorting; 0 means unlimited).
            order: The sorting order. "phased" (default for Gemini) or "grouped" (for human displays).
            active_only: When True, drop bulky ASSISTANT/TOOL rows of already-compressed turns
                while retaining USER/SYSTEM rows (including their ``summary_node_id``) so logical
                turn segmentation preserves turn count, indices, compressed flags, and
                ``<recent_user_messages>`` content.

        Returns:
            List of Message objects ordered by the requested sorting mechanism.
        """
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            if active_only:
                cursor.execute(
                    """
                    SELECT * FROM messages
                    WHERE session_id = ? AND type != ?
                      AND (summary_node_id IS NULL OR role IN (?, ?))
                    """,
                    (session_id, MessageType.SESSION_RENAME, MessageRole.USER, MessageRole.SYSTEM),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM messages
                    WHERE session_id = ? AND type != ?
                    """,
                    (session_id, MessageType.SESSION_RENAME),
                )
            rows = cursor.fetchall()
            all_msgs = [self._row_to_message(row) for row in rows]

            all_msgs = sort_session_messages(all_msgs, order)

            return all_msgs[-limit:] if limit else all_msgs

    def get_message(self, message_id: str) -> Message | None:
        """Retrieve a specific message by its unique ID."""
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM messages WHERE id = ?", (message_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_message(row)
            return None

    def get_messages_by_filters(
        self,
        filters: dict[str, Any],
        exclude_statuses: list[str] | None = None,
        exclude_roles: list[str] | None = None,
    ) -> list[Message]:
        """Retrieve messages matching specific key-value filters and excluding specified statuses or roles.

        Args:
            filters: Dictionary of column=value criteria.
            exclude_statuses: Optional list of status strings to exclude.
            exclude_roles: Optional list of role strings to exclude.

        Returns:
            List of matching Message objects.
        """
        with self.connection_provider.connection() as conn:
            query = "SELECT * FROM messages"
            params: list[Any] = []
            clauses = []
            if filters:
                for k, v in filters.items():
                    clauses.append(f"{k} = ?")
                    params.append(v)
            if exclude_statuses:
                placeholders = ", ".join("?" for _ in exclude_statuses)
                clauses.append(f"status NOT IN ({placeholders})")
                params.extend(exclude_statuses)
            if exclude_roles:
                placeholders = ", ".join("?" for _ in exclude_roles)
                clauses.append(f"role NOT IN ({placeholders})")
                params.extend(exclude_roles)
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY timestamp ASC"

            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            return [self._row_to_message(row) for row in rows]

    def get_pending_deferred_media_messages(self, chatbot_id: str, since_timestamp: float) -> list[Message]:
        """Retrieve recent assistant messages for ``chatbot_id`` containing ``deferred_media`` metadata.

        Args:
            chatbot_id: Target chatbot identifier.
            since_timestamp: Minimum Unix timestamp to scan (e.g. ``now - 86400``).

        Returns:
            List of matching Message objects ordered chronologically.
        """
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM messages
                WHERE chatbot_id = ?
                  AND role = ?
                  AND timestamp >= ?
                  AND metadata LIKE '%"deferred_media"%'
                ORDER BY timestamp ASC
                """,
                (chatbot_id, MessageRole.ASSISTANT, since_timestamp),
            )
            rows = cursor.fetchall()
            return [self._row_to_message(row) for row in rows]

    def recover_orphaned_processing_messages(self, threshold_seconds: float = 300.0) -> int:
        """Recover messages that got stuck in 'processing' status by reverting them to 'pending_agent'.

        Args:
            threshold_seconds: Messages with 'processing' status older than this threshold in seconds will be reverted.

        Returns:
            The number of messages recovered.
        """
        with self.connection_provider.connection() as conn:
            with conn:
                cutoff = time.time() - threshold_seconds
                cursor = conn.execute(
                    f"UPDATE messages SET status = '{MessageStatus.PENDING_AGENT}' "
                    f"WHERE status = '{MessageStatus.PROCESSING}' AND timestamp < ?",
                    (cutoff,),
                )
                return cursor.rowcount

    def get_session_turns_count(self, session_id: str) -> int:
        """Count the number of user turns in a session.

        Args:
            session_id: The target session identifier.

        Returns:
            The count of user messages.
        """
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = '{MessageRole.USER}'",
                (session_id,),
            )
            return cursor.fetchone()[0]

    def get_session_turn_anchors(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve list of user and system messages with their ID, role and timestamp.

        Args:
            session_id: Target session ID.

        Returns:
            List of dicts containing message id, role and timestamp.
        """
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT id, role, timestamp FROM messages "
                f"WHERE session_id = ? AND role IN ('{MessageRole.USER}', '{MessageRole.SYSTEM}') "
                f"ORDER BY timestamp ASC",
                (session_id,),
            )
            rows = cursor.fetchall()
            return [{"id": r["id"], "role": r["role"], "timestamp": r["timestamp"]} for r in rows]

    def get_session_skill_anchor_ids(self, session_id: str) -> list[str]:
        """Retrieve the user message IDs of turns that loaded a skill successfully.

        Args:
            session_id: Target session ID.

        Returns:
            List of turn anchor message IDs.
        """
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT DISTINCT tc.parent_id FROM messages tr
                JOIN messages tc ON tr.parent_id = tc.id
                WHERE tr.session_id = ?
                  AND tr.role = '{MessageRole.TOOL}'
                  AND tr.type = '{MessageType.TOOL_RESULT}'
                  AND json_extract(tr.metadata, '$.tool_name') = 'use_skill'
                  AND json_extract(tr.metadata, '$.tool_error') IS NULL
                """,
                (session_id,),
            )
            return [r[0] for r in cursor.fetchall() if r[0]]

    def get_orphaned_tool_calls(self, session_id: str) -> list[Message]:
        """Retrieve all orphaned tool call messages in a session.

        Args:
            session_id: Target session ID.

        Returns:
            List of orphaned tool call Message objects.
        """
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT * FROM messages
                WHERE session_id = ?
                  AND type = '{MessageType.TOOL_CALL}'
                  AND id NOT IN (
                      SELECT parent_id FROM messages
                      WHERE session_id = ?
                        AND type = '{MessageType.TOOL_RESULT}'
                        AND parent_id IS NOT NULL
                  )
                """,
                (session_id, session_id),
            )
            rows = cursor.fetchall()
            return [self._row_to_message(row) for row in rows]

    def get_session_history_by_ranges(
        self, session_id: str, ranges: list[tuple[float, float | None]], order: Literal["phased", "grouped"] = "phased"
    ) -> list[Message]:
        """Retrieve historical messages for specific timestamp ranges in a session, sorted logically.

        Args:
            session_id: Session ID to query.
            ranges: List of (start_timestamp, end_timestamp_or_None) tuples.
            order: The sorting order. "phased" or "grouped".

        Returns:
            List of Message objects, logically ordered.
        """
        if not ranges:
            return []

        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            clauses = []
            params: list[Any] = [session_id]
            for start, end in ranges:
                if end is None:
                    clauses.append("(timestamp >= ?)")
                    params.append(start)
                else:
                    clauses.append("(timestamp >= ? AND timestamp < ?)")
                    params.extend([start, end])

            query = f"""
                SELECT * FROM messages
                WHERE session_id = ?
                  AND type != ?
                  AND ({" OR ".join(clauses)})
            """
            params.insert(1, MessageType.SESSION_RENAME)
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            all_msgs = [self._row_to_message(row) for row in rows]

            all_msgs = sort_session_messages(all_msgs, order)

            return all_msgs

    def get_memory_files_by_role(
        self,
        role: str,
        *,
        visibility: str | None = None,
        exclude_pinned: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve indexed memory file metadata rows for a role.

        Args:
            role: Role persona identifier.
            visibility: Optional exact visibility filter (e.g., `'PINNED'`, `'DEFAULT'`).
            exclude_pinned: If True, exclude `'PINNED'` and `'HIDDEN'` rows.

        Returns:
            List of dictionaries representing rows from ``memory_files``.
        """
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            clauses = ["role = ?"]
            params: list[Any] = [role]
            if visibility is not None:
                clauses.append("visibility = ?")
                params.append(visibility)
            elif exclude_pinned:
                clauses.append("visibility = 'DEFAULT'")
            cursor.execute(
                f"""
                SELECT id, role, rel_path, description, visibility, size_bytes, content_hash, mtime, updated_at
                FROM memory_files
                WHERE {" AND ".join(clauses)}
                ORDER BY mtime DESC, rel_path ASC
                """,
                tuple(params),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_memory_role_signature(self, role: str) -> tuple[int, float]:
        """Return `(file_count, max_updated_at)` for `role` to validate in-memory caches cheaply."""
        with self.connection_provider.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(MAX(updated_at), 0.0) FROM memory_files WHERE role = ?",
                (role,),
            ).fetchone()
            return (int(row[0] or 0), float(row[1] or 0.0)) if row else (0, 0.0)

    def upsert_memory_file(
        self,
        role: str,
        rel_path: str,
        description: str,
        content_hash: str,
        mtime: float,
        updated_at: float,
        chunks: list[tuple[int, str, bytes]],
        *,
        visibility: str = "DEFAULT",
        size_bytes: int = 0,
    ) -> int:
        """Insert or update a memory file entry and replace its chunk embeddings atomically.

        Args:
            role: Role persona identifier.
            rel_path: Path relative to the role's memories directory (normalized with '/').
            description: Single-line summary from frontmatter or first heading.
            content_hash: SHA-256 hex digest of the raw file content.
            mtime: Filesystem modification timestamp.
            updated_at: Index update timestamp.
            chunks: List of ``(chunk_index, content, embedding_bytes)`` tuples.
            visibility: `'DEFAULT'`, `'PINNED'`, or `'HIDDEN'`.
            size_bytes: UTF-8 byte length of the raw file.

        Returns:
            The ``memory_files.id`` primary key of the upserted row.
        """
        norm_vis = (visibility or "DEFAULT").strip().upper()
        if norm_vis not in {"DEFAULT", "PINNED", "HIDDEN"}:
            norm_vis = "DEFAULT"
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO memory_files
                    (role, rel_path, description, visibility, size_bytes, content_hash, mtime, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(role, rel_path) DO UPDATE SET
                        description = excluded.description,
                        visibility = excluded.visibility,
                        size_bytes = excluded.size_bytes,
                        content_hash = excluded.content_hash,
                        mtime = excluded.mtime,
                        updated_at = excluded.updated_at
                    """,
                    (role, rel_path, description, norm_vis, max(0, int(size_bytes)), content_hash, mtime, updated_at),
                )
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id FROM memory_files WHERE role = ? AND rel_path = ?",
                    (role, rel_path),
                )
                row = cursor.fetchone()
                if not row:
                    raise RuntimeError(f"Failed to retrieve memory_files id for {role}/{rel_path}")
                file_id = int(row["id"])

                conn.execute("DELETE FROM memory_chunks WHERE file_id = ?", (file_id,))
                if chunks:
                    conn.executemany(
                        """
                        INSERT INTO memory_chunks (file_id, chunk_index, content, embedding)
                        VALUES (?, ?, ?, ?)
                        """,
                        [(file_id, idx, text, emb) for idx, text, emb in chunks],
                    )
                return file_id

    def delete_memory_files_by_ids(self, file_ids: list[int]) -> None:
        """Delete memory file entries (and their cascaded chunks) by ID.

        Args:
            file_ids: List of ``memory_files.id`` values to delete.
        """
        if not file_ids:
            return
        placeholders = ",".join("?" for _ in file_ids)
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute(
                    f"DELETE FROM memory_files WHERE id IN ({placeholders})",
                    tuple(file_ids),
                )

    def clear_role_memory_index(self, role: str | None = None) -> None:
        """Delete all indexed memory files and chunks (for a specific role or all roles).

        Args:
            role: Optional role name. If None, clears all roles.
        """
        with self.connection_provider.connection() as conn:
            with conn:
                if role:
                    conn.execute("DELETE FROM memory_files WHERE role = ?", (role,))
                else:
                    conn.execute("DELETE FROM memory_files")

    def get_memory_chunks_for_role(self, role: str, *, include_pinned: bool = True) -> list[sqlite3.Row]:
        """Fetch memory chunks joined with their parent file metadata for a role.

        Args:
            role: Role persona identifier.
            include_pinned: When False, return only `visibility = 'DEFAULT'` chunks (excluding PINNED and HIDDEN).

        Returns:
            List of sqlite3.Row objects with file and chunk fields.
        """
        vis_clause = "AND mf.visibility != 'HIDDEN'" if include_pinned else "AND mf.visibility = 'DEFAULT'"
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT
                    mf.id AS file_id,
                    mf.rel_path AS rel_path,
                    mf.description AS description,
                    mf.visibility AS visibility,
                    mf.size_bytes AS size_bytes,
                    mf.mtime AS mtime,
                    mf.updated_at AS updated_at,
                    mc.chunk_index AS chunk_index,
                    mc.content AS content,
                    mc.embedding AS chunk_embedding
                FROM memory_chunks mc
                JOIN memory_files mf ON mc.file_id = mf.id
                WHERE mf.role = ? {vis_clause}
                ORDER BY mf.id ASC, mc.chunk_index ASC
                """,
                (role,),
            )
            return cursor.fetchall()

    def save_ui_prompt(self, session_id: str, message_id: str, spec: dict[str, Any]) -> None:
        """Insert or update the spec of an interactive UI prompt (created_at is kept on update)."""
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO ui_prompts (session_id, message_id, spec_json, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(session_id, message_id) DO UPDATE SET spec_json = excluded.spec_json
                    """,
                    (session_id, message_id, json.dumps(spec, ensure_ascii=False), time.time()),
                )

    def get_ui_prompt(self, session_id: str, message_id: str) -> dict[str, Any] | None:
        """Load the spec of an interactive UI prompt, or None if unknown."""
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT spec_json FROM ui_prompts WHERE session_id = ? AND message_id = ?",
                (session_id, message_id),
            )
            row = cursor.fetchone()
            if not row:
                return None
            try:
                spec = json.loads(row["spec_json"])
            except (TypeError, ValueError):
                logger.warning(f"Corrupt ui_prompts spec for session={session_id} message={message_id}")
                return None
            return spec if isinstance(spec, dict) else None

    def set_channel_role(self, chatbot_id: str, channel_id: str, role: str) -> None:
        """Bind a roleplay persona to a chatbot channel/thread."""
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO channel_roles (chatbot_id, channel_id, role)
                    VALUES (?, ?, ?)
                    """,
                    (chatbot_id, channel_id, role),
                )

    def get_channel_role(self, chatbot_id: str, channel_id: str) -> str | None:
        """Retrieve the roleplay persona bound directly to a chatbot channel/thread."""
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role FROM channel_roles WHERE chatbot_id = ? AND channel_id = ?",
                (chatbot_id, channel_id),
            )
            row = cursor.fetchone()
            return row["role"] if row else None

    def get_channel_role_with_inheritance(
        self,
        chatbot_id: str,
        channel_id: str,
        session_id: str | None = None,
        *,
        default: str | None = None,
    ) -> str:
        """Retrieve the active role bound to a channel/thread with parent inheritance support."""
        resolved_default = default if default is not None else self._get_chatbot_default_role(chatbot_id)

        # 1. Direct lookup
        role = self.get_channel_role(chatbot_id, channel_id)
        if role:
            return role

        # 2. Discord parent channel inheritance (scan most recent user message that carries parent_channel_id,
        # so UI button / simulated choice messages without parent_channel_id do not shadow earlier thread messages)
        if chatbot_id.startswith("discord"):
            with self.connection_provider.connection() as conn:
                try:
                    cursor = conn.cursor()
                    if session_id:
                        cursor.execute(
                            "SELECT json_extract(metadata, '$.parent_channel_id') AS parent_id "
                            "FROM messages WHERE session_id = ? AND role = 'user' "
                            "AND json_extract(metadata, '$.parent_channel_id') IS NOT NULL "
                            "AND json_extract(metadata, '$.parent_channel_id') != '' "
                            "ORDER BY timestamp DESC LIMIT 1",
                            (session_id,),
                        )
                    else:
                        cursor.execute(
                            "SELECT json_extract(metadata, '$.parent_channel_id') AS parent_id "
                            "FROM messages WHERE chatbot_id = ? AND channel_id = ? AND role = 'user' "
                            "AND json_extract(metadata, '$.parent_channel_id') IS NOT NULL "
                            "AND json_extract(metadata, '$.parent_channel_id') != '' "
                            "ORDER BY timestamp DESC LIMIT 1",
                            (chatbot_id, channel_id),
                        )
                    row = cursor.fetchone()
                    if row and row["parent_id"]:
                        role = self.get_channel_role(chatbot_id, str(row["parent_id"]))
                        if role:
                            return role
                except Exception as e:
                    logger.warning(f"Failed to retrieve parent channel role: {e}")

        # 3. Fallback to the session's own persisted role_name if no explicit channel/parent role was found
        if session_id:
            with self.connection_provider.connection() as conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT role_name FROM sessions WHERE id = ?", (session_id,))
                    row = cursor.fetchone()
                    if row and row["role_name"] and row["role_name"] not in ("default", resolved_default):
                        return str(row["role_name"])
                except Exception as e:
                    logger.warning(f"Failed to retrieve session role fallback: {e}")

        return resolved_default

    def get_channel_by_session(self, session_id: str) -> tuple[str, str] | None:
        """Retrieve the chatbot_id and channel_id mapping for a given session ID."""
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT chatbot_id, channel_id FROM channel_sessions WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            return (row["chatbot_id"], row["channel_id"]) if row else None

    def get_last_message_timestamp(self, chatbot_id: str, channel_id: str) -> float | None:
        """Retrieve the timestamp of the most recent user or assistant message in a channel."""
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT timestamp FROM messages
                WHERE chatbot_id = ?
                  AND (channel_id = ? OR json_extract(metadata, '$.parent_channel_id') = ?)
                  AND role IN ('user', 'assistant')
                ORDER BY timestamp DESC LIMIT 1
                """,
                (chatbot_id, channel_id, channel_id),
            )
            row = cursor.fetchone()
            return row["timestamp"] if row else None

    def get_cronjob_sent_stats_today(self, chatbot_id: str, channel_id: str) -> tuple[int, float | None]:
        """Retrieve count and last timestamp of cron messages sent today (local time) in a channel."""
        now = datetime.datetime.now()
        # Local midnight timestamp
        midnight = datetime.datetime(now.year, now.month, now.day)
        midnight_ts = midnight.timestamp()

        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(*), MAX(timestamp) FROM messages
                WHERE chatbot_id = ?
                  AND (channel_id = ? OR json_extract(metadata, '$.parent_channel_id') = ?)
                  AND sender = 'Cronjob'
                  AND timestamp >= ?
                """,
                (chatbot_id, channel_id, channel_id, midnight_ts),
            )
            row = cursor.fetchone()
            count = row[0] if row and row[0] is not None else 0
            last_ts = row[1] if row and row[1] is not None else None
            return count, last_ts

    # --- Cross-Session Digests, FTS5, and Memory Extraction Cursor DAOs ---

    _DIGEST_SCALAR_COLS = (
        "session_id, role, chatbot_id, channel_id, title, started_at, last_active_at, "
        "turn_count, last_user_msgs, context_policy, summary, open_threads, summary_upto_ts, updated_at"
    )
    _MSG_EXPLICIT_COLS = (
        "id, session_id, chatbot_id, channel_id, sender, role, type, content, metadata, "
        "timestamp, status, parent_id, summary_node_id"
    )

    def upsert_session_digest(
        self,
        session_id: str,
        role: str,
        *,
        chatbot_id: str | None = None,
        channel_id: str | None = None,
        title: str | None = None,
        started_at: float | None = None,
        last_active_at: float | None = None,
        turn_count: int | None = None,
        last_user_msgs: list[str] | str | None = None,
        context_policy: str | None = None,
        summary: str | None = None,
        open_threads: list[str] | str | None = None,
        summary_upto_ts: float | None = None,
        embedding: bytes | None = None,
        updated_at: float | None = None,
    ) -> None:
        """Insert or partially update a `session_digests` record and refresh `session_digests_fts`."""
        if not self.digests_available:
            return
        if chatbot_id and chatbot_id.strip().lower() == "cronjob":
            return
        now = updated_at if updated_at is not None else time.time()
        safe_role = role or "default"

        with self.connection_provider.connection() as conn:
            with conn:
                existing = conn.execute(
                    f"SELECT {self._DIGEST_SCALAR_COLS} FROM session_digests WHERE session_id = ?",
                    (session_id,),
                ).fetchone()

                if existing is None:
                    eff_chatbot = chatbot_id
                    eff_channel = channel_id
                    eff_title = title if title is not None else ""
                    eff_started = float(started_at if started_at is not None else now)
                    eff_active = float(last_active_at if last_active_at is not None else eff_started)
                    eff_turns = int(turn_count if turn_count is not None else 0)
                    if isinstance(last_user_msgs, list):
                        eff_user_msgs = json.dumps(last_user_msgs, ensure_ascii=False)
                    elif isinstance(last_user_msgs, str):
                        eff_user_msgs = last_user_msgs
                    else:
                        eff_user_msgs = "[]"
                    eff_policy = context_policy or "full"
                    eff_summary = summary if summary is not None else ""
                    if isinstance(open_threads, list):
                        eff_threads = json.dumps(open_threads, ensure_ascii=False)
                    elif isinstance(open_threads, str):
                        eff_threads = open_threads
                    else:
                        eff_threads = "[]"
                    eff_upto = summary_upto_ts

                    conn.execute(
                        """
                        INSERT INTO session_digests (
                            session_id, role, chatbot_id, channel_id, title,
                            started_at, last_active_at, turn_count, last_user_msgs,
                            context_policy, summary, open_threads, summary_upto_ts,
                            embedding, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            safe_role,
                            eff_chatbot,
                            eff_channel,
                            eff_title,
                            eff_started,
                            eff_active,
                            eff_turns,
                            eff_user_msgs,
                            eff_policy,
                            eff_summary,
                            eff_threads,
                            eff_upto,
                            embedding,
                            now,
                        ),
                    )
                else:
                    eff_role = safe_role if role else str(existing["role"])
                    eff_chatbot = chatbot_id if chatbot_id is not None else existing["chatbot_id"]
                    eff_channel = channel_id if channel_id is not None else existing["channel_id"]
                    eff_title = title if title is not None else str(existing["title"] or "")
                    eff_started = float(started_at if started_at is not None else existing["started_at"])
                    eff_active = (
                        max(float(existing["last_active_at"]), float(last_active_at))
                        if last_active_at is not None
                        else float(existing["last_active_at"])
                    )
                    eff_turns = int(turn_count if turn_count is not None else existing["turn_count"])
                    if isinstance(last_user_msgs, list):
                        eff_user_msgs = json.dumps(last_user_msgs, ensure_ascii=False)
                    elif isinstance(last_user_msgs, str):
                        eff_user_msgs = last_user_msgs
                    else:
                        eff_user_msgs = str(existing["last_user_msgs"] or "[]")
                    eff_policy = context_policy if context_policy is not None else str(existing["context_policy"])
                    eff_summary = summary if summary is not None else str(existing["summary"] or "")
                    if isinstance(open_threads, list):
                        eff_threads = json.dumps(open_threads, ensure_ascii=False)
                    elif isinstance(open_threads, str):
                        eff_threads = open_threads
                    else:
                        eff_threads = str(existing["open_threads"] or "[]")
                    eff_upto = summary_upto_ts if summary_upto_ts is not None else existing["summary_upto_ts"]

                    if embedding is not None:
                        conn.execute(
                            """
                            UPDATE session_digests SET
                                role = ?, chatbot_id = ?, channel_id = ?, title = ?,
                                started_at = ?, last_active_at = ?, turn_count = ?,
                                last_user_msgs = ?, context_policy = ?, summary = ?,
                                open_threads = ?, summary_upto_ts = ?, embedding = ?, updated_at = ?
                            WHERE session_id = ?
                            """,
                            (
                                eff_role,
                                eff_chatbot,
                                eff_channel,
                                eff_title,
                                eff_started,
                                eff_active,
                                eff_turns,
                                eff_user_msgs,
                                eff_policy,
                                eff_summary,
                                eff_threads,
                                eff_upto,
                                embedding,
                                now,
                                session_id,
                            ),
                        )
                    else:
                        conn.execute(
                            """
                            UPDATE session_digests SET
                                role = ?, chatbot_id = ?, channel_id = ?, title = ?,
                                started_at = ?, last_active_at = ?, turn_count = ?,
                                last_user_msgs = ?, context_policy = ?, summary = ?,
                                open_threads = ?, summary_upto_ts = ?, updated_at = ?
                            WHERE session_id = ?
                            """,
                            (
                                eff_role,
                                eff_chatbot,
                                eff_channel,
                                eff_title,
                                eff_started,
                                eff_active,
                                eff_turns,
                                eff_user_msgs,
                                eff_policy,
                                eff_summary,
                                eff_threads,
                                eff_upto,
                                now,
                                session_id,
                            ),
                        )
                    safe_role = eff_role

                self._refresh_digest_fts_in_tx(
                    conn,
                    session_id=session_id,
                    role=safe_role,
                    title=eff_title,
                    summary=eff_summary,
                    last_user_msgs_json=eff_user_msgs,
                )

    def get_session_digest(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve scalar columns of a single `session_digests` row (never selecting `embedding`)."""
        if not self.digests_available:
            return None
        with self.connection_provider.connection() as conn:
            row = conn.execute(
                f"SELECT {self._DIGEST_SCALAR_COLS} FROM session_digests WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return dict(row) if row is not None else None

    def get_recent_session_digests(
        self,
        role: str,
        exclude_session_id: str | None = None,
        since_ts: float = 0.0,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve recent non-cronjob `session_digests` rows for `role` (never selecting `embedding`)."""
        if not self.digests_available or limit <= 0:
            return []
        with self.connection_provider.connection() as conn:
            clauses = ["role = ?", "last_active_at >= ?", "COALESCE(LOWER(chatbot_id), '') <> 'cronjob'"]
            params: list[Any] = [role or "default", float(since_ts)]
            if exclude_session_id:
                clauses.append("session_id <> ?")
                params.append(exclude_session_id)
            params.append(int(limit))
            rows = conn.execute(
                f"""
                SELECT {self._DIGEST_SCALAR_COLS}
                FROM session_digests
                WHERE {" AND ".join(clauses)}
                ORDER BY last_active_at DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_role_digest_embeddings(
        self,
        role: str,
        exclude_session_id: str | None = None,
        limit: int = 200,
    ) -> list[sqlite3.Row]:
        """Fetch bounded non-isolated episode digest embeddings for ambient Tier 3 episode hits."""
        if not self.digests_available or limit <= 0:
            return []
        with self.connection_provider.connection() as conn:
            clauses = [
                "role = ?",
                "context_policy <> 'isolated'",
                "embedding IS NOT NULL",
                "COALESCE(LOWER(chatbot_id), '') <> 'cronjob'",
            ]
            params: list[Any] = [role or "default"]
            if exclude_session_id:
                clauses.append("session_id <> ?")
                params.append(exclude_session_id)
            params.append(int(limit))
            return conn.execute(
                f"""
                SELECT session_id, title, summary, last_active_at, context_policy, embedding AS chunk_embedding
                FROM session_digests
                WHERE {" AND ".join(clauses)}
                ORDER BY last_active_at DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()

    def fts_insert_messages(self, entries: Sequence[dict[str, Any]]) -> int:
        """Idempotently insert conversational message rows into `messages_fts`.

        Args:
            entries: Sequence of dicts containing `message_id`, `session_id`, `persona_role`,
                `context_policy`, `speaker_role`, `ts`, `raw_excerpt`, and `text`.

        Returns:
            Number of newly inserted FTS rows.
        """
        if not self.fts_available or not entries:
            return 0
        valid = [e for e in entries if e.get("message_id") and e.get("session_id") and e.get("text")]
        if not valid:
            return 0

        by_session: dict[str, list[dict[str, Any]]] = {}
        for item in valid:
            by_session.setdefault(str(item["session_id"]), []).append(item)

        inserted = 0
        with self.connection_provider.connection() as conn:
            with conn:
                for sid, items in by_session.items():
                    safe_sid = sid.replace('"', '""')
                    existing_ids = {
                        str(r[0])
                        for r in conn.execute(
                            "SELECT message_id FROM messages_fts WHERE messages_fts MATCH ?",
                            (f'session_id:"{safe_sid}"',),
                        ).fetchall()
                    }
                    to_insert = [
                        (
                            str(it["message_id"]),
                            sid,
                            str(it.get("persona_role") or "default"),
                            str(it.get("context_policy") or "full"),
                            str(it.get("speaker_role") or "user"),
                            float(it.get("ts") or 0.0),
                            str(it.get("raw_excerpt") or ""),
                            str(it["text"]),
                        )
                        for it in items
                        if str(it["message_id"]) not in existing_ids
                    ]
                    if to_insert:
                        conn.executemany(
                            """
                            INSERT INTO messages_fts (
                                message_id, session_id, persona_role, context_policy,
                                speaker_role, ts, raw_excerpt, text
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            to_insert,
                        )
                        inserted += len(to_insert)
        return inserted

    def fts_delete_session(self, session_id: str) -> None:
        """Remove all FTS5 index entries belonging to `session_id`."""
        if not self.fts_available:
            return
        safe_sid = session_id.replace('"', '""')
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute("DELETE FROM messages_fts WHERE messages_fts MATCH ?", (f'session_id:"{safe_sid}"',))
                conn.execute("DELETE FROM session_digests_fts WHERE session_id = ?", (session_id,))

    def fts_search_messages(
        self,
        role: str,
        query: str,
        limit: int = 40,
        *,
        include_isolated: bool = False,
        exclude_session_id: str | None = None,
        since_ts: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search `messages_fts` scoped inside `MATCH` by `persona_role` and `context_policy` (`C2`, `R2-2`)."""
        if not self.fts_available or limit <= 0:
            return []
        fts_q = fts_tokenize(query, for_query=True)
        if not fts_q:
            return []

        safe_role = (role or "default").replace('"', '""')
        match_expr = f'(persona_role:"{safe_role}" AND text:({fts_q}))'
        if not include_isolated:
            match_expr = f'({match_expr}) NOT context_policy:"isolated"'
        if exclude_session_id:
            safe_excl = exclude_session_id.replace('"', '""')
            match_expr = f'({match_expr}) NOT session_id:"{safe_excl}"'

        sql = """
            SELECT
                message_id, session_id, persona_role, context_policy,
                speaker_role, ts, raw_excerpt, bm25(messages_fts) AS rank
            FROM messages_fts
            WHERE messages_fts MATCH ?
        """
        params: list[Any] = [match_expr]
        if since_ts is not None:
            sql += " AND CAST(ts AS REAL) >= ?"
            params.append(float(since_ts))
        sql += " ORDER BY rank ASC LIMIT ?"
        params.append(int(limit))

        with self.connection_provider.connection() as conn:
            try:
                rows = conn.execute(sql, tuple(params)).fetchall()
                return [dict(r) for r in rows]
            except sqlite3.OperationalError as e:
                logger.warning(f"fts_search_messages query failed ({match_expr!r}): {e}")
                return []

    def fts_search_digests(
        self,
        role: str,
        query: str,
        limit: int = 5,
        *,
        include_isolated: bool = False,
        exclude_session_id: str | None = None,
        since_ts: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search `session_digests_fts` joined with scalar `session_digests` metadata (Stage 1)."""
        if not self.fts_available or not self.digests_available or limit <= 0:
            return []
        fts_q = fts_tokenize(query, for_query=True)
        if not fts_q:
            return []

        safe_role = (role or "default").replace('"', '""')
        match_expr = f'role:"{safe_role}" AND text:({fts_q})'
        clauses = [
            "session_digests_fts MATCH ?",
            "sd.role = ?",
            "COALESCE(LOWER(sd.chatbot_id), '') <> 'cronjob'",
        ]
        params: list[Any] = [match_expr, role or "default"]
        if not include_isolated:
            clauses.append("sd.context_policy <> 'isolated'")
        if exclude_session_id:
            clauses.append("sd.session_id <> ?")
            params.append(exclude_session_id)
        if since_ts is not None:
            clauses.append("sd.last_active_at >= ?")
            params.append(float(since_ts))
        params.append(int(limit))

        sql = f"""
            SELECT
                sd.session_id, sd.role, sd.chatbot_id, sd.channel_id, sd.title,
                sd.started_at, sd.last_active_at, sd.turn_count, sd.last_user_msgs,
                sd.context_policy, sd.summary, sd.open_threads, sd.summary_upto_ts, sd.updated_at,
                bm25(session_digests_fts) AS rank
            FROM session_digests_fts
            JOIN session_digests sd ON sd.session_id = session_digests_fts.session_id
            WHERE {" AND ".join(clauses)}
            ORDER BY rank ASC, sd.last_active_at DESC
            LIMIT ?
        """
        with self.connection_provider.connection() as conn:
            try:
                rows = conn.execute(sql, tuple(params)).fetchall()
                return [dict(r) for r in rows]
            except sqlite3.OperationalError as e:
                logger.warning(f"fts_search_digests query failed ({match_expr!r}): {e}")
                return []

    def get_messages_text_by_ids(self, message_ids: Sequence[str]) -> dict[str, str]:
        """Hydrate `id -> content` for `<= 10` displayed snippet IDs via PK lookup (never `embedding`; `R2-3`)."""
        ids = [str(mid) for mid in message_ids if mid][:10]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self.connection_provider.connection() as conn:
            rows = conn.execute(
                f"SELECT id, content FROM messages WHERE id IN ({placeholders})",
                tuple(ids),
            ).fetchall()
            return {str(r["id"]): str(r["content"] or "") for r in rows}

    def get_session_view_messages(
        self,
        session_id: str,
        around_message_id: str | None = None,
        window: int = 6,
        max_messages: int = 20,
    ) -> list[Message]:
        """Fetch bounded conversational messages around `around_message_id` (explicit columns, never `embedding`)."""
        cap = max(1, min(20, int(max_messages)))
        win = max(1, int(window))
        with self.connection_provider.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT {self._MSG_EXPLICIT_COLS}
                FROM messages
                WHERE session_id = ?
                  AND type NOT IN (?, ?)
                ORDER BY timestamp ASC, id ASC
                """,
                (session_id, MessageType.THOUGHT, MessageType.SESSION_RENAME),
            ).fetchall()
            all_msgs = sort_session_messages([self._row_to_message(r) for r in rows], "grouped")
            if not all_msgs:
                return []
            if around_message_id:
                center_idx = next((i for i, m in enumerate(all_msgs) if m.id == around_message_id), None)
                if center_idx is not None:
                    lo = max(0, center_idx - win)
                    hi = min(len(all_msgs), center_idx + win + 1)
                    sliced = all_msgs[lo:hi]
                    if len(sliced) > cap:
                        half = cap // 2
                        new_lo = max(0, center_idx - half)
                        sliced = all_msgs[new_lo : new_lo + cap]
                    return sliced
            take = min(cap, win * 2 + 1)
            return all_msgs[-take:]

    def get_extraction_cursor(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve the `memory_extraction_cursors` row for `session_id`."""
        with self.connection_provider.connection() as conn:
            try:
                row = conn.execute(
                    "SELECT * FROM memory_extraction_cursors WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                return dict(row) if row is not None else None
            except sqlite3.OperationalError:
                return None

    def upsert_extraction_cursor(
        self,
        session_id: str,
        role: str,
        *,
        last_extracted_msg_ts: float | None = None,
        last_extracted_msg_id: str | None = None,
        pending_user_turns: int | None = None,
        pending_chars: int | None = None,
        last_run_at: float | None = None,
        last_reason: str | None = None,
        tokens_spent: int | None = None,
        failures: int | None = None,
    ) -> dict[str, Any]:
        """Insert or update `memory_extraction_cursors` for `session_id` and return the updated row."""
        with self.connection_provider.connection() as conn:
            with conn:
                prev = conn.execute(
                    "SELECT * FROM memory_extraction_cursors WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                eff_role = role or (str(prev["role"]) if prev else "default")
                eff_ts = (
                    float(last_extracted_msg_ts)
                    if last_extracted_msg_ts is not None
                    else (float(prev["last_extracted_msg_ts"]) if prev else 0.0)
                )
                eff_id = (
                    str(last_extracted_msg_id)
                    if last_extracted_msg_id is not None
                    else (str(prev["last_extracted_msg_id"]) if prev else "")
                )
                eff_turns = (
                    max(0, int(pending_user_turns))
                    if pending_user_turns is not None
                    else (int(prev["pending_user_turns"]) if prev else 0)
                )
                eff_chars = (
                    max(0, int(pending_chars))
                    if pending_chars is not None
                    else (int(prev["pending_chars"]) if prev else 0)
                )
                eff_run_at = (
                    float(last_run_at)
                    if last_run_at is not None
                    else (prev["last_run_at"] if prev else None)
                )
                eff_reason = (
                    str(last_reason)
                    if last_reason is not None
                    else (prev["last_reason"] if prev else None)
                )
                eff_tokens = (
                    max(0, int(tokens_spent))
                    if tokens_spent is not None
                    else (int(prev["tokens_spent"]) if prev else 0)
                )
                eff_failures = (
                    max(0, int(failures))
                    if failures is not None
                    else (int(prev["failures"]) if prev else 0)
                )
                conn.execute(
                    """
                    INSERT INTO memory_extraction_cursors (
                        session_id, role, last_extracted_msg_ts, last_extracted_msg_id,
                        pending_user_turns, pending_chars, last_run_at, last_reason,
                        tokens_spent, failures
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        role = excluded.role,
                        last_extracted_msg_ts = excluded.last_extracted_msg_ts,
                        last_extracted_msg_id = excluded.last_extracted_msg_id,
                        pending_user_turns = excluded.pending_user_turns,
                        pending_chars = excluded.pending_chars,
                        last_run_at = excluded.last_run_at,
                        last_reason = excluded.last_reason,
                        tokens_spent = excluded.tokens_spent,
                        failures = excluded.failures
                    """,
                    (
                        session_id,
                        eff_role,
                        eff_ts,
                        eff_id,
                        eff_turns,
                        eff_chars,
                        eff_run_at,
                        eff_reason,
                        eff_tokens,
                        eff_failures,
                    ),
                )
                updated = conn.execute(
                    "SELECT * FROM memory_extraction_cursors WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                return dict(updated) if updated else {}

    def list_extraction_cursors(
        self,
        role: str | None = None,
        min_pending_turns: int = 0,
    ) -> list[dict[str, Any]]:
        """List `memory_extraction_cursors` rows optionally filtered by `role` and `pending_user_turns`."""
        with self.connection_provider.connection() as conn:
            try:
                clauses = ["pending_user_turns >= ?"]
                params: list[Any] = [int(min_pending_turns)]
                if role is not None:
                    clauses.append("role = ?")
                    params.append(role)
                rows = conn.execute(
                    f"""
                    SELECT * FROM memory_extraction_cursors
                    WHERE {" AND ".join(clauses)}
                    ORDER BY pending_user_turns DESC, session_id ASC
                    """,
                    tuple(params),
                ).fetchall()
                return [dict(r) for r in rows]
            except sqlite3.OperationalError:
                return []

    def get_unextracted_messages(
        self,
        session_id: str,
        after_ts: float = 0.0,
        after_id: str = "",
        safe_upper: float | tuple[float, str] | None = None,
        limit: int = 200,
    ) -> list[Message]:
        """Select completed, unextracted messages strictly before any in-flight USER turn (`C4`, `R2-1`).

        Bounds `(after_ts, after_id) < (timestamp, id) <= safe_upper`, where `safe_upper` is strictly
        before the earliest USER anchor with `status IN ('pending_agent', 'processing')`
        (`MessageStatus.PENDING_AGENT`, `MessageStatus.PROCESSING` — never `'pending'`).
        Selects explicit scalar columns only (never `embedding`), ordered by `(timestamp ASC, id ASC)`.
        """
        if limit <= 0:
            return []
        with self.connection_provider.connection() as conn:
            inflight_anchor = conn.execute(
                """
                SELECT timestamp, id
                FROM messages
                WHERE session_id = ?
                  AND role = ?
                  AND status IN (?, ?)
                ORDER BY timestamp ASC, id ASC
                LIMIT 1
                """,
                (
                    session_id,
                    MessageRole.USER,
                    MessageStatus.PENDING_AGENT,
                    MessageStatus.PROCESSING,
                ),
            ).fetchone()

            clauses = [
                "session_id = ?",
                "type != ?",
                "COALESCE(LOWER(chatbot_id), '') <> 'cronjob'",
                "NOT (role = 'user' AND LOWER(sender) IN ('cronjob', 'system'))",
                "json_extract(metadata, '$.prompt_blocked') IS NULL",
                "(timestamp > ? OR (timestamp = ? AND id > ?))",
            ]
            params: list[Any] = [
                session_id,
                MessageType.SESSION_RENAME,
                float(after_ts),
                float(after_ts),
                str(after_id or ""),
            ]

            if inflight_anchor is not None:
                inf_ts = float(inflight_anchor["timestamp"])
                inf_id = str(inflight_anchor["id"])
                clauses.append("(timestamp < ? OR (timestamp = ? AND id < ?))")
                params.extend([inf_ts, inf_ts, inf_id])

            if isinstance(safe_upper, tuple) and len(safe_upper) == 2:
                up_ts, up_id = float(safe_upper[0]), str(safe_upper[1])
                clauses.append("(timestamp < ? OR (timestamp = ? AND id <= ?))")
                params.extend([up_ts, up_ts, up_id])
            elif isinstance(safe_upper, (int, float)):
                clauses.append("timestamp <= ?")
                params.append(float(safe_upper))

            eff_limit = max(1, min(200, int(limit)))
            params.append(eff_limit + 1)
            rows = conn.execute(
                f"""
                SELECT {self._MSG_EXPLICIT_COLS}
                FROM messages
                WHERE {" AND ".join(clauses)}
                ORDER BY timestamp ASC, id ASC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
            msgs = [self._row_to_message(r) for r in rows]
            if len(msgs) > eff_limit:
                next_msg = msgs.pop()
                if next_msg.role != MessageRole.USER:
                    last_user_idx = next(
                        (i for i in range(len(msgs) - 1, -1, -1) if msgs[i].role == MessageRole.USER),
                        None,
                    )
                    if last_user_idx is not None and last_user_idx > 0:
                        msgs = msgs[:last_user_idx]
            return msgs

    def get_session_tool_calls(self, session_id: str) -> list[Message]:
        """Fetch all `TOOL_CALL` rows of `session_id` ordered by `(timestamp ASC, id ASC)` (never `embedding`)."""
        with self.connection_provider.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT {self._MSG_EXPLICIT_COLS}
                FROM messages
                WHERE session_id = ?
                  AND type = ?
                ORDER BY timestamp ASC, id ASC
                """,
                (session_id, MessageType.TOOL_CALL),
            ).fetchall()
            return [self._row_to_message(r) for r in rows]

    def log_extraction(
        self,
        *,
        role: str,
        session_id: str | None,
        reason: str,
        started_at: float,
        finished_at: float,
        status: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        ops_applied: int = 0,
        error: str | None = None,
    ) -> None:
        """Append an observability entry to `memory_extraction_log`."""
        with self.connection_provider.connection() as conn:
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO memory_extraction_log (
                            role, session_id, reason, started_at, finished_at,
                            status, input_tokens, output_tokens, ops_applied, error
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            role,
                            session_id,
                            reason,
                            float(started_at),
                            float(finished_at),
                            status,
                            int(input_tokens),
                            int(output_tokens),
                            int(ops_applied),
                            error,
                        ),
                    )
            except sqlite3.OperationalError as e:
                logger.warning(f"Failed writing memory_extraction_log: {e}")

    def get_daily_extraction_tokens(self, role: str, since_ts: float | None = None) -> int:
        """Return total extraction tokens (`input_tokens + output_tokens`) spent by `role` since local midnight."""
        if since_ts is None:
            now = datetime.datetime.now()
            since_ts = datetime.datetime(now.year, now.month, now.day).timestamp()
        with self.connection_provider.connection() as conn:
            try:
                row = conn.execute(
                    """
                    SELECT COALESCE(SUM(COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)), 0)
                    FROM memory_extraction_log
                    WHERE role = ? AND started_at >= ?
                    """,
                    (role, float(since_ts)),
                ).fetchone()
                return int(row[0] or 0) if row else 0
            except sqlite3.OperationalError:
                return 0

    def get_recent_extraction_logs(
        self,
        role: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return most recent `memory_extraction_log` rows."""
        with self.connection_provider.connection() as conn:
            try:
                if role:
                    rows = conn.execute(
                        "SELECT * FROM memory_extraction_log WHERE role = ? ORDER BY id DESC LIMIT ?",
                        (role, int(limit)),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM memory_extraction_log ORDER BY id DESC LIMIT ?",
                        (int(limit),),
                    ).fetchall()
                return [dict(r) for r in rows]
            except sqlite3.OperationalError:
                return []

    def get_fts_backfill_state(self, name: str = "messages_fts") -> dict[str, Any] | None:
        """Retrieve `fts_backfill_state` checkpoint row."""
        with self.connection_provider.connection() as conn:
            try:
                row = conn.execute("SELECT * FROM fts_backfill_state WHERE name = ?", (name,)).fetchone()
                return dict(row) if row is not None else None
            except sqlite3.OperationalError:
                return None

    def upsert_fts_backfill_state(
        self,
        name: str,
        last_ts: float,
        last_id: str,
        done: bool,
    ) -> None:
        """Persist `fts_backfill_state` checkpoint."""
        with self.connection_provider.connection() as conn:
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO fts_backfill_state (name, last_ts, last_id, done)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(name) DO UPDATE SET
                            last_ts = excluded.last_ts,
                            last_id = excluded.last_id,
                            done = excluded.done
                        """,
                        (name, float(last_ts), str(last_id), 1 if done else 0),
                    )
            except sqlite3.OperationalError as e:
                logger.warning(f"Failed updating fts_backfill_state: {e}")

    def _row_to_summary_node(self, row: sqlite3.Row) -> SummaryNode:
        """Convert a sqlite Row to a SummaryNode Pydantic model."""
        return SummaryNode(
            id=row["id"],
            session_id=row["session_id"],
            level=row["level"],
            summary=row["summary"],
            start_timestamp=row["start_timestamp"],
            end_timestamp=row["end_timestamp"],
            token_count=row["token_count"],
            source_token_count=row["source_token_count"],
            parent_id=row["parent_id"],
            created_at=row["created_at"],
        )

    def insert_summary_node(self, node: SummaryNode) -> None:
        """Insert a new summary node into the database."""
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO summary_nodes (
                        id, session_id, level, summary, start_timestamp, end_timestamp,
                        token_count, source_token_count, parent_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node.id,
                        node.session_id,
                        node.level,
                        node.summary,
                        node.start_timestamp,
                        node.end_timestamp,
                        node.token_count,
                        node.source_token_count,
                        node.parent_id,
                        node.created_at,
                    ),
                )

    def update_summary_node_parent(self, node_id: str, parent_id: str) -> None:
        """Set the parent ID of an existing summary node."""
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute(
                    "UPDATE summary_nodes SET parent_id = ? WHERE id = ?",
                    (parent_id, node_id),
                )

    def get_root_summary_nodes(self, session_id: str, level: int | None = None) -> list[SummaryNode]:
        """Retrieve all root summary nodes (parent_id IS NULL) for a session, sorted by start_timestamp ASC."""
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            if level is not None:
                cursor.execute(
                    """
                    SELECT * FROM summary_nodes
                    WHERE session_id = ? AND level = ? AND parent_id IS NULL
                    ORDER BY start_timestamp ASC
                    """,
                    (session_id, level),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM summary_nodes
                    WHERE session_id = ? AND parent_id IS NULL
                    ORDER BY start_timestamp ASC
                    """,
                    (session_id,),
                )
            rows = cursor.fetchall()
            return [self._row_to_summary_node(r) for r in rows]

    def get_all_summary_nodes(self, session_id: str) -> list[SummaryNode]:
        """Retrieve all summary nodes for a session, sorted by creation time."""
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM summary_nodes WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            )
            rows = cursor.fetchall()
            return [self._row_to_summary_node(r) for r in rows]

    def delete_session_summary_nodes(self, session_id: str) -> None:
        """Delete all summary nodes for a session."""
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute("DELETE FROM summary_nodes WHERE session_id = ?", (session_id,))

    def update_messages_summary_node(self, message_ids: list[str], summary_node_id: str | None) -> None:
        """Associate multiple messages with a specific summary node ID."""
        if not message_ids:
            return
        with self.connection_provider.connection() as conn:
            with conn:
                placeholders = ",".join("?" for _ in message_ids)
                conn.execute(
                    f"UPDATE messages SET summary_node_id = ? WHERE id IN ({placeholders})",
                    (summary_node_id, *message_ids),
                )


class AsyncDatabaseManager:
    """Asynchronous wrapper around DatabaseManager running blocking calls in executor threads."""

    def __init__(self, sync_db: DatabaseManager) -> None:
        """Initialize AsyncDatabaseManager wrapping a synchronous DatabaseManager.

        Args:
            sync_db: The underlying synchronous DatabaseManager instance.
        """
        self.sync_db = sync_db
        self._on_message_saved_listeners: list[Callable[[Message], Awaitable[None]]] = []

    def register_on_message_saved(self, callback: Callable[[Message], Awaitable[None]]) -> None:
        """Register a callback to be triggered when a message is successfully saved."""
        self._on_message_saved_listeners.append(callback)

    async def verify_db(self) -> None:
        """Verify the database connection and schema."""
        await asyncio.to_thread(self.sync_db.verify_db)

    async def init_tables(self, overwrite: bool = False) -> None:
        """Initialize database tables.

        Args:
            overwrite: Whether to overwrite existing tables.
        """
        await asyncio.to_thread(self.sync_db.init_tables, overwrite)

    async def create_session(self, session: Session) -> None:
        """Create a new session in the database.

        Args:
            session: The session to create.
        """
        await asyncio.to_thread(self.sync_db.create_session, session)

    async def get_session(self, session_id: str) -> Session | None:
        """Retrieve a session by its ID.

        Args:
            session_id: The ID of the session to retrieve.

        Returns:
            The session if found, else None.
        """
        return await asyncio.to_thread(self.sync_db.get_session, session_id)

    async def update_session_updated_at(self, session_id: str, updated_at: float) -> None:
        """Update the updated_at timestamp of a session.

        Args:
            session_id: The ID of the session.
            updated_at: The new timestamp.
        """
        await asyncio.to_thread(self.sync_db.update_session_updated_at, session_id, updated_at)

    async def update_session_title(self, session_id: str, title: str) -> None:
        """Update the title of a session.

        Args:
            session_id: The ID of the session.
            title: The new title.
        """
        await asyncio.to_thread(self.sync_db.update_session_title, session_id, title)


    async def list_sessions(self) -> list[Session]:
        """List all sessions.

        Returns:
            A list of all sessions.
        """
        return await asyncio.to_thread(self.sync_db.list_sessions)

    async def get_latest_session(self) -> Session | None:
        """Retrieve the most recently updated session.

        Returns:
            The latest session if any, else None.
        """
        return await asyncio.to_thread(self.sync_db.get_latest_session)

    async def get_session_by_channel(self, chatbot_id: str, channel_id: str) -> Session | None:
        """Retrieve the active session for a given chatbot and channel.

        Args:
            chatbot_id: The ID of the chatbot.
            channel_id: The ID of the channel.

        Returns:
            The active session if found, else None.
        """
        return await asyncio.to_thread(self.sync_db.get_session_by_channel, chatbot_id, channel_id)

    async def get_last_session_by_channel_and_role(self, chatbot_id: str, channel_id: str, role: str) -> Session | None:
        """Retrieve the most recent session for a chatbot channel under a specific role.

        Args:
            chatbot_id: The ID of the chatbot.
            channel_id: The ID of the channel.
            role: The role name.

        Returns:
            The Session object if found, None otherwise.
        """
        return await asyncio.to_thread(self.sync_db.get_last_session_by_channel_and_role, chatbot_id, channel_id, role)

    async def set_active_session_for_channel(self, chatbot_id: str, channel_id: str, session_id: str) -> None:
        """Set the active session for a given chatbot and channel.

        Args:
            chatbot_id: The ID of the chatbot.
            channel_id: The ID of the channel.
            session_id: The ID of the session to make active.
        """
        await asyncio.to_thread(self.sync_db.set_active_session_for_channel, chatbot_id, channel_id, session_id)

    async def delete_session(self, session_id: str) -> None:
        """Delete a session and its associated messages.

        Args:
            session_id: The ID of the session to delete.
        """
        await asyncio.to_thread(self.sync_db.delete_session, session_id)

    async def delete_messages_by_ids(self, message_ids: list[str]) -> None:
        """Delete specific messages from the database by their IDs.

        Args:
            message_ids: List of message ID strings to delete.
        """
        await asyncio.to_thread(self.sync_db.delete_messages_by_ids, message_ids)

    async def delete_messages_before_timestamp(
        self, session_id: str, timestamp: float, exclude_ids: list[str] | None = None
    ) -> int:
        """Delete all messages in a session that are older than a specific timestamp.

        Args:
            session_id: The session identifier.
            timestamp: Cutoff unix timestamp. Messages older than this are deleted.
            exclude_ids: Optional list of message IDs to protect from deletion.

        Returns:
            The number of messages deleted.
        """
        return await asyncio.to_thread(
            self.sync_db.delete_messages_before_timestamp, session_id, timestamp, exclude_ids
        )

    async def save_message(self, msg: Message) -> None:
        """Save a message to the database.

        Args:
            msg: The message to save.
        """
        await asyncio.to_thread(self.sync_db.save_message, msg)
        for listener in self._on_message_saved_listeners:
            try:
                await listener(msg)
            except Exception as e:
                logger.error(f"Error in on_message_saved listener: {e}", exc_info=True)

    async def update_message_status(self, message_id: str, status: str) -> None:
        """Update the status of a message.

        Args:
            message_id: The ID of the message.
            status: The new status.
        """
        await asyncio.to_thread(self.sync_db.update_message_status, message_id, status)

    async def claim_message(self, message_id: str, new_status: str, expected_statuses: list[str]) -> bool:
        """Atomically claim a message by updating its status if it is in one of the expected statuses.

        Args:
            message_id: The ID of the message.
            new_status: The new status to set.
            expected_statuses: The statuses the message is expected to be in.

        Returns:
            True if the message was successfully claimed, False otherwise.
        """
        return await asyncio.to_thread(self.sync_db.claim_message, message_id, new_status, expected_statuses)

    async def update_message_metadata(self, message_id: str, metadata: dict[str, Any]) -> None:
        """Update the metadata of a message.

        Args:
            message_id: The ID of the message.
            metadata: The new metadata dict to merge/set.
        """
        await asyncio.to_thread(self.sync_db.update_message_metadata, message_id, metadata)

    async def update_session_system_prompt(self, session_id: str, content: str) -> None:
        """Update the system prompt of a session.

        Args:
            session_id: The ID of the session.
            content: The new system prompt content.
        """
        await asyncio.to_thread(self.sync_db.update_session_system_prompt, session_id, content)

    async def get_session_messages(self, session_id: str) -> list[Message]:
        """Retrieve all raw persisted messages for a session."""
        return await asyncio.to_thread(self.sync_db.get_session_messages, session_id)

    async def get_session_history(
        self,
        session_id: str,
        limit: int = 20,
        order: Literal["phased", "grouped"] = "phased",
        active_only: bool = False,
    ) -> list[Message]:
        """Retrieve message history for a session.

        Args:
            session_id: The ID of the session.
            limit: Maximum number of messages to return.
            order: The ordering strategy ('phased' or 'grouped').
            active_only: Whether to omit compressed ASSISTANT/TOOL rows while keeping USER/SYSTEM anchors.

        Returns:
            A list of messages.
        """
        return await asyncio.to_thread(self.sync_db.get_session_history, session_id, limit, order, active_only)

    async def get_message(self, message_id: str) -> Message | None:
        """Retrieve a specific message by its unique ID.

        Args:
            message_id: The ID of the message.

        Returns:
            The message if found, else None.
        """
        return await asyncio.to_thread(self.sync_db.get_message, message_id)

    async def get_messages_by_filters(
        self,
        filters: dict[str, Any],
        limit: int | None = None,
        exclude_roles: list[str] | None = None,
    ) -> list[Message]:
        """Retrieve messages matching specific filters.

        Args:
            filters: A dictionary of field-value filters.
            limit: Optional maximum number of messages to return.
            exclude_roles: Optional list of roles to exclude.

        Returns:
            A list of matching messages.
        """
        return await asyncio.to_thread(self.sync_db.get_messages_by_filters, filters, limit, exclude_roles)

    async def get_pending_deferred_media_messages(self, chatbot_id: str, since_timestamp: float) -> list[Message]:
        """Retrieve recent assistant messages for ``chatbot_id`` containing ``deferred_media`` metadata.

        Args:
            chatbot_id: Target chatbot identifier.
            since_timestamp: Minimum Unix timestamp to scan (e.g. ``now - 86400``).

        Returns:
            A list of matching messages ordered chronologically.
        """
        return await asyncio.to_thread(
            self.sync_db.get_pending_deferred_media_messages,
            chatbot_id,
            since_timestamp,
        )

    async def recover_orphaned_processing_messages(self, threshold_seconds: float = 300.0) -> int:
        """Recover messages stuck in 'processing' status for too long.

        Args:
            threshold_seconds: The age threshold in seconds to consider a message orphaned.

        Returns:
            The number of recovered messages.
        """
        return await asyncio.to_thread(self.sync_db.recover_orphaned_processing_messages, threshold_seconds)

    async def get_session_turns_count(self, session_id: str) -> int:
        """Get the number of turns in a session.

        Args:
            session_id: The ID of the session.

        Returns:
            The number of turns.
        """
        return await asyncio.to_thread(self.sync_db.get_session_turns_count, session_id)

    async def get_session_turn_anchors(self, session_id: str) -> list[dict[str, Any]]:
        """Get turn anchors for a session.

        Args:
            session_id: The ID of the session.

        Returns:
            A list of turn anchor dictionaries.
        """
        return await asyncio.to_thread(self.sync_db.get_session_turn_anchors, session_id)

    async def get_session_skill_anchor_ids(self, session_id: str) -> list[str]:
        """Get skill anchor IDs for a session.

        Args:
            session_id: The ID of the session.

        Returns:
            A list of skill anchor IDs.
        """
        return await asyncio.to_thread(self.sync_db.get_session_skill_anchor_ids, session_id)

    async def get_orphaned_tool_calls(self, session_id: str) -> list[Message]:
        """Get orphaned tool calls in a session.

        Args:
            session_id: The ID of the session.

        Returns:
            A list of orphaned tool call messages.
        """
        return await asyncio.to_thread(self.sync_db.get_orphaned_tool_calls, session_id)

    async def get_session_history_by_ranges(
        self,
        session_id: str,
        ranges: list[tuple[float, float | None]],
        order: Literal["phased", "grouped"] = "phased",
    ) -> list[Message]:
        """Retrieve session history within specific timestamp ranges.

        Args:
            session_id: The ID of the session.
            ranges: A list of (start, end) timestamp tuples.
            order: The ordering strategy ('phased' or 'grouped').

        Returns:
            A list of messages within the ranges.
        """
        return await asyncio.to_thread(self.sync_db.get_session_history_by_ranges, session_id, ranges, order)

    async def get_memory_files_by_role(
        self,
        role: str,
        *,
        visibility: str | None = None,
        exclude_pinned: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve indexed memory file metadata rows for a role."""
        return await asyncio.to_thread(
            lambda: self.sync_db.get_memory_files_by_role(role, visibility=visibility, exclude_pinned=exclude_pinned)
        )

    async def upsert_memory_file(
        self,
        role: str,
        rel_path: str,
        description: str,
        content_hash: str,
        mtime: float,
        updated_at: float,
        chunks: list[tuple[int, str, bytes]],
        *,
        visibility: str = "DEFAULT",
        size_bytes: int = 0,
    ) -> int:
        """Insert or update a memory file entry and replace its chunk embeddings atomically."""
        return await asyncio.to_thread(
            lambda: self.sync_db.upsert_memory_file(
                role,
                rel_path,
                description,
                content_hash,
                mtime,
                updated_at,
                chunks,
                visibility=visibility,
                size_bytes=size_bytes,
            )
        )

    async def delete_memory_files_by_ids(self, file_ids: list[int]) -> None:
        """Delete memory file entries (and their cascaded chunks) by ID."""
        await asyncio.to_thread(self.sync_db.delete_memory_files_by_ids, file_ids)

    async def clear_role_memory_index(self, role: str | None = None) -> None:
        """Delete all indexed memory files and chunks (for a specific role or all roles)."""
        await asyncio.to_thread(self.sync_db.clear_role_memory_index, role)

    async def get_memory_chunks_for_role(self, role: str, *, include_pinned: bool = True) -> list[sqlite3.Row]:
        """Fetch memory chunks joined with their parent file metadata for a role."""
        return await asyncio.to_thread(
            lambda: self.sync_db.get_memory_chunks_for_role(role, include_pinned=include_pinned)
        )

    async def save_ui_prompt(self, session_id: str, message_id: str, spec: dict[str, Any]) -> None:
        """Insert or update the spec of an interactive UI prompt."""
        await asyncio.to_thread(self.sync_db.save_ui_prompt, session_id, message_id, spec)

    async def get_ui_prompt(self, session_id: str, message_id: str) -> dict[str, Any] | None:
        """Load the spec of an interactive UI prompt, or None if unknown."""
        return await asyncio.to_thread(self.sync_db.get_ui_prompt, session_id, message_id)

    async def set_channel_role(self, chatbot_id: str, channel_id: str, role: str) -> None:
        """Set the role for a specific channel.

        Args:
            chatbot_id: The ID of the chatbot.
            channel_id: The ID of the channel.
            role: The role name.
        """
        await asyncio.to_thread(self.sync_db.set_channel_role, chatbot_id, channel_id, role)

    async def get_channel_role(self, chatbot_id: str, channel_id: str) -> str | None:
        """Get the role for a specific channel.

        Args:
            chatbot_id: The ID of the chatbot.
            channel_id: The ID of the channel.

        Returns:
            The role name if set, else None.
        """
        return await asyncio.to_thread(self.sync_db.get_channel_role, chatbot_id, channel_id)

    async def get_channel_role_with_inheritance(
        self,
        chatbot_id: str,
        channel_id: str,
        session_id: str | None = None,
        *,
        default: str | None = None,
    ) -> str:
        """Get the role for a channel, inheriting from parent channels if necessary.

        Args:
            chatbot_id: The ID of the chatbot.
            channel_id: The ID of the channel.
            session_id: Optional session ID to help resolve role.
            default: Optional explicit fallback role override.

        Returns:
            The resolved role name.
        """
        return await asyncio.to_thread(
            lambda: self.sync_db.get_channel_role_with_inheritance(
                chatbot_id, channel_id, session_id, default=default
            )
        )

    async def get_channel_by_session(self, session_id: str) -> tuple[str, str] | None:
        """Get the chatbot ID and channel ID associated with a session.

        Args:
            session_id: The ID of the session.

        Returns:
            A tuple of (chatbot_id, channel_id) if found, else None.
        """
        return await asyncio.to_thread(self.sync_db.get_channel_by_session, session_id)

    async def get_last_message_timestamp(self, chatbot_id: str, channel_id: str) -> float | None:
        """Get the timestamp of the last message in a channel.

        Args:
            chatbot_id: The ID of the chatbot.
            channel_id: The ID of the channel.

        Returns:
            The timestamp if any message exists, else None.
        """
        return await asyncio.to_thread(self.sync_db.get_last_message_timestamp, chatbot_id, channel_id)

    async def get_cronjob_sent_stats_today(self, chatbot_id: str, channel_id: str) -> tuple[int, float | None]:
        """Get cronjob sending statistics for today.

        Args:
            chatbot_id: The ID of the chatbot.
            channel_id: The ID of the channel.

        Returns:
            A tuple of (count_today, last_sent_timestamp).
        """
        return await asyncio.to_thread(self.sync_db.get_cronjob_sent_stats_today, chatbot_id, channel_id)

    async def upsert_session_digest(self, session_id: str, role: str, **kwargs: Any) -> None:
        """Insert or update a session digest asynchronously."""
        await asyncio.to_thread(lambda: self.sync_db.upsert_session_digest(session_id, role, **kwargs))

    async def get_session_digest(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve a session digest asynchronously."""
        return await asyncio.to_thread(self.sync_db.get_session_digest, session_id)

    async def get_recent_session_digests(
        self,
        role: str,
        exclude_session_id: str | None = None,
        since_ts: float = 0.0,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve recent session digests for `role` asynchronously."""
        return await asyncio.to_thread(
            self.sync_db.get_recent_session_digests, role, exclude_session_id, since_ts, limit
        )

    async def fts_insert_messages(self, entries: Sequence[dict[str, Any]]) -> int:
        """Insert message FTS entries asynchronously."""
        return await asyncio.to_thread(self.sync_db.fts_insert_messages, entries)

    async def fts_delete_session(self, session_id: str) -> None:
        """Delete FTS entries for `session_id` asynchronously."""
        await asyncio.to_thread(self.sync_db.fts_delete_session, session_id)

    async def fts_search_messages(self, role: str, query: str, limit: int = 40, **kwargs: Any) -> list[dict[str, Any]]:
        """Search `messages_fts` asynchronously."""
        return await asyncio.to_thread(lambda: self.sync_db.fts_search_messages(role, query, limit, **kwargs))

    async def fts_search_digests(self, role: str, query: str, limit: int = 5, **kwargs: Any) -> list[dict[str, Any]]:
        """Search `session_digests_fts` asynchronously."""
        return await asyncio.to_thread(lambda: self.sync_db.fts_search_digests(role, query, limit, **kwargs))

    async def get_messages_text_by_ids(self, message_ids: Sequence[str]) -> dict[str, str]:
        """Hydrate message text for `<= 10` IDs asynchronously."""
        return await asyncio.to_thread(self.sync_db.get_messages_text_by_ids, message_ids)

    async def get_session_view_messages(
        self,
        session_id: str,
        around_message_id: str | None = None,
        window: int = 6,
        max_messages: int = 20,
    ) -> list[Message]:
        """Retrieve bounded messages around `around_message_id` asynchronously."""
        return await asyncio.to_thread(
            self.sync_db.get_session_view_messages, session_id, around_message_id, window, max_messages
        )

    async def get_extraction_cursor(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve extraction cursor asynchronously."""
        return await asyncio.to_thread(self.sync_db.get_extraction_cursor, session_id)

    async def upsert_extraction_cursor(self, session_id: str, role: str, **kwargs: Any) -> dict[str, Any]:
        """Upsert extraction cursor asynchronously."""
        return await asyncio.to_thread(lambda: self.sync_db.upsert_extraction_cursor(session_id, role, **kwargs))

    async def get_unextracted_messages(
        self,
        session_id: str,
        after_ts: float = 0.0,
        after_id: str = "",
        safe_upper: float | tuple[float, str] | None = None,
        limit: int = 200,
    ) -> list[Message]:
        """Retrieve safe unextracted messages asynchronously."""
        return await asyncio.to_thread(
            self.sync_db.get_unextracted_messages, session_id, after_ts, after_id, safe_upper, limit
        )

    async def get_session_tool_calls(self, session_id: str) -> list[Message]:
        """Retrieve all `TOOL_CALL` messages of `session_id` asynchronously."""
        return await asyncio.to_thread(self.sync_db.get_session_tool_calls, session_id)

    async def log_extraction(self, **kwargs: Any) -> None:
        """Append an extraction log entry asynchronously."""
        await asyncio.to_thread(lambda: self.sync_db.log_extraction(**kwargs))

    async def insert_summary_node(self, node: SummaryNode) -> None:
        """Insert a new summary node into the database (async)."""
        await asyncio.to_thread(self.sync_db.insert_summary_node, node)

    async def update_summary_node_parent(self, node_id: str, parent_id: str) -> None:
        """Set the parent ID of an existing summary node (async)."""
        await asyncio.to_thread(self.sync_db.update_summary_node_parent, node_id, parent_id)

    async def get_root_summary_nodes(self, session_id: str, level: int | None = None) -> list[SummaryNode]:
        """Retrieve all root summary nodes (parent_id IS NULL) for a session, sorted by start_timestamp ASC (async)."""
        return await asyncio.to_thread(self.sync_db.get_root_summary_nodes, session_id, level)

    async def get_all_summary_nodes(self, session_id: str) -> list[SummaryNode]:
        """Retrieve all summary nodes for a session, sorted by creation time (async)."""
        return await asyncio.to_thread(self.sync_db.get_all_summary_nodes, session_id)

    async def delete_session_summary_nodes(self, session_id: str) -> None:
        """Delete all summary nodes for a session (async)."""
        await asyncio.to_thread(self.sync_db.delete_session_summary_nodes, session_id)

    async def update_messages_summary_node(self, message_ids: list[str], summary_node_id: str | None) -> None:
        """Associate multiple messages with a specific summary node ID (async)."""
        await asyncio.to_thread(self.sync_db.update_messages_summary_node, message_ids, summary_node_id)
