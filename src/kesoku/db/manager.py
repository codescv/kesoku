import asyncio
import datetime
import json
import logging
import os
import shutil
import sqlite3
import time
from typing import Any, Literal

from kesoku.agent.history_sorter import sort_session_messages
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db.connection import ConnectionProvider
from kesoku.db.models import CrossSessionContext, Message, Session

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Encapsulates all SQLite database schema and CRUD operations for Kesoku."""

    def __init__(self, db_path: str) -> None:
        """Initialize DatabaseManager with SQLite file path.

        Args:
            db_path: Absolute or relative filesystem path to SQLite database file.
        """
        self.db_path = db_path
        self.connection_provider = ConnectionProvider(db_path)

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
                cursor.execute("PRAGMA table_info(sessions)")
                sess_columns = [row["name"] for row in cursor.fetchall()]
                if "role_name" not in sess_columns:
                    conn.execute("ALTER TABLE sessions ADD COLUMN role_name TEXT")
                # Clean up redundant single-column session index since idx_messages_session_timestamp supersedes it
                conn.execute("DROP INDEX IF EXISTS idx_messages_session")
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
                # Ensure agent_memories table exists
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        category TEXT NOT NULL,
                        key TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        updated_at REAL NOT NULL,
                        role TEXT NOT NULL DEFAULT 'default',
                        UNIQUE(category, key, role)
                    );
                    """
                )
                # Ensure cross_session_contexts table exists
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cross_session_contexts (
                        role TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        updated_at REAL NOT NULL,
                        status TEXT NOT NULL DEFAULT 'idle'
                    );
                    """
                )
                # Migrate legacy 'global' role to 'default'
                conn.execute("UPDATE OR REPLACE agent_memories SET role = 'default' WHERE role = 'global';")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_memories_category_role ON agent_memories(category, role);"
                )
        except Exception as e:
            logger.error(f"Failed to apply database schema migrations: {e}")
            raise RuntimeError(f"Database schema migration error: {e}") from e

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
                    f"Database at '{self.db_path}' is invalid or corrupt. "
                    "Please run 'kesoku init' first."
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
                            parent_id TEXT
                        );
                        """
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
                        CREATE TABLE IF NOT EXISTS cross_session_contexts (
                            role TEXT PRIMARY KEY,
                            content TEXT NOT NULL,
                            updated_at REAL NOT NULL,
                            status TEXT NOT NULL DEFAULT 'idle'
                        );
                        """
                    )
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_status ON messages(status);")
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_messages_session_timestamp ON messages(session_id, timestamp);"
                    )
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(chatbot_id, channel_id);")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);")
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
                role = self.get_channel_role_with_inheritance(chatbot_id, channel_id, session_id)
                conn.execute(
                    """
                    UPDATE sessions
                    SET role_name = ?
                    WHERE id = ?
                    """,
                    (role, session_id),
                )
                logger.info(
                    f"Explicitly bound channel '{chatbot_id}:{channel_id}' "
                    f"to active session '{session_id}' (role: {role})"
                )

    def delete_session(self, session_id: str) -> None:
        """Delete a session and all its associated messages from the database.

        Args:
            session_id: The unique session identifier to delete.
        """
        with self.connection_provider.connection() as conn:
            with conn:
                # Delete all messages belonging to this session
                conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
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
                     content, metadata, timestamp, status, parent_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def get_session_history(
        self, session_id: str, limit: int = 20, order: Literal["phased", "grouped"] = "phased"
    ) -> list[Message]:
        """Retrieve historical messages for a specific session ordered by logical conversational turn.

        Args:
            session_id: Session ID to query.
            limit: Max messages count.
            order: The sorting order. "phased" (default for Gemini) or "grouped" (for human displays).

        Returns:
            List of Message objects ordered by the requested sorting mechanism.
        """
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ?
                """,
                (session_id,),
            )
            rows = cursor.fetchall()
            all_msgs = [self._row_to_message(row) for row in rows]

            all_msgs = sort_session_messages(all_msgs, order)

            return all_msgs[-limit:] if limit else all_msgs

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
                  AND ({" OR ".join(clauses)})
            """
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            all_msgs = [self._row_to_message(row) for row in rows]

            all_msgs = sort_session_messages(all_msgs, order)

            return all_msgs

    # Agent Memory CRUD
    def upsert_agent_memory(self, category: str, key: str, title: str, content: str, role: str = "default") -> None:
        """Atomically insert or replace an agent memory record."""
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO agent_memories
                    (category, key, title, content, updated_at, role)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (category, key, title, content, time.time(), role),
                )

    def get_agent_memory(self, category: str, key: str, role: str = "default") -> dict[str, Any] | None:
        """Retrieve a specific agent memory record."""
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM agent_memories WHERE category = ? AND key = ? AND role = ?",
                (category, key, role),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_agent_memories(self, category: str | None = None, role: str | None = None) -> list[dict[str, Any]]:
        """Retrieve agent memories, optionally filtered by category and/or role.

        If role is provided, retrieves both default memories and role-specific memories.
        """
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM agent_memories"
            params = []
            clauses = []

            if category:
                clauses.append("category = ?")
                params.append(category)

            if role:
                clauses.append("(role = 'default' OR role = ?)")
                params.append(role)

            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY updated_at DESC"

            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def delete_agent_memory(self, category: str, key: str, role: str = "default") -> None:
        """Delete a specific agent memory record."""
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute(
                    "DELETE FROM agent_memories WHERE category = ? AND key = ? AND role = ?",
                    (category, key, role),
                )

    def search_role_memories(self, role: str, query_text: str) -> list[dict[str, Any]]:
        """Search agent memories for a role matching query_text in content or title.

        Includes default memories.
        """
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            sql = """
                SELECT * FROM agent_memories
                WHERE (role = 'default' OR role = ?)
                  AND (content LIKE ? OR title LIKE ?)
                ORDER BY updated_at DESC
            """
            like_query = f"%{query_text}%"
            cursor.execute(sql, (role, like_query, like_query))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def search_role_messages(self, role: str, query_text: str) -> list[Message]:
        """Search user/assistant text messages for a role matching query_text.

        Excludes thoughts, tool calls, and tool results.
        """
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            sql = """
                SELECT m.* FROM messages m
                JOIN sessions s ON m.session_id = s.id
                WHERE COALESCE(s.role_name, 'default') = ?
                  AND m.role IN ('user', 'assistant')
                  AND m.type = 'text'
                  AND m.content LIKE ?
                ORDER BY m.timestamp DESC
            """
            like_query = f"%{query_text}%"
            cursor.execute(sql, (role, like_query))
            rows = cursor.fetchall()
            return [self._row_to_message(row) for row in rows]

    def get_allowed_memory_categories(self) -> set[str]:
        """Retrieves the set of all currently permitted or existing memory categories."""
        categories = {"progress", "user_preferences", "memo"}
        try:
            memories = self.get_agent_memories()
            for m in memories:
                categories.add(m["category"])
        except Exception as e:
            logger.warning(f"Failed to fetch existing categories from database: {e}")
        return categories

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

    def get_channel_role_with_inheritance(self, chatbot_id: str, channel_id: str, session_id: str | None = None) -> str:
        """Retrieve the active role bound to a channel/thread with parent inheritance support."""
        # 1. Direct lookup
        role = self.get_channel_role(chatbot_id, channel_id)
        if role:
            return role

        # 2. Discord parent channel inheritance
        if chatbot_id.startswith("discord") and session_id:
            with self.connection_provider.connection() as conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT metadata FROM messages WHERE session_id = ? AND role = 'user' "
                        "ORDER BY timestamp DESC LIMIT 1",
                        (session_id,),
                    )
                    row = cursor.fetchone()
                    if row:
                        meta = json.loads(row["metadata"])
                        parent_id = meta.get("parent_channel_id")
                        if parent_id:
                            role = self.get_channel_role(chatbot_id, parent_id)
                            if role:
                                return role
                except Exception as e:
                    logger.warning(f"Failed to retrieve parent channel role: {e}")

        return "default"

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

    def get_role_messages_since(
        self,
        role: str,
        since_timestamp: float,
        exclude_session_id: str | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """Retrieve high-value conversational messages for a role since a timestamp.

        Args:
            role: The persona role to query.
            since_timestamp: Fetch messages created after this unix timestamp.
            exclude_session_id: Optional session ID to exclude from the query.
            limit: Optional maximum number of messages to return.

        Returns:
            List of matching Message objects.
        """
        with self.connection_provider.connection() as conn:
            query = """
                SELECT m.* FROM messages m
                JOIN channel_sessions cs ON m.session_id = cs.session_id
                LEFT JOIN channel_roles cr_direct
                  ON cs.chatbot_id = cr_direct.chatbot_id
                 AND cs.channel_id = cr_direct.channel_id
                LEFT JOIN (
                    SELECT session_id,
                           json_extract(metadata, '$.parent_channel_id') as parent_channel_id
                    FROM messages
                    WHERE role = 'user' AND metadata LIKE '%parent_channel_id%'
                    GROUP BY session_id
                ) parent_info
                  ON m.session_id = parent_info.session_id
                 AND cs.chatbot_id = 'discord'
                LEFT JOIN channel_roles cr_parent
                  ON cs.chatbot_id = cr_parent.chatbot_id
                 AND parent_info.parent_channel_id = cr_parent.channel_id
                WHERE COALESCE(cr_direct.role, cr_parent.role, 'default') = ?
                  AND m.timestamp > ?
                  AND m.role IN ('user', 'assistant')
                  AND m.type = 'text'
            """
            params: list[Any] = [role, since_timestamp]
            if exclude_session_id:
                query += " AND m.session_id != ?"
                params.append(exclude_session_id)

            query += " ORDER BY m.timestamp ASC"
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)

            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            return [self._row_to_message(row) for row in rows]

    def get_role_session_ids(self, role: str) -> list[str]:
        """Retrieve all session IDs associated with a specific persona role.

        Args:
            role: The persona role to query.

        Returns:
            List of session ID strings.
        """
        with self.connection_provider.connection() as conn:
            query = """
                SELECT id FROM sessions WHERE role_name = ?
                UNION
                SELECT DISTINCT m.session_id
                FROM messages m
                LEFT JOIN channel_roles cr ON m.chatbot_id = cr.chatbot_id AND m.channel_id = cr.channel_id
                WHERE COALESCE(cr.role, 'default') = ?
                UNION
                SELECT DISTINCT cs.session_id
                FROM channel_sessions cs
                LEFT JOIN channel_roles cr_direct
                  ON cs.chatbot_id = cr_direct.chatbot_id
                 AND cs.channel_id = cr_direct.channel_id
                LEFT JOIN (
                    SELECT session_id,
                           json_extract(metadata, '$.parent_channel_id') as parent_channel_id
                    FROM messages
                    WHERE role = 'user' AND metadata LIKE '%parent_channel_id%'
                    GROUP BY session_id
                ) parent_info
                  ON cs.session_id = parent_info.session_id
                 AND cs.chatbot_id = 'discord'
                LEFT JOIN channel_roles cr_parent
                  ON cs.chatbot_id = cr_parent.chatbot_id
                 AND parent_info.parent_channel_id = cr_parent.channel_id
                WHERE COALESCE(cr_direct.role, cr_parent.role, 'default') = ?
            """
            cursor = conn.cursor()
            cursor.execute(query, (role, role, role))
            rows = cursor.fetchall()
            return [row[0] for row in rows if row[0]]


    # Cross Session Context CRUD
    def get_cross_session_context(self, role: str) -> CrossSessionContext | None:
        """Retrieve the cross-session context for a specific role.

        Args:
            role: The persona role identifier.

        Returns:
            The CrossSessionContext object if found, None otherwise.
        """
        with self.connection_provider.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM cross_session_contexts WHERE role = ?",
                (role,),
            )
            row = cursor.fetchone()
            if row:
                return CrossSessionContext(
                    role=row["role"],
                    content=row["content"],
                    updated_at=row["updated_at"],
                    status=row["status"],
                )
            return None

    def upsert_cross_session_context(self, role: str, content: str) -> None:
        """Insert or replace a cross-session context record.

        Args:
            role: Persona role identifier.
            content: The text content of the consolidated summary context.
        """
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cross_session_contexts
                    (role, content, updated_at, status)
                    VALUES (?, ?, ?, 'idle')
                    """,
                    (role, content, time.time()),
                )

    def claim_cross_session_context_for_update(self, role: str) -> bool:
        """Atomically claim lock to update cross-session context for a role.

        If a lock is already held ('updating') but older than 5 minutes (300s),
        forcibly resets the lock to 'idle' and re-claims it to prevent deadlocks.

        Args:
            role: The target persona role identifier.

        Returns:
            True if lock was claimed successfully, False otherwise.
        """
        with self.connection_provider.connection() as conn:
            with conn:
                now = time.time()
                # 1. Ensure a row for this role exists in the table (INSERT OR IGNORE)
                conn.execute(
                    """
                    INSERT OR IGNORE INTO cross_session_contexts
                    (role, content, updated_at, status)
                    VALUES (?, '', 0.0, 'idle')
                    """,
                    (role,),
                )
                # 2. Self-heal stale locks older than 300 seconds (5 minutes)
                conn.execute(
                    """
                    UPDATE cross_session_contexts
                    SET status = 'idle'
                    WHERE role = ? AND status = 'updating' AND (? - updated_at) > 300
                    """,
                    (role, now),
                )
                # 3. Try to atomically lock the record (CAS)
                cursor = conn.execute(
                    """
                    UPDATE cross_session_contexts
                    SET status = 'updating', updated_at = ?
                    WHERE role = ? AND status = 'idle'
                    """,
                    (now, role),
                )
                return cursor.rowcount == 1

    def release_cross_session_context_lock(self, role: str, content: str, updated_at: float | None = None) -> None:
        """Release lock on cross-session context, updating the summary content.

        Args:
            role: Persona role identifier.
            content: The newly consolidated summary content.
            updated_at: Optional timestamp to set as the new checkpoint. Defaults to current time.
        """
        with self.connection_provider.connection() as conn:
            with conn:
                ts = updated_at if updated_at is not None else time.time()
                conn.execute(
                    """
                    UPDATE cross_session_contexts
                    SET content = ?, updated_at = ?, status = 'idle'
                    WHERE role = ?
                    """,
                    (content, ts, role),
                )

    def recover_orphaned_context_locks(self) -> int:
        """Reset any lingering updating locks from past server crashes.

        Returns:
            The number of locks recovered.
        """
        with self.connection_provider.connection() as conn:
            with conn:
                cursor = conn.execute("UPDATE cross_session_contexts SET status = 'idle' WHERE status = 'updating'")
                return cursor.rowcount


class AsyncDatabaseManager:
    """Asynchronous wrapper around DatabaseManager running blocking calls in executor threads."""

    def __init__(self, sync_db: DatabaseManager) -> None:
        """Initialize AsyncDatabaseManager wrapping a synchronous DatabaseManager.

        Args:
            sync_db: The underlying synchronous DatabaseManager instance.
        """
        self.sync_db = sync_db

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

    async def get_session_history(
        self,
        session_id: str,
        limit: int = 20,
        order: Literal["phased", "grouped"] = "phased",
    ) -> list[Message]:
        """Retrieve message history for a session.

        Args:
            session_id: The ID of the session.
            limit: Maximum number of messages to return.
            order: The ordering strategy ('phased' or 'grouped').

        Returns:
            A list of messages.
        """
        return await asyncio.to_thread(self.sync_db.get_session_history, session_id, limit, order)

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

    async def upsert_agent_memory(
        self, category: str, key: str, title: str, content: str, role: str = "default"
    ) -> None:
        """Upsert an agent memory entry.

        Args:
            category: The category of the memory.
            key: The key of the memory.
            title: The title of the memory.
            content: The content of the memory.
            role: The role associated with the memory.
        """
        await asyncio.to_thread(self.sync_db.upsert_agent_memory, category, key, title, content, role)

    async def get_agent_memory(self, category: str, key: str, role: str = "default") -> dict[str, Any] | None:
        """Retrieve a specific agent memory entry.

        Args:
            category: The category of the memory.
            key: The key of the memory.
            role: The role associated with the memory.

        Returns:
            The memory entry dict if found, else None.
        """
        return await asyncio.to_thread(self.sync_db.get_agent_memory, category, key, role)

    async def get_agent_memories(self, category: str | None = None, role: str | None = None) -> list[dict[str, Any]]:
        """Retrieve all agent memories, optionally filtered by category or role.

        Args:
            category: Optional category filter.
            role: Optional role filter.

        Returns:
            A list of memory entry dicts.
        """
        return await asyncio.to_thread(self.sync_db.get_agent_memories, category, role)

    async def delete_agent_memory(self, category: str, key: str, role: str = "default") -> None:
        """Delete an agent memory entry.

        Args:
            category: The category of the memory.
            key: The key of the memory.
            role: The role associated with the memory.
        """
        await asyncio.to_thread(self.sync_db.delete_agent_memory, category, key, role)

    async def search_role_memories(self, role: str, query_text: str) -> list[dict[str, Any]]:
        """Search agent memories for a role matching query_text."""
        return await asyncio.to_thread(self.sync_db.search_role_memories, role, query_text)

    async def search_role_messages(self, role: str, query_text: str) -> list[Message]:
        """Search messages for a role matching query_text."""
        return await asyncio.to_thread(self.sync_db.search_role_messages, role, query_text)

    async def get_allowed_memory_categories(self) -> set[str]:
        """Retrieves the set of all currently permitted or existing memory categories.

        Returns:
            A set of allowed categories.
        """
        return await asyncio.to_thread(self.sync_db.get_allowed_memory_categories)

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
        self, chatbot_id: str, channel_id: str, session_id: str | None = None
    ) -> str:
        """Get the role for a channel, inheriting from parent channels if necessary.

        Args:
            chatbot_id: The ID of the chatbot.
            channel_id: The ID of the channel.
            session_id: Optional session ID to help resolve role.

        Returns:
            The resolved role name.
        """
        return await asyncio.to_thread(
            self.sync_db.get_channel_role_with_inheritance, chatbot_id, channel_id, session_id
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

    async def get_role_messages_since(
        self,
        role: str,
        since_timestamp: float,
        exclude_session_id: str | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """Retrieve messages for a role since a specific timestamp.

        Args:
            role: The role name.
            since_timestamp: The start timestamp.
            exclude_session_id: Optional session ID to exclude.
            limit: Optional maximum number of messages to return.

        Returns:
            A list of messages.
        """
        return await asyncio.to_thread(
            self.sync_db.get_role_messages_since, role, since_timestamp, exclude_session_id, limit
        )

    async def get_cross_session_context(self, role: str) -> CrossSessionContext | None:
        """Retrieve cross-session context for a role.

        Args:
            role: The role name.

        Returns:
            The cross-session context if found, else None.
        """
        return await asyncio.to_thread(self.sync_db.get_cross_session_context, role)

    async def upsert_cross_session_context(self, role: str, content: str) -> None:
        """Upsert cross-session context for a role.

        Args:
            role: The role name.
            content: The context content.
        """
        await asyncio.to_thread(self.sync_db.upsert_cross_session_context, role, content)

    async def claim_cross_session_context_for_update(self, role: str) -> bool:
        """Claim cross-session context for update by acquiring a lock.

        Args:
            role: The role name.

        Returns:
            True if the lock was acquired, False otherwise.
        """
        return await asyncio.to_thread(self.sync_db.claim_cross_session_context_for_update, role)

    async def release_cross_session_context_lock(
        self, role: str, content: str, updated_at: float | None = None
    ) -> None:
        """Release the cross-session context lock and update the content.

        Args:
            role: The role name.
            content: The new content.
            updated_at: Optional timestamp to set for the update.
        """
        await asyncio.to_thread(self.sync_db.release_cross_session_context_lock, role, content, updated_at)

    async def recover_orphaned_context_locks(self) -> int:
        """Recover orphaned cross-session context locks.

        Returns:
            The number of recovered locks.
        """
        return await asyncio.to_thread(self.sync_db.recover_orphaned_context_locks)

    async def get_cross_session_memory_updates(
        self,
        role: str,
        exclude_session_id: str,
    ) -> tuple[CrossSessionContext | None, list[Message]]:
        """Get cross-session memory updates for a role, excluding a specific session.

        Args:
            role: The role name.
            exclude_session_id: The session ID to exclude from updates.

        Returns:
            A tuple of (stored_context, new_messages).
        """
        stored_ctx = await self.get_cross_session_context(role)
        last_updated = stored_ctx.updated_at if stored_ctx else 0.0

        new_messages = await self.get_role_messages_since(
            role=role,
            since_timestamp=last_updated,
            exclude_session_id=exclude_session_id,
        )
        return stored_ctx, new_messages

    async def get_role_session_ids(self, role: str) -> list[str]:
        """Retrieve all session IDs associated with a specific persona role (async).

        Args:
            role: The persona role to query.

        Returns:
            List of session ID strings.
        """
        return await asyncio.to_thread(self.sync_db.get_role_session_ids, role)


