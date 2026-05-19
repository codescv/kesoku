"""Database models and SQLite persistence manager for Kesoku AI Agent."""

import json
import logging
import os
import re
import shutil
import sqlite3
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from kesoku.constants import (
    ROLE_ASSISTANT,
    ROLE_TOOL,
    ROLE_USER,
    STATUS_PENDING,
    TYPE_TEXT,
    TYPE_THOUGHT,
    TYPE_TOOL_CALL,
    TYPE_TOOL_RESULT,
)

logger = logging.getLogger(__name__)


class Message(BaseModel):
    """Represents a conversational message within the Kesoku framework."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(..., description="Internal unique conversational session identifier")
    chatbot_id: str = Field(..., description="Unique identifier of the chatbot platform/instance (e.g., 'cli')")
    channel_id: str = Field(..., description="External platform-specific channel or room identifier")
    sender: str = Field(..., description="Sender identifier or username")
    role: Literal["user", "assistant", "tool", "system"] = Field(
        default=ROLE_USER, description="Role of the message sender"
    )
    type: Literal["text", "thought", "tool_call", "tool_result"] = Field(
        default=TYPE_TEXT, description="Type of message content or action"
    )
    content: str = Field(..., description="Text content of the message")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata for platform-specific attributes (e.g., attachments, guild_id)",
    )
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp of when the message was created")
    status: Literal[
        "pending", "processing", "processed", "delivered", "interrupted", "pending_agent", "responded", "error"
    ] = Field(default=STATUS_PENDING, description="Current lifecycle status of the message")
    parent_id: str | None = Field(default=None, description="Links tool results or followups to specific message/call")


class Session(BaseModel):
    """Represents a persistent conversational chat session."""

    id: str = Field(..., description="Unique alphanumeric identifier for the session")
    title: str = Field(..., description="Summary title or first message snippet")
    created_at: float = Field(default_factory=time.time, description="Creation unix timestamp")
    updated_at: float = Field(default_factory=time.time, description="Last updated unix timestamp")

    @property
    def workspace_name(self) -> str:
        """Generate a unique, escaped folder name for the session workspace.

        Returns:
            An escaped folder name string.
        """
        escaped = re.sub(r"[^\w\-]", "_", self.title)
        escaped = re.sub(r"_+", "_", escaped)
        escaped = escaped.strip("_")
        if not escaped:
            escaped = "session"
        title_escaped = escaped[:20].strip("_")
        ts_str = time.strftime("%y%m%d-%H-%M", time.localtime(self.created_at))
        return f"{ts_str}_{title_escaped}_{self.id}"


class DatabaseManager:
    """Encapsulates all SQLite database schema and CRUD operations for Kesoku."""

    def __init__(self, db_path: str) -> None:
        """Initialize DatabaseManager with SQLite file path.

        Args:
            db_path: Absolute or relative filesystem path to SQLite database file.
        """
        self.db_path = db_path

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
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
            if not cursor.fetchone():
                raise RuntimeError(
                    f"Database at '{self.db_path}' is missing required tables. Please run 'kesoku init' first."
                )
            self._ensure_migrations(conn)
        except sqlite3.DatabaseError:
            raise RuntimeError(f"Database at '{self.db_path}' is invalid or corrupt. Please run 'kesoku init' first.")
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Obtain a configured SQLite database connection.

        Returns:
            A configured sqlite3.Connection instance.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

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

        conn = self._get_connection()
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
                        updated_at REAL NOT NULL
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_status ON messages(status);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(chatbot_id, channel_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);")
            self._ensure_migrations(conn)
            logger.info(f"Database schema initialized successfully at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise
        finally:
            conn.close()

    # Session CRUD
    def create_session(self, session: Session) -> None:
        """Persist a new chat session record.

        Args:
            session: The Session object to store.
        """
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO sessions
                    (id, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session.id, session.title, session.created_at, session.updated_at),
                )
        finally:
            conn.close()

    def get_session(self, session_id: str) -> Session | None:
        """Retrieve a chat session record by ID.

        Args:
            session_id: Session ID to query.

        Returns:
            The Session object if found, None otherwise.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            if row:
                return Session(
                    id=row["id"],
                    title=row["title"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            return None
        finally:
            conn.close()

    def update_session_updated_at(self, session_id: str, updated_at: float) -> None:
        """Update the updated_at timestamp for a session.

        Args:
            session_id: Target session ID.
            updated_at: New timestamp float.
        """
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (updated_at, session_id))
        finally:
            conn.close()

    def list_sessions(self) -> list[Session]:
        """List all chat sessions ordered by most recently updated.

        Returns:
            List of Session objects.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions ORDER BY updated_at DESC")
            rows = cursor.fetchall()
            return [
                Session(
                    id=row["id"],
                    title=row["title"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]
        finally:
            conn.close()

    def get_latest_session(self) -> Session | None:
        """Retrieve the most recently updated chat session.

        Returns:
            The Session object if available, None otherwise.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions ORDER BY updated_at DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                return Session(
                    id=row["id"],
                    title=row["title"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            return None
        finally:
            conn.close()

    def get_session_by_channel(self, chatbot_id: str, channel_id: str) -> Session | None:
        """Retrieve the chat session associated with a specific chatbot channel.

        Args:
            chatbot_id: Unique identifier of the chatbot.
            channel_id: Channel or room identifier.

        Returns:
            The Session object if found, None otherwise.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT s.* FROM sessions s
                JOIN messages m ON s.id = m.session_id
                WHERE m.chatbot_id = ? AND m.channel_id = ?
                ORDER BY m.timestamp DESC LIMIT 1
                """,
                (chatbot_id, channel_id),
            )
            row = cursor.fetchone()
            if row:
                return Session(
                    id=row["id"],
                    title=row["title"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            return None
        finally:
            conn.close()

    def delete_session(self, session_id: str) -> None:
        """Delete a session and all its associated messages from the database.

        Args:
            session_id: The unique session identifier to delete.
        """
        conn = self._get_connection()
        try:
            with conn:
                # Delete all messages belonging to this session
                conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                # Delete the session itself
                conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        finally:
            conn.close()

    # Message CRUD
    def save_message(self, msg: Message) -> None:
        """Persist a new conversational message record.

        Args:
            msg: The Message object to store.
        """
        conn = self._get_connection()
        try:
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
        finally:
            conn.close()

    def update_message_status(self, message_id: str, status: str) -> None:
        """Update the operational lifecycle status of a message.

        Args:
            message_id: Target message ID.
            status: New status string.
        """
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("UPDATE messages SET status = ? WHERE id = ?", (status, message_id))
        finally:
            conn.close()

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
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ?
                """,
                (session_id,),
            )
            rows = cursor.fetchall()
            all_msgs = [
                Message(
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
                    status=row["status"],  # type: ignore
                    parent_id=row["parent_id"],
                )
                for row in rows
            ]

            # Build map for fast root lookup
            msg_map = {m.id: m for m in all_msgs}

            def get_root_timestamp(m: Message) -> float:
                curr = m
                while curr.parent_id and curr.parent_id in msg_map:
                    curr = msg_map[curr.parent_id]
                return curr.timestamp

            def get_sorting_phase(m: Message) -> int:
                if m.role == ROLE_TOOL and m.type == TYPE_TOOL_CALL:
                    return 1
                elif m.role == ROLE_TOOL and m.type == TYPE_TOOL_RESULT:
                    return 2
                elif m.role == ROLE_ASSISTANT and m.type != TYPE_THOUGHT:
                    return 3
                return 0

            def get_tool_group_timestamp(m: Message) -> float:
                if m.parent_id and m.parent_id in msg_map:
                    parent_msg = msg_map[m.parent_id]
                    if parent_msg.role == ROLE_TOOL and parent_msg.type == TYPE_TOOL_CALL:
                        return parent_msg.timestamp
                return m.timestamp

            # Sort logically by turn root timestamp, then by the requested ordering
            if order == "phased":
                all_msgs.sort(
                    key=lambda m: (
                        get_root_timestamp(m),
                        get_sorting_phase(m),
                        get_tool_group_timestamp(m),
                        m.timestamp,
                    )
                )
            else:
                all_msgs.sort(key=lambda m: (get_root_timestamp(m), get_tool_group_timestamp(m), m.timestamp))

            return all_msgs[-limit:] if limit else all_msgs
        finally:
            conn.close()

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
        conn = self._get_connection()
        try:
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
            return [
                Message(
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
                    status=row["status"],  # type: ignore
                    parent_id=row["parent_id"],
                )
                for row in rows
            ]
        finally:
            conn.close()
