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

from kesoku.constants import MessageRole, MessageStatus, MessageType

logger = logging.getLogger(__name__)


class Message(BaseModel):
    """Represents a conversational message within the Kesoku framework."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(..., description="Internal unique conversational session identifier")
    chatbot_id: str = Field(..., description="Unique identifier of the chatbot platform/instance (e.g., 'cli')")
    channel_id: str = Field(..., description="External platform-specific channel or room identifier")
    sender: str = Field(..., description="Sender identifier or username")
    role: MessageRole = Field(
        default=MessageRole.USER, description="Role of the message sender"
    )
    type: MessageType = Field(
        default=MessageType.TEXT, description="Type of message content or action"
    )
    content: str = Field(..., description="Text content of the message")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata for platform-specific attributes (e.g., attachments, guild_id)",
    )
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp of when the message was created")
    status: MessageStatus = Field(default=MessageStatus.PENDING, description="Current lifecycle status of the message")
    parent_id: str | None = Field(default=None, description="Links tool results or followups to specific message/call")


class Session(BaseModel):
    """Represents a persistent conversational chat session."""

    id: str = Field(..., description="Unique alphanumeric identifier for the session")
    title: str = Field(..., description="Summary title or first message snippet")
    created_at: float = Field(default_factory=time.time, description="Creation unix timestamp")
    updated_at: float = Field(default_factory=time.time, description="Last updated unix timestamp")
    system_prompt: str = Field(default="", description="The main system prompt instructions for the session")

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


class AgentMemory(BaseModel):
    """Represents a structured agent memory record in the SQLite database."""

    id: int | None = Field(default=None, description="Database autoincrement primary key")
    category: str = Field(..., description="Category: 'progress', 'learnings', 'notes' etc.")
    key: str = Field(..., description="snake_case unique identifier")
    title: str = Field(..., description="Human-readable label or title for the entry")
    content: str = Field(..., description="Markdown text or structured content payload")
    updated_at: float = Field(default_factory=time.time, description="Unix timestamp of last update")
    role: str = Field(default="default", description="Optional roleplay-specific character persona binding")


def _sort_session_messages(all_msgs: list[Message], order: Literal["phased", "grouped"]) -> list[Message]:
    """Sort historical messages for a specific session ordered logically.

    Args:
        all_msgs: A list of Message objects to sort.
        order: Sorting mechanism ("phased" or "grouped").

    Returns:
        A list of logically sorted Message objects.
    """
    if not all_msgs:
        return []

    msg_map = {m.id: m for m in all_msgs}

    def get_root_timestamp(m: Message) -> float:
        curr = m
        while curr.parent_id and curr.parent_id in msg_map:
            curr = msg_map[curr.parent_id]
        return curr.timestamp

    def get_tool_group_timestamp(m: Message) -> float:
        if m.parent_id and m.parent_id in msg_map:
            parent_msg = msg_map[m.parent_id]
            if parent_msg.role == MessageRole.TOOL and parent_msg.type == MessageType.TOOL_CALL:
                return parent_msg.timestamp
        return m.timestamp

    if order == "grouped":
        # Grouped sorting simply sorts by root turn timestamp, then tool group timestamp, then actual timestamp
        return sorted(all_msgs, key=lambda m: (get_root_timestamp(m), get_tool_group_timestamp(m), m.timestamp))

    # Phased sorting logic (default for LLM inputs):
    # 1. Group messages by turn root timestamp
    turns: dict[float, list[Message]] = {}
    for msg in all_msgs:
        root_ts = get_root_timestamp(msg)
        turns.setdefault(root_ts, []).append(msg)

    # 2. Sort each turn individually
    for root_ts, turn_msgs in turns.items():
        # Identify all tool calls in the current turn
        tc_map = {
            m.id: m for m in turn_msgs
            if m.role == MessageRole.TOOL and m.type == MessageType.TOOL_CALL
        }

        # Collect and sort tool results by parent tool call timestamp
        tr_msgs = [
            m for m in turn_msgs
            if m.role == MessageRole.TOOL and m.type == MessageType.TOOL_RESULT
        ]
        tr_msgs.sort(
            key=lambda m: tc_map[m.parent_id].timestamp if m.parent_id in tc_map else m.timestamp
        )

        # Group tool results into logical execution batches
        batches: list[list[Message]] = []
        current_batch: list[Message] = []
        last_ts = None

        for tr in tr_msgs:
            parent_tc = tc_map.get(tr.parent_id)
            ts = parent_tc.timestamp if parent_tc else tr.timestamp

            if last_ts is None:
                current_batch.append(tr)
            elif ts - last_ts < 0.5:
                current_batch.append(tr)
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [tr]
            last_ts = ts

        if current_batch:
            batches.append(current_batch)

        # Determine maximum timestamp boundary (cutoff) for each batch
        batch_cutoffs = [max(tr.timestamp for tr in batch) for batch in batches] if batches else []

        # Determine which iteration round a message belongs to based on these boundaries
        def get_iteration_index(m: Message) -> int:
            idx = 0
            for cutoff in batch_cutoffs:
                if m.timestamp > cutoff:
                    idx += 1
            return idx

        # Define the phase sorting inside a single iteration round
        def get_sorting_phase(m: Message) -> float:
            if m.role == MessageRole.TOOL and m.type == MessageType.TOOL_CALL:
                return 1.0
            elif m.role == MessageRole.TOOL and m.type == MessageType.TOOL_RESULT:
                return 2.0
            elif m.role == MessageRole.ASSISTANT and m.type == MessageType.THOUGHT:
                return 0.0
            elif m.role == MessageRole.ASSISTANT and m.type != MessageType.THOUGHT:
                return 3.0
            return 0.0

        # Sort turn messages in place
        turn_msgs.sort(
            key=lambda m: (
                get_iteration_index(m),
                get_sorting_phase(m),
                get_tool_group_timestamp(m),
                m.timestamp,
            )
        )

    # 3. Flatten all sorted turns chronologically
    sorted_msgs = []
    for r_ts in sorted(turns.keys()):
        sorted_msgs.extend(turns[r_ts])

    return sorted_msgs



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

        # Enable WAL (Write-Ahead Logging) mode for high concurrent read/write
        conn.execute("PRAGMA journal_mode=WAL;")

        # Use NORMAL synchronous mode for faster WAL writes while maintaining app-level crash safety
        conn.execute("PRAGMA synchronous=NORMAL;")

        # Enforce foreign key constraints
        conn.execute("PRAGMA foreign_keys=ON;")

        # Set busy timeout to 5000ms to avoid database locked exceptions
        conn.execute("PRAGMA busy_timeout=5000;")

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
                        updated_at REAL NOT NULL,
                        system_prompt TEXT NOT NULL DEFAULT ''
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
                conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_status ON messages(status);")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_session_timestamp "
                    "ON messages(session_id, timestamp);"
                )
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
                    (id, title, created_at, updated_at, system_prompt)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (session.id, session.title, session.created_at, session.updated_at, session.system_prompt),
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
                    system_prompt=row["system_prompt"],
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
                    system_prompt=row["system_prompt"],
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
                    system_prompt=row["system_prompt"],
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
                JOIN channel_sessions cs ON s.id = cs.session_id
                WHERE cs.chatbot_id = ? AND cs.channel_id = ?
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
                    system_prompt=row["system_prompt"],
                )
            return None
        finally:
            conn.close()

    def set_active_session_for_channel(self, chatbot_id: str, channel_id: str, session_id: str) -> None:
        """Bind a session as the active session for a chatbot channel (UPSERT).

        Args:
            chatbot_id: Unique chatbot platform identifier.
            channel_id: External channel or thread identifier.
            session_id: Unique session identifier to bind.
        """
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO channel_sessions (chatbot_id, channel_id, session_id)
                    VALUES (?, ?, ?)
                    ON CONFLICT(chatbot_id, channel_id) DO UPDATE SET session_id=excluded.session_id
                    """,
                    (chatbot_id, channel_id, session_id),
                )
                logger.info(f"Explicitly bound channel '{chatbot_id}:{channel_id}' to active session '{session_id}'")
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

    def claim_message(self, message_id: str, new_status: str, expected_statuses: list[str]) -> bool:
        """Atomically update message status only if it is currently in one of expected_statuses.

        Args:
            message_id: Target message ID.
            new_status: The new status to set.
            expected_statuses: The list of valid current statuses.

        Returns:
            True if exactly one message was updated, False otherwise.
        """
        conn = self._get_connection()
        try:
            with conn:
                placeholders = ", ".join("?" for _ in expected_statuses)
                cursor = conn.execute(
                    f"UPDATE messages SET status = ? WHERE id = ? AND status IN ({placeholders})",
                    [new_status, message_id, *expected_statuses],
                )
                return cursor.rowcount == 1
        finally:
            conn.close()


    def update_message_metadata(self, message_id: str, metadata: dict[str, Any]) -> None:
        """Update the metadata dictionary of a message.

        Args:
            message_id: Target message ID.
            metadata: Dictionary representing the new metadata.
        """
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("UPDATE messages SET metadata = ? WHERE id = ?", (json.dumps(metadata), message_id))
        finally:
            conn.close()

    def update_session_system_prompt(self, session_id: str, content: str) -> None:
        """Update the main system prompt content for an existing session.

        Args:
            session_id: Target session ID.
            content: New system prompt content string.
        """
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    "UPDATE sessions SET system_prompt = ? WHERE id = ?",
                    (content, session_id),
                )
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

            all_msgs = _sort_session_messages(all_msgs, order)

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

    def recover_orphaned_processing_messages(self, threshold_seconds: float = 300.0) -> int:
        """Recover messages that got stuck in 'processing' status by reverting them to 'pending_agent'.

        Args:
            threshold_seconds: Messages with 'processing' status older than this threshold in seconds will be reverted.

        Returns:
            The number of messages recovered.
        """
        conn = self._get_connection()
        try:
            with conn:
                cutoff = time.time() - threshold_seconds
                cursor = conn.execute(
                    f"UPDATE messages SET status = '{MessageStatus.PENDING_AGENT}' "
                    f"WHERE status = '{MessageStatus.PROCESSING}' AND timestamp < ?",
                    (cutoff,),
                )
                return cursor.rowcount
        finally:
            conn.close()

    def get_session_turns_count(self, session_id: str) -> int:
        """Count the number of user turns in a session.

        Args:
            session_id: The target session identifier.

        Returns:
            The count of user messages.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = '{MessageRole.USER}'",
                (session_id,),
            )
            return cursor.fetchone()[0]
        finally:
            conn.close()

    def get_session_turn_anchors(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve list of user and system messages with their ID, role and timestamp.

        Args:
            session_id: Target session ID.

        Returns:
            List of dicts containing message id, role and timestamp.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT id, role, timestamp FROM messages
                WHERE session_id = ? AND role IN ('{MessageRole.USER}', '{MessageRole.SYSTEM}')
                ORDER BY timestamp ASC
                """,
                (session_id,),
            )
            rows = cursor.fetchall()
            return [{"id": r["id"], "role": r["role"], "timestamp": r["timestamp"]} for r in rows]
        finally:
            conn.close()

    def get_session_skill_anchor_ids(self, session_id: str) -> list[str]:
        """Retrieve the user message IDs of turns that loaded a skill successfully.

        Args:
            session_id: Target session ID.

        Returns:
            List of turn anchor message IDs.
        """
        conn = self._get_connection()
        try:
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
        finally:
            conn.close()

    def get_orphaned_tool_calls(self, session_id: str) -> list[Message]:
        """Retrieve all orphaned tool call messages in a session.

        Args:
            session_id: Target session ID.

        Returns:
            List of orphaned tool call Message objects.
        """
        conn = self._get_connection()
        try:
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

        conn = self._get_connection()
        try:
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
                  AND ({' OR '.join(clauses)})
            """
            cursor.execute(query, tuple(params))
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

            all_msgs = _sort_session_messages(all_msgs, order)

            return all_msgs
        finally:
            conn.close()

    # Agent Memory CRUD
    def upsert_agent_memory(self, category: str, key: str, title: str, content: str, role: str = "default") -> None:
        """Atomically insert or replace an agent memory record."""
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO agent_memories
                    (category, key, title, content, updated_at, role)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (category, key, title, content, time.time(), role),
                )
        finally:
            conn.close()

    def get_agent_memory(self, category: str, key: str, role: str = "default") -> dict[str, Any] | None:
        """Retrieve a specific agent memory record."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM agent_memories WHERE category = ? AND key = ? AND role = ?",
                (category, key, role),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()

    def get_agent_memories(self, category: str | None = None, role: str | None = None) -> list[dict[str, Any]]:
        """Retrieve agent memories, optionally filtered by category and/or role.

        If role is provided, retrieves both default memories and role-specific memories.
        """
        conn = self._get_connection()
        try:
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
            query += " ORDER BY key ASC"

            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def delete_agent_memory(self, category: str, key: str, role: str = "default") -> None:
        """Delete a specific agent memory record."""
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    "DELETE FROM agent_memories WHERE category = ? AND key = ? AND role = ?",
                    (category, key, role),
                )
        finally:
            conn.close()

    def set_channel_role(self, chatbot_id: str, channel_id: str, role: str) -> None:
        """Bind a roleplay persona to a chatbot channel/thread."""
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO channel_roles (chatbot_id, channel_id, role)
                    VALUES (?, ?, ?)
                    """,
                    (chatbot_id, channel_id, role),
                )
        finally:
            conn.close()

    def get_channel_role(self, chatbot_id: str, channel_id: str) -> str | None:
        """Retrieve the roleplay persona bound directly to a chatbot channel/thread."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role FROM channel_roles WHERE chatbot_id = ? AND channel_id = ?",
                (chatbot_id, channel_id),
            )
            row = cursor.fetchone()
            return row["role"] if row else None
        finally:
            conn.close()

    def get_channel_role_with_inheritance(self, chatbot_id: str, channel_id: str, session_id: str | None = None) -> str:
        """Retrieve the active role bound to a channel/thread with parent inheritance support."""
        # 1. Direct lookup
        role = self.get_channel_role(chatbot_id, channel_id)
        if role:
            return role

        # 2. Discord parent channel inheritance
        if chatbot_id == "discord" and session_id:
            conn = self._get_connection()
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
            finally:
                conn.close()

        return "default"

    def get_channel_by_session(self, session_id: str) -> tuple[str, str] | None:
        """Retrieve the chatbot_id and channel_id mapping for a given session ID."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT chatbot_id, channel_id FROM channel_sessions WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            return (row["chatbot_id"], row["channel_id"]) if row else None
        finally:
            conn.close()


