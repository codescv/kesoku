"""Gateway module for Kesoku AI Agent.

Orchestrates incoming/outgoing messages between chatbots and SQLite persistence
via DatabaseManager using a Pure Broker Pattern.
"""

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from typing import Any

from kesoku.config import WorkspaceConfig, get_config
from kesoku.constants import (
    DEFAULT_SYSTEM_PROMPT,
    ROLE_SYSTEM,
    STATUS_PROCESSED,
    STATUS_RESPONDED,
    TYPE_TEXT,
)
from kesoku.db import DatabaseManager, Message, Session
from kesoku.logger import setup_logger

logger = setup_logger(__name__)


class Listener:
    """Internal container for an active subscriber to the Gateway broker."""

    def __init__(self, filter_func: Callable[[Message], bool]) -> None:
        """Initialize a listener with an async queue and filter function."""
        self.queue: asyncio.Queue[Message] = asyncio.Queue()
        self.filter_func = filter_func


class Gateway:
    """Manages message ingestion, routing, and persistence across chatbots using a stateless broker pattern."""

    def __init__(self, workspace_config: WorkspaceConfig | None = None) -> None:
        """Initialize the Gateway.

        Args:
            workspace_config: Configuration settings for the workspace database and paths.
        """
        if workspace_config is None:
            workspace_config = get_config().workspace
        self.workspace_config = workspace_config
        self.db_path = self.workspace_config.db_path
        self._listeners: list[Listener] = []
        self.db = DatabaseManager(self.db_path)
        self.db.verify_db()

    async def create_session(
        self,
        session_id: str | None = None,
        title: str = "New Session",
        system_prompt: str | None = None,
        created_at: float | None = None,
    ) -> Session:
        """Create a new chat session record in SQLite and initialize system instructions.

        Args:
            session_id: Optional unique identifier. If None, a random 8-char hex is generated.
            title: Summary title or first message snippet for the session.
            system_prompt: Optional defining system prompt instructions.
            created_at: Optional initial creation timestamp float.

        Returns:
            The created Session instance.
        """
        if session_id is None:
            session_id = uuid.uuid4().hex[:8]
        now = created_at if created_at is not None else time.time()
        sess = Session(id=session_id, title=title, created_at=now, updated_at=now)
        await asyncio.to_thread(self.db.create_session, sess)

        # Save initial system prompt as the first message in the session
        sys_msg = Message(
            session_id=session_id,
            chatbot_id="system",
            channel_id="system",
            sender="System",
            role=ROLE_SYSTEM,
            type=TYPE_TEXT,
            content=system_prompt or DEFAULT_SYSTEM_PROMPT,
            status=STATUS_RESPONDED,
            # Use a timestamp slightly in the past to ensure system message always comes first
            timestamp=now - 0.01,
        )
        await asyncio.to_thread(self.db.save_message, sys_msg)
        logger.debug(f"Created new chat session: {session_id} ({title})")
        return sess

    async def get_session(self, session_id: str) -> Session | None:
        """Retrieve a chat session record by ID.

        Args:
            session_id: Session ID to look up.

        Returns:
            Session object if found, None otherwise.
        """
        return await asyncio.to_thread(self.db.get_session, session_id)

    async def get_session_by_channel(self, chatbot_id: str, channel_id: str) -> Session | None:
        """Retrieve the chat session associated with a specific chatbot channel.

        Args:
            chatbot_id: Chatbot identifier.
            channel_id: Channel identifier.

        Returns:
            The Session object if found, None otherwise.
        """
        return await asyncio.to_thread(self.db.get_session_by_channel, chatbot_id, channel_id)

    async def update_session_updated_at(self, session_id: str) -> None:
        """Update the updated_at timestamp for a session.

        Args:
            session_id: Target session ID.
        """
        now = time.time()
        await asyncio.to_thread(self.db.update_session_updated_at, session_id, now)
        logger.debug(f"Updated session {session_id} timestamp.")

    async def list_sessions(self) -> list[Session]:
        """List all chat sessions ordered by most recently updated.

        Returns:
            A list of Session objects.
        """
        return await asyncio.to_thread(self.db.list_sessions)

    async def get_latest_session(self) -> Session | None:
        """Retrieve the most recently updated chat session.

        Returns:
            Session object of the latest session if available, None otherwise.
        """
        return await asyncio.to_thread(self.db.get_latest_session)

    async def post(self, message: Message) -> Message:
        """Post a message to storage and broadcast to active matching listeners.

        Args:
            message: The Message instance to post.

        Returns:
            The posted Message.
        """
        await asyncio.to_thread(self.db.save_message, message)
        logger.debug(f"Gateway posted message {message.id} ({message.role}:{message.sender})")

        for listener in self._listeners:
            if listener.filter_func(message):
                await listener.queue.put(message)

        return message

    async def listen(self, **filters: Any) -> AsyncGenerator[Message, None]:
        """Subscribe to messages matching specified filter criteria.

        Yields messages from storage (offline recovery / initial pending) and incoming real-time posts.

        Args:
            filters: Key-value attribute matches (e.g. role='user', status='pending').
        """

        def filter_func(msg: Message) -> bool:
            for k, v in filters.items():
                if getattr(msg, k, None) != v:
                    return False
            return True

        listener = Listener(filter_func)
        self._listeners.append(listener)
        seen_ids = set()

        # Offline recovery / initial pending fetch
        pending_messages = await asyncio.to_thread(self.db.get_messages_by_filters, filters)
        for msg in pending_messages:
            await listener.queue.put(msg)

        try:
            while True:
                msg = await listener.queue.get()
                if msg.id not in seen_ids:
                    seen_ids.add(msg.id)
                    yield msg
        finally:
            if listener in self._listeners:
                self._listeners.remove(listener)

    async def mark_message_processed(self, message_id: str) -> None:
        """Mark a completed user prompt as 'processed' in SQLite storage.

        Args:
            message_id: Unique ID of the user message.
        """
        await asyncio.to_thread(self.db.update_message_status, message_id, STATUS_PROCESSED)
        logger.debug(f"Message {message_id} marked as processed.")

    async def update_message_status(self, message_id: str, status: str) -> None:
        """Update the status of a message in storage.

        Args:
            message_id: Message ID.
            status: New status string.
        """
        await asyncio.to_thread(self.db.update_message_status, message_id, status)
        logger.debug(f"Message {message_id} status updated to {status}.")

    async def get_session_history(self, session_id: str, limit: int = 20) -> list[Message]:
        """Retrieve historical messages for a specific internal session.

        Args:
            session_id: Internal session identifier.
            limit: Maximum number of recent messages to fetch.

        Returns:
            A list of Message objects ordered by timestamp.
        """
        return await asyncio.to_thread(self.db.get_session_history, session_id, limit)
