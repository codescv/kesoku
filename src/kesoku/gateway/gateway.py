"""Gateway module for Kesoku AI Agent.

Orchestrates incoming/outgoing messages between chatbots and SQLite persistence
via DatabaseManager using a Pure Broker Pattern.
"""

import asyncio
import os
import shutil
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from typing import Any

from kesoku.agent.prompt import build_sys_prompt
from kesoku.constants import MessageRole, MessageStatus
from kesoku.context import KesokuContext
from kesoku.db import Message, Session
from kesoku.logger import setup_logger

logger = setup_logger(__name__)


class Listener:
    """Internal container for an active subscriber to the Gateway broker."""

    def __init__(self, filter_func: Callable[[Message], bool], maxsize: int = 1000) -> None:
        """Initialize a listener with an async queue and filter function."""
        self.queue: asyncio.Queue[Message] = asyncio.Queue(maxsize=maxsize)
        self.filter_func = filter_func


class Gateway:
    """Manages message ingestion, routing, and persistence across chatbots using a stateless broker pattern."""

    def __init__(
        self,
        context: KesokuContext | None = None,
    ) -> None:
        """Initialize the Gateway.

        Args:
            context: Runtime context container.
        """
        self.context = context or KesokuContext()
        self.workspace_config = self.context.config.workspace
        self.db = self.context.db

        self.db_path = self.workspace_config.db_path
        self._listeners: set[Listener] = set()
        self.db.verify_db()
        self.agent: Any | None = None

    def register_agent(self, agent: Any) -> None:
        """Register an active Agent dispatcher instance.

        Args:
            agent: The active Agent instance.
        """
        self.agent = agent

    async def create_session(
        self,
        session_id: str | None = None,
        title: str = "New Session",
        system_prompt: str | None = None,
        custom_prompt: str | None = None,
        created_at: float | None = None,
        chatbot_id: str | None = None,
        channel_id: str | None = None,
        role: str | None = None,
    ) -> Session:
        """Create a new chat session record in SQLite and initialize system instructions.

        Args:
            session_id: Optional unique identifier. If None, a random 8-char hex is generated.
            title: Summary title or first message snippet for the session.
            system_prompt: Optional defining system prompt instructions (pre-built).
            custom_prompt: Optional custom instructions to include in the built system prompt.
            created_at: Optional initial creation timestamp float.
            chatbot_id: Optional chatbot platform/instance identifier.
            channel_id: Optional external channel identifier.
            role: Optional pre-resolved character persona binding.

        Returns:
            The created Session instance.
        """
        if session_id is None:
            session_id = uuid.uuid4().hex[:8]
        now = created_at if created_at is not None else time.time()
        sess = Session(id=session_id, title=title, created_at=now, updated_at=now)

        # Build system prompt if not provided
        if system_prompt is None:
            system_prompt = build_sys_prompt(custom_prompt=custom_prompt, session=sess, role=role)
        sess.system_prompt = system_prompt

        await asyncio.to_thread(self.db.create_session, sess)
        logger.debug(f"Created new chat session: {session_id} ({title})")

        # If chatbot/channel parameters are provided, explicitly bind the active session mapping
        if chatbot_id and channel_id:
            await self.set_active_session_for_channel(chatbot_id, channel_id, session_id)

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

    async def set_active_session_for_channel(self, chatbot_id: str, channel_id: str, session_id: str) -> None:
        """Bind a session as the active session for a chatbot channel.

        Args:
            chatbot_id: Unique chatbot platform identifier.
            channel_id: External channel or thread identifier.
            session_id: Unique session identifier to bind.
        """
        await asyncio.to_thread(self.db.set_active_session_for_channel, chatbot_id, channel_id, session_id)

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

        for listener in list(self._listeners):
            if listener in self._listeners:
                if listener.filter_func(message):
                    try:
                        listener.queue.put_nowait(message)
                    except asyncio.QueueFull:
                        logger.warning(
                            f"Listener queue is full (maxsize={listener.queue.maxsize}). "
                            f"Dropping message {message.id} for this listener."
                        )

        return message

    async def listen(
        self,
        exclude_statuses: list[str] | None = None,
        exclude_roles: list[str] | None = None,
        **filters: Any,
    ) -> AsyncGenerator[Message, None]:
        """Subscribe to messages matching specified filter criteria.

        Yields messages from storage (offline recovery / initial pending) and incoming real-time posts.

        Args:
            exclude_statuses: Optional list of message lifecycle statuses to ignore.
            exclude_roles: Optional list of message roles to ignore.
            filters: Key-value attribute matches (e.g. role='user', status='pending').
        """

        def filter_func(msg: Message) -> bool:
            if exclude_statuses and msg.status in exclude_statuses:
                return False
            if exclude_roles and msg.role in exclude_roles:
                return False
            for k, v in filters.items():
                if getattr(msg, k, None) != v:
                    return False
            return True

        listener = Listener(filter_func)
        self._listeners.add(listener)
        seen_ids = set()

        # Offline recovery / initial pending fetch
        # Agent dispatcher recovers MessageStatus.PENDING_AGENT user messages,
        # while chatbot adapters recover MessageStatus.PENDING assistant responses.
        is_agent_query = filters.get("role") == MessageRole.USER
        recovery_status = MessageStatus.PENDING_AGENT if is_agent_query else MessageStatus.PENDING

        recovery_filters = {**filters}
        if "status" not in recovery_filters:
            recovery_filters["status"] = recovery_status

        pending_messages = await asyncio.to_thread(
            self.db.get_messages_by_filters, recovery_filters, None, exclude_roles
        )
        for msg in pending_messages:
            if msg.id not in seen_ids:
                seen_ids.add(msg.id)
                yield msg

        try:
            while True:
                msg = await listener.queue.get()
                if msg.id not in seen_ids:
                    seen_ids.add(msg.id)
                    yield msg
        finally:
            self._listeners.discard(listener)

    async def mark_message_processed(self, message_id: str) -> None:
        """Mark a completed user prompt as 'processed' in SQLite storage.

        Args:
            message_id: Unique ID of the user message.
        """
        await asyncio.to_thread(self.db.update_message_status, message_id, MessageStatus.PROCESSED)
        logger.debug(f"Message {message_id} marked as processed.")

    async def update_message_status(self, message_id: str, status: str) -> None:
        """Update the status of a message in storage.

        Args:
            message_id: Message ID.
            status: New status string.
        """
        await asyncio.to_thread(self.db.update_message_status, message_id, status)
        logger.debug(f"Message {message_id} status updated to {status}.")

    async def claim_message(self, message_id: str, new_status: str, expected_statuses: list[str]) -> bool:
        """Atomically update message status only if it is currently in one of expected_statuses.

        Args:
            message_id: Target message ID.
            new_status: The new status to set.
            expected_statuses: The list of valid current statuses.

        Returns:
            True if exactly one message was updated, False otherwise.
        """
        return await asyncio.to_thread(self.db.claim_message, message_id, new_status, expected_statuses)

    async def abort_session(self, session_id: str) -> None:
        """Abort any active processing for the session (e.g. if the channel is deleted).

        Args:
            session_id: Target session ID.
        """
        try:
            await self.context.active_jobs.stop_all_for_session(session_id)
        except Exception as e:
            logger.warning(f"Failed to clean up background jobs in abort_session: {e}")

        if self.agent:
            logger.info(f"Aborting active session worker for session {session_id} due to lost channel.")
            await self.agent.stop_session_worker(session_id, immediate=True)

        # Mark all outstanding (pending, processing, pending_agent) messages in this session as MessageStatus.ERROR
        history = await self.get_session_history(session_id, limit=0)
        for msg in history:
            if msg.status in (MessageStatus.PENDING_AGENT, MessageStatus.PROCESSING, MessageStatus.PENDING):
                await self.update_message_status(msg.id, MessageStatus.ERROR)


    async def update_message_metadata(self, message_id: str, metadata: dict[str, Any]) -> None:
        """Update the metadata of a message in storage.

        Args:
            message_id: Message ID.
            metadata: Dictionary representing the new metadata.
        """
        await asyncio.to_thread(self.db.update_message_metadata, message_id, metadata)
        logger.debug(f"Message {message_id} metadata updated.")

    async def update_session_system_prompt(self, session_id: str, system_prompt: str) -> None:
        """Update the main system prompt message for an existing session in the database.

        Args:
            session_id: Unique conversational session identifier.
            system_prompt: New system prompt content.
        """
        await asyncio.to_thread(self.db.update_session_system_prompt, session_id, system_prompt)
        logger.debug(f"Session {session_id} system prompt updated.")

    async def get_session_history(self, session_id: str, limit: int = 20, order: str = "phased") -> list[Message]:
        """Retrieve historical messages for a specific internal session.

        Args:
            session_id: Internal session identifier.
            limit: Maximum number of recent messages to fetch.
            order: The sorting order ("phased" or "grouped").

        Returns:
            A list of Message objects ordered by the sorting mechanism.
        """
        return await asyncio.to_thread(self.db.get_session_history, session_id, limit, order)

    async def get_session_turns_count(self, session_id: str) -> int:
        """Retrieve the count of user turns in the current session.

        Args:
            session_id: Target session identifier.

        Returns:
            The count of user messages.
        """
        return await asyncio.to_thread(self.db.get_session_turns_count, session_id)

    async def get_session_turn_anchors(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve list of user and system messages with their ID, role and timestamp.

        Args:
            session_id: Target session ID.

        Returns:
            List of dicts containing message id, role and timestamp.
        """
        return await asyncio.to_thread(self.db.get_session_turn_anchors, session_id)

    async def get_session_skill_anchor_ids(self, session_id: str) -> list[str]:
        """Retrieve the user message IDs of turns that loaded a skill successfully.

        Args:
            session_id: Target session ID.

        Returns:
            List of turn anchor message IDs.
        """
        return await asyncio.to_thread(self.db.get_session_skill_anchor_ids, session_id)

    async def get_orphaned_tool_calls(self, session_id: str) -> list[Message]:
        """Retrieve all orphaned tool call messages in a session.

        Args:
            session_id: Target session ID.

        Returns:
            List of orphaned tool call Message objects.
        """
        return await asyncio.to_thread(self.db.get_orphaned_tool_calls, session_id)

    async def get_session_history_by_ranges(
        self, session_id: str, ranges: list[tuple[float, float | None]], order: str = "phased"
    ) -> list[Message]:
        """Retrieve historical messages for specific timestamp ranges in a session, sorted logically.

        Args:
            session_id: Session ID to query.
            ranges: List of (start_timestamp, end_timestamp_or_None) tuples.
            order: Sorting order ("phased" or "grouped").

        Returns:
            List of Message objects, logically ordered.
        """
        return await asyncio.to_thread(self.db.get_session_history_by_ranges, session_id, ranges, order)


    async def delete_session(self, session_id: str) -> None:
        """Delete a session, its message history from database, and its workspace from disk.

        Args:
            session_id: The target session ID.
        """
        try:
            await self.context.active_jobs.stop_all_for_session(session_id)
        except Exception as e:
            logger.warning(f"Failed to clean up background jobs in delete_session: {e}")

        session = await self.get_session(session_id)
        if session:
            # Delete the staging session workspace folder from disk
            workspace_dir = os.path.join(self.workspace_config.sessions_dir, session.workspace_name)
            # Check existence asynchronously using to_thread
            if await asyncio.to_thread(os.path.exists, workspace_dir):
                try:
                    # Recursively delete directory asynchronously via to_thread
                    await asyncio.to_thread(shutil.rmtree, workspace_dir)
                    logger.debug(f"Deleted session workspace directory: {workspace_dir}")
                except Exception as e:
                    logger.warning(f"Failed to delete session workspace directory {workspace_dir}: {e}")

            # Delete the SQLite database records
            await asyncio.to_thread(self.db.delete_session, session_id)
            logger.info(f"Successfully deleted session {session_id} from database.")

    async def set_channel_role(self, chatbot_id: str, channel_id: str, role: str) -> None:
        """Bind a roleplay persona to a chatbot channel/thread."""
        await asyncio.to_thread(self.db.set_channel_role, chatbot_id, channel_id, role)

    async def get_channel_role(self, chatbot_id: str, channel_id: str) -> str | None:
        """Retrieve the roleplay persona bound directly to a chatbot channel/thread."""
        return await asyncio.to_thread(self.db.get_channel_role, chatbot_id, channel_id)

    async def get_channel_role_with_inheritance(
        self,
        chatbot_id: str,
        channel_id: str,
        session_id: str | None = None,
    ) -> str:
        """Retrieve the active role bound to a channel/thread with parent inheritance support."""
        return await asyncio.to_thread(self.db.get_channel_role_with_inheritance, chatbot_id, channel_id, session_id)

    async def get_channel_by_session(self, session_id: str) -> tuple[str, str] | None:
        """Retrieve the chatbot_id and channel_id mapping for a given session ID."""
        return await asyncio.to_thread(self.db.get_channel_by_session, session_id)

    async def get_last_message_timestamp(self, chatbot_id: str, channel_id: str) -> float | None:
        """Retrieve the timestamp of the most recent user or assistant message in a channel."""
        return await asyncio.to_thread(self.db.get_last_message_timestamp, chatbot_id, channel_id)

    async def get_cronjob_sent_stats_today(self, chatbot_id: str, channel_id: str) -> tuple[int, float | None]:
        """Retrieve count and last timestamp of cron messages sent today in a channel."""
        return await asyncio.to_thread(self.db.get_cronjob_sent_stats_today, chatbot_id, channel_id)


