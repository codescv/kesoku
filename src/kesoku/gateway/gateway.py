"""Gateway module for Kesoku AI Agent.

Orchestrates incoming/outgoing messages between chatbots and SQLite persistence
via DatabaseManager using a Pure Broker Pattern.
"""

import asyncio
import os
import shutil
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from kesoku.agent.prompt import build_sys_prompt
from kesoku.constants import MessageRole, MessageStatus
from kesoku.context import KesokuContext
from kesoku.db import Message, Session
from kesoku.logger import setup_logger

logger = setup_logger(__name__)

StreamHandler = Callable[[str, str, str, str | None], Awaitable[None]]

DEFAULT_STREAM_MIN_INTERVAL_SEC = 1.5
"""Default throttle between streamed preview updates when a chatbot does not register its own."""


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
        self._stream_handlers: dict[str, StreamHandler] = {}
        self._stream_min_intervals: dict[str, float] = {}
        self._deferred_media_chatbots: set[str] = set()
        self._chatbots: dict[str, Any] = {}
        self.context.sync_db.verify_db()
        self.agent: Any | None = None
        self.cron_manager: Any | None = None

    def register_chatbot(self, chatbot: Any) -> None:
        """Register an active chatbot adapter instance for session lifecycle callbacks.

        Args:
            chatbot: Chatbot adapter instance with a ``chatbot_id`` attribute.
        """
        chatbot_id = getattr(chatbot, "chatbot_id", None)
        if isinstance(chatbot_id, str) and chatbot_id:
            self._chatbots[chatbot_id] = chatbot

    def register_deferred_media(self, chatbot_id: str, supported: bool = True) -> None:
        """Register whether ``chatbot_id`` supports deferred media sidecars (``.pending.json``).

        Args:
            chatbot_id: Unique chatbot identifier (e.g., 'discord', 'wechat').
            supported: True when the adapter renders deferred placeholders/watchers.
        """
        if supported:
            self._deferred_media_chatbots.add(chatbot_id)
        else:
            self._deferred_media_chatbots.discard(chatbot_id)

    def supports_deferred_media(self, chatbot_id: str) -> bool:
        """Return True if ``chatbot_id`` has registered deferred-media support."""
        return chatbot_id in self._deferred_media_chatbots

    def register_stream_handler(self, chatbot_id: str, handler: StreamHandler) -> None:
        """Register an ephemeral streaming preview handler for a chatbot adapter.

        Args:
            chatbot_id: Unique chatbot identifier (e.g., 'discord').
            handler: Async callback `(channel_id, session_id, turn_id, text_or_none) -> None`.
        """
        self._stream_handlers[chatbot_id] = handler

    def register_stream_min_interval(self, chatbot_id: str, seconds: float) -> None:
        """Register the minimum interval between streamed preview updates for `chatbot_id`.

        Args:
            chatbot_id: Unique chatbot identifier.
            seconds: Minimum seconds between preview publishes (clamped to >= 0).
        """
        self._stream_min_intervals[chatbot_id] = max(0.0, float(seconds))

    def get_stream_min_interval(self, chatbot_id: str) -> float:
        """Return the streamed preview throttle interval for `chatbot_id` (default 1.5s)."""
        return self._stream_min_intervals.get(chatbot_id, DEFAULT_STREAM_MIN_INTERVAL_SEC)

    def register_memory_policy_override_provider(self, provider: Callable[[str], str | None] | None) -> None:
        """Register an authoritative per-session memory policy override provider on the runtime context.

        Args:
            provider: Callable `(session_id) -> policy | None`, or None to unregister.
        """
        self.context.memory_policy_override_provider = provider

    def has_stream_handler(self, chatbot_id: str) -> bool:
        """Return True if a live stream preview handler is registered for `chatbot_id`."""
        return chatbot_id in self._stream_handlers

    async def publish_stream(
        self,
        chatbot_id: str,
        channel_id: str,
        session_id: str,
        turn_id: str,
        text: str | None,
    ) -> None:
        """Forward an ephemeral streaming preview update (or None to discard) to a chatbot handler.

        Args:
            chatbot_id: Target chatbot identifier.
            channel_id: Target channel or thread identifier.
            session_id: Active conversational session identifier.
            turn_id: Initiating user message ID for the turn.
            text: Accumulated response text to display, or None to clear any active preview.
        """
        handler = self._stream_handlers.get(chatbot_id)
        if handler is None:
            return
        try:
            await handler(channel_id, session_id, turn_id, text)
        except Exception as e:
            logger.warning(f"Stream handler for chatbot '{chatbot_id}' failed (session={session_id}): {e}")

    def register_agent(self, agent: Any) -> None:
        """Register an active Agent dispatcher instance.

        Args:
            agent: The active Agent instance.
        """
        self.agent = agent

    def register_cron_manager(self, cron_manager: Any) -> None:
        """Register the active CronManager instance.

        Args:
            cron_manager: The active CronManager instance.
        """
        self.cron_manager = cron_manager

    async def list_sessions_by_chatbots(
        self,
        chatbot_ids: list[str],
        role: str | None = None,
        cursor: tuple[float, str] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List sessions bound to any of `chatbot_ids` (newest first, keyset pagination).

        Args:
            chatbot_ids: Chatbot identifiers whose channel bindings qualify a session.
            role: Optional persona role filter.
            cursor: Optional `(updated_at, session_id)` keyset cursor.
            limit: Maximum number of sessions to return.

        Returns:
            Session summary dicts including all channel bindings (`chatbot_ids`).
        """
        return await self.db.list_sessions_by_chatbots(chatbot_ids, role, cursor, limit)

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
            session_id: Optional unique identifier. If None, a random 8-char hex is generated by Session model.
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
        now = created_at if created_at is not None else time.time()
        sess_kwargs = {"title": title, "created_at": now, "updated_at": now}
        if session_id is not None:
            sess_kwargs["id"] = session_id
        sess = Session(**sess_kwargs)

        if role is None:
            if chatbot_id and channel_id:
                role = await self.db.get_channel_role_with_inheritance(
                    chatbot_id,
                    channel_id,
                    sess.id,
                    default=self.context.config.get_chatbot_default_role(chatbot_id),
                )
            elif chatbot_id:
                role = self.context.config.get_chatbot_default_role(chatbot_id)
        if role:
            sess.role_name = role

        # Build system prompt if not provided
        if system_prompt is None:
            system_prompt = build_sys_prompt(custom_prompt=custom_prompt, session=sess, role=role)
        sess.system_prompt = system_prompt

        await self.db.create_session(sess)
        logger.debug(f"Created new chat session: {sess.id} ({title})")

        # If chatbot/channel parameters are provided, explicitly bind the active session mapping
        if chatbot_id and channel_id:
            await self.db.set_active_session_for_channel(chatbot_id, channel_id, sess.id)



        return sess

    async def post(self, message: Message) -> Message:
        """Post a message to storage and broadcast to active matching listeners.

        Args:
            message: The Message instance to post.

        Returns:
            The posted Message.
        """
        await self.db.save_message(message)
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

        pending_messages = await self.db.get_messages_by_filters(recovery_filters, None, exclude_roles)
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

    async def _cancel_session_deferred_media(self, session_id: str) -> None:
        """Cancel active deferred-media watchers and terminate detached workers for ``session_id``."""
        from kesoku.gateway.chatbot.deferred_media import cancel_session_deferred_entries_in_db

        for bot in list(self._chatbots.values()):
            cancel_fn = getattr(bot, "cancel_deferred_for_session", None)
            if callable(cancel_fn):
                try:
                    await cancel_fn(session_id)
                except Exception as e:
                    logger.warning(f"Failed to cancel deferred media on chatbot for session {session_id}: {e}")

        try:
            await cancel_session_deferred_entries_in_db(self.db, session_id)
        except Exception as e:
            logger.warning(f"Failed DB-level deferred media cleanup for session {session_id}: {e}")

    async def abort_session(self, session_id: str) -> None:
        """Abort any active processing for the session (e.g. if the channel is deleted).

        Args:
            session_id: Target session ID.
        """
        await self._cancel_session_deferred_media(session_id)

        try:
            await self.context.active_jobs.stop_all_for_session(session_id)
        except Exception as e:
            logger.warning(f"Failed to clean up background jobs in abort_session: {e}")

        if self.agent:
            logger.info(f"Aborting active session worker for session {session_id} due to lost channel.")
            await self.agent.stop_session_worker(session_id, immediate=True)

        # Mark all outstanding (pending, processing, pending_agent) messages in this session as MessageStatus.ERROR
        messages = await self.db.get_messages_by_filters(filters={"session_id": session_id})
        for msg in messages:
            if msg.status in (MessageStatus.PENDING_AGENT, MessageStatus.PROCESSING, MessageStatus.PENDING):
                await self.db.update_message_status(msg.id, MessageStatus.ERROR)


    async def delete_session(self, session_id: str) -> None:
        """Delete a session, its message history from database, and its workspace from disk.

        Args:
            session_id: The target session ID.
        """
        await self._cancel_session_deferred_media(session_id)

        if self.agent:
            try:
                await self.agent.stop_session_worker(session_id, immediate=True)
            except Exception as e:
                logger.warning(f"Failed to stop session worker in delete_session: {e}")

        try:
            from kesoku.agent.turn_executor import discard_session_cache_state

            discard_session_cache_state(session_id)
        except Exception:
            pass

        try:
            await self.context.active_jobs.stop_all_for_session(session_id)
        except Exception as e:
            logger.warning(f"Failed to clean up background jobs in delete_session: {e}")

        session = await self.db.get_session(session_id)
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
            await self.db.delete_session(session_id)
            logger.info(f"Successfully deleted session {session_id} from database.")
