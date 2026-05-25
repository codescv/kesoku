"""Autonomous Agent loop for Kesoku AI Agent framework.

Orchestrates polling pending messages from the gateway, invoking LLM, executing
tool calls, and returning responses using SessionWorker concurrency and
anti-stall mechanisms.
"""

import asyncio
import os

from kesoku.agent.tool_runner import ToolRunner
from kesoku.agent.tools import ToolContext
from kesoku.agent.turn_executor import TurnExecutor
from kesoku.agent.turn_logger import TurnLogger
from kesoku.constants import (
    ROLE_USER,
    STATUS_ERROR,
    STATUS_INTERRUPTED,
    STATUS_PENDING,
    STATUS_PENDING_AGENT,
    STATUS_PROCESSING,
)
from kesoku.context import KesokuContext
from kesoku.db import Message
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger

logger = setup_logger(__name__)


class SessionWorker:
    """Dedicated asynchronous worker handling message queues and tool execution for a single conversational session."""

    def __init__(
        self,
        session_id: str,
        gateway: Gateway,
        context: KesokuContext | None = None,
    ) -> None:
        """Initialize SessionWorker.

        Args:
            session_id: Internal session identifier.
            gateway: Gateway instance.
            context: Optional runtime context container.
        """
        self.session_id = session_id
        self.gateway = gateway
        self.context = context or getattr(gateway, "context", KesokuContext())
        self.queue: asyncio.Queue[Message] = asyncio.Queue()
        self._running = False
        self.task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        """Check if the session worker is currently running.

        Returns:
            True if running, False otherwise.
        """
        return self._running

    def start(self) -> None:
        """Start the session worker background processing loop."""
        self._running = True
        self.task = asyncio.create_task(self._worker_loop())
        logger.info(f"Started SessionWorker for session {self.session_id}")

    async def enqueue(self, msg: Message) -> None:
        """Enqueue a user message for processing.

        Args:
            msg: The user Message.
        """
        await self.queue.put(msg)

    def stop(self) -> None:
        """Stop the worker loop and cancel pending tasks."""
        self._running = False
        if self.task and not self.task.done():
            self.task.cancel()

    def queue_empty(self) -> bool:
        """Check if the message queue is empty.

        Returns:
            True if empty, False otherwise.
        """
        return self.queue.empty()

    async def drain_queue_and_pivot(self, current_msg: Message) -> Message:
        """Drain pending messages from the queue and pivot to the latest one.

        Marks earlier messages in the queue as interrupted.

        Args:
            current_msg: The message currently being processed.

        Returns:
            The latest message to process.
        """
        if self.queue.empty():
            return current_msg

        new_msgs = []
        while True:
            try:
                new_msgs.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        for m in new_msgs[:-1]:
            await self.gateway.update_message_status(m.id, STATUS_INTERRUPTED)
        await self.gateway.update_message_status(current_msg.id, STATUS_INTERRUPTED)
        latest_msg = new_msgs[-1]
        await self.gateway.update_message_status(latest_msg.id, STATUS_PROCESSING)
        logger.info(f"Thought interruption detected in session {self.session_id}! Pivoting to {latest_msg.id}")
        return latest_msg

    async def _worker_loop(self) -> None:
        while self._running:
            try:
                msg = await self.queue.get()
                await self.gateway.update_message_status(msg.id, STATUS_PROCESSING)
                await self._process_turn(msg)
            except asyncio.CancelledError:
                self._running = False
                break
            except Exception as e:
                logger.error(f"Error in SessionWorker for session {self.session_id}: {e}", exc_info=True)
                await asyncio.sleep(1.0)



    async def _process_turn(self, current_msg: Message) -> None:
        session = await self.gateway.get_session(self.session_id)
        if not session:
            logger.error(f"Session {self.session_id} not found in database. Aborting message processing.")
            await self.gateway.update_message_status(current_msg.id, STATUS_ERROR)
            return

        folder_name = session.workspace_name
        tool_context = ToolContext(session_id=self.session_id, session_workspace=folder_name)

        cfg = self.context.config
        session_staging_dir = os.path.realpath(  # noqa: ASYNC240
            os.path.join(cfg.workspace.sessions_dir, folder_name)
        )
        os.makedirs(session_staging_dir, exist_ok=True)  # noqa: ASYNC240

        tool_runner = ToolRunner(self.context.tool_registry, tool_context)
        turn_logger = TurnLogger(self.session_id, session_staging_dir)
        turn_executor = TurnExecutor(
            session_id=self.session_id,
            gateway=self.gateway,
            tool_runner=tool_runner,
            turn_logger=turn_logger,
            context=self.context,
        )

        await turn_executor.process_turn(
            current_msg=current_msg,
            worker=self,
            session_staging_dir=session_staging_dir,
        )


class Agent:
    """Core autonomous agent dispatcher loop orchestrating SessionWorkers."""

    def __init__(
        self,
        gateway: Gateway,
        context: KesokuContext | None = None,
    ) -> None:
        """Initialize the Agent dispatcher.

        Args:
            gateway: The Gateway instance providing message queues and persistence.
            context: Optional runtime context container.
        """
        self.context = context or getattr(gateway, "context", None) or KesokuContext()
        self.gateway = gateway
        self.workers: dict[str, SessionWorker] = {}
        self._running = False
        self._master_task: asyncio.Task[None] | None = None
        self.gateway.register_agent(self)

    async def start(self) -> None:
        """Start the master listener loop dispatching messages to SessionWorkers."""
        self._running = True
        self._master_task = asyncio.current_task()
        logger.info("Kesoku Agent master dispatcher loop started.")

        try:
            async for msg in self.gateway.listen(role=ROLE_USER):
                if not self._running:
                    break

                if msg.status in (STATUS_PENDING, STATUS_PENDING_AGENT):
                    logger.debug(f"Dispatcher dispatching message {msg.id} for session {msg.session_id}")

                    worker = self.workers.get(msg.session_id)
                    if worker is None or not worker.running:
                        worker = SessionWorker(
                            session_id=msg.session_id,
                            gateway=self.gateway,
                            context=self.context,
                        )
                        self.workers[msg.session_id] = worker
                        worker.start()

                    await worker.enqueue(msg)
        except asyncio.CancelledError:
            logger.info("Agent master dispatcher loop cancelled.")
        finally:
            self._running = False
            self.stop_all_workers()

    async def stop_session_worker(self, session_id: str) -> None:
        """Stop the active session worker for the given session ID.

        Args:
            session_id: Unique identifier for the session worker.
        """
        worker = self.workers.get(session_id)
        if worker:
            worker.stop()
            self.workers.pop(session_id, None)

    def stop_all_workers(self) -> None:
        """Stop all active session workers."""
        for worker in list(self.workers.values()):
            worker.stop()
        self.workers.clear()

    def stop(self) -> None:
        """Signal the agent dispatcher to stop and cancel worker tasks."""
        self._running = False
        if self._master_task and not self._master_task.done():
            self._master_task.cancel()
        self.stop_all_workers()
