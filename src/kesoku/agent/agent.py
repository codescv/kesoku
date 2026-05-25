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
from kesoku.constants import MessageRole, MessageStatus
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
        self._processing_turn = False
        self._turn_finished_event = asyncio.Event()

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

    async def stop(self, grace_period: float = 5.0, immediate: bool = False) -> None:
        """Stop the worker loop gracefully, waiting up to grace_period seconds for the current turn.

        Args:
            grace_period: Maximum seconds to wait for active turn to complete.
            immediate: Whether to cancel the worker immediately without waiting.
        """
        self._running = False
        if self.task and not self.task.done():
            if self._processing_turn and not immediate:
                try:
                    async with asyncio.timeout(grace_period):
                        await self._turn_finished_event.wait()
                except TimeoutError:
                    logger.warning(
                        f"SessionWorker graceful stop timed out after {grace_period} seconds "
                        f"for session {self.session_id}"
                    )
            if not self.task.done():
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
            await self.gateway.update_message_status(m.id, MessageStatus.INTERRUPTED)
        await self.gateway.update_message_status(current_msg.id, MessageStatus.INTERRUPTED)
        latest_msg = new_msgs[-1]
        logger.info(f"Thought interruption detected in session {self.session_id}! Pivoting to {latest_msg.id}")
        return latest_msg

    async def _worker_loop(self) -> None:
        while self._running:
            try:
                msg = await self.queue.get()
                self._processing_turn = True
                self._turn_finished_event.clear()
                try:
                    await self._process_turn(msg)
                finally:
                    self._processing_turn = False
                    self._turn_finished_event.set()
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
            await self.gateway.update_message_status(current_msg.id, MessageStatus.ERROR)
            return

        folder_name = session.workspace_name
        tool_context = ToolContext(
            session_id=self.session_id,
            session_workspace=folder_name,
            original_msg_id=current_msg.id,
            active_jobs=self.context.active_jobs,
        )

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

        # Recover orphaned processing messages at startup
        recovered_count = await asyncio.to_thread(
            self.gateway.db.recover_orphaned_processing_messages
        )
        if recovered_count > 0:
            logger.info(f"Recovered {recovered_count} orphaned processing messages back to pending_agent status.")

        try:
            async for msg in self.gateway.listen(role=MessageRole.USER):
                if not self._running:
                    break

                if msg.status == MessageStatus.PENDING_AGENT:
                    # Atomically claim the message to prevent duplicate delivery/processing
                    success = await self.gateway.claim_message(
                        msg.id, MessageStatus.PROCESSING, [MessageStatus.PENDING_AGENT]
                    )
                    if not success:
                        logger.debug(
                            f"Message {msg.id} already claimed by another process/worker or already processed."
                        )
                        continue

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
            await self.stop_all_workers()

    async def stop_session_worker(self, session_id: str, immediate: bool = False) -> None:
        """Stop the active session worker for the given session ID.

        Args:
            session_id: Unique identifier for the session worker.
            immediate: Whether to stop the worker immediately.
        """
        try:
            await self.context.active_jobs.stop_all_for_session(session_id)
        except Exception as e:
            logger.warning(f"Failed to clean up background jobs in stop_session_worker: {e}")

        worker = self.workers.get(session_id)
        if worker:
            await worker.stop(immediate=immediate)
            self.workers.pop(session_id, None)

    async def stop_all_workers(self) -> None:
        """Stop all active session workers."""
        workers = list(self.workers.values())
        if workers:
            await asyncio.gather(*(worker.stop() for worker in workers), return_exceptions=True)
        self.workers.clear()

    def stop(self) -> None:
        """Signal the agent dispatcher to stop and cancel worker tasks."""
        self._running = False
        if self._master_task and not self._master_task.done():
            self._master_task.cancel()
        else:
            asyncio.create_task(self.stop_all_workers())
