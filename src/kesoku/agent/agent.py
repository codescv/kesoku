"""Autonomous Agent loop for Kesoku AI Agent framework.

Orchestrates polling pending messages from the gateway, invoking LLM, executing
tool calls, and returning responses using SessionWorker concurrency and
anti-stall mechanisms.
"""

import asyncio
import inspect
import json
import re
import time
from typing import Any

from kesoku.agent.llm import BaseLLM, get_llm
from kesoku.agent.tools import ToolContext, ToolRegistry, default_registry
from kesoku.constants import (
    ROLE_ASSISTANT,
    ROLE_TOOL,
    ROLE_USER,
    STATUS_ERROR,
    STATUS_INTERRUPTED,
    STATUS_PENDING,
    STATUS_PENDING_AGENT,
    STATUS_PROCESSING,
    STATUS_RESPONDED,
    TYPE_TEXT,
    TYPE_THOUGHT,
    TYPE_TOOL_CALL,
    TYPE_TOOL_RESULT,
)
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
        llm: BaseLLM,
        tool_registry: ToolRegistry,
        dispatcher: Any,
    ) -> None:
        """Initialize SessionWorker.

        Args:
            session_id: Internal session identifier.
            gateway: Gateway instance.
            llm: LLM backend interface.
            tool_registry: Tool/skill registry.
            dispatcher: Parent Agent dispatcher reference.
        """
        self.session_id = session_id
        self.gateway = gateway
        self.llm = llm
        self.tool_registry = tool_registry
        self.dispatcher = dispatcher
        self.queue: asyncio.Queue[Message] = asyncio.Queue()
        self.running = False
        self.task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the session worker background processing loop."""
        self.running = True
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
        self.running = False
        if self.task and not self.task.done():
            self.task.cancel()

    async def _drain_queue_and_pivot(self, current_msg: Message) -> Message:
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
        while self.running:
            try:
                msg = await self.queue.get()
                await self.gateway.update_message_status(msg.id, STATUS_PROCESSING)
                await self._process_turn(msg)
            except asyncio.CancelledError:
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in SessionWorker for session {self.session_id}: {e}", exc_info=True)
                await asyncio.sleep(1.0)

    async def _process_turn(self, current_msg: Message) -> None:
        chatbot_id = current_msg.chatbot_id
        channel_id = current_msg.channel_id

        session = await self.gateway.get_session(self.session_id)
        if not session:
            logger.error(f"Session {self.session_id} not found in database. Aborting message processing.")
            await self.gateway.update_message_status(current_msg.id, STATUS_ERROR)
            return

        folder_name = session.workspace_name
        tool_context = ToolContext(session_id=self.session_id, session_workspace=folder_name)

        while self.running:
            # Check-in before atomic action (Thought Interruption)
            latest_msg = await self._drain_queue_and_pivot(current_msg)
            if latest_msg != current_msg:
                current_msg = latest_msg

            # Retrieve recent history from DB (which includes current_msg and any tool turns)
            history = await self.gateway.get_session_history(self.session_id, limit=50)

            tools_list = self.tool_registry.get_tools_list()

            # LLM inference
            res = await self.llm.generate(
                history=history,
                tools=tools_list,
            )

            # Check if LLM requested tool calls
            if res.tool_calls:
                thought_text = res.thought or res.content
                if thought_text:
                    thought_msg = Message(
                        session_id=self.session_id,
                        chatbot_id=chatbot_id,
                        channel_id=channel_id,
                        sender="Kesoku",
                        role=ROLE_ASSISTANT,
                        type=TYPE_THOUGHT,
                        content=thought_text,
                        status=STATUS_RESPONDED,
                        parent_id=current_msg.id,
                    )
                    await self.gateway.post(thought_msg)

                tool_call_msgs = []
                for call in res.tool_calls:
                    logger.info(f"Executing requested tool call: '{call.name}' with args {call.arguments}")
                    call_args_json = json.dumps(call.arguments, indent=2, ensure_ascii=False)
                    tool_call_msg = Message(
                        session_id=self.session_id,
                        chatbot_id=chatbot_id,
                        channel_id=channel_id,
                        sender="Kesoku",
                        role=ROLE_TOOL,
                        type=TYPE_TOOL_CALL,
                        content=f"Calling tool `{call.name}` with arguments:\n```json\n{call_args_json}\n```",
                        status=STATUS_RESPONDED,
                        parent_id=current_msg.id,
                        metadata={
                            "tool_name": call.name,
                            "tool_arguments": call.arguments,
                            "thought_signature": call.thought_signature,
                        },
                    )
                    await self.gateway.post(tool_call_msg)
                    tool_call_msgs.append((call, tool_call_msg))

                async def _exec_tool(call: Any, tc_msg: Message) -> Message:
                    """Execute a single tool call asynchronously and return the resulting Message.

                    Args:
                        call: ToolCallRequest instance.
                        tc_msg: The corresponding ToolCall Message.

                    Returns:
                        A Message representing either successful tool result or error.
                    """
                    if not self.queue.empty():
                        logger.info(f"Interruption prior to tool '{call.name}'. Aborting tool execution.")
                    try:
                        tool_func = self.tool_registry.get_tool(call.name)
                        call_kwargs = dict(call.arguments)
                        sig = inspect.signature(tool_func)
                        if "context" in sig.parameters:
                            call_kwargs["context"] = tool_context
                        # Atomic tool execution
                        result = await asyncio.to_thread(tool_func, **call_kwargs)
                        return Message(
                            session_id=self.session_id,
                            chatbot_id=chatbot_id,
                            channel_id=channel_id,
                            sender=call.name,
                            role=ROLE_TOOL,
                            type=TYPE_TOOL_RESULT,
                            content=f"Tool `{call.name}` returned:\n```\n{result}\n```",
                            status=STATUS_RESPONDED,
                            parent_id=tc_msg.id,
                            metadata={"tool_name": call.name, "tool_result": str(result)},
                        )
                    except Exception as te:
                        logger.error(f"Error executing tool '{call.name}': {te}")
                        return Message(
                            session_id=self.session_id,
                            chatbot_id=chatbot_id,
                            channel_id=channel_id,
                            sender=call.name,
                            role=ROLE_TOOL,
                            type=TYPE_TOOL_RESULT,
                            content=f"Tool `{call.name}` error:\n```\n{te}\n```",
                            status=STATUS_RESPONDED,
                            parent_id=tc_msg.id,
                            metadata={"tool_name": call.name, "tool_error": str(te)},
                        )

                exec_tasks = [_exec_tool(call, tc_msg) for call, tc_msg in tool_call_msgs]
                if not self.queue.empty():
                    logger.info("Interruption detected before launching concurrent tool execution.")
                    for coro in exec_tasks:
                        coro.close()
                    break
                result_msgs = await asyncio.gather(*exec_tasks)
                for rm in result_msgs:
                    await self.gateway.post(rm)

                continue
            else:
                if res.thought:
                    thought_msg = Message(
                        session_id=self.session_id,
                        chatbot_id=chatbot_id,
                        channel_id=channel_id,
                        sender="Kesoku",
                        role=ROLE_ASSISTANT,
                        type=TYPE_THOUGHT,
                        content=res.thought,
                        status=STATUS_RESPONDED,
                        parent_id=current_msg.id,
                    )
                    await self.gateway.post(thought_msg)

                final_content = res.content
                if not final_content:
                    final_content = "Processed request successfully."

                final_msg = Message(
                    session_id=self.session_id,
                    chatbot_id=chatbot_id,
                    channel_id=channel_id,
                    sender="Kesoku",
                    role=ROLE_ASSISTANT,
                    type=TYPE_TEXT,
                    content=final_content,
                    status=STATUS_PENDING,
                    parent_id=current_msg.id,
                )
                await self.gateway.post(final_msg)
                await self.gateway.mark_message_processed(current_msg.id)
                break


class Agent:
    """Core autonomous agent dispatcher loop orchestrating SessionWorkers."""

    def __init__(
        self,
        gateway: Gateway,
        llm: BaseLLM | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        """Initialize the Agent dispatcher.

        Args:
            gateway: The Gateway instance providing message queues and persistence.
            llm: The LLM backend interface. If None, initializes via get_llm().
            tool_registry: Registry of available tools/skills. If None, initializes default_registry.
        """
        if llm is None:
            llm = get_llm()
        if tool_registry is None:
            tool_registry = default_registry

        self.gateway = gateway
        self.llm = llm
        self.tool_registry = tool_registry
        self.workers: dict[str, SessionWorker] = {}
        self._running = False
        self._master_task: asyncio.Task[None] | None = None

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
                            llm=self.llm,
                            tool_registry=self.tool_registry,
                            dispatcher=self,
                        )
                        self.workers[msg.session_id] = worker
                        worker.start()

                    await worker.enqueue(msg)
        except asyncio.CancelledError:
            logger.info("Agent master dispatcher loop cancelled.")
        finally:
            self._running = False
            self.stop_all_workers()

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
