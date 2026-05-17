"""Autonomous Agent loop for Kesoku AI Agent framework.

Orchestrates polling pending messages from the gateway, invoking LLM, executing
tool calls, and returning responses using SessionWorker concurrency and
anti-stall mechanisms.
"""

import asyncio
import json
from typing import Any

from kesoku.agent.llm import BaseLLM, GeminiLLM
from kesoku.agent.tools import ToolRegistry, default_registry
from kesoku.constants import (
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL,
    ROLE_USER,
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

DEFAULT_SYSTEM_PROMPT = """You are Kesoku Agent, a helpful, highly capable autonomous AI assistant.
You can use available tools to calculate equations, search information, and answer user questions precisely.
"""


class SessionWorker:
    """Dedicated asynchronous worker handling message queues and tool execution for a single conversational session."""

    def __init__(
        self,
        session_id: str,
        gateway: Gateway,
        llm: BaseLLM,
        tool_registry: ToolRegistry,
        system_prompt: str,
        dispatcher: Any,
    ) -> None:
        """Initialize SessionWorker.

        Args:
            session_id: Internal session identifier.
            gateway: Gateway instance.
            llm: LLM backend interface.
            tool_registry: Tool/skill registry.
            system_prompt: System prompt instructions.
            dispatcher: Parent Agent dispatcher reference.
        """
        self.session_id = session_id
        self.gateway = gateway
        self.llm = llm
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
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

    async def _worker_loop(self) -> None:
        while self.running:
            try:
                # 1. Pull: Get latest user message(s) from queue
                msg = await self.queue.get()
                user_messages = [msg]
                while not self.queue.empty():
                    user_messages.append(self.queue.get_nowait())

                # Update status to processing / interrupted
                for m in user_messages[:-1]:
                    await self.gateway.update_message_status(m.id, STATUS_INTERRUPTED)
                current_msg = user_messages[-1]
                await self.gateway.update_message_status(current_msg.id, STATUS_PROCESSING)

                # Process turn
                await self._process_turn(current_msg)

            except asyncio.CancelledError:
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in SessionWorker for session {self.session_id}: {e}", exc_info=True)
                await asyncio.sleep(1.0)

    async def _process_turn(self, current_msg: Message) -> None:
        chatbot_id = current_msg.chatbot_id
        channel_id = current_msg.channel_id
        original_user_msg = current_msg

        while self.running:
            # 3. Check-in before atomic action (Thought Interruption)
            if not self.queue.empty():
                new_msgs = []
                while not self.queue.empty():
                    new_msgs.append(self.queue.get_nowait())
                for m in new_msgs[:-1]:
                    await self.gateway.update_message_status(m.id, STATUS_INTERRUPTED)
                await self.gateway.update_message_status(original_user_msg.id, STATUS_INTERRUPTED)
                current_msg = new_msgs[-1]
                original_user_msg = current_msg
                await self.gateway.update_message_status(current_msg.id, STATUS_PROCESSING)
                logger.info(f"Thought interruption detected in session {self.session_id}! Pivoting to {current_msg.id}")

            # Retrieve recent history from DB
            history = await self.gateway.get_session_history(self.session_id, limit=20)
            # Exclude current_msg if it's in history
            history = [m for m in history if m.id != current_msg.id]

            tools_list = self.tool_registry.get_tools_list()

            # Step 2: Perform ONE atomic action (LLM inference)
            res = await self.llm.generate(
                prompt=current_msg.content,
                system_prompt=self.system_prompt,
                history=history,
                tools=tools_list,
            )

            # Check if LLM requested tool calls
            if res.tool_calls:
                tool_execution_summaries = []
                interrupted_in_tools = False

                if res.content:
                    thought_msg = Message(
                        session_id=self.session_id,
                        chatbot_id=chatbot_id,
                        channel_id=channel_id,
                        sender="Kesoku",
                        role=ROLE_ASSISTANT,
                        type=TYPE_THOUGHT,
                        content=res.content,
                        status=STATUS_RESPONDED,
                        parent_id=original_user_msg.id,
                    )
                    await self.gateway.post(thought_msg)

                for call in res.tool_calls:
                    # 3. Check-in before each tool execution (Thought Interruption between LLM calls / tool runs)
                    if not self.queue.empty():
                        logger.info(f"Interruption prior to tool '{call.name}'. Aborting tool execution to pivot.")
                        interrupted_in_tools = True
                        break

                    logger.info(f"Executing requested tool call: '{call.name}' with args {call.arguments}")
                    call_args_json = json.dumps(call.arguments, indent=2)
                    tool_call_msg = Message(
                        session_id=self.session_id,
                        chatbot_id=chatbot_id,
                        channel_id=channel_id,
                        sender="Kesoku",
                        role=ROLE_TOOL,
                        type=TYPE_TOOL_CALL,
                        content=f"Calling tool `{call.name}` with arguments:\n```json\n{call_args_json}\n```",
                        status=STATUS_RESPONDED,
                        parent_id=original_user_msg.id,
                    )
                    await self.gateway.post(tool_call_msg)

                    try:
                        tool_func = self.tool_registry.get_tool(call.name)
                        # Atomic tool execution (Never Kill Mid-Tool)
                        result = await asyncio.to_thread(tool_func, **call.arguments)
                        tool_execution_summaries.append(f"Tool '{call.name}' returned: {result}")
                        tool_result_msg = Message(
                            session_id=self.session_id,
                            chatbot_id=chatbot_id,
                            channel_id=channel_id,
                            sender=call.name,
                            role=ROLE_TOOL,
                            type=TYPE_TOOL_RESULT,
                            content=f"Tool `{call.name}` returned:\n```\n{result}\n```",
                            status=STATUS_RESPONDED,
                            parent_id=original_user_msg.id,
                        )
                        await self.gateway.post(tool_result_msg)
                    except Exception as te:
                        logger.error(f"Error executing tool '{call.name}': {te}")
                        tool_execution_summaries.append(f"Tool '{call.name}' error: {te}")
                        tool_error_msg = Message(
                            session_id=self.session_id,
                            chatbot_id=chatbot_id,
                            channel_id=channel_id,
                            sender=call.name,
                            role=ROLE_TOOL,
                            type=TYPE_TOOL_RESULT,
                            content=f"Tool `{call.name}` error:\n```\n{te}\n```",
                            status=STATUS_RESPONDED,
                            parent_id=original_user_msg.id,
                        )
                        await self.gateway.post(tool_error_msg)

                if interrupted_in_tools:
                    # Let outer while self.running loop pick up the new user message in queue
                    continue

                # Check-in: check queue after tool execution
                if not self.queue.empty():
                    new_msgs = []
                    while not self.queue.empty():
                        new_msgs.append(self.queue.get_nowait())
                    for m in new_msgs[:-1]:
                        await self.gateway.update_message_status(m.id, STATUS_INTERRUPTED)
                    await self.gateway.update_message_status(original_user_msg.id, STATUS_INTERRUPTED)
                    current_msg = new_msgs[-1]
                    original_user_msg = current_msg
                    await self.gateway.update_message_status(current_msg.id, STATUS_PROCESSING)
                    logger.info(f"Interruption after tool execution! Pivoting to new message {current_msg.id}")
                    continue

                # Formulate followup prompt for LLM to generate final response
                followup_prompt = (
                    f"User request was: {original_user_msg.content}\n"
                    f"Tool execution results:\n"
                    + "\n".join(tool_execution_summaries)
                    + "\nPlease formulate the final response to the user based on these results."
                )
                current_msg = Message(
                    session_id=self.session_id,
                    chatbot_id=chatbot_id,
                    channel_id=channel_id,
                    sender="System",
                    role=ROLE_SYSTEM,
                    type=TYPE_TEXT,
                    content=followup_prompt,
                    status=STATUS_RESPONDED,
                    parent_id=original_user_msg.id,
                )
                await self.gateway.post(current_msg)
                continue

            else:
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
                )
                await self.gateway.post(final_msg)
                await self.gateway.mark_message_responded(original_user_msg.id)
                break


class Agent:
    """Core autonomous agent dispatcher loop orchestrating SessionWorkers."""

    def __init__(
        self,
        gateway: Gateway,
        llm: BaseLLM | None = None,
        tool_registry: ToolRegistry | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        """Initialize the Agent dispatcher.

        Args:
            gateway: The Gateway instance providing message queues and persistence.
            llm: The LLM backend interface. If None, initializes GeminiLLM.
            tool_registry: Registry of available tools/skills. If None, initializes default_registry.
            system_prompt: Defining system instructions for the agent.
        """
        if llm is None:
            llm = GeminiLLM()
        if tool_registry is None:
            tool_registry = default_registry

        self.gateway = gateway
        self.llm = llm
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
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
                            system_prompt=self.system_prompt,
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
