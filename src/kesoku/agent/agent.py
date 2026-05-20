"""Autonomous Agent loop for Kesoku AI Agent framework.

Orchestrates polling pending messages from the gateway, invoking LLM, executing
tool calls, and returning responses using SessionWorker concurrency and
anti-stall mechanisms.
"""

import asyncio
import inspect
import json
import time
from typing import Any

from kesoku.agent.llm import BaseLLM, get_llm
from kesoku.agent.tools import ToolContext, ToolRegistry, default_registry
from kesoku.config import get_config
from kesoku.constants import (
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
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
        self.default_llm = llm
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

    async def _get_session_turns_count(self) -> int:
        """Get the number of conversational turns (user messages) in the current session.

        Returns:
            The count of user messages in this session.
        """
        raw_history = await self.gateway.get_session_history(self.session_id, limit=0)
        return len([m for m in raw_history if m.role == ROLE_USER])

    def _resolve_llm(self, current_msg: Message) -> BaseLLM:
        """Resolve the appropriate LLM instance for the current message, applying overrides.

        Args:
            current_msg: The active user message initiating the turn.

        Returns:
            A BaseLLM instance to use for this turn.
        """
        if current_msg.chatbot_id == "discord":
            channel_id = current_msg.channel_id
            channel_name = current_msg.metadata.get("channel_name", "")
            parent_id = current_msg.metadata.get("parent_channel_id")
            parent_name = current_msg.metadata.get("parent_channel_name")

            cfg = get_config()
            for override in cfg.discord.channels:
                identifiers = {channel_id, channel_name}
                if parent_id:
                    identifiers.add(parent_id)
                if parent_name:
                    identifiers.add(parent_name)
                if any(ident in override.channels for ident in identifiers if ident):
                    if override.llm:
                        logger.info(
                            f"Applying LLM override '{override.llm}' for Discord channel {channel_id} "
                            f"('{channel_name}')"
                        )
                        try:
                            return get_llm(override.llm)
                        except Exception as e:
                            logger.error(f"Failed to get override LLM provider '{override.llm}': {e}")

        return self.default_llm

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

        start_time = time.time()
        turn_tool_calls = 0
        turn_tokens = 0
        last_context_tokens = 0

        try:
            while self.running:
                # Check-in before atomic action (Thought Interruption)
                latest_msg = await self._drain_queue_and_pivot(current_msg)
                if latest_msg != current_msg:
                    current_msg = latest_msg

                # Resolve LLM dynamically for the current message (applying channel overrides)
                llm = self._resolve_llm(current_msg)

                # Retrieve and build the cleaned, prioritized, and aligned session history
                history = await self._build_clean_history()

                tools_list = self.tool_registry.get_tools_list()

                # LLM inference
                res = await llm.generate(
                    history=history,
                    tools=tools_list,
                )

                # Accumulate token metrics
                if res.prompt_tokens:
                    last_context_tokens = res.prompt_tokens
                if res.total_tokens:
                    turn_tokens += res.total_tokens

                # Check if LLM requested tool calls
                if res.tool_calls:
                    turn_tool_calls += len(res.tool_calls)
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
                        metadata={
                            "turn_metrics": {
                                "session_turns": await self._get_session_turns_count(),
                                "context_tokens": last_context_tokens,
                                "turn_tool_calls": turn_tool_calls,
                                "turn_tokens": turn_tokens,
                                "turn_time": time.time() - start_time,
                                "status": "finished",
                            }
                        },
                    )
                    await self.gateway.post(final_msg)
                    await self.gateway.mark_message_processed(current_msg.id)
                    break
        except asyncio.CancelledError:
            # Interrupted turn: save turn metrics to the initiating user message
            turn_metrics = {
                "session_turns": await self._get_session_turns_count(),
                "context_tokens": last_context_tokens,
                "turn_tool_calls": turn_tool_calls,
                "turn_tokens": turn_tokens,
                "turn_time": time.time() - start_time,
                "status": "interrupted",
            }
            history = await self.gateway.get_session_history(self.session_id, limit=20)
            user_msg = None
            for msg in reversed(history):
                if msg.role == ROLE_USER:
                    user_msg = msg
                    break
            if user_msg:
                user_msg.metadata["turn_metrics"] = turn_metrics
                await self.gateway.update_message_metadata(user_msg.id, user_msg.metadata)
            raise


    async def _build_clean_history(
        self,
        max_turns: int | None = None,
        pin_initial_turns: int | None = None,
        pin_recent_turns: int | None = None,
    ) -> list[Message]:
        """Retrieve, clean up, and format the conversational history for the LLM.

        Resolves orphaned tool calls, handles initial turns pinning, applies priority-based turn dropping,
        recovers loaded skills, and slides the turn window.

        Example Turn-Based Truncation for 100 Turns (max_turns=30, pin_initial_turns=3, pin_recent_turns=10):
        - System Prompt (kept at history[0])
        - Pinned initial Turns 1, 2, and 3 (retained in full)
        - Pinned recovered skill Turns (e.g., Turn 5 that loaded 'role-playing', recovered in full)
        - Candidate Turns 74 to 90 (stripped of thoughts and resolved tools, keeping only user/assistant text)
        - Candidate Turns 91 to 100 (kept in 100% full execution detail: prompts, thoughts, tool calls/results)

        Args:
            max_turns: Maximum logical turns allowed in context history. If None, uses config setting.
            pin_initial_turns: Number of initial turns to pin at the start. If None, uses config setting.
            pin_recent_turns: Number of latest turns to keep in full detail. If None, uses config setting.

        Returns:
            A list of cleanly structured, prioritized, and aligned Message objects for the LLM.
        """
        cfg = get_config().agent.history

        if max_turns is None:
            max_turns = cfg.max_turns
        if pin_initial_turns is None:
            pin_initial_turns = cfg.pin_initial_turns
        if pin_recent_turns is None:
            pin_recent_turns = cfg.pin_recent_turns

        # 1. Detects and heals orphaned tool calls by posting a synthesized interruption result.
        raw_history = await self.gateway.get_session_history(self.session_id, limit=0)
        tool_calls = [m for m in raw_history if m.type == TYPE_TOOL_CALL]
        tool_results_parent_ids = {m.parent_id for m in raw_history if m.type == TYPE_TOOL_RESULT and m.parent_id}

        healed = False
        for tc in tool_calls:
            if tc.id not in tool_results_parent_ids:
                logger.warning(
                    f"Found orphaned tool call {tc.id} (tool: {tc.metadata.get('tool_name')}). "
                    "Synthesizing interruption response."
                )
                tool_name = tc.metadata.get("tool_name", "unknown_tool")
                interrupted_msg = Message(
                    session_id=self.session_id,
                    chatbot_id=tc.chatbot_id,
                    channel_id=tc.channel_id,
                    sender=tool_name,
                    role=ROLE_TOOL,
                    type=TYPE_TOOL_RESULT,
                    content=f"Tool `{tool_name}` execution was interrupted due to service restart.",
                    status=STATUS_RESPONDED,
                    parent_id=tc.id,
                    metadata={
                        "tool_name": tool_name,
                        "tool_error": "Tool execution was interrupted due to service restart.",
                    },
                )
                await self.gateway.post(interrupted_msg)
                healed = True

        if healed:
            raw_history = await self.gateway.get_session_history(self.session_id, limit=0)

        # 2. Always preserves the initial system message(s) at the start.
        system_msg = None
        for m in raw_history:
            if m.role == ROLE_SYSTEM:
                system_msg = m
                break

        conv_msgs = [m for m in raw_history if (not system_msg or m.id != system_msg.id) and m.role != ROLE_SYSTEM]

        # 3. Groups messages into complete logical turns (User prompt -> ... -> before next user prompt).
        turns: list[list[Message]] = []
        current_turn: list[Message] = []
        for m in conv_msgs:
            if m.role == ROLE_USER:
                if current_turn:
                    turns.append(current_turn)
                current_turn = [m]
            else:
                if current_turn:
                    current_turn.append(m)
                else:
                    current_turn = [m]
        if current_turn:
            turns.append(current_turn)

        # 4. Pins the first K turns immediately following the system message.
        pinned_turns = turns[:pin_initial_turns]
        candidate_turns = turns[pin_initial_turns:]

        # 5. Drops thoughts and intermediate tool turns for older turns, treating parallel batches atomically.
        cutoff_idx = max(0, len(candidate_turns) - pin_recent_turns)
        older_candidate_turns = candidate_turns[:cutoff_idx]
        recent_candidate_turns = candidate_turns[cutoff_idx:]

        tc_to_tr = {}
        for m in raw_history:
            if m.type == TYPE_TOOL_RESULT and m.parent_id:
                tc_to_tr[m.parent_id] = m

        turn_to_tcs = {}
        for m in raw_history:
            if m.type == TYPE_TOOL_CALL and m.parent_id:
                turn_to_tcs.setdefault(m.parent_id, []).append(m)

        resolved_turns_without_skill = set()
        for user_msg_id, tcs in turn_to_tcs.items():
            is_resolved = all(tc.id in tc_to_tr for tc in tcs)
            has_skill = any(tc.metadata.get("tool_name") == "use_skill" for tc in tcs)
            if is_resolved and not has_skill:
                resolved_turns_without_skill.add(user_msg_id)

        clean_older_candidate_turns = []
        for turn in older_candidate_turns:
            dropped_ids = set()
            user_prompt = next((m for m in turn if m.role == ROLE_USER), None)
            user_prompt_id = user_prompt.id if user_prompt else None

            for m in turn:
                if m.role == ROLE_ASSISTANT and m.type == TYPE_THOUGHT:
                    dropped_ids.add(m.id)
                    continue
                if m.type == TYPE_TOOL_CALL and user_prompt_id and user_prompt_id in resolved_turns_without_skill:
                    dropped_ids.add(m.id)
                    continue
                if m.type == TYPE_TOOL_RESULT and m.parent_id:
                    parent_tc = next((tc for tcs in turn_to_tcs.values() for tc in tcs if tc.id == m.parent_id), None)
                    if parent_tc and parent_tc.parent_id in resolved_turns_without_skill:
                        dropped_ids.add(m.id)
                        continue

            clean_turn = [m for m in turn if m.id not in dropped_ids]
            clean_older_candidate_turns.append(clean_turn)

        clean_candidate_turns = clean_older_candidate_turns + recent_candidate_turns

        # 6. Trims/aligns history strictly by turn count, naturally preserving user-message start.
        allowed_turns = max_turns - pin_initial_turns
        if allowed_turns <= 0:
            suffix_turns = []
            discarded_candidate_turns = clean_candidate_turns
        else:
            suffix_idx = max(0, len(clean_candidate_turns) - allowed_turns)
            suffix_turns = clean_candidate_turns[suffix_idx:]
            discarded_candidate_turns = clean_candidate_turns[:suffix_idx]

        # 7. Recovers any pinned skill use (use_skill) turns completely and atomically.
        recovered_turns = []
        for turn in discarded_candidate_turns:
            has_completed_skill = False
            for m in turn:
                if (
                    m.type == TYPE_TOOL_RESULT
                    and m.metadata.get("tool_name") == "use_skill"
                    and "tool_error" not in m.metadata
                ):
                    has_completed_skill = True
                    break
            if has_completed_skill:
                recovered_turns.append(turn)

        # 8. Flatten turns and construct the final chronological context history.
        final_history = []
        if system_msg:
            final_history.append(system_msg)
        for turn in pinned_turns:
            final_history.extend(turn)
        for turn in recovered_turns:
            final_history.extend(turn)
        for turn in suffix_turns:
            final_history.extend(turn)

        return final_history


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
