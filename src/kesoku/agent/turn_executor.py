"""Orchestrates conversational turn execution, including LLM inference, thought logging, and tool calling."""

import asyncio
import datetime
import json
import os
import time
import traceback
from typing import TYPE_CHECKING

import tzlocal

from kesoku.agent.compressor import HistoryCompressor
from kesoku.agent.history import (
    build_history,
    prepare_history_for_llm,
)
from kesoku.agent.llm import BaseLLM, LLMResponse
from kesoku.agent.tool_runner import ToolRunner
from kesoku.agent.turn_logger import TurnLogger
from kesoku.config import KesokuConfig
from kesoku.context import KesokuContext
from kesoku.utils.async_fs import async_exists, async_read_text_file, async_write_text_file
from kesoku.utils.text import truncate_middle

if TYPE_CHECKING:
    from kesoku.agent.agent import SessionWorker
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger

logger = setup_logger(__name__)
MAX_TOTAL_CROSS_SESSION_CONTEXT_LENGTH = 3000
MAX_CHATBOT_ERROR_MESSAGE_LENGTH = 500


class TurnExecutor:
    """Orchestrates conversational turn execution, including LLM inference, thought logging, and tool calling."""

    def __init__(
        self,
        session_id: str,
        gateway: Gateway,
        tool_runner: ToolRunner,
        turn_logger: TurnLogger | None = None,
        context: KesokuContext | None = None,
    ) -> None:
        """Initialize TurnExecutor.

        Args:
            session_id: Unique conversational session identifier.
            gateway: Gateway instance.
            tool_runner: Tool runner handling actual tool execution.
            turn_logger: Optional logger to output detailed YAML logs of the turns.
            context: Optional runtime context container.
        """
        self.session_id = session_id
        self.gateway = gateway
        self.tool_runner = tool_runner
        self.turn_logger = turn_logger
        self.context = context or getattr(gateway, "context", KesokuContext())

    async def _get_session_turns_count(self) -> int:
        """Retrieve the count of user turns in the current session.

        Returns:
            The count of user messages.
        """
        return await self.context.db.get_session_turns_count(self.session_id)

    async def _is_bootstrap_turn(self, history: list[Message], current_msg: Message) -> bool:
        """Determine if this is a Bootstrap Turn (first turn of session or idle > 30 mins)."""
        turn_count = await self._get_session_turns_count()
        if turn_count <= 1:
            return True
        non_current_msgs = [m for m in history if m.id != current_msg.id]
        if non_current_msgs:
            last_msg_time = non_current_msgs[-1].timestamp
            # 1800 seconds = 30 minutes inactivity
            if current_msg.timestamp - last_msg_time > 1800:
                return True
        else:
            return True
        return False

    def _resolve_llm(self, current_msg: Message) -> BaseLLM:
        """Resolve the appropriate LLM instance for the current message, applying overrides.

        Args:
            current_msg: The active user message initiating the turn.

        Returns:
            A BaseLLM instance to use for this turn.
        """
        cfg = self.context.config
        discord_cfg = cfg.get_discord_config(current_msg.chatbot_id)
        if discord_cfg:
            channel_id = current_msg.channel_id
            channel_name = current_msg.metadata.get("channel_name", "")
            parent_id = current_msg.metadata.get("parent_channel_id")
            parent_name = current_msg.metadata.get("parent_channel_name")

            for override in discord_cfg.channels:
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
                            return self.context.get_llm(provider=override.llm)
                        except Exception as e:
                            logger.error(f"Failed to get override LLM provider '{override.llm}': {e}")

        return self.context.get_llm()

    async def process_turn(
        self,
        current_msg: Message,
        worker: "SessionWorker",
        session_staging_dir: str,
    ) -> None:
        """Process a single conversational turn.

        Args:
            current_msg: Active user message initiating the turn.
            worker: The SessionWorker handling this conversational session.
            session_staging_dir: Path to session staging directory for outputs/logs.
        """
        chatbot_id = current_msg.chatbot_id
        channel_id = current_msg.channel_id

        cfg = self.context.config
        start_time = time.time()
        turn_tool_calls = 0
        turn_tokens = 0
        last_context_tokens = 0
        last_cached_tokens = 0

        active_cache_name = getattr(worker, "active_cache_name", None)
        if active_cache_name is not None and not isinstance(active_cache_name, str):
            active_cache_name = None
        cached_messages_len = getattr(worker, "cached_messages_len", 0)
        if not isinstance(cached_messages_len, int):
            cached_messages_len = 0

        nudged = False
        try:
            while worker.running:
                # Check-in before atomic action (Thought Interruption)
                prev_msg_id = current_msg.id
                current_msg = await worker.drain_queue_and_pivot(current_msg)
                if current_msg.id != prev_msg_id:
                    logger.info(
                        f"Pivoted from message {prev_msg_id} to {current_msg.id} during turn loop. "
                        f"Resetting turn metrics and nudge flag."
                    )
                    self.tool_runner.tool_context.original_msg_id = current_msg.id
                    nudged = False
                    turn_tool_calls = 0
                    turn_tokens = 0
                    last_context_tokens = 0
                    last_cached_tokens = 0
                    start_time = time.time()
                    if active_cache_name:
                        try:
                            # We can resolve LLM for deletion using current_msg resolved LLM or recreate it
                            temp_llm = self._resolve_llm(current_msg)
                            await temp_llm.delete_cache(active_cache_name)
                        except Exception as de:
                            logger.warning(f"Failed to delete cache on pivot: {de}")
                        active_cache_name = None
                        cached_messages_len = 0
                        worker.active_cache_name = None
                        worker.active_cache_llm = None
                        worker.cached_messages_len = 0

                # Resolve LLM dynamically for the current message
                llm = self._resolve_llm(current_msg)

                # Retrieve system prompt directly from session
                session = await self.context.db.get_session(self.session_id)
                system_prompt = session.system_prompt if session else None

                tools_list = self.tool_runner.tool_registry.get_tools_list()

                # Retrieve and build the raw session history
                history = await build_history(
                    gateway=self.gateway,
                    session_id=self.session_id,
                    heal_orphans=True,
                )
                # Check and automatically compact history in-place if it exceeds threshold
                history, compacted_occurred = await self._check_and_auto_compact_history(
                    history=history,
                    system_prompt=system_prompt,
                    tools_list=tools_list,
                    llm=llm,
                    cfg=cfg,
                    current_msg=current_msg,
                )

                # If history was compacted, the existing cache is obsolete and must be deleted.
                if compacted_occurred and active_cache_name:
                    logger.info("Session history was compacted. Deleting obsolete context cache...")
                    try:
                        await llm.delete_cache(active_cache_name)
                    except Exception as de:
                        logger.warning(f"Failed to delete obsolete cache: {de}")
                    active_cache_name = None
                    cached_messages_len = 0
                    worker.active_cache_name = None
                    worker.active_cache_llm = None
                    worker.cached_messages_len = 0

                await self._inject_context_and_trigger_consolidation(history, current_msg, llm)

                # Prepare history for the LLM by stripping thoughts and attachments dynamically
                llm_history = prepare_history_for_llm(history)

                # Setup context cache if enabled, Gemini model, and not already created
                if cfg.gemini.context_caching and llm.__class__.__name__ == "GeminiLLM" and active_cache_name is None:
                    prefix_messages = []
                    last_user_idx = None
                    for idx, msg in enumerate(llm_history):
                        if msg.role == MessageRole.USER:
                            last_user_idx = idx
                    if last_user_idx is not None:
                        prefix_messages = llm_history[:last_user_idx]

                    if prefix_messages:
                        prefix_tokens = llm.count_tokens(
                            system_prompt=system_prompt,
                            history=prefix_messages,
                        )
                        if prefix_tokens >= cfg.gemini.context_caching_threshold:
                            logger.info(
                                f"Session prefix has {prefix_tokens} tokens. Creating explicit context cache..."
                            )
                            active_cache_name = await llm.create_cache(
                                contents=prefix_messages,
                                system_prompt=system_prompt,
                                tools=tools_list,
                                display_name=f"kesoku_{self.session_id}",
                                ttl_seconds=cfg.gemini.context_caching_ttl,
                            )
                            if active_cache_name:
                                cached_messages_len = len(prefix_messages)
                                worker.active_cache_name = active_cache_name
                                worker.active_cache_llm = llm
                                worker.cached_messages_len = cached_messages_len

                # If the prepared history length is somehow less than cached_messages_len, the cache is obsolete.
                if active_cache_name and len(llm_history) < cached_messages_len:
                    logger.info("History length is shorter than cache prefix. Deleting obsolete context cache...")
                    try:
                        await llm.delete_cache(active_cache_name)
                    except Exception as de:
                        logger.warning(f"Failed to delete obsolete cache: {de}")
                    active_cache_name = None
                    cached_messages_len = 0
                    worker.active_cache_name = None
                    worker.active_cache_llm = None
                    worker.cached_messages_len = 0

                # Partition history if using context cache
                logger.info(
                    f"Context caching debug: active_cache_name={active_cache_name}, "
                    f"llm_history_len={len(llm_history)}, cached_messages_len={cached_messages_len}"
                )
                if active_cache_name and len(llm_history) >= cached_messages_len:
                    generate_history = llm_history[cached_messages_len:]
                    generate_system_prompt = system_prompt
                    generate_tools = None
                    generate_cached_content = active_cache_name
                else:
                    generate_history = llm_history
                    generate_system_prompt = system_prompt
                    generate_tools = tools_list
                    generate_cached_content = None

                # LLM inference
                try:
                    res = await llm.generate(
                        system_prompt=generate_system_prompt,
                        history=generate_history,
                        tools=generate_tools,
                        cached_content=generate_cached_content,
                    )
                except Exception as e:
                    if active_cache_name and ("expired" in str(e) or "cache" in str(e).lower()):
                        logger.warning(
                            f"LLM generation failed, possibly due to cache expiration: {e}. "
                            "Retrying without context cache."
                        )
                        active_cache_name = None
                        cached_messages_len = 0
                        worker.active_cache_name = None
                        worker.active_cache_llm = None
                        worker.cached_messages_len = 0
                        generate_history = llm_history
                        generate_system_prompt = system_prompt
                        generate_tools = tools_list
                        generate_cached_content = None
                        res = await llm.generate(
                            system_prompt=generate_system_prompt,
                            history=generate_history,
                            tools=generate_tools,
                            cached_content=generate_cached_content,
                        )
                    else:
                        raise

                # Log the raw LLM turn using TurnLogger if enabled
                if cfg.agent.raw_llm_logs and self.turn_logger:
                    try:
                        self.turn_logger.log_llm_turn(
                            llm_provider=llm.__class__.__name__,
                            history=generate_history,
                            tools=tools_list,
                            response=res,
                            system_prompt=generate_system_prompt,
                        )
                    except Exception as le:
                        logger.error(f"Failed to log LLM turn: {le}", exc_info=True)

                # Accumulate token metrics
                last_context_tokens = res.prompt_tokens or 0
                last_cached_tokens = res.cached_tokens or 0
                if res.total_tokens:
                    turn_tokens += res.total_tokens

                if res.tool_calls:
                    turn_tool_calls += len(res.tool_calls)
                    should_continue = await self._execute_tool_calls(res, current_msg, worker)
                    if should_continue:
                        continue
                    else:
                        break
                else:
                    should_continue, nudged = await self._handle_final_response(
                        res=res,
                        current_msg=current_msg,
                        last_context_tokens=last_context_tokens,
                        last_cached_tokens=last_cached_tokens,
                        turn_tool_calls=turn_tool_calls,
                        turn_tokens=turn_tokens,
                        start_time=start_time,
                        llm=llm,
                        nudged=nudged,
                    )
                    if should_continue:
                        continue
                    else:
                        break
        except asyncio.CancelledError:
            # Interrupted turn: save turn metrics to the initiating user message
            turn_metrics = {
                "session_turns": await self._get_session_turns_count(),
                "context_tokens": last_context_tokens,
                "cached_tokens": last_cached_tokens,
                "turn_tool_calls": turn_tool_calls,
                "turn_tokens": turn_tokens,
                "turn_time": time.time() - start_time,
                "status": "interrupted",
            }
            history = await self.context.db.get_session_history(self.session_id, limit=20)
            user_msg = None
            for msg in reversed(history):
                if msg.role == MessageRole.USER:
                    user_msg = msg
                    break
            if user_msg:
                user_msg.metadata["turn_metrics"] = turn_metrics
                await self.context.db.update_message_metadata(user_msg.id, user_msg.metadata)
            raise
        except Exception as e:
            logger.error(f"Error in session turn {self.session_id}: {e}", exc_info=True)
            tb_str = traceback.format_exc()
            error_filename = f"error_{current_msg.id}.txt"
            error_file_path = os.path.join(session_staging_dir, error_filename)
            try:
                await async_write_text_file(error_file_path, tb_str)
            except Exception as fe:
                logger.error(f"Failed to write error traceback file to {error_file_path}: {fe}")

            full_error_msg = f"⚠️ An error occurred while processing your request: {e}"
            hint = f"\n\nFull error log saved to staging directory: {error_filename}"
            max_err_len = MAX_CHATBOT_ERROR_MESSAGE_LENGTH - len(hint)
            truncated_error = truncate_middle(
                full_error_msg,
                max_err_len,
                "\n\n... [Error Truncated for Brevity] ...\n\n",
            )
            truncated_content = truncated_error + hint

            error_msg = Message(
                session_id=self.session_id,
                chatbot_id=chatbot_id,
                channel_id=channel_id,
                sender="Kesoku",
                role=MessageRole.ASSISTANT,
                type=MessageType.TEXT,
                content=truncated_content,
                status=MessageStatus.PENDING,
                parent_id=current_msg.id,
            )
            await self.gateway.post(error_msg)
            await self.context.db.update_message_status(current_msg.id, MessageStatus.ERROR)

    async def _inject_context_and_trigger_consolidation(
        self,
        history: list[Message],
        current_msg: Message,
        llm: BaseLLM,
    ) -> Message | None:
        """Inject context and user preferences, triggering background consolidation if needed.

        Modifies the history list in-place by prepending context block to the latest user message.

        Returns:
            The modified latest User Message, or None if no user message was found to inject context into.
        """
        latest_user_msg = None
        for msg in reversed(history):
            if msg.role == MessageRole.USER:
                latest_user_msg = msg
                break

        if not latest_user_msg:
            return None

        active_role = await self.context.db.get_channel_role_with_inheritance(
            current_msg.chatbot_id,
            current_msg.channel_id,
            self.session_id,
        )

        # 1. Calculate if this is a Bootstrap Turn (first turn of session or idle > 30 mins)
        is_bootstrap = await self._is_bootstrap_turn(history, current_msg)

        # 2. Read role-based preferences.md if needed (bootstrap turn or every 4 turns)
        turn_count = await self._get_session_turns_count()
        inject_preferences = is_bootstrap or (turn_count > 0 and turn_count % 4 == 1)
        instructions_prefix = ""
        if inject_preferences and active_role:
            roles_dir = self.context.config.workspace.roles_dir
            if not os.path.isabs(roles_dir) and self.context.config.agent_working_dir:
                roles_dir = os.path.join(self.context.config.agent_working_dir, roles_dir)
            pref_path = os.path.join(roles_dir, active_role, "preferences.md")
            if await async_exists(pref_path):
                try:
                    role_prefs_content = await async_read_text_file(pref_path)
                    role_prefs_content = role_prefs_content.strip()
                    if role_prefs_content:
                        instructions_prefix = f"<instructions>\n{role_prefs_content}\n</instructions>\n"
                except Exception as e:
                    logger.warning(f"Failed to read preferences.md for role '{active_role}': {e}")

        # 3. Prepend Consolidated Passive Synchronization, Preferences, and Context Compression Guidelines
        # (if Bootstrap)
        full_prefix = ""
        if is_bootstrap:
            lines = [
                '<background_context type="sync_guidelines">',
                "- Use `memory_grep(query)` to search active memories and past chat messages for this role.",
                "- Use `view_message(message_id)` to inspect full historical messages when needed.",
                "</background_context>",
            ]
            full_prefix = "\n".join(lines)



        msg_idx = history.index(latest_user_msg)
        copied_msg = latest_user_msg.model_copy()
        msg_time = datetime.datetime.fromtimestamp(copied_msg.timestamp).astimezone()
        time_str = msg_time.strftime("%Y-%m-%d %H:%M:%S (%A) %Z")
        sender_name = copied_msg.metadata.get("sender_name") or copied_msg.sender
        if sender_name.lower() == "cronjob":
            sender_name = "system"
        try:
            tz_name = tzlocal.get_localzone_name()
        except Exception:
            tz_name = msg_time.tzname() or "UTC"

        copied_msg.content = (
            f"{instructions_prefix}"
            f"{full_prefix}"
            f'<current_message from="{sender_name}" time="{time_str}" timezone="{tz_name}">\n'
            f"{copied_msg.content}\n"
            "</current_message>"
        )
        history[msg_idx] = copied_msg
        latest_user_msg = copied_msg
        logger.info(
            f"Wrapped active user message {copied_msg.id} in <current_message "
            f'from="{sender_name}" time="{time_str}" timezone="{tz_name}"> '
            f"(bootstrap: {is_bootstrap})"
        )

        return latest_user_msg

    async def _check_and_auto_compact_history(
        self,
        history: list[Message],
        system_prompt: str | None,
        tools_list: list,
        llm: BaseLLM,
        cfg: KesokuConfig,
        current_msg: Message,
    ) -> tuple[list[Message], bool]:
        """Check context window usage and automatically compact history using custom turn-based compressor.

        Returns:
            A tuple of (active message history list, compacted_occurred bool).
        """
        if not history:
            return history, False

        compressor = HistoryCompressor(self.context.db)

        # Trigger automatic compaction
        compacted_occurred = await compressor.auto_compact_session(
            session_id=self.session_id,
            history=history,
            llm=llm,
            config=cfg,
        )

        # Retrieve all root summary nodes from the database for this session
        root_summaries = await self.context.db.get_root_summary_nodes(self.session_id)

        if not root_summaries:
            return history, compacted_occurred

        # Re-segment and assemble the scaffold context
        turns = compressor.segment_turns(history)
        protect_front = cfg.agent.protect_front_turns
        protect_tail = cfg.agent.protect_tail_turns

        # Assemble Protected Head
        protected_head = []
        for t in turns[:protect_front]:
            protected_head.extend(t)

        # Assemble Protected Tail
        protected_tail = []
        if len(turns) > protect_front:
            tail_start = max(protect_front, len(turns) - protect_tail)
            for t in turns[tail_start:]:
                protected_tail.extend(t)

        # Assemble Buffer (uncompacted turns in the middle)
        buffer = []
        if len(turns) > protect_front + protect_tail:
            middle_turns = turns[protect_front:-protect_tail]
            for t in middle_turns:
                if not any(msg.summary_node_id is not None for msg in t):
                    buffer.extend(t)

        # Format hierarchical summaries
        scaffold_parts = [
            "[Note: This conversation uses custom turn-based context management. "
            "Earlier turns have been compacted into hierarchical summaries below. "
            "You have access to search past messages and memories using standard search tools.]\n"
        ]
        for node in root_summaries:
            depth_label = {0: "Recent", 1: "Session Arc", 2: "Durable"}.get(node.level, f"Level-{node.level}")
            scaffold_parts.append(f"\n[{depth_label} Summary (Level {node.level}, node {node.id})]\n{node.summary}")
        scaffold_content = "\n".join(scaffold_parts)

        # Build assembled messages in memory
        assembled_history = []
        assembled_history.extend(protected_head)

        # Append Scaffold as a USER message with an ASSISTANT acknowledgement
        scaffold_msg = Message(
            session_id=self.session_id,
            chatbot_id=current_msg.chatbot_id,
            channel_id=current_msg.channel_id,
            sender="system",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content=scaffold_content,
            status=MessageStatus.RESPONDED,
            metadata={"is_scaffold": True},
        )
        ack_msg = Message(
            session_id=self.session_id,
            chatbot_id=current_msg.chatbot_id,
            channel_id=current_msg.channel_id,
            sender="assistant",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content="Understood. I have access to the full conversation history.",
            status=MessageStatus.RESPONDED,
            metadata={"is_scaffold_ack": True},
        )
        assembled_history.append(scaffold_msg)
        assembled_history.append(ack_msg)

        assembled_history.extend(buffer)
        assembled_history.extend(protected_tail)

        return assembled_history, compacted_occurred

    async def _execute_tool_calls(
        self,
        res: LLMResponse,
        current_msg: Message,
        worker: "SessionWorker",
    ) -> bool:
        """Execute the requested tool calls concurrently, posting thought, call, and result messages.

        Returns:
            True if process_turn should continue (loop back to LLM generation),
            False if turn execution was interrupted or transitioned to a new session (should break).
        """
        chatbot_id = current_msg.chatbot_id
        channel_id = current_msg.channel_id

        thought_text = res.thought or res.content
        if thought_text:
            thought_msg = Message(
                session_id=self.session_id,
                chatbot_id=chatbot_id,
                channel_id=channel_id,
                sender="Kesoku",
                role=MessageRole.ASSISTANT,
                type=MessageType.THOUGHT,
                content=thought_text,
                status=MessageStatus.RESPONDED,
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
                role=MessageRole.TOOL,
                type=MessageType.TOOL_CALL,
                content=f"Calling tool `{call.name}` with arguments:\n```json\n{call_args_json}\n```",
                status=MessageStatus.RESPONDED,
                parent_id=current_msg.id,
                metadata={
                    "tool_name": call.name,
                    "tool_arguments": call.arguments,
                    "thought_signature": call.thought_signature,
                    "tool_call_id": call.tool_call_id,
                },
            )
            await self.gateway.post(tool_call_msg)
            tool_call_msgs.append((call, tool_call_msg))

        exec_tasks = [
            self.tool_runner.execute_tool(
                call,
                tc_msg,
                is_interrupted=lambda: not worker.queue_empty(),
            )
            for call, tc_msg in tool_call_msgs
        ]
        if not worker.queue_empty():
            logger.info("Interruption detected before launching concurrent tool execution.")
            for coro in exec_tasks:
                coro.close()
            # Mark the initiating message as interrupted to prevent it from being stuck in processing status
            await self.context.db.update_message_status(current_msg.id, MessageStatus.INTERRUPTED)
            return False

        result_msgs = await asyncio.gather(*exec_tasks)
        for idx, rm in enumerate(result_msgs):
            rm.timestamp = time.time() + (idx * 0.001)
            await self.gateway.post(rm)

        # Check if history was compacted and session was transitioned
        tool_ctx = self.tool_runner.tool_context
        val = getattr(tool_ctx, "transitioned_to_session", None)
        if isinstance(val, str):
            new_session_id = val
            logger.info(
                f"Session '{self.session_id}' transitioned to '{new_session_id}'. "
                "Aborting remaining turn steps and stopping old session worker."
            )
            # Mark the initiating message as processed
            await self.context.db.update_message_status(current_msg.id, MessageStatus.PROCESSED)

            # Schedule worker stop task in the background so the current execution exits cleanly first
            if self.gateway.agent:
                asyncio.create_task(self.gateway.agent.stop_session_worker(self.session_id, immediate=True))
            return False

        return True

    async def _handle_final_response(
        self,
        res: LLMResponse,
        current_msg: Message,
        last_context_tokens: int,
        last_cached_tokens: int,
        turn_tool_calls: int,
        turn_tokens: int,
        start_time: float,
        llm: BaseLLM,
        nudged: bool,
    ) -> tuple[bool, bool]:
        """Handle final assistant response including thought logging, nudge logic, and metrics embedding.

        Returns:
            tuple[should_continue, new_nudged_flag]:
                - should_continue=True: loop should continue (i.e. nudged LLM for retry).
                - should_continue=False: loop should break (i.e. successfully processed turn).
        """
        chatbot_id = current_msg.chatbot_id
        channel_id = current_msg.channel_id

        if res.thought:
            thought_msg = Message(
                session_id=self.session_id,
                chatbot_id=chatbot_id,
                channel_id=channel_id,
                sender="Kesoku",
                role=MessageRole.ASSISTANT,
                type=MessageType.THOUGHT,
                content=res.thought,
                status=MessageStatus.RESPONDED,
                parent_id=current_msg.id,
            )
            await self.gateway.post(thought_msg)

        final_content = res.content
        if not final_content.strip():
            if not nudged:
                logger.info(f"LLM returned empty content in session {self.session_id}. Nudging model.")
                nudge_msg = Message(
                    session_id=self.session_id,
                    chatbot_id=chatbot_id,
                    channel_id=channel_id,
                    sender="System",
                    role=MessageRole.SYSTEM,
                    type=MessageType.TEXT,
                    content=(
                        "Your previous response had empty content. Please provide a final "
                        "user-facing response summarizing your results/actions."
                    ),
                    status=MessageStatus.RESPONDED,
                    parent_id=current_msg.id,
                )
                await self.gateway.post(nudge_msg)
                return True, True  # should_continue=True, nudged=True
            else:
                logger.warning(
                    f"LLM returned empty content again after nudge in session {self.session_id}. Using fallback."
                )
                final_content = "Processed request successfully."

        limit = getattr(llm, "context_window_limit", 1048576)
        context_percent = (last_context_tokens / limit) * 100 if limit else 0.0

        final_msg = Message(
            session_id=self.session_id,
            chatbot_id=chatbot_id,
            channel_id=channel_id,
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content=final_content,
            status=MessageStatus.PENDING,
            parent_id=current_msg.id,
            metadata={
                "turn_metrics": {
                    "session_turns": await self._get_session_turns_count(),
                    "context_tokens": last_context_tokens,
                    "cached_tokens": last_cached_tokens,
                    "context_limit": limit,
                    "context_percent": context_percent,
                    "turn_tool_calls": turn_tool_calls,
                    "turn_tokens": turn_tokens,
                    "turn_time": time.time() - start_time,
                    "status": "finished",
                }
            },
        )
        await self.gateway.post(final_msg)
        await self.context.db.update_message_status(current_msg.id, MessageStatus.PROCESSED)
        return False, nudged  # should_continue=False, nudged unmodified
