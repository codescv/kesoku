"""Orchestrates conversational turn execution, including LLM inference, thought logging, and tool calling."""

import asyncio
import json
import time
from typing import TYPE_CHECKING

from kesoku.agent.history import build_clean_history
from kesoku.agent.llm import BaseLLM
from kesoku.agent.tool_runner import ToolRunner
from kesoku.agent.turn_logger import TurnLogger
from kesoku.context import KesokuContext

if TYPE_CHECKING:
    from kesoku.agent.agent import SessionWorker
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger
from kesoku.utils.text import truncate_context_middle

logger = setup_logger(__name__)

MAX_TOTAL_USER_PREFERENCES_LENGTH = 500
MAX_TOTAL_CROSS_SESSION_CONTEXT_LENGTH = 3000


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
        return await self.gateway.get_session_turns_count(self.session_id)

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

            cfg = self.context.config
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
                    nudged = False
                    turn_tool_calls = 0
                    turn_tokens = 0
                    last_context_tokens = 0
                    last_cached_tokens = 0
                    start_time = time.time()

                # Resolve LLM dynamically for the current message
                llm = self._resolve_llm(current_msg)

                # Retrieve and build the cleaned, prioritized, and aligned session history
                history = await build_clean_history(
                    gateway=self.gateway,
                    session_id=self.session_id,
                )

                latest_user_msg = await self._inject_context_and_trigger_consolidation(
                    history, current_msg, llm
                )

                # Retrieve system prompt directly from session
                session = await self.gateway.get_session(self.session_id)
                system_prompt = session.system_prompt if session else None

                tools_list = self.tool_runner.tool_registry.get_tools_list()

                # Calculate and monitor context window limit
                await self._monitor_context_window(
                    history, latest_user_msg, system_prompt, tools_list, llm, cfg
                )

                # LLM inference
                res = await llm.generate(
                    system_prompt=system_prompt,
                    history=history,
                    tools=tools_list,
                )

                # Log the raw LLM turn using TurnLogger if enabled
                if cfg.agent.raw_llm_logs and self.turn_logger:
                    try:
                        self.turn_logger.log_llm_turn(
                            llm_provider=llm.__class__.__name__,
                            history=history,
                            tools=tools_list,
                            response=res,
                            system_prompt=system_prompt,
                        )
                    except Exception as le:
                        logger.error(f"Failed to log LLM turn: {le}", exc_info=True)

                # Accumulate token metrics
                if res.prompt_tokens:
                    last_context_tokens = res.prompt_tokens
                if res.cached_tokens:
                    last_cached_tokens = res.cached_tokens
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
            history = await self.gateway.get_session_history(self.session_id, limit=20)
            user_msg = None
            for msg in reversed(history):
                if msg.role == MessageRole.USER:
                    user_msg = msg
                    break
            if user_msg:
                user_msg.metadata["turn_metrics"] = turn_metrics
                await self.gateway.update_message_metadata(user_msg.id, user_msg.metadata)
            raise
        except Exception as e:
            logger.error(f"Error in session turn {self.session_id}: {e}", exc_info=True)
            error_msg = Message(
                session_id=self.session_id,
                chatbot_id=chatbot_id,
                channel_id=channel_id,
                sender="Kesoku",
                role=MessageRole.ASSISTANT,
                type=MessageType.TEXT,
                content=f"⚠️ An error occurred while processing your request: {e}",
                status=MessageStatus.PENDING,
                parent_id=current_msg.id,
            )
            await self.gateway.post(error_msg)
            await self.gateway.update_message_status(current_msg.id, MessageStatus.ERROR)

    async def _summarize_cross_session_context_bg(
        self, role: str, current_context: str, since_timestamp: float
    ) -> None:
        """Runs an asynchronous background task to summarize and consolidate memory context.

        Args:
            role: Persona role identifier.
            current_context: The current summarized context content.
            since_timestamp: Only fetch new messages created after this timestamp.
        """
        try:
            logger.info(f"Starting background memory consolidation for role '{role}' since {since_timestamp}")
            # 1. Fetch messages since since_timestamp, capped at 200 to prevent LLM/prompt overrun
            history_msgs = await asyncio.to_thread(
                self.gateway.db.get_role_messages_since,
                role=role,
                since_timestamp=since_timestamp,
                exclude_session_id=None,
                limit=200,
            )

            if not history_msgs:
                logger.info(f"No new messages to consolidate for role '{role}'. Releasing lock.")
                await asyncio.to_thread(
                    self.gateway.db.release_cross_session_context_lock,
                    role,
                    current_context,
                )
                return

            # 2. Build consolidation prompt
            history_log = "\n".join(
                f"[{time.strftime('%m-%d %H:%M', time.localtime(m.timestamp))}] {m.sender}: {m.content}"
                for m in history_msgs
            )
            prompt = (
                "You are an expert memory consolidator for a roleplay companion agent.\n"
                f'Current Consolidated Event Timeline:\n"""\n{current_context or "None"}\n"""\n\n'
                f'New Chat History since last update:\n"""\n{history_log}\n"""\n\n'
                "Task: Combine the current event timeline and the new chat history into a single, "
                "highly concise, chronological timeline/log of events and stories.\n"
                "Rules:\n"
                "- Focus EXCLUSIVELY on concrete stories, events, interesting happenings, "
                "milestones reached, topics discussed, and promises made during the conversation.\n"
                "- STRICTLY PROHIBITED: Do not create any sections, headers, or bullet points for "
                "'User Profile', 'Preferences', 'Rules', 'Settings', or 'Interface Configurations'. "
                "Any such data must be completely discarded and MUST NOT be summarized.\n"
                "- Keep the consolidated timeline highly compact and strictly under 300 words.\n"
                "- As new events are integrated, you MUST aggressively prune and discard older, "
                "resolved, or minor events to maintain the 500-word limit. Only keep the most "
                "significant historical milestones and active ongoing stories.\n"
                "- Drop trivial pleasantries, greeting exchanges, and temporary topics.\n"
                "- Output a direct, highly clean, bullet-pointed markdown timeline of events."
            )

            # 3. Invoke LLM
            llm = self.context.get_llm()
            res = await llm.generate(
                system_prompt="You are an expert background memory consolidator.",
                prompt=prompt,
            )
            new_summary = res.content.strip()

            if not new_summary:
                logger.warning(
                    f"Consolidation returned empty response for role '{role}'. Releasing lock without change."
                )
                await asyncio.to_thread(
                    self.gateway.db.release_cross_session_context_lock,
                    role,
                    current_context,
                )
                return

            # 4. Save new consolidated summary to DB, releasing lock and checkpointing at the last digested message
            checkpoint_ts = history_msgs[-1].timestamp
            await asyncio.to_thread(
                self.gateway.db.release_cross_session_context_lock,
                role,
                new_summary,
                checkpoint_ts,
            )
            logger.info(f"Successfully consolidated and updated CrossSessionContext for role '{role}'.")
        except Exception as e:
            logger.error(f"Error in background memory consolidation for role '{role}': {e}", exc_info=True)
            # Ensure lock is safely released even on crash or failure
            try:
                await asyncio.to_thread(
                    self.gateway.db.release_cross_session_context_lock,
                    role,
                    current_context,
                )
            except Exception as le:
                logger.critical(f"Critical failure: failed to release lock during cleanup for role '{role}': {le}")

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

        active_role = await self.gateway.get_channel_role_with_inheritance(
            current_msg.chatbot_id,
            current_msg.channel_id,
            self.session_id,
        )

        # 1. Calculate if this is a Bootstrap Turn (first turn of session or idle > 30 mins)
        turn_count = await self._get_session_turns_count()
        is_bootstrap = False
        if turn_count <= 1:
            is_bootstrap = True
        else:
            non_current_msgs = [m for m in history if m.id != current_msg.id]
            if non_current_msgs:
                last_msg_time = non_current_msgs[-1].timestamp
                # 1800 seconds = 30 minutes inactivity
                if current_msg.timestamp - last_msg_time > 1800:
                    is_bootstrap = True
            else:
                is_bootstrap = True

        # 2. Query user preferences only if this is a bootstrap turn
        pref_content = ""
        if is_bootstrap:
            user_prefs = await asyncio.to_thread(
                self.gateway.db.get_agent_memories,
                category="user_preferences",
                role=active_role,
            )
            if user_prefs:
                pref_content = "\n".join(
                    f"- {pref['title']}: {pref['content']}" for pref in user_prefs
                )
                if len(pref_content) > MAX_TOTAL_USER_PREFERENCES_LENGTH:
                    pref_content = pref_content[: MAX_TOTAL_USER_PREFERENCES_LENGTH - 3] + "..."

        # 3. Prepend Sync Guidelines and Preferences to the latest user message
        if is_bootstrap and "[Background Context: Sync Guidelines]" not in latest_user_msg.content:
            guidelines_prefix = (
                "[Background Context: Sync Guidelines]\n"
                "======\n"
                "# Passive Synchronization Guidelines:\n"
                f"- 💡 You are playing the active persona role: {active_role}.\n"
                "- 💡 You have access to the `view_cross_session_memory` tool, which retrieves a summarized chronological timeline of recent events, chats, and developments that occurred in other channels/threads.\n"
                "- 💡 If the user's current request below refers to external threads, other chats, or events you cannot locate in this session's history, you MUST call `view_cross_session_memory` to synchronize your context before providing a response.\n"
                "======\n\n"
            )

            pref_prefix = ""
            if pref_content:
                pref_prefix = (
                    "[User Preferences]\n"
                    f"{pref_content}\n\n"
                )

            full_prefix = guidelines_prefix + pref_prefix + "[Current Request]\n"

            msg_idx = history.index(latest_user_msg)
            copied_msg = latest_user_msg.model_copy()
            copied_msg.content = full_prefix + copied_msg.content
            history[msg_idx] = copied_msg
            latest_user_msg = copied_msg
            logger.info(
                f"Prepended bootstrap context blocks (guidelines & preferences) into user message {copied_msg.id}"
            )

        # 4. Background Consolidation Trigger (Unchanged)
        # Retrieve Cross-Session Memory context parameters solely to check and trigger consolidation asynchronously
        stored_ctx = await asyncio.to_thread(
            self.gateway.db.get_cross_session_context,
            active_role,
        )
        stored_content = stored_ctx.content if stored_ctx else ""
        last_updated = stored_ctx.updated_at if stored_ctx else 0.0
        lock_status = stored_ctx.status if stored_ctx else "idle"

        new_messages = await asyncio.to_thread(
            self.gateway.db.get_role_messages_since,
            role=active_role,
            since_timestamp=last_updated,
            exclude_session_id=self.session_id,
        )

        try:
            new_msg_tokens = await asyncio.to_thread(
                llm.count_tokens,
                prompt=None,
                system_prompt=None,
                history=new_messages,
            )
        except Exception as te:
            logger.warning(f"Failed to count new message tokens: {te}")
            new_msg_tokens = llm.estimate_tokens_fallback(history=new_messages)

        now_ts = time.time()
        has_timeout = (now_ts - last_updated > 1800) and len(new_messages) > 0
        has_token_overrun = new_msg_tokens > 4000

        should_consolidate = has_timeout or has_token_overrun

        if should_consolidate and lock_status == "idle":
            locked = await asyncio.to_thread(
                self.gateway.db.claim_cross_session_context_for_update,
                active_role,
            )
            if locked:
                logger.info(f"Claimed lock for cross-session context update on role '{active_role}'")
                asyncio.create_task(
                    self._summarize_cross_session_context_bg(active_role, stored_content, last_updated)
                )

        return latest_user_msg


    async def _monitor_context_window(
        self,
        history: list[Message],
        latest_user_msg: Message | None,
        system_prompt: str | None,
        tools_list: list,
        llm: BaseLLM,
        cfg,
    ) -> int:
        """Estimate context window usage and prepend context monitoring warning to latest_user_msg in place if needed.

        Returns:
            The calculated or estimated context token count.
        """
        try:
            context_tokens = await asyncio.to_thread(
                llm.count_tokens,
                prompt=None,
                system_prompt=system_prompt,
                history=history,
                tools=tools_list,
            )
        except Exception as te:
            logger.warning(f"Failed to count context tokens: {te}")
            context_tokens = llm.estimate_tokens_fallback(None, system_prompt, history)

        limit = getattr(llm, "context_window_limit", 1048576)
        percentage = (context_tokens / limit) * 100

        threshold = cfg.agent.compact_history_warning_threshold
        if latest_user_msg and percentage >= threshold:
            monitor_suffix = (
                f"\n\n[Context Monitor: Currently using {context_tokens:,} tokens, "
                f"which is {percentage:.1f}% of your {limit:,} window limit. "
                "It is highly recommended that you call the 'compact_history' "
                "tool now to reset the context window.]"
            )
            if "[Context Monitor:" not in latest_user_msg.content:
                # Clone to avoid mutating shared database/cached states in place
                msg_idx = history.index(latest_user_msg)
                copied_msg = latest_user_msg.model_copy()
                copied_msg.content += monitor_suffix
                history[msg_idx] = copied_msg
                logger.info(
                    f"Injected context monitor warning into user message {copied_msg.id}: "
                    f"{context_tokens} tokens ({percentage:.1f}%)"
                )

        return context_tokens

    async def _execute_tool_calls(
        self,
        res,
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
            return False

        result_msgs = await asyncio.gather(*exec_tasks)
        for rm in result_msgs:
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
            await self.gateway.mark_message_processed(current_msg.id)

            # Schedule worker stop task in the background so the current execution exits cleanly first
            if self.gateway.agent:
                asyncio.create_task(self.gateway.agent.stop_session_worker(self.session_id, immediate=True))
            return False

        return True

    async def _handle_final_response(
        self,
        res,
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
                    f"LLM returned empty content again after nudge in session "
                    f"{self.session_id}. Using fallback."
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
        await self.gateway.mark_message_processed(current_msg.id)
        return False, nudged  # should_continue=False, nudged unmodified


