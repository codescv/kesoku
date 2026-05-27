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

logger = setup_logger(__name__)


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
                current_msg = await worker.drain_queue_and_pivot(current_msg)

                # Resolve LLM dynamically for the current message
                llm = self._resolve_llm(current_msg)

                # Retrieve and build the cleaned, prioritized, and aligned session history
                history = await build_clean_history(
                    gateway=self.gateway,
                    session_id=self.session_id,
                )

                 # Retrieve system prompt directly from session
                session = await self.gateway.get_session(self.session_id)
                system_prompt = session.system_prompt if session else None

                tools_list = self.tool_runner.tool_registry.get_tools_list()

                # Calculate/estimate context tokens asynchronously to avoid blocking the event loop
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

                # Inject context monitor warning into the latest user message in history
                latest_user_msg = None
                for msg in reversed(history):
                    if msg.role == MessageRole.USER:
                        latest_user_msg = msg
                        break

                if latest_user_msg:
                    monitor_suffix = (
                        f"\n\n[Context Monitor: Currently using {context_tokens:,} tokens, "
                        f"which is {percentage:.1f}% of your {limit:,} window limit. "
                        f"Please call 'compact_history' tool if you are close to the limit "
                        f"to reset the context window.]"
                    )
                    if "[Context Monitor:" not in latest_user_msg.content:
                        # Clone the user message to avoid mutating shared database/cached states in place
                        msg_idx = history.index(latest_user_msg)
                        copied_msg = latest_user_msg.model_copy()
                        copied_msg.content += monitor_suffix
                        history[msg_idx] = copied_msg
                        logger.info(
                            f"Injected context monitor warning into user message {copied_msg.id}: "
                            f"{context_tokens} tokens ({percentage:.1f}%)"
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
                        break
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
                            asyncio.create_task(
                                self.gateway.agent.stop_session_worker(self.session_id, immediate=True)
                            )
                        break

                    continue
                else:
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
                    if not final_content:
                        if not nudged:
                            logger.info(
                                f"LLM returned empty content in session {self.session_id}. "
                                f"Nudging model."
                            )
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
                            nudged = True
                            continue
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
