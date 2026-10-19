"""Sub-agent runner for bounded, read-only exploration and investigation."""

import asyncio
import json
import logging
import os
from typing import Any

import yaml
from pydantic import BaseModel

import kesoku.agent.llm as kesoku_llm
from kesoku.agent.tool_runner import ToolRunner
from kesoku.agent.tools.registry import ToolContext, ToolRegistry, default_registry
from kesoku.agent.turn_logger import TurnLogger
from kesoku.config import get_config
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.utils.path import PathResolver
from kesoku.utils.text import truncate_middle

logger = logging.getLogger(__name__)


class SubagentResult(BaseModel):
    """Result of a sub-agent run."""

    success: bool
    report: str
    error: str | None = None
    steps: int = 0
    tool_calls: int = 0
    tokens: int = 0
    transcript_name: str = ""


def _resolve_subagent_llm(context: ToolContext, cfg: Any) -> kesoku_llm.BaseLLM:
    """Resolve the LLM instance for the sub-agent, preserving KesokuContext/MeteredLLM when present."""
    model_override = cfg.agent.subagent_model_name or None
    kesoku_ctx = getattr(context, "kesoku_context", None) or getattr(
        getattr(context, "gateway", None), "context", None
    )
    if kesoku_ctx is not None and hasattr(kesoku_ctx, "get_llm"):
        return kesoku_ctx.get_llm(model_name=model_override)
    return kesoku_llm.get_llm(model_name=model_override)


class SubagentRunner:
    """Runner for sub-agent investigations."""

    @staticmethod
    async def run(
        task: str,
        tools: list[str],
        max_steps: int,
        context: ToolContext,
        context_hint: str = "",
    ) -> str:
        """Run a sub-agent investigation and return its formatted report string."""
        cfg = getattr(getattr(context, "kesoku_context", None), "config", None) or get_config()
        context.subagents_launched_this_turn += 1
        subagent_id = os.urandom(4).hex()

        awd = PathResolver.resolve(cfg.agent_working_dir)
        if context.session_workspace:
            staging_dir = PathResolver.get_session_staging_dir(context.session_workspace)
        else:
            staging_dir = PathResolver.resolve(cfg.workspace.sessions_dir)
            os.makedirs(staging_dir, exist_ok=True)

        sys_prompt = (
            "# Kesoku Sub-agent\n\n"
            f"# Agent Working Directory\n> AWD='{awd}'\n"
            "You are working in the agent working directory (AWD).\n"
            "This is where you find the files you need by default.\n\n"
            f"# Session Staging Directory\n> STAGING_DIR='{staging_dir}'\n"
            "Files generated dynamically MUST be written to STAGING_DIR.\n\n"
            "# Rules\n"
            "- You are a read-only investigation sub-agent. You cannot ask the user questions and cannot delegate.\n"
            f"- You have a strict limit of {max_steps} steps. Finish your investigation before hitting the limit.\n"
            "- When you are done, your final answer MUST be a concise report (<= 1500 chars) with facts, "
            "file paths + line numbers, and exact error strings.\n"
            "- Include an explicit 'not found / uncertain' section if you could not answer the task completely.\n"
            "- Do not write files or modify anything.\n"
        )

        user_prompt = f"{task}\n\nContext hint:\n{context_hint}" if context_hint.strip() else task
        messages: list[Message] = [
            Message(
                session_id=context.session_id,
                chatbot_id="subagent",
                channel_id="subagent",
                sender="User",
                role=MessageRole.USER,
                type=MessageType.TEXT,
                content=user_prompt,
                status=MessageStatus.PROCESSED,
                parent_id=None,
            )
        ]

        sub_registry = ToolRegistry()
        allowed_callables = []
        for name in tools:
            try:
                func = default_registry.get_tool(name)
                sub_registry.register(func)
                allowed_callables.append(func)
            except KeyError:
                pass
        schema_tools = sub_registry.get_tools_list()

        sub_context = ToolContext(
            session_id=context.session_id,
            session_workspace=context.session_workspace,
            original_msg_id=context.original_msg_id,
            active_jobs=context.active_jobs,
            gateway=context.gateway,
            is_subagent=True,
            foreground_only=True,
            interrupt_check=context.interrupt_check,
            subagents_launched_this_turn=context.subagents_launched_this_turn,
            kesoku_context=getattr(context, "kesoku_context", None),
        )
        runner = ToolRunner(sub_registry, sub_context)

        llm = _resolve_subagent_llm(context, cfg)
        turn_logger = TurnLogger(context.session_id, staging_dir) if cfg.agent.raw_llm_logs else None

        transcript_name = f"subagent_{subagent_id}.log.yaml"
        transcript_path = os.path.join(staging_dir, transcript_name)
        transcript_steps: list[dict[str, Any]] = []

        total_tokens = 0
        steps_taken = 0
        tool_calls_count = 0
        report = ""
        success = False
        error_msg = ""

        for step in range(max_steps):
            if context.interrupt_check and context.interrupt_check():
                error_msg = "Sub-agent aborted: interrupted"
                break

            active_tools = schema_tools if (step < max_steps - 1 and schema_tools) else None

            try:
                res = await llm.generate(
                    system_prompt=sys_prompt,
                    history=messages,
                    tools=active_tools,
                )
                steps_taken += 1
                step_tokens = res.total_tokens or ((res.prompt_tokens or 0) + (res.candidates_tokens or 0))
                total_tokens += step_tokens
                context.subagent_llm_calls = getattr(context, "subagent_llm_calls", 0) + 1
                context.subagent_tokens = getattr(context, "subagent_tokens", 0) + step_tokens

                if turn_logger:
                    try:
                        turn_logger.log_llm_turn(
                            llm_provider=f"{llm.__class__.__name__}/subagent",
                            history=list(messages),
                            tools=active_tools or [],
                            response=res,
                            system_prompt=sys_prompt,
                        )
                    except Exception as le:
                        logger.warning(f"Failed to log sub-agent LLM turn: {le}")

                call_names = [c.name for c in (res.tool_calls or [])]
                transcript_steps.append(
                    {
                        "step": step + 1,
                        "tool_calls": call_names,
                        "content_chars": len(res.content or ""),
                        "tokens": step_tokens,
                    }
                )

                if res.tool_calls and active_tools is not None:
                    thought_text = res.thought or res.content
                    if thought_text:
                        thought_meta: dict[str, Any] = {}
                        if res.thought_signature:
                            thought_meta["thought_signature"] = res.thought_signature
                        messages.append(
                            Message(
                                session_id=context.session_id,
                                chatbot_id="subagent",
                                channel_id="subagent",
                                sender="Kesoku",
                                role=MessageRole.ASSISTANT,
                                type=MessageType.THOUGHT,
                                content=thought_text,
                                status=MessageStatus.RESPONDED,
                                metadata=thought_meta,
                            )
                        )

                    tool_call_pairs = []
                    for call in res.tool_calls:
                        tool_calls_count += 1
                        call_args_json = json.dumps(call.arguments, indent=2, ensure_ascii=False)
                        tc_msg = Message(
                            session_id=context.session_id,
                            chatbot_id="subagent",
                            channel_id="subagent",
                            sender="Kesoku",
                            role=MessageRole.TOOL,
                            type=MessageType.TOOL_CALL,
                            content=f"Calling tool `{call.name}` with arguments:\n```json\n{call_args_json}\n```",
                            status=MessageStatus.RESPONDED,
                            metadata={
                                "tool_name": call.name,
                                "tool_arguments": call.arguments,
                                "thought_signature": call.thought_signature,
                                "tool_call_id": call.tool_call_id,
                            },
                        )
                        messages.append(tc_msg)
                        tool_call_pairs.append((call, tc_msg))

                    if context.interrupt_check and context.interrupt_check():
                        error_msg = "Sub-agent aborted: interrupted"
                        break

                    exec_tasks = [
                        runner.execute_tool(call, tc_msg, is_interrupted=context.interrupt_check)
                        for call, tc_msg in tool_call_pairs
                    ]
                    results = await asyncio.gather(*exec_tasks)
                    for rm in results:
                        messages.append(rm)
                else:
                    if res.content:
                        report = res.content
                        success = True
                        break
                    if res.thought and not report:
                        report = res.thought
            except Exception as e:
                error_msg = f"LLM error: {e}"
                logger.exception("Sub-agent error")
                break

        if not success and not error_msg:
            error_msg = "Reached max steps without returning a final report."

        try:
            transcript_data = {
                "subagent_id": subagent_id,
                "task": task,
                "tools": tools,
                "steps_taken": steps_taken,
                "tool_calls": tool_calls_count,
                "tokens": total_tokens,
                "success": success,
                "error": error_msg or None,
                "steps": transcript_steps,
            }
            with open(transcript_path, "w", encoding="utf-8") as f:  # noqa: ASYNC230
                yaml.safe_dump(transcript_data, f, allow_unicode=True, sort_keys=False)
        except Exception as e:
            logger.warning(f"Failed to write sub-agent transcript {transcript_path}: {e}")

        report = truncate_middle(report, 1500) if report else ""

        if error_msg == "Sub-agent aborted: interrupted":
            return f"Sub-agent aborted: interrupted\n{report}".strip()
        if success:
            return (
                f"[sub-agent {subagent_id}: {steps_taken} steps, {tool_calls_count} tool calls, "
                f"{total_tokens} tokens; transcript: {transcript_name}]\n{report}"
            )
        partial = f"\nPartial findings:\n{report}" if report else ""
        return f"Sub-agent {subagent_id} failed: {error_msg}{partial}"
