import json
import logging
import os

from pydantic import BaseModel

from kesoku.agent.llm import get_llm
from kesoku.agent.tool_runner import ToolRunner
from kesoku.agent.tools.registry import ToolContext, default_registry
from kesoku.agent.turn_logger import TurnLogger
from kesoku.config import get_config
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message

logger = logging.getLogger(__name__)

class SubagentResult(BaseModel):
    """Result of a sub-agent run."""
    success: bool
    report: str
    error: str | None = None

class SubagentRunner:
    """Runner for sub-agent investigations."""
    @staticmethod
    async def run(
        task: str,
        tools: list[str],
        max_steps: int,
        context: ToolContext
    ) -> str:
        """Run a sub-agent investigation."""
        cfg = get_config()
        # Increment counter
        context.subagents_launched_this_turn += 1
        subagent_id = os.urandom(4).hex()

        # Build system prompt
        sys_prompt = "# Kesoku Sub-agent\n\n"

        awd = cfg.agent_working_dir
        sys_prompt += f"# Agent Working Directory\n> AWD='{awd}'\nYou are working in the agent working directory (AWD).\nThis is where you find the files you need by default.\n\n"  # noqa: E501

        if context.session_workspace:
            staging_dir = os.path.abspath(os.path.join(cfg.workspace.sessions_dir, context.session_workspace))  # noqa: ASYNC240
            sys_prompt += f"# Session Staging Directory\n> STAGING_DIR='{staging_dir}'\nFiles generated dynamically MUST be written to STAGING_DIR.\n\n"  # noqa: E501
        else:
            staging_dir = cfg.workspace.sessions_dir

        sys_prompt += (
            "# Rules\n"
            "- You are a read-only investigation sub-agent. You cannot ask the user questions and cannot delegate.\n"
            f"- You have a strict limit of {max_steps} steps. Finish your investigation before hitting the limit.\n"
            "- When you are done, your final answer MUST be a concise report (<= 1500 chars) with facts, file paths + line numbers, exact error strings.\n"  # noqa: E501
            "- Include an explicit 'not found / uncertain' section if you could not answer the task completely.\n"
            "- Do not write files or modify anything.\n"
        )

        # Build messages history
        messages: list[Message] = []
        messages.append(
            Message(
                session_id=context.session_id,
                chatbot_id="subagent",
                channel_id="subagent",
                sender="User",
                role=MessageRole.USER,
                type=MessageType.TEXT,
                content=task,
                status=MessageStatus.PROCESSED,
                parent_id=None
            )
        )

        # Build tools
        allowed_callables = []
        for name in tools:
            try:
                func = default_registry.get_tool(name)
                allowed_callables.append(func)
            except KeyError:
                pass

        # Setup ToolRunner
        sub_context = ToolContext(
            session_id=context.session_id,
            session_workspace=context.session_workspace,
            original_msg_id=context.original_msg_id,
            active_jobs=context.active_jobs,
            gateway=context.gateway,
            is_subagent=True,
            foreground_only=True,
            interrupt_check=context.interrupt_check,
            subagents_launched_this_turn=context.subagents_launched_this_turn
        )
        runner = ToolRunner(default_registry, sub_context)

        llm = get_llm(model_name=cfg.agent.subagent_model_name or None)
        turn_logger = None
        if cfg.agent.raw_llm_logs:
            turn_logger = TurnLogger(context.session_id, staging_dir)

        transcript_path = os.path.join(staging_dir, f"subagent_{subagent_id}.log.yaml")
        transcript = [f"task: {task}\ntools: {tools}\nsteps:"]

        total_tokens = 0
        report = ""
        success = False
        error_msg = ""

        for step in range(max_steps):
            if context.interrupt_check and context.interrupt_check():
                error_msg = "Sub-agent aborted: interrupted"
                break

            active_tools = allowed_callables if step < max_steps - 1 else None

            try:
                res = await llm.generate(
                    system_prompt=sys_prompt,
                    history=messages,
                    tools=active_tools
                )

                if turn_logger:
                    await turn_logger.log_llm_turn(
                        llm_provider=f"{llm.__class__.__name__}/subagent",
                        sys_prompt=sys_prompt,
                        model_name=cfg.agent.subagent_model_name or cfg.gemini.model_name,
                        history=messages,
                        tools=active_tools,
                        response=res,
                        tokens_used=0,
                        elapsed_seconds=0
                    )

                # Note: `res` structure is `LLMResponse`. usage_metadata is not always there, but we can do our best.
                # Just keeping total_tokens 0 for now as it's optional.

                call_names = [c.name for c in res.tool_calls]
                transcript.append(f"- step: {step}\n  calls: {call_names}\n  content_len: {len(res.content)}")

                if res.thought:
                    messages.append(Message(
                        session_id=context.session_id,
                        chatbot_id="subagent",
                        channel_id="subagent",
                        sender="Kesoku",
                        role=MessageRole.ASSISTANT,
                        type=MessageType.THOUGHT,
                        content=res.thought,
                        status=MessageStatus.PROCESSED,
                    ))

                if res.tool_calls:
                    for call in res.tool_calls:
                        messages.append(Message(
                            session_id=context.session_id,
                            chatbot_id="subagent",
                            channel_id="subagent",
                            sender="Kesoku",
                            role=MessageRole.TOOL,
                            type=MessageType.TOOL_CALL,
                            content=f"Calling {call.name}({json.dumps(call.arguments)})",
                            status=MessageStatus.PROCESSED,
                            metadata={"tool_name": call.name, "tool_call_id": call.tool_call_id}
                        ))

                    exec_tasks = [runner.execute_tool(call, messages[-1], is_interrupted=context.interrupt_check) for call in res.tool_calls]  # noqa: E501
                    import asyncio
                    results = await asyncio.gather(*exec_tasks)

                    for rm in results:
                        messages.append(rm)

                else:
                    if res.content:
                        report = res.content
                        success = True
                        break

            except Exception as e:
                error_msg = f"LLM error: {e}"
                logger.exception("Subagent error")
                break

        if not success and not error_msg:
            error_msg = "Reached max steps without returning a final report."

        try:
            with open(transcript_path, "w") as f:  # noqa: ASYNC230
                f.write("\n".join(transcript))
        except Exception:
            pass

        from kesoku.utils.text import truncate_middle
        report = truncate_middle(report, 1500)

        if success:
            return f"[sub-agent {subagent_id}: {step+1} steps, {len([m for m in messages if m.type == MessageType.TOOL_CALL])} tool calls, {total_tokens} tokens; transcript: subagent_{subagent_id}.log.yaml]\n{report}"  # noqa: E501
        else:
            return f"Sub-agent {subagent_id} failed: {error_msg}\nPartial findings:\n{report}"
