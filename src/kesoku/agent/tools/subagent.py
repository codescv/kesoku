"""Tool wrapper for delegating read-only investigations to a sub-agent."""

from kesoku.agent.tools.registry import ToolContext, default_registry
from kesoku.config import get_config

_NEVER_ALLOWED = frozenset({"delegate_subagent", "chat_search", "view_message"})


@default_registry.register
async def delegate_subagent(
    task: str,
    context_hint: str = "",
    tools: list[str] | None = None,
    max_turns: int = 8,
    max_steps: int | None = None,
    context: ToolContext | None = None,
) -> str:
    """Run a focused sub-agent for an investigation (large log/codebase searches, API look-ups) and return only.

    Returns its concise report (<= 1500 chars) to this conversation. It is intended for investigation and must not
    modify files (its shell is not sandboxed, so instruct it accordingly); it cannot talk to the user or delegate
    further, and its commands run in the foreground only. Use it when the exploration would otherwise flood this
    conversation with tool output. One sub-agent runs at a time; at most 3 per turn.
    """
    if not context:
        return "Error: missing context."

    if context.is_subagent:
        return "Error: nested delegation is not allowed."

    cfg = getattr(getattr(context, "kesoku_context", None), "config", None) or get_config()

    if getattr(context, "running_subagents", 0) >= cfg.agent.subagent_max_concurrent:
        return "Error: one sub-agent at a time; wait for the running one."

    if context.subagents_launched_this_turn >= cfg.agent.subagent_max_per_turn:
        return (
            f"Error: max {cfg.agent.subagent_max_per_turn} sub-agents per turn reached; "
            "continue directly in this conversation."
        )

    allowed = set(cfg.agent.subagent_allowed_tools) - _NEVER_ALLOWED
    if cfg.agent.subagent_allow_write:
        allowed.add("update_file")

    if tools is None:
        requested_tools = [t for t in cfg.agent.subagent_allowed_tools if t in allowed]
    else:
        requested_tools = []
        for t in tools:
            if t in _NEVER_ALLOWED or t not in allowed:
                return (
                    f"Error: tool '{t}' is not permitted for sub-agents. "
                    f"Allowed tools: {', '.join(sorted(allowed))}"
                )
            if t not in requested_tools:
                requested_tools.append(t)

    effective_steps = max_steps if max_steps is not None else max_turns
    effective_steps = max(1, min(effective_steps, cfg.agent.subagent_max_steps))

    from kesoku.agent.subagent import SubagentRunner

    context.running_subagents = getattr(context, "running_subagents", 0) + 1
    try:
        return await SubagentRunner.run(
            task=task,
            tools=requested_tools,
            max_steps=effective_steps,
            context=context,
            context_hint=context_hint,
        )
    finally:
        context.running_subagents = max(0, getattr(context, "running_subagents", 1) - 1)
