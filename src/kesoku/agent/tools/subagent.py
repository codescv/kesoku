from kesoku.agent.tools.registry import ToolContext, default_registry
from kesoku.config import get_config


@default_registry.register
async def delegate_subagent(
    task: str,
    tools: list[str] | None = None,
    max_steps: int = 10,
    context: ToolContext | None = None
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

    cfg = get_config()

    if context.subagents_launched_this_turn >= cfg.agent.subagent_max_per_turn:
        return f"Error: max sub-agents per turn ({cfg.agent.subagent_max_per_turn}) reached."

    allowed = set(cfg.agent.subagent_allowed_tools)
    if cfg.agent.subagent_allow_write:
        allowed.add("update_file")

    requested_tools = []
    if tools:
        for t in tools:
            if t in ("delegate_subagent", "chat_search", "view_message"):
                continue
            if t in allowed:
                requested_tools.append(t)
            else:
                return f"Error: tool '{t}' is not permitted for sub-agents."

    max_steps = max(1, min(max_steps, cfg.agent.subagent_max_steps))

    # Check concurrency (we can track it on context.gateway or something, but the simplest way is a global lock)
    # Actually, the simplest is to just check if there's a running subagent on the TurnExecutor.
    # We can use asyncio.Lock() stored on the TurnExecutor?
    # Wait, tool calls in the same batch are concurrent, so we can use a class-level or context-level lock.

    from kesoku.agent.subagent import SubagentRunner

    # We pass the lock check to the runner, or use a lock here.
    return await SubagentRunner.run(
        task=task,
        tools=requested_tools,
        max_steps=max_steps,
        context=context
    )
