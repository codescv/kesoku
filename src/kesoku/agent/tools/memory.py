"""Skills and role Markdown memory search tools for Kesoku AI Agent."""

import asyncio
import datetime
import logging

from kesoku.agent.memory_store import search_role_memories
from kesoku.agent.skills import SkillManager
from kesoku.agent.tools.registry import ToolContext, default_registry

logger = logging.getLogger(__name__)

skill_manager = SkillManager()


@default_registry.register
def list_skills(context: ToolContext | None = None) -> str:
    """List all valid skills supported on the current host OS, including the active role's own skills.

    Args:
        context: Optional tool execution context.

    Returns:
        Formatted summary of available skills.
    """
    skills = skill_manager.list_skills(role=_resolve_session_role_sync(context))
    if not skills:
        return "No skills available or supported on this platform."
    lines = ["=== Available Skills ==="]
    for s in skills:
        lines.append(f"- {s['name']} (v{s['version']}): {s['description']}")
    return "\n".join(lines)


@default_registry.register
def use_skill(skill_name: str, context: ToolContext | None = None) -> str:
    """Retrieve the complete instructions and absolute directory path for a specific skill.

    Args:
        skill_name: Name of the skill.
        context: Optional tool execution context.

    Returns:
        Complete markdown instructions and absolute path header for the skill.
    """
    try:
        _, content = skill_manager.get_skill(skill_name, role=_resolve_session_role_sync(context))
        return content
    except Exception as e:
        return f"Failed to load skill '{skill_name}': {e}"


def _resolve_session_role_sync(context: ToolContext | None) -> str | None:
    """Resolve the session's persona role synchronously (for sync tools); None when unknown.

    Used to expose role-scoped skills (`roles/<role>/skills/`) only to sessions of that role.
    """
    if not context or not context.gateway or not context.session_id:
        return None
    try:
        sync_db = getattr(context.gateway.db, "sync_db", None)
        if sync_db is None:
            return None
        sess = sync_db.get_session(context.session_id)
        return sess.role_name if sess and sess.role_name else None
    except Exception as e:
        logger.warning(f"Failed to resolve session role for skills: {e}")
        return None


async def _resolve_role(role_param: str | None, context: ToolContext | None) -> str:
    """Resolve the correct role scope based on the context rules."""
    if context and context.gateway:
        db = context.gateway.db
        if context.session_id:
            try:
                sess = await db.get_session(context.session_id)
                if sess and sess.role_name and sess.role_name != "default":
                    return sess.role_name
            except Exception as e:
                logger.warning(f"Failed to resolve session role: {e}")

        if context.original_msg_id:
            try:
                msg_list = await db.get_messages_by_filters({"id": context.original_msg_id})
                if msg_list:
                    msg = msg_list[0]
                    return await db.get_channel_role_with_inheritance(
                        msg.chatbot_id, msg.channel_id, context.session_id
                    )
            except Exception as e:
                logger.warning(f"Failed to resolve active role: {e}")
        if context.session_id:
            try:
                mapping = await db.get_channel_by_session(context.session_id)
                if mapping:
                    chatbot_id, channel_id = mapping
                    return await db.get_channel_role_with_inheritance(chatbot_id, channel_id, context.session_id)
            except Exception as e:
                logger.warning(f"Failed to resolve active role from session_id: {e}")

    return role_param if role_param else "default"


@default_registry.register
async def memory_search(
    query: str,
    limit: int = 10,
    context: ToolContext | None = None,
) -> str:
    """Search the active role's Markdown memories (`$MEMORY_DIR`) by natural-language query or keywords.

    Returns matching memory file paths (`$MEMORY_DIR/...`), update timestamps, relevance scores, and
    1-line descriptions. Use `read_file` on a returned path to inspect its full contents. Pass `query="*"`
    to list all memories for the active role.

    Args:
        query: Natural-language search query or keywords (or `"*"` to list all memories).
        limit: Maximum number of matching memory files to return (default: 10).
        context: Injected tool execution context.

    Returns:
        Formatted list of matching memory files with their paths and descriptions.
    """
    if not context or not context.gateway:
        return "Error: ToolContext is missing."

    active_role = await _resolve_role(role_param=None, context=context)
    db = context.gateway.db
    sync_db = getattr(db, "sync_db", db)

    try:
        threshold = 0.45
        if context.gateway.context and context.gateway.context.config:
            threshold = min(0.45, context.gateway.context.config.agent.search_threshold)

        matches = await asyncio.to_thread(
            search_role_memories,
            sync_db,
            active_role,
            query,
            limit=limit,
            min_score=threshold,
        )

        if not matches:
            return f"No memory files matching '{query}' found for role '{active_role}'."

        lines = [
            f"🧠 **Memory Search Results for '{query}' (Role: {active_role})**",
            "Use `read_file` on a `$MEMORY_DIR/...` path below to read its full contents:",
        ]
        for m in matches:
            updated_str = datetime.datetime.fromtimestamp(m.mtime).astimezone().strftime("%Y-%m-%d %H:%M")
            lines.append(
                f'- `{m.memory_path}` (updated: {updated_str}, relevance: {m.score:.2f})\n'
                f"  description: {m.description}"
            )
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Failed to execute memory_search: {e}", exc_info=True)
        return f"Error executing memory_search: {e}"
