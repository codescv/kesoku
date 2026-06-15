"""Memory management, role persona switching, and skill listing tools for Kesoku AI Agent."""

import logging
import re
import time
from typing import Literal

from kesoku.agent.skills import SkillManager
from kesoku.agent.tools.registry import ToolContext, default_registry

MemoryCategory = Literal["progress", "user_preferences", "memo"]

logger = logging.getLogger(__name__)

MAX_MEMORY_CONTENT_LENGTH = 500

skill_manager = SkillManager()


@default_registry.register
def list_skills(context: ToolContext | None = None) -> str:
    """List all valid skills in skills_dir supported on the current host operating system.

    Args:
        context: Optional tool execution context.

    Returns:
        Formatted summary of available skills.
    """
    skills = skill_manager.list_skills()
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
        _, content = skill_manager.get_skill(skill_name)
        return content
    except Exception as e:
        return f"Failed to load skill '{skill_name}': {e}"


def sanitize_key(input_key: str) -> str:
    """Sanitizes key by lowercasing, stripping, and replacing invalid characters with underscores."""
    clean_key = re.sub(r"[^a-z0-9_]", "_", input_key.lower().strip())
    clean_key = re.sub(r"_+", "_", clean_key).strip("_")
    return clean_key


def validate_key(key: str) -> bool:
    """Verifies if the key strictly contains only lowercase letters, underscores, and numbers."""
    return bool(re.match(r"^[a-z0-9_]+$", key))


async def _resolve_memory_role(category: str, role_param: str | None, context: ToolContext | None) -> str:
    """Resolve the correct role scope based on the memory category and context rules."""
    category = category.strip().lower()

    # Rule 1: Standard categories ALWAYS use "default" role
    if category == "progress":
        return "default"

    # Rule 2: user_preferences and memo categories use current channel's active role
    if category in {"user_preferences", "memo"}:
        if context and context.gateway:
            db = context.gateway.db
            if context.original_msg_id:
                try:
                    msg_list = await db.get_messages_by_filters({"id": context.original_msg_id})
                    if msg_list:
                        msg = msg_list[0]
                        return await db.get_channel_role_with_inheritance(
                            msg.chatbot_id, msg.channel_id, context.session_id
                        )
                except Exception as e:
                    logger.warning(f"Failed to resolve active role for memory category {category}: {e}")
            if context.session_id:
                try:
                    mapping = await db.get_channel_by_session(context.session_id)
                    if mapping:
                        chatbot_id, channel_id = mapping
                        return await db.get_channel_role_with_inheritance(
                            chatbot_id, channel_id, context.session_id
                        )
                except Exception as e:
                    logger.warning(f"Failed to resolve active role from session_id for memory category {category}: {e}")
        # Fallback
        return role_param if role_param else "default"

    # For other categories, return the passed role parameter
    return role_param if role_param else "default"


@default_registry.register
async def list_memories(
    category: MemoryCategory,
    role: str | None = None,
    context: ToolContext | None = None,
) -> str:
    """List all active memory keys and titles under the specified category.

    Args:
        category: The memory category (e.g., 'progress', 'learnings', 'user_preferences').
        role: Optional roleplay persona scope (defaults to None for implicit channel/default persona).
        context: Injected tool execution context.

    Returns:
        A clean list of active keys, titles, and their last updated timestamps.
    """
    if not context:
        return "Error: ToolContext is missing."

    resolved_role = await _resolve_memory_role(category, role, context)
    db = context.gateway.db
    try:
        memories = await db.get_agent_memories(category=category, role=resolved_role)
        if not memories:
            return f"No memories found in category '{category}' for role scope '{resolved_role}'."

        lines = [f"=== Memories in '{category}' (scope: {resolved_role}) ==="]
        for m in memories:
            updated_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m["updated_at"]))
            lines.append(f'- key: `{m["key"]}` | title: "{m["title"]}" | updated: {updated_str} | scope: {m["role"]}')
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Failed to list memories: {e}", exc_info=True)
        return f"Error listing memories: {e}"


@default_registry.register
async def view_memory(
    category: MemoryCategory,
    key: str | None = None,
    role: str | None = None,
    context: ToolContext | None = None,
) -> str:
    """Retrieve detailed content for a specific memory key, or dynamically render all memories in a category.

    If `key` is provided, returns the content of that specific record.
    If `key` is None (or omitted), dynamically aggregates all entries inside that category under the implicit
    correct role/default scope and formats them into a beautiful, readable Markdown block.

    Args:
        category: The memory category (e.g., 'progress', 'learnings', 'user_preferences').
        key: Optional unique snake_case key. If omitted, renders all entries.
        role: Optional roleplay persona scope (defaults to None for implicit channel/default persona).
        context: Injected tool execution context.

    Returns:
        The content of the specific memory entry, or a compiled Markdown text block of all entries.
    """
    if not context:
        return "Error: ToolContext is missing."

    resolved_role = await _resolve_memory_role(category, role, context)
    db = context.gateway.db
    try:
        if key:
            if not validate_key(key):
                return (
                    f"Error: Invalid Key '{key}'.\n"
                    "Memory keys must strictly contain only lowercase letters, "
                    "underscores, and numbers (regex: ^[a-z0-9_]+$)."
                )
            sanitized = sanitize_key(key)
            mem = await db.get_agent_memory(category=category, key=sanitized, role=resolved_role)
            if not mem:
                return f"No memory found for category='{category}', key='{sanitized}', role='{resolved_role}'."
            return f"=== Memory: {mem['title']} (key: {mem['key']}, scope: {mem['role']}) ===\n{mem['content']}"

        memories = await db.get_agent_memories(category=category, role=resolved_role)
        if not memories:
            return f"No memories found in category '{category}' for role scope '{resolved_role}'."

        lines = [f"# Category: {category} (scope: {resolved_role})"]
        for m in memories:
            lines.append(f"\n## {m['title']} (key: `{m['key']}`, scope: `{m['role']}`)")
            lines.append(m["content"].strip())
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Failed to view memory: {e}", exc_info=True)
        return f"Error viewing memory: {e}"


@default_registry.register
async def update_memory(
    category: MemoryCategory,
    key: str,
    title: str,
    content: str,
    old_content: str | None = None,
    role: str | None = None,
    context: ToolContext | None = None,
) -> str:
    """Insert or atomically replace a memory record in the SQLite database.

    This executes a transactionally isolated UPSERT SQL statement, completely protecting other keys.

    Note: To prevent memory bloat, each memory entry's content MUST be kept concise.
    To prevent accidental overwrites, updating an existing memory requires providing the
    exact current content in `old_content`.

    Args:
        category: The memory category (e.g., 'progress', 'learnings', 'user_preferences', 'memo').
        key: The unique snake_case key.
        title: Human-readable label or title for the entry.
        content: Detailed markdown or JSON content payload (must be <= 200 characters).
        old_content: The exact expected current content of the memory if it already exists.
            Must match the existing content to succeed. Required for updating existing keys.
        role: Optional roleplay persona scope (defaults to None for implicit channel/default persona).
        context: Injected tool execution context.

    Returns:
        A success message confirming the sanitized key, title, and role scope.
    """
    if not context:
        return "Error: ToolContext is missing."

    resolved_role = await _resolve_memory_role(category, role, context)
    db = context.gateway.db
    try:
        allowed = await db.get_allowed_memory_categories()
        if category not in allowed:
            return (
                f"Write Denied: Category '{category}' is not recognized.\n"
                f"You can only use the following configured categories: {sorted(list(allowed))}.\n"
                "Please choose one of the allowed categories."
            )

        if not validate_key(key):
            return (
                f"Error: Invalid Key '{key}'.\n"
                "Memory keys must strictly contain only lowercase letters, "
                "underscores, and numbers (regex: ^[a-z0-9_]+$)."
            )

        if len(content) > MAX_MEMORY_CONTENT_LENGTH:
            return (
                f"Error: Content length ({len(content)} characters) exceeds the maximum limit "
                f"of {MAX_MEMORY_CONTENT_LENGTH} characters. Please keep the memory extremely concise and try again."
            )

        sanitized_key = sanitize_key(key)

        # Optimistic locking check to prevent overwrites
        existing = await db.get_agent_memory(category=category, key=sanitized_key, role=resolved_role)
        if existing:
            existing_content = existing.get("content")
            if old_content is None:
                return (
                    f"Write Denied: Memory already exists for key '{sanitized_key}' in category '{category}'.\n"
                    f"To prevent overwriting, you MUST provide the `old_content` parameter "
                    f"matching the current content. Use `view_memory` to read it first."
                )
            if existing_content != old_content:
                return (
                    f"Write Denied: The provided `old_content` does not match the actual existing content "
                    f"for key '{sanitized_key}' in category '{category}'.\n"
                    f"Actual content: {existing_content!r}\n"
                    f"Provided old_content: {old_content!r}\n"
                    f"Please call `view_memory` to get the latest content, and try again."
                )

        await db.upsert_agent_memory(
            category=category,
            key=sanitized_key,
            title=title,
            content=content,
            role=resolved_role,
        )
        return (
            f"Memory successfully saved!\n"
            f"  Category: `{category}`\n"
            f"  Key: `{sanitized_key}`\n"
            f'  Title: "{title}"\n'
            f"  Scope: `{resolved_role}`"
        )
    except Exception as e:
        logger.error(f"Failed to update memory: {e}", exc_info=True)
        return f"Error updating memory: {e}"


@default_registry.register
async def delete_memory(
    category: MemoryCategory,
    key: str,
    role: str | None = None,
    context: ToolContext | None = None,
) -> str:
    """Atomically delete a specific memory entry under a category and implicit correct role scope.

    Args:
        category: The memory category (e.g., 'progress', 'learnings', 'user_preferences').
        key: The unique snake_case key to delete.
        role: Optional roleplay persona scope (defaults to None for implicit channel/default persona).
        context: Injected tool execution context.

    Returns:
        A confirmation message.
    """
    if not context:
        return "Error: ToolContext is missing."

    resolved_role = await _resolve_memory_role(category, role, context)
    db = context.gateway.db
    try:
        if not validate_key(key):
            return (
                f"Error: Invalid Key '{key}'.\n"
                "Memory keys must strictly contain only lowercase letters, "
                "underscores, and numbers (regex: ^[a-z0-9_]+$)."
            )

        sanitized = sanitize_key(key)
        mem = await db.get_agent_memory(category=category, key=sanitized, role=resolved_role)
        if not mem:
            return (
                f"No memory entry found for category='{category}', key='{sanitized}', role='{resolved_role}' to delete."
            )

        await db.delete_agent_memory(category=category, key=sanitized, role=resolved_role)
        return f"Memory successfully deleted: category='{category}', key='{sanitized}', role='{resolved_role}'."
    except Exception as e:
        logger.error(f"Failed to delete memory: {e}", exc_info=True)
        return f"Error deleting memory: {e}"


@default_registry.register
async def view_chat_history_summary(context: ToolContext) -> str:
    """Retrieve the consolidated chat history summary and recent active messages for the current role persona.

    Use this tool to read global context and synchronize knowledge about external conversations,
    stories, and milestones that occurred across all channels/threads.

    Args:
        context: Injected tool execution context (automatically resolved).

    Returns:
        A formatted Markdown string containing the consolidated summary and recent active messages.
    """
    if not context or not context.gateway:
        return "Error: ToolContext or Gateway is missing."

    # 1. Resolve active role using helper
    active_role = await _resolve_memory_role(category="user_preferences", role_param=None, context=context)

    try:
        # 2. Query consolidated timeline and recent messages from Gateway
        stored_ctx, new_messages = await context.gateway.db.get_cross_session_memory_updates(
            role=active_role,
            exclude_session_id=context.session_id,
        )
        timeline_content = stored_ctx.content if stored_ctx else ""

        lines = [f"=== Consolidated Chat History Summary (Role: {active_role}) ==="]
        if timeline_content.strip():
            lines.append(timeline_content.strip())
        else:
            lines.append("No consolidated history recorded yet.")

        if new_messages:
            lines.append("\n=== Recent Active Messages ===")
            for m in new_messages:
                time_str = time.strftime("%m-%d %H:%M", time.localtime(m.timestamp))
                lines.append(f"- [{time_str}] {m.sender}: {m.content}")

        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Failed to retrieve chat history summary: {e}", exc_info=True)
        return f"Error retrieving chat history summary: {e}"


@default_registry.register
async def memory_grep(
    query: str,
    context: ToolContext | None = None,
) -> str:
    """Search active memories and past chat messages for the current role matching the query.

    Args:
        query: Text to search for.
        context: Injected tool execution context.

    Returns:
        Formatted markdown summary of matching memories and messages.
    """
    if not context or not context.gateway:
        return "Error: ToolContext is missing."

    active_role = await _resolve_memory_role(category="user_preferences", role_param=None, context=context)
    db = context.gateway.db

    try:
        memories = await db.search_role_memories(active_role, query)
        messages = await db.search_role_messages(active_role, query)

        if not memories and not messages:
            return f"No memories or messages matching '{query}' found for role '{active_role}'."

        lines = [f"🔍 **Search Results for '{query}' (Role: {active_role})**"]

        if memories:
            lines.append("\n### 🧠 Matching Memories")
            for m in memories:
                updated_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m["updated_at"]))
                lines.append(f"- **[{m['category']}]** `{m['key']}`: \"{m['title']}\" (updated: {updated_str})")
                content_snippet = m["content"].strip().replace("\n", " ")
                if len(content_snippet) > 100:
                    content_snippet = content_snippet[:100] + "..."
                lines.append(f"  > {content_snippet}")

        if messages:
            lines.append("\n### 💬 Matching Messages")
            for m in messages:
                time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m.timestamp))
                sender_str = f"**{m.sender}** ({m.role})"
                content_snippet = m.content.strip().replace("\n", " ")
                if len(content_snippet) > 150:
                    content_snippet = content_snippet[:150] + "..."
                lines.append(f"- [{time_str}] {sender_str} in session `{m.session_id}`:")
                lines.append(f"  > {content_snippet}")

        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Failed to execute memory_grep: {e}", exc_info=True)
        return f"Error executing memory_grep: {e}"
