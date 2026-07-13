"""Skills and chat history search tools for Kesoku AI Agent."""

import logging
import time

from kesoku.agent.skills import SkillManager
from kesoku.agent.tools.registry import ToolContext, default_registry
from kesoku.utils.text import truncate_middle
from kesoku.utils.time_utils import parse_time_to_timestamp

logger = logging.getLogger(__name__)

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


async def _resolve_role(role_param: str | None, context: ToolContext | None) -> str:
    """Resolve the correct role scope based on the context rules."""
    if context and context.gateway:
        db = context.gateway.db
        if context.session_id:
            try:
                sess = await db.get_session(context.session_id)
                if sess and sess.role_name:
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
async def chat_search(
    query: str,
    start_time: str | None = None,
    end_time: str | None = None,
    limit: int = 20,
    context: ToolContext | None = None,
) -> str:
    """Perform hybrid semantic/keyword search over the history messages for the current role.

    Supports wildcard searches (* or empty query) to retrieve the latest messages,
    and filtering by optional time range (start_time, end_time).

    Args:
        query: Query text. If empty or '*', matches all messages (wildcard).
        start_time: Optional start ISO 8601 date/time.
        end_time: Optional end ISO 8601 date/time.
        limit: Max number of matching messages to retrieve (default: 20).
        context: Injected tool execution context.

    Returns:
        Formatted markdown summary of matching messages ordered by similarity or time.
    """
    if not context or not context.gateway:
        return "Error: ToolContext is missing."

    active_role = await _resolve_role(role_param=None, context=context)
    db = context.gateway.db

    start_ts = parse_time_to_timestamp(start_time)
    end_ts = parse_time_to_timestamp(end_time)

    try:
        threshold = 0.55
        if context and context.gateway and context.gateway.context and context.gateway.context.config:
            threshold = context.gateway.context.config.agent.search_threshold

        messages = await db.search_role_messages_semantic(
            active_role,
            query,
            start_time=start_ts,
            end_time=end_ts,
            limit=limit,
            threshold=threshold,
        )

        if not messages:
            time_filter_str = ""
            if start_time or end_time:
                time_filter_str = f" in time range [{start_time or ''} to {end_time or ''}]"
            return f"No messages matching '{query}' found for role '{active_role}'{time_filter_str}."

        is_wildcard = not query or query.strip() == "*"
        title_prefix = "Search Results" if is_wildcard else "Semantic Search Results"
        lines = [f"🔍 **{title_prefix} for '{query}' (Role: {active_role})**"]

        if messages:
            lines.append("\n### 💬 Matching Messages")
            for m in messages:
                time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m.timestamp))
                sender_str = f"**{m.sender}** ({m.role})"
                score_str = ""
                if "similarity_score" in m.metadata and not is_wildcard:
                    score_str = f" (score: {m.metadata['similarity_score']:.4f})"
                lines.append(f"- [{time_str}] {sender_str}{score_str} (ID: `{m.id}`) in session `{m.session_id}`:")
                parts = []
                if m.metadata.get("prev_chunk"):
                    parts.append(f"... {m.metadata['prev_chunk']}")
                parts.append(m.content)
                if m.metadata.get("post_chunk"):
                    parts.append(f"{m.metadata['post_chunk']} ...")

                combined_content = " ".join(parts)
                truncated_content = truncate_middle(combined_content, 500)
                content_indented = "\n".join(f"  > {line}" for line in truncated_content.splitlines())
                lines.append(content_indented)

        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Failed to execute chat_search: {e}", exc_info=True)
        return f"Error executing chat_search: {e}"


@default_registry.register
async def view_message(
    message_id: str,
    context: ToolContext | None = None,
) -> str:
    """Retrieve the complete content of a specific historical chat message by its database ID.

    Args:
        message_id: The unique database ID of the message.
        context: Injected tool execution context.

    Returns:
        Full text and operational status of the requested message.
    """
    if not context or not context.gateway:
        return "Error: ToolContext is missing."

    try:
        db = context.gateway.db
        msg = await db.get_message(message_id)
        if not msg:
            return f"Message with ID '{message_id}' not found."

        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(msg.timestamp))
        return (
            f"💬 **Message Details**\n"
            f"- **ID**: `{msg.id}`\n"
            f"- **Session ID**: `{msg.session_id}`\n"
            f"- **Timestamp**: {time_str}\n"
            f"- **Sender**: {msg.sender} ({msg.role})\n"
            f"- **Type**: {msg.type}\n"
            f"- **Content**:\n"
            f"```\n"
            f"{msg.content}\n"
            f"```"
        )
    except Exception as e:
        logger.error(f"Failed to view message {message_id}: {e}", exc_info=True)
        return f"Error retrieving message details: {e}"
