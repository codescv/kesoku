"""Skills, role Markdown memory search, and episodic cross-session search/view tools for Kesoku."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import re
import time
from typing import Any

from kesoku.agent.memory_store import search_role_memories
from kesoku.agent.session_digest import is_excluded_from_memory
from kesoku.agent.skills import SkillManager
from kesoku.agent.tools.registry import ToolContext, default_registry
from kesoku.config import KesokuConfig, get_config
from kesoku.constants import MessageRole, MessageType
from kesoku.db.manager import DatabaseManager
from kesoku.db.models import Session
from kesoku.utils.text import strip_furigana, strip_injected_blocks

logger = logging.getLogger(__name__)

skill_manager = SkillManager()

_SNIPPET_QUERY_TERMS_RE = re.compile(
    r"[\u3005-\u3007\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+|[a-zA-Z0-9_]+"
)


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
    """Resolve the session's persona role synchronously (for sync tools); None when unknown."""
    if not context or not context.gateway or not context.session_id:
        return None
    try:
        sync_db = getattr(context.gateway.db, "sync_db", None)
        if sync_db is None:
            return None
        gw_ctx = getattr(context.gateway, "context", None)
        cfg = gw_ctx.config if gw_ctx is not None else None
        sess = sync_db.get_session(context.session_id)
        mapping = sync_db.get_channel_by_session(context.session_id)
        bot_default = cfg.get_chatbot_default_role(mapping[0]) if (cfg is not None and mapping) else "default"
        if sess and sess.role_name and sess.role_name not in ("default", bot_default):
            return sess.role_name
        if mapping:
            chatbot_id, channel_id = mapping
            return sync_db.get_channel_role_with_inheritance(
                chatbot_id, channel_id, context.session_id, default=bot_default
            )
        return sess.role_name if sess and sess.role_name else None
    except Exception as e:
        logger.warning(f"Failed to resolve session role for skills: {e}")
        return None


async def _resolve_role(role_param: str | None, context: ToolContext | None) -> str:
    """Resolve the active canonical persona role from `ToolContext`."""
    if context and context.gateway:
        db = context.gateway.db
        gw_ctx = getattr(context.gateway, "context", None)
        cfg = gw_ctx.config if gw_ctx is not None else None
        sess = None
        if context.session_id:
            try:
                sess = await db.get_session(context.session_id)
            except Exception as e:
                logger.warning(f"Failed to resolve session role: {e}")

        if context.original_msg_id:
            try:
                msg_list = await db.get_messages_by_filters({"id": context.original_msg_id})
                if msg_list:
                    msg = msg_list[0]
                    bot_default = cfg.get_chatbot_default_role(msg.chatbot_id) if cfg is not None else None
                    if sess and sess.role_name and sess.role_name not in ("default", bot_default or "default"):
                        return sess.role_name
                    return await db.get_channel_role_with_inheritance(
                        msg.chatbot_id, msg.channel_id, context.session_id, default=bot_default
                    )
            except Exception as e:
                logger.warning(f"Failed to resolve active role: {e}")
        if context.session_id:
            try:
                mapping = await db.get_channel_by_session(context.session_id)
                if mapping:
                    chatbot_id, channel_id = mapping
                    bot_default = cfg.get_chatbot_default_role(chatbot_id) if cfg is not None else None
                    if sess and sess.role_name and sess.role_name not in ("default", bot_default or "default"):
                        return sess.role_name
                    return await db.get_channel_role_with_inheritance(
                        chatbot_id, channel_id, context.session_id, default=bot_default
                    )
            except Exception as e:
                logger.warning(f"Failed to resolve active role from session_id: {e}")
        if sess and sess.role_name and sess.role_name != "default":
            return sess.role_name

    return role_param if role_param else "default"


def resolve_canonical_session_role_sync(
    sync_db: DatabaseManager,
    cfg: KesokuConfig,
    session: Session,
) -> tuple[str, bool]:
    """Resolve `(canonical_role, is_cronjob)` for any historical or live `Session` (`H2`)."""
    raw_role = (session.role_name or "").strip()
    if raw_role in ("", "Role Name", "None"):
        raw_role = ""

    mapping = sync_db.get_channel_by_session(session.id)
    chatbot_id = mapping[0] if mapping else ""
    channel_id = mapping[1] if mapping else ""

    if not chatbot_id:
        with sync_db.connection_provider.connection() as conn:
            m_row = conn.execute(
                "SELECT chatbot_id, channel_id, sender, metadata FROM messages "
                "WHERE session_id = ? ORDER BY timestamp ASC LIMIT 1",
                (session.id,),
            ).fetchone()
            if m_row is not None:
                chatbot_id = str(m_row["chatbot_id"] or "")
                channel_id = str(m_row["channel_id"] or "")
                try:
                    meta = json.loads(m_row["metadata"]) if m_row["metadata"] else {}
                except Exception:
                    meta = {}
                if is_excluded_from_memory(chatbot_id, str(m_row["sender"] or ""), meta):
                    return "default", True

    if chatbot_id.strip().lower() == "cronjob":
        return "default", True

    bot_default = cfg.get_chatbot_default_role(chatbot_id) if chatbot_id else "default"
    if raw_role and raw_role not in ("default", bot_default):
        return raw_role, False
    if chatbot_id and channel_id:
        resolved = sync_db.get_channel_role_with_inheritance(
            chatbot_id, channel_id, session.id, default=bot_default
        )
        if resolved and resolved not in ("Role Name", "None"):
            return resolved, False
    return (raw_role or bot_default or "default"), False


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

        list_all = (query or "").strip() in ("", "*")
        matches = await asyncio.to_thread(
            lambda: search_role_memories(
                sync_db,
                active_role,
                query,
                limit=limit,
                min_score=threshold,
                include_pinned=list_all,
                allow_sync=True,
            )
        )

        if not matches:
            return f"No memory files matching '{query}' found for role '{active_role}'."

        lines = [
            f"🧠 **Memory Search Results for '{query}' (Role: {active_role})**",
            "Use `read_file` on a `$MEMORY_DIR/...` path below to read its full contents:",
        ]
        for m in matches:
            updated_str = datetime.datetime.fromtimestamp(m.mtime).astimezone().strftime("%Y-%m-%d %H:%M")
            vis_badge = f" [{m.visibility}]" if m.visibility and m.visibility != "DEFAULT" else ""
            kind_badge = f" ({m.kind})" if m.kind and m.kind != "file" else ""
            lines.append(
                f"- `{m.memory_path}`{vis_badge}{kind_badge} (updated: {updated_str}, relevance: {m.score:.2f})\n"
                f"  description: {m.description}"
            )
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Failed to execute memory_search: {e}", exc_info=True)
        return f"Error executing memory_search: {e}"


def _extract_snippet_terms(query: str) -> list[str]:
    """Extract lowercase query terms (plus CJK bigrams) for snippet window centering."""
    cleaned = strip_furigana(strip_injected_blocks(query or "")).lower()
    terms: list[str] = []
    for m in _SNIPPET_QUERY_TERMS_RE.finditer(cleaned):
        tok = m.group(0)
        if tok not in terms:
            terms.append(tok)
        if len(tok) > 2 and not tok.isascii():
            for i in range(len(tok) - 1):
                bg = tok[i : i + 2]
                if bg not in terms:
                    terms.append(bg)
    return terms


def _contains_any_term(text: str, terms: list[str]) -> bool:
    if not text or not terms:
        return bool(text)
    low = text.lower()
    low_furi = strip_furigana(text).lower()
    return any(t in low or t in low_furi for t in terms)


def _render_snippet_window(text: str, terms: list[str], radius: int = 120) -> str:
    """Slice `±radius` chars around the first matched term in `text`."""
    cleaned = " ".join(strip_injected_blocks(text or "").split()).strip()
    if not cleaned:
        return ""
    low = cleaned.lower()
    hit_idx = -1
    hit_len = 0
    for t in terms:
        pos = low.find(t)
        if pos != -1:
            hit_idx = pos
            hit_len = len(t)
            break
    if hit_idx == -1:
        furi = " ".join(strip_furigana(cleaned).split())
        furi_low = furi.lower()
        for t in terms:
            pos = furi_low.find(t)
            if pos != -1:
                cleaned = furi
                hit_idx = pos
                hit_len = len(t)
                break

    if hit_idx == -1:
        return (cleaned[: radius * 2] + "…") if len(cleaned) > radius * 2 else cleaned

    start = max(0, hit_idx - radius)
    end = min(len(cleaned), hit_idx + hit_len + radius)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(cleaned) else ""
    return f"{prefix}{cleaned[start:end].strip()}{suffix}"


def _execute_session_search_sync(
    sync_db: DatabaseManager,
    *,
    role: str,
    query: str,
    current_session_id: str | None,
    days: int | None,
    limit: int,
    include_isolated: bool,
) -> str:
    """Run Stage-1 (`session_digests_fts`) + Stage-2 (`messages_fts`) search with PK snippet hydration."""
    if not sync_db.fts_available:
        return "Session FTS search is currently unavailable on this database."

    since_ts = (time.time() - float(days) * 86400.0) if (days is not None and days > 0) else None
    cap = max(1, min(20, int(limit)))

    digest_hits = sync_db.fts_search_digests(
        role=role,
        query=query,
        limit=cap,
        include_isolated=include_isolated,
        exclude_session_id=current_session_id,
        since_ts=since_ts,
    )
    msg_hits = sync_db.fts_search_messages(
        role=role,
        query=query,
        limit=40,
        include_isolated=include_isolated,
        exclude_session_id=current_session_id,
        since_ts=since_ts,
    )

    if not digest_hits and not msg_hits:
        return f"No past sessions matching '{query}' found for role '{role}'."

    ordered_sids: list[str] = []
    digest_by_sid: dict[str, dict[str, Any]] = {}
    msgs_by_sid: dict[str, list[dict[str, Any]]] = {}

    for dh in digest_hits:
        sid = str(dh["session_id"])
        digest_by_sid[sid] = dh
        if sid not in ordered_sids:
            ordered_sids.append(sid)

    for mh in msg_hits:
        sid = str(mh["session_id"])
        bucket = msgs_by_sid.setdefault(sid, [])
        if len(bucket) < 2:
            bucket.append(mh)
        if sid not in ordered_sids:
            ordered_sids.append(sid)

    selected_sids = ordered_sids[:cap]
    terms = _extract_snippet_terms(query)

    # Hydrate full message text ONLY for <= 10 displayed rows whose 400-char raw_excerpt lacks the hit (R2-3)
    hydrate_ids: list[str] = []
    for sid in selected_sids:
        if sid not in digest_by_sid:
            d_row = sync_db.get_session_digest(sid)
            if d_row is not None:
                digest_by_sid[sid] = d_row
        for mh in msgs_by_sid.get(sid, ())[:2]:
            excerpt = str(mh.get("raw_excerpt") or "")
            if not _contains_any_term(excerpt, terms) and len(hydrate_ids) < 10:
                hydrate_ids.append(str(mh["message_id"]))

    hydrated_texts = sync_db.get_messages_text_by_ids(hydrate_ids) if hydrate_ids else {}

    out_lines: list[str] = [
        f"🔎 **Session Search Results for '{query}' (Role: {role})**",
        "Use `session_view(session_id, around_message_id=...)` to inspect surrounding conversation turns:",
    ]

    for sid in selected_sids:
        d_info = digest_by_sid.get(sid)
        sess = sync_db.get_session(sid) if d_info is None else None
        if d_info and str(d_info.get("chatbot_id") or "").lower() == "cronjob":
            continue
        title = (
            str(d_info.get("title") or "")
            if d_info
            else (sess.title if sess else "Untitled")
        ) or "Untitled"
        title = " ".join(title.split()).replace('"', "'")
        m_list = msgs_by_sid.get(sid, [])[:2]

        ref_ts = (
            float(m_list[0]["ts"])
            if m_list
            else (float(d_info["last_active_at"]) if d_info else (sess.updated_at if sess else 0.0))
        )
        date_str = (
            datetime.datetime.fromtimestamp(ref_ts).astimezone().strftime("%Y-%m-%d")
            if ref_ts > 0
            else "unknown-date"
        )
        iso_badge = (
            " [isolated]"
            if (d_info and d_info.get("context_policy") == "isolated")
            or any(m.get("context_policy") == "isolated" for m in m_list)
            else ""
        )

        if m_list:
            for mh in m_list:
                mid = str(mh["message_id"])
                speaker = str(mh.get("speaker_role") or "user")
                source_text = hydrated_texts.get(mid) or str(mh.get("raw_excerpt") or "")
                snippet = _render_snippet_window(source_text, terms, radius=120)
                out_lines.append(
                    f'- #session:{sid[:8]} "{title}" {date_str}{iso_badge} — '
                    f"{snippet} ({speaker}) [msg:{mid}]"
                )
        elif d_info:
            summary = " ".join(str(d_info.get("summary") or "").split())
            snip = _render_snippet_window(summary or title, terms, radius=120)
            out_lines.append(f'- #session:{sid[:8]} "{title}" {date_str}{iso_badge} — {snip} (digest)')

    return "\n".join(out_lines)


@default_registry.register
async def session_search(
    query: str,
    days: int | None = None,
    limit: int = 5,
    include_isolated: bool = False,
    context: ToolContext | None = None,
) -> str:
    """Search sibling sessions of the active role by keywords or phrases (FTS5 BM25).

    Excludes the current session, other personas' sessions, cronjob sessions, and `isolated`
    turns (unless `include_isolated=True`). Follow up with `session_view` to read full turns.

    Args:
        query: Keywords or phrase to search across session digests and transcripts.
        days: Optional lookback window in days (None searches all history).
        limit: Maximum number of sessions to return (default: 5).
        include_isolated: Whether to include turns from `isolated` context policies.
        context: Injected tool execution context.

    Returns:
        Grouped session search results with snippets and `#session:<id>` / `[msg:<id>]` handles.
    """
    if not context or not context.gateway:
        return "Error: ToolContext is missing."
    gw_ctx = getattr(context.gateway, "context", None)
    cfg = gw_ctx.config if gw_ctx is not None else get_config()
    if not cfg.agent.session_search_enabled:
        return "session_search is disabled in configuration."

    if not query or not query.strip():
        return "Error: query must be non-empty."

    active_role = await _resolve_role(role_param=None, context=context)
    sync_db = getattr(context.gateway.db, "sync_db", context.gateway.db)
    eff_limit = limit if limit > 0 else int(cfg.agent.session_search_max_results or 5)

    try:
        return await asyncio.to_thread(
            lambda: _execute_session_search_sync(
                sync_db,
                role=active_role,
                query=query,
                current_session_id=context.session_id,
                days=days,
                limit=eff_limit,
                include_isolated=include_isolated,
            )
        )
    except Exception as e:
        logger.error(f"Failed executing session_search: {e}", exc_info=True)
        return f"Error executing session_search: {e}"


def _execute_session_view_sync(
    sync_db: DatabaseManager,
    cfg: KesokuConfig,
    *,
    active_role: str,
    raw_session_id: str,
    around_message_id: str | None,
    window: int,
    include_isolated: bool,
) -> str:
    """Execute role- and policy-guarded `session_view` lookup (`M2`)."""
    clean_id = (raw_session_id or "").strip()
    if clean_id.startswith("#session:"):
        clean_id = clean_id[len("#session:") :].strip()
    if not clean_id:
        return "Session not found."

    digest = sync_db.get_session_digest(clean_id)
    sess = sync_db.get_session(clean_id)

    # Support 8-char `#session:<id8>` prefix lookup strictly within `active_role`'s digests if exact ID wasn't found
    if digest is None and sess is None and len(clean_id) >= 6:
        with sync_db.connection_provider.connection() as conn:
            row = conn.execute(
                "SELECT session_id FROM session_digests WHERE role = ? AND session_id LIKE ? LIMIT 1",
                (active_role, f"{clean_id}%"),
            ).fetchone()
            if row is not None:
                clean_id = str(row["session_id"])
                digest = sync_db.get_session_digest(clean_id)
                sess = sync_db.get_session(clean_id)

    if digest is None and sess is None:
        return f"Session '{clean_id}' not found."

    # Authorization guard 1: cronjob exclusion + canonical role check
    if digest is not None:
        if str(digest.get("chatbot_id") or "").strip().lower() == "cronjob":
            return f"Session '{clean_id}' not found."
        target_role = str(digest.get("role") or "default")
        target_policy = str(digest.get("context_policy") or "full")
    else:
        assert sess is not None
        target_role, is_cron = resolve_canonical_session_role_sync(sync_db, cfg, sess)
        if is_cron:
            return f"Session '{clean_id}' not found."
        target_policy = "full"

    if target_role != active_role:
        return f"Session '{clean_id}' not found."

    # Authorization guard 2: context_policy isolation check
    if target_policy == "isolated" and not include_isolated:
        return f"Session '{clean_id}' not found."

    msgs = sync_db.get_session_view_messages(
        clean_id,
        around_message_id=around_message_id,
        window=max(1, int(window)),
        max_messages=20,
    )
    if not msgs:
        return f"Session '{clean_id}' has no viewable messages."

    # Filter out cronjob/Notification rows and any per-turn isolated rows when include_isolated=False
    filtered_msgs = []
    skip_isolated_turn = False
    for m in msgs:
        if m.sender == "Notification" or is_excluded_from_memory(m):
            continue
        if m.role == MessageRole.USER:
            turn_pol = str((m.metadata or {}).get("context_policy") or "full")
            skip_isolated_turn = turn_pol == "isolated" and not include_isolated
        if skip_isolated_turn:
            continue
        filtered_msgs.append(m)

    if not filtered_msgs:
        return f"Session '{clean_id}' not found."

    title = (
        str(digest.get("title") or "")
        if digest
        else (sess.title if sess else "Untitled")
    ) or "Untitled"

    header = f'📖 **Session #session:{clean_id[:8]} "{title}"** (showing {len(filtered_msgs)} message(s)):'
    lines = [header]

    for m in filtered_msgs[:20]:
        ts_str = datetime.datetime.fromtimestamp(m.timestamp).astimezone().strftime("%Y-%m-%d %H:%M")
        if m.type == MessageType.TOOL_CALL:
            t_name = (m.metadata or {}).get("tool_name") or m.sender or "tool"
            lines.append(f"[{ts_str}] [msg:{m.id}] tool_call: {t_name}")
        elif m.type == MessageType.TOOL_RESULT:
            raw_res = " ".join(strip_injected_blocks(m.content or "").split())
            if len(raw_res) > 200:
                raw_res = raw_res[:200] + "…[truncated]"
            lines.append(f"[{ts_str}] [msg:{m.id}] tool_result: {raw_res}")
        else:
            cleaned = strip_injected_blocks(m.content or "")
            lines.append(f"[{ts_str}] [msg:{m.id}] {m.role}: {cleaned}")

    return "\n".join(lines)


@default_registry.register
async def session_view(
    session_id: str,
    around_message_id: str | None = None,
    window: int = 6,
    include_isolated: bool = False,
    context: ToolContext | None = None,
) -> str:
    """View bounded conversational messages from a sibling session of the active role.

    Enforces strict role, cronjob, and `context_policy` authorization guards (`<= 20` messages,
    tool outputs truncated to `<= 200` chars).

    Args:
        session_id: Target session ID or `#session:<id8>` handle.
        around_message_id: Optional message ID (`[msg:<id>]` from `session_search`) to center the window on.
        window: Number of messages before and after `around_message_id` (default: 6; total capped at 20).
        include_isolated: Whether to allow viewing sessions/turns marked `isolated`.
        context: Injected tool execution context.

    Returns:
        Formatted transcript excerpt, or `"Session '...' not found."` if unauthorized or absent.
    """
    if not context or not context.gateway:
        return "Error: ToolContext is missing."
    gw_ctx = getattr(context.gateway, "context", None)
    cfg = gw_ctx.config if gw_ctx is not None else get_config()
    if not cfg.agent.session_search_enabled:
        return "session_view is disabled in configuration."

    active_role = await _resolve_role(role_param=None, context=context)
    sync_db = getattr(context.gateway.db, "sync_db", context.gateway.db)

    try:
        return await asyncio.to_thread(
            lambda: _execute_session_view_sync(
                sync_db,
                cfg,
                active_role=active_role,
                raw_session_id=session_id,
                around_message_id=around_message_id,
                window=window,
                include_isolated=include_isolated,
            )
        )
    except Exception as e:
        logger.error(f"Failed executing session_view: {e}", exc_info=True)
        return f"Error executing session_view: {e}"
