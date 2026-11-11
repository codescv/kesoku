"""Tier 0b deterministic cross-session rolling digest (`<recent_sessions>`) and post-turn FTS indexing."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

from kesoku.constants import MessageRole, MessageType
from kesoku.db.manager import AsyncDatabaseManager, DatabaseManager
from kesoku.db.models import Message, Session
from kesoku.utils.text import fts_tokenize, strip_injected_blocks

logger = logging.getLogger(__name__)

MAX_LAST_USER_MSGS = 3
MAX_LAST_USER_MSG_CHARS = 160
MAX_RAW_EXCERPT_CHARS = 400


def is_excluded_from_memory(
    msg_or_chatbot_id: Message | str | None,
    sender: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> bool:
    """Return True if a message or session belongs to a `cronjob` or `system` source (`H2`)."""
    if isinstance(msg_or_chatbot_id, Message):
        chatbot_id = str(msg_or_chatbot_id.chatbot_id or "")
        raw_sender = str(msg_or_chatbot_id.sender or "")
        meta = msg_or_chatbot_id.metadata if isinstance(msg_or_chatbot_id.metadata, Mapping) else {}
    else:
        chatbot_id = str(msg_or_chatbot_id or "")
        raw_sender = str(sender or "")
        meta = metadata if isinstance(metadata, Mapping) else {}

    if chatbot_id.strip().lower() == "cronjob":
        return True
    if raw_sender.strip().lower() in ("cronjob", "system"):
        return True
    sender_name = str(meta.get("sender_name") or "").strip().lower()
    if sender_name in ("cronjob", "system"):
        return True
    return bool(meta.get("is_cronjob"))


def _unwrap_sync_db(db: DatabaseManager | AsyncDatabaseManager) -> DatabaseManager:
    """Extract the underlying synchronous `DatabaseManager`."""
    if isinstance(db, DatabaseManager):
        return db
    return getattr(db, "sync_db", db)  # type: ignore[return-value]


def _clean_user_snippet(text: str, max_chars: int = MAX_LAST_USER_MSG_CHARS) -> str:
    """Clean a user message to a single-line snippet of at most `max_chars` characters."""
    cleaned = strip_injected_blocks(text or "")
    collapsed = " ".join(cleaned.split()).strip()
    if len(collapsed) > max_chars:
        return collapsed[: max_chars - 1].rstrip() + "…"
    return collapsed


def _parse_json_str_list(raw: Any) -> list[str]:
    """Parse a JSON array column (or Python list) into a list of non-empty strings."""
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    return []


def _escape_attr(value: str) -> str:
    """Escape XML attribute special characters."""
    return str(value).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


class DigestUpdater:
    """Maintains `session_digests`, `session_digests_fts`, and post-turn `messages_fts` entries."""

    @staticmethod
    def touch_at_turn_start_sync(
        db: DatabaseManager,
        session: Session | None,
        role: str,
        current_msg: Message,
    ) -> None:
        """Upsert `session_digests` deterministically at turn start (~1 ms, zero LLM calls)."""
        if is_excluded_from_memory(current_msg):
            return
        session_id = current_msg.session_id
        safe_role = role or (session.role_name if session and session.role_name else "default")
        existing = db.get_session_digest(session_id)
        prev_msgs = _parse_json_str_list(existing["last_user_msgs"]) if existing else []

        snippet = _clean_user_snippet(current_msg.content)
        if snippet and (not prev_msgs or prev_msgs[-1] != snippet):
            prev_msgs.append(snippet)
        prev_msgs = prev_msgs[-MAX_LAST_USER_MSGS:]

        started_at = (
            float(existing["started_at"])
            if existing
            else float(session.created_at if session else current_msg.timestamp)
        )
        title = (
            str(session.title)
            if (session and session.title)
            else (str(existing["title"]) if existing else "")
        )
        db.upsert_session_digest(
            session_id=session_id,
            role=safe_role,
            chatbot_id=current_msg.chatbot_id,
            channel_id=current_msg.channel_id,
            title=title,
            started_at=started_at,
            last_active_at=float(current_msg.timestamp),
            last_user_msgs=prev_msgs,
            updated_at=time.time(),
        )

    @classmethod
    async def touch_at_turn_start(
        cls,
        db: DatabaseManager | AsyncDatabaseManager,
        session: Session | None,
        role: str,
        current_msg: Message,
    ) -> None:
        """Fire-and-forget async wrapper updating `session_digests` at turn start (`H3`)."""
        try:
            sync_db = _unwrap_sync_db(db)
            await asyncio.to_thread(cls.touch_at_turn_start_sync, sync_db, session, role, current_msg)
        except Exception as e:
            logger.warning(f"DigestUpdater.touch_at_turn_start failed for session '{current_msg.session_id}': {e}")

    @staticmethod
    def finalize_after_turn_sync(
        db: DatabaseManager,
        session_id: str,
        role: str,
        context_policy: str,
        turn_messages: Sequence[Message],
        current_user_msg: Message | None = None,
    ) -> None:
        """Persist post-turn `turn_count`, `context_policy`, user message metadata, and `messages_fts` rows."""
        if current_user_msg is not None and is_excluded_from_memory(current_user_msg):
            return
        if turn_messages and all(is_excluded_from_memory(m) for m in turn_messages if m.role == MessageRole.USER):
            return

        safe_role = role or "default"
        eff_policy = context_policy or "full"

        if current_user_msg is not None:
            meta = dict(current_user_msg.metadata or {})
            if meta.get("context_policy") != eff_policy:
                meta["context_policy"] = eff_policy
                current_user_msg.metadata = meta
                db.update_message_metadata(current_user_msg.id, meta)

        turn_count = db.get_session_turns_count(session_id)
        last_ts = max((float(m.timestamp) for m in turn_messages), default=time.time())
        sess = db.get_session(session_id)

        existing = db.get_session_digest(session_id)
        user_msgs_list: list[str] | None = None
        if existing is None and current_user_msg is not None:
            snip = _clean_user_snippet(current_user_msg.content)
            user_msgs_list = [snip] if snip else []

        db.upsert_session_digest(
            session_id=session_id,
            role=safe_role,
            chatbot_id=current_user_msg.chatbot_id if current_user_msg else None,
            channel_id=current_user_msg.channel_id if current_user_msg else None,
            title=sess.title if sess else None,
            started_at=sess.created_at if sess else last_ts,
            last_active_at=last_ts,
            turn_count=turn_count,
            last_user_msgs=user_msgs_list,
            context_policy=eff_policy,
            updated_at=time.time(),
        )

        fts_entries: list[dict[str, Any]] = []
        for msg in turn_messages:
            if msg.role not in (MessageRole.USER, MessageRole.ASSISTANT) or msg.type != MessageType.TEXT:
                continue
            if msg.sender == "Notification" or is_excluded_from_memory(msg):
                continue
            if isinstance(msg.metadata, Mapping) and msg.metadata.get("prompt_blocked"):
                continue
            cleaned = strip_injected_blocks(msg.content or "")
            if not cleaned:
                continue
            tokenized = fts_tokenize(cleaned, for_query=False)
            if not tokenized:
                continue
            excerpt = " ".join(cleaned.split())[:MAX_RAW_EXCERPT_CHARS]
            fts_entries.append(
                {
                    "message_id": msg.id,
                    "session_id": session_id,
                    "persona_role": safe_role,
                    "context_policy": eff_policy,
                    "speaker_role": str(msg.role),
                    "ts": float(msg.timestamp),
                    "raw_excerpt": excerpt,
                    "text": tokenized,
                }
            )
        if fts_entries:
            db.fts_insert_messages(fts_entries)

    @classmethod
    async def finalize_after_turn(
        cls,
        db: DatabaseManager | AsyncDatabaseManager,
        session_id: str,
        role: str,
        context_policy: str,
        turn_messages: Sequence[Message],
        current_user_msg: Message | None = None,
    ) -> None:
        """Async wrapper finalizing `session_digests` and indexing the turn into `messages_fts`."""
        try:
            sync_db = _unwrap_sync_db(db)
            await asyncio.to_thread(
                cls.finalize_after_turn_sync,
                sync_db,
                session_id,
                role,
                context_policy,
                turn_messages,
                current_user_msg,
            )
        except Exception as e:
            logger.warning(f"DigestUpdater.finalize_after_turn failed for session '{session_id}': {e}")


def fetch_recent_digests(
    db: DatabaseManager,
    role: str,
    current_session_id: str | None = None,
    window_days: int = 7,
    limit: int = 5,
    now_ts: float | None = None,
) -> list[dict[str, Any]]:
    """Fetch recent sibling `session_digests` rows for `role` within `window_days`."""
    if limit <= 0 or window_days <= 0:
        return []
    ref_now = float(now_ts if now_ts is not None else time.time())
    since_ts = ref_now - float(window_days) * 86400.0
    return db.get_recent_session_digests(
        role=role or "default",
        exclude_session_id=current_session_id,
        since_ts=since_ts,
        limit=limit,
    )


def should_render(
    is_bootstrap: bool,
    prev_user_ts: float | None,
    rows: Sequence[Mapping[str, Any]],
) -> bool:
    """Return True only on bootstrap turns or when a sibling session interleaved after `prev_user_ts`."""
    if not rows:
        return False
    if is_bootstrap or prev_user_ts is None:
        return True
    cutoff = float(prev_user_ts)
    return any(float(r.get("last_active_at") or 0.0) > cutoff for r in rows)


def extract_bootstrap_extra_context(
    rows: Sequence[Mapping[str, Any]],
    max_sessions: int = 2,
    max_chars: int = 600,
) -> str:
    """Collect `last_user_msgs` from top non-isolated digests for short/referential bootstrap queries."""
    snippets: list[str] = []
    used_sessions = 0
    for row in rows:
        if str(row.get("context_policy") or "full") == "isolated":
            continue
        msgs = _parse_json_str_list(row.get("last_user_msgs"))
        if not msgs:
            continue
        title = str(row.get("title") or "").strip()
        joined = " / ".join(msgs[-2:])
        snippets.append(f"{title}: {joined}" if title else joined)
        used_sessions += 1
        if used_sessions >= max_sessions:
            break
    out = " | ".join(snippets)
    return out[:max_chars]


def format_recent_sessions_block(
    role: str,
    rows: Sequence[Mapping[str, Any]],
    max_chars: int = 1200,
) -> str:
    """Render `<recent_sessions>` within `max_chars` (`[isolated]` sessions render title-only)."""
    if not rows or max_chars <= 0:
        return ""

    safe_role = _escape_attr(role or "default")
    header = (
        f'<recent_sessions role="{safe_role}" '
        'note="Other recent sessions of this role (not this one). '
        'Use session_search / session_view for details. Never mention this block to the user.">\n'
    )
    footer = "</recent_sessions>\n"
    overhead = len(header) + len(footer)
    if max_chars <= overhead + 20:
        return ""

    entry_blocks: list[str] = []
    used = overhead

    for row in rows:
        sid = str(row.get("session_id") or "")[:8]
        raw_title = " ".join(str(row.get("title") or "Untitled").split()).replace('"', "'")
        if len(raw_title) > 60:
            raw_title = raw_title[:59].rstrip() + "…"
        last_ts = float(row.get("last_active_at") or 0.0)
        ts_str = (
            datetime.datetime.fromtimestamp(last_ts).astimezone().strftime("%Y-%m-%d %H:%M")
            if last_ts > 0
            else "unknown"
        )
        turns = int(row.get("turn_count") or 0)
        policy = str(row.get("context_policy") or "full")

        if policy == "isolated":
            entry = f'- #session:{sid} "{raw_title}" last: {ts_str} turns: {turns} [isolated]\n'
        else:
            lines = [f'- #session:{sid} "{raw_title}" last: {ts_str} turns: {turns}']
            summary = " ".join(str(row.get("summary") or "").split()).strip()
            open_threads = _parse_json_str_list(row.get("open_threads"))
            if open_threads and "待办" not in summary:
                threads_str = "、".join(open_threads[:3])
                summary = f"{summary}；待办：{threads_str}" if summary else f"待办：{threads_str}"
            if summary:
                if len(summary) > 220:
                    summary = summary[:219].rstrip() + "…"
                lines.append(f"  summary: {summary}")
            user_msgs = _parse_json_str_list(row.get("last_user_msgs"))
            if user_msgs:
                last_u = user_msgs[-1].replace('"', "'")
                if len(last_u) > 160:
                    last_u = last_u[:159].rstrip() + "…"
                lines.append(f'  last_user: "{last_u}"')
            entry = "\n".join(lines) + "\n"

        if used + len(entry) > max_chars:
            # Try fitting at least the compact 1-line header if nothing was added yet
            compact = (
                f'- #session:{sid} "{raw_title}" last: {ts_str} turns: {turns}'
                + (" [isolated]\n" if policy == "isolated" else "\n")
            )
            if not entry_blocks and used + len(compact) <= max_chars:
                entry_blocks.append(compact)
            break

        entry_blocks.append(entry)
        used += len(entry)

    if not entry_blocks:
        return ""
    return header + "".join(entry_blocks) + footer
