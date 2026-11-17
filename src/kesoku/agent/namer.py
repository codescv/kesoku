"""Module for automatically generating descriptive titles for sessions.

Naming is hybrid and source-aware (`Session.title_source`):

* `manual`      - the user named the session; never overwritten.
* `context`     - a mode/skill declared `session_title` frontmatter in `$STAGING_DIR/context/*.md`; the first
                  such theme is kept for the rest of the session (later chapter/scene changes never rename it).
* `auto`        - an LLM-generated topic title; final unless a `context` title appears.
* `provisional` - an LLM title produced while the user was still in a menu/setup step; a later turn replaces it.
* `""`          - never named.
"""

import json
import logging
import re
from typing import Any

from kesoku.agent.context_dir import extract_context_session_title
from kesoku.agent.llm import BaseLLM
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import AsyncDatabaseManager, Message
from kesoku.gateway.gateway import Gateway

logger = logging.getLogger(__name__)

TITLE_SOURCE_PROVISIONAL = "provisional"
TITLE_SOURCE_AUTO = "auto"
TITLE_SOURCE_CONTEXT = "context"
TITLE_SOURCE_MANUAL = "manual"

# LLM naming runs only during the opening turns of a session.
_MAX_NAMING_TURNS = 6
# At most this many transcript messages (≈ 3 turns), each truncated, are sent to the naming model.
_MAX_TRANSCRIPT_MESSAGES = 6
_MESSAGE_MAX_CHARS = 350
_TITLE_MAX_CHARS = 40

_SPOILER_BLOCK_RE = re.compile(r"<spoiler>.*?</spoiler>", re.DOTALL | re.IGNORECASE)
_SPOILER_TAG_RE = re.compile(r"</?spoiler>", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```$", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("“", "”"), ("「", "」"), ("『", "』"), ("《", "》"))

_TITLE_PROMPT = (
    "Name the conversation below with a short, concrete topic title.\n"
    "Rules:\n"
    "- Write the title in the same language the user writes in.\n"
    "- Chinese: 4-10 characters (never more than 14). English: 2-5 words.\n"
    "- Use concrete topic nouns/phrases (the subject, story, task or question actually being discussed).\n"
    "- Never use meta/process labels that describe the conversation mechanics instead of its topic, such as "
    '"模式启动", "世界观选择", "开启对话", "Interactive Mode", "New Chat".\n'
    "- No quotes, no trailing punctuation, no prefixes.\n"
    'Return ONLY a JSON object: {"title": "...", "provisional": false}\n'
    'Set "provisional" to true ONLY when the user has just opened a menu or setup step and has not picked a '
    "concrete topic yet (so the title should be replaced once they choose); otherwise false."
)


def _clean_text(content: str) -> str:
    """Remove spoiler blocks/tags and truncate a transcript message to `_MESSAGE_MAX_CHARS`."""
    text = _SPOILER_BLOCK_RE.sub("", content or "")
    text = _SPOILER_TAG_RE.sub("", text).strip()
    if len(text) > _MESSAGE_MAX_CHARS:
        text = text[:_MESSAGE_MAX_CHARS].rstrip() + "…"
    return text


def _clean_title(raw: str) -> str:
    """Normalize a generated title: first non-empty line, surrounding quotes removed, length capped."""
    title = ""
    for line in (raw or "").splitlines():
        if line.strip():
            title = line.strip()
            break
    for opening, closing in _QUOTE_PAIRS:
        if len(title) >= 2 and title.startswith(opening) and title.endswith(closing):
            title = title[len(opening) : -len(closing)].strip()
    return title[:_TITLE_MAX_CHARS].strip()


def parse_title_response(content: str) -> tuple[str, bool]:
    """Parse the naming model reply into `(title, provisional)`.

    Accepts a JSON object `{"title": ..., "provisional": ...}` (optionally wrapped in a markdown code fence);
    any other reply is treated as a plain-text title with `provisional = False`.
    """
    text = (content or "").strip()
    fence = _CODE_FENCE_RE.match(text)
    body = fence.group(1).strip() if fence else text
    data: Any = None
    for candidate in (body, *(m.group(0) for m in [_JSON_OBJECT_RE.search(body)] if m)):
        try:
            data = json.loads(candidate)
            break
        except (TypeError, ValueError):
            continue
    if isinstance(data, dict) and isinstance(data.get("title"), str):
        return _clean_title(data["title"]), data.get("provisional") is True
    return _clean_title(body), False


def _collect_transcript(history: list[Message]) -> list[Message] | None:
    """Pick the opening/latest user & assistant messages to name from (None without a full exchange).

    Keeps every eligible message when there are at most `_MAX_TRANSCRIPT_MESSAGES`; otherwise keeps the first
    user message plus the latest messages so a topic chosen after a setup menu is visible.
    """
    eligible = [
        m
        for m in history
        if m.role in (MessageRole.USER, MessageRole.ASSISTANT)
        and not (m.metadata or {}).get("is_scaffold")
        and (m.content or "").strip()
    ]
    first_user = next((i for i, m in enumerate(eligible) if m.role == MessageRole.USER), None)
    if first_user is None:
        return None
    eligible = eligible[first_user:]
    if not any(m.role == MessageRole.ASSISTANT for m in eligible):
        return None
    if len(eligible) > _MAX_TRANSCRIPT_MESSAGES:
        eligible = eligible[:1] + eligible[-(_MAX_TRANSCRIPT_MESSAGES - 1) :]
    return eligible


class SessionNamer:
    """Handles automatic session renaming from context frontmatter or the opening conversation turns."""

    def __init__(self, db: AsyncDatabaseManager, gateway: Gateway, llm: BaseLLM) -> None:
        """Initialize SessionNamer.

        Args:
            db: AsyncDatabaseManager instance.
            gateway: Gateway instance.
            llm: BaseLLM instance (a lightweight model is sufficient).
        """
        self.db = db
        self.gateway = gateway
        self.llm = llm

    async def auto_rename_session(self, session_id: str, staging_dir: str | None = None) -> bool:
        """Attempt to automatically (re)name the session.

        Order: manual titles and the initial context theme are locked; a context `session_title` wins over
        LLM titles; otherwise an LLM names the session during the opening turns (provisional titles are
        refined on later turns).

        Args:
            session_id: The ID of the session to rename.
            staging_dir: Optional session staging directory scanned for `context/*.md` `session_title`.

        Returns:
            True if the session title changed (and a rename was posted), False otherwise.
        """
        try:
            session = await self.db.get_session(session_id)
            src = (session.title_source if session else "") or ""
            if src in (TITLE_SOURCE_MANUAL, TITLE_SOURCE_CONTEXT):
                return False

            if staging_dir:
                ctx_title = await extract_context_session_title(staging_dir)
                if ctx_title:
                    await self.db.update_session_title(session_id, ctx_title, title_source=TITLE_SOURCE_CONTEXT)
                    if session is None or session.title != ctx_title:
                        logger.info(f"Named session {session_id} from context frontmatter: '{ctx_title}'")
                        await self._post_rename(session_id, ctx_title)
                        return True
                    return False

            if src not in ("", TITLE_SOURCE_PROVISIONAL):
                return False
            return await self._llm_rename(session_id, session.title if session else None)
        except Exception as e:
            logger.error(f"Failed to auto-name session {session_id}: {e}", exc_info=True)
            return False

    async def _llm_rename(self, session_id: str, current_title: str | None) -> bool:
        """Generate a title with the LLM from the opening turns and persist it (see `auto_rename_session`)."""
        turns_count = await self.db.get_session_turns_count(session_id)
        if not 1 <= turns_count <= _MAX_NAMING_TURNS:
            return False

        history = await self.db.get_session_history(session_id, limit=0)
        transcript = _collect_transcript(history)
        if not transcript:
            logger.warning(f"Could not find an opening exchange for auto-naming in session {session_id}")
            return False

        lines = [
            f"{'User' if m.role == MessageRole.USER else 'Assistant'}: {_clean_text(m.content)}" for m in transcript
        ]
        prompt = f"{_TITLE_PROMPT}\n\n" + "\n".join(lines)
        naming_history = [
            Message(
                session_id=session_id,
                chatbot_id="system",
                channel_id="system",
                sender="system",
                role=MessageRole.USER,
                content=prompt,
            )
        ]

        logger.info(f"Generating auto title for session {session_id}...")
        res = await self.llm.generate(system_prompt=None, history=naming_history, tools=None)
        new_title, provisional = parse_title_response(res.content or "")
        if not new_title:
            logger.warning(f"LLM returned empty title for session {session_id}")
            return False

        source = TITLE_SOURCE_PROVISIONAL if provisional else TITLE_SOURCE_AUTO
        await self.db.update_session_title(session_id, new_title, title_source=source)
        if current_title == new_title:
            return False
        logger.info(f"Auto-named session {session_id} to '{new_title}' ({source})")
        await self._post_rename(session_id, new_title, history)
        return True

    async def _post_rename(self, session_id: str, title: str, history: list[Message] | None = None) -> None:
        """Post a `SESSION_RENAME` system message so chat platforms rename the external channel/thread."""
        if history is None:
            history = await self.db.get_session_history(session_id, limit=20)
        chatbot_id = channel_id = None
        for msg in reversed(history):
            if msg.chatbot_id and msg.channel_id and msg.chatbot_id != "system":
                chatbot_id, channel_id = msg.chatbot_id, msg.channel_id
                break
        if chatbot_id is None:
            binding = await self.db.get_channel_by_session(session_id)
            if binding:
                chatbot_id, channel_id = binding
        if not chatbot_id or not channel_id:
            logger.info(f"Session {session_id} has no bound channel; skipping external rename")
            return
        await self.gateway.post(
            Message(
                session_id=session_id,
                chatbot_id=chatbot_id,
                channel_id=channel_id,
                sender="System",
                role=MessageRole.SYSTEM,
                type=MessageType.SESSION_RENAME,
                content=title,
                status=MessageStatus.PENDING,
            )
        )
