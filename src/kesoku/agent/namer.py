"""Module for automatically generating descriptive titles for sessions."""

import logging

from kesoku.agent.llm import BaseLLM
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import AsyncDatabaseManager, Message
from kesoku.gateway.gateway import Gateway

logger = logging.getLogger(__name__)

_TITLE_PROMPT = (
    "Generate a short, descriptive title (3-7 words) for a conversation that starts with the "
    "following exchange. The title should capture the main topic or intent. "
    "Return ONLY the title text, nothing else. No quotes, no punctuation at the end, no prefixes."
)


class SessionNamer:
    """Handles automatic session renaming based on initial conversation exchange."""

    def __init__(self, db: AsyncDatabaseManager, gateway: Gateway, llm: BaseLLM) -> None:
        """Initialize SessionNamer.

        Args:
            db: AsyncDatabaseManager instance.
            gateway: Gateway instance.
            llm: BaseLLM instance.
        """
        self.db = db
        self.gateway = gateway
        self.llm = llm

    async def auto_rename_session(self, session_id: str) -> bool:
        """Attempt to automatically rename the session based on first turn exchange.

        Args:
            session_id: The ID of the session to rename.

        Returns:
            True if renaming was successful, False otherwise.
        """
        try:
            turns_count = await self.db.get_session_turns_count(session_id)
            if turns_count not in (1, 2):
                return False

            # Get the first turn exchange
            history = await self.db.get_session_history(session_id, limit=0)
            user_msg = None
            assistant_msg = None
            for msg in history:
                if msg.role == MessageRole.USER and not msg.metadata.get("is_scaffold"):
                    if user_msg is None:
                        user_msg = msg
                elif msg.role == MessageRole.ASSISTANT and user_msg is not None:
                    if assistant_msg is None:
                        assistant_msg = msg
                        break

            if not user_msg or not assistant_msg:
                logger.warning(f"Could not find first turn exchange for auto-naming in session {session_id}")
                return False

            # Truncate content to 500 chars to avoid sending too much data to LLM
            user_content = user_msg.content[:500]
            assistant_content = assistant_msg.content[:500]

            prompt = (
                f"{_TITLE_PROMPT}\n\n"
                f"User: {user_content}\n"
                f"Assistant: {assistant_content}"
            )

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
            new_title = res.content.strip()

            # Clean up potential quotes in the generated title
            if new_title.startswith('"') and new_title.endswith('"'):
                new_title = new_title[1:-1]
            if new_title.startswith("'") and new_title.endswith("'"):
                new_title = new_title[1:-1]

            if new_title:
                logger.info(f"Auto-named session {session_id} to '{new_title}'")
                await self.db.update_session_title(session_id, new_title)

                # Post a system message to gateway to trigger external platform rename
                rename_msg = Message(
                    session_id=session_id,
                    chatbot_id=assistant_msg.chatbot_id,
                    channel_id=assistant_msg.channel_id,
                    sender="System",
                    role=MessageRole.SYSTEM,
                    type=MessageType.SESSION_RENAME,
                    content=new_title,
                    status=MessageStatus.PENDING,
                )
                await self.gateway.post(rename_msg)
                return True
            else:
                logger.warning(f"LLM returned empty title for session {session_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to auto-name session {session_id}: {e}", exc_info=True)
            return False
