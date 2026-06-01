"""History cleaning and optimization utilities for Kesoku."""

import datetime
import logging
import os
from typing import Literal

from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.gateway.gateway import Gateway

logger = logging.getLogger(__name__)


async def build_clean_history(
    gateway: Gateway,
    session_id: str,
    order: Literal["phased", "grouped"] = "phased",
    heal_orphans: bool = True,
) -> list[Message]:
    """Retrieve, clean up, and format the conversational history for the LLM.

    Heals orphaned tool calls, groups messages into complete logical turns,
    strips thoughts from all completed turns, strips attachments from historical user prompts,
    and returns the simplified history.

    Args:
        gateway: Gateway instance to interact with storage.
        session_id: Unique conversational session identifier.
        order: The sorting order of the history ("phased" or "grouped").
        heal_orphans: If True, heals orphaned tool calls in the database.

    Returns:
        A list of cleanly structured, prioritized, and aligned Message objects for the LLM.
    """
    # 1. Detects and heals orphaned tool calls using optimized database query.
    if heal_orphans:
        orphaned_calls = await gateway.db.get_orphaned_tool_calls(session_id)
        for tc in orphaned_calls:
            logger.warning(
                f"Found orphaned tool call {tc.id} (tool: {tc.metadata.get('tool_name')}). "
                "Synthesizing interruption response."
            )
            tool_name = tc.metadata.get("tool_name", "unknown_tool")
            interrupted_msg = Message(
                session_id=session_id,
                chatbot_id=tc.chatbot_id,
                channel_id=tc.channel_id,
                sender=tool_name,
                role=MessageRole.TOOL,
                type=MessageType.TOOL_RESULT,
                content=f"Tool `{tool_name}` execution was interrupted due to service restart.",
                status=MessageStatus.RESPONDED,
                parent_id=tc.id,
                metadata={
                    "tool_name": tool_name,
                    "tool_error": "Tool execution was interrupted due to service restart.",
                },
            )
            await gateway.post(interrupted_msg)

    # 2. Load entire chronological history (limit=0 retrieves all messages in the session)
    raw_history = await gateway.db.get_session_history(session_id, limit=0, order=order)
    # Exclude system-generated assistant notification messages from LLM prompt history
    raw_history = [m for m in raw_history if not (m.role == MessageRole.ASSISTANT and m.sender == "Notification")]

    # 3. Groups messages into complete logical turns (User/System prompt -> ... -> before next user/system prompt).
    turns: list[list[Message]] = []
    current_turn: list[Message] = []
    for m in raw_history:
        if m.role in (MessageRole.USER, MessageRole.SYSTEM):
            if current_turn:
                turns.append(current_turn)
            current_turn = [m]
        else:
            if current_turn:
                current_turn.append(m)
            else:
                current_turn = [m]
    if current_turn:
        turns.append(current_turn)

    # 4. Clean each turn logically based on its completion status.
    # The very last turn in the list is the active turn, which is kept in full detail.
    # All prior turns are completed turns, which are cleaned.
    cleaned_turns = []
    num_turns = len(turns)
    for idx, turn in enumerate(turns):
        is_latest = idx == num_turns - 1

        if is_latest:
            # Keep all thoughts and details in the latest/active turn
            cleaned_turns.append(turn)
        else:
            # Completed turn: strip thoughts and clean historical user attachments
            clean_turn = []
            for m in turn:
                # Drop assistant thoughts in completed turns
                if m.role == MessageRole.ASSISTANT and m.type == MessageType.THOUGHT:
                    continue

                # Strip attachments from historical user messages to optimize context size
                if m.role == MessageRole.USER:
                    attachments = m.metadata.get("attachments")
                    if attachments:
                        filenames = [os.path.basename(att.get("path", "file")) for att in attachments]
                        placeholder = f"\n\n[Attachments stripped from history: {', '.join(filenames)}]"
                        if placeholder not in m.content:
                            m.content += placeholder
                        # Copy metadata dict to avoid mutating shared in-memory state
                        m.metadata = dict(m.metadata)
                        m.metadata.pop("attachments", None)

                clean_turn.append(m)
            cleaned_turns.append(clean_turn)

    # 5. Flatten turns and construct the final chronological context history.
    final_history = []
    for turn in cleaned_turns:
        for m in turn:
            if m.role == MessageRole.USER:
                m_copy = m.model_copy()
                msg_time = datetime.datetime.fromtimestamp(m_copy.timestamp).astimezone()
                time_str = msg_time.strftime("%Y-%m-%d %H:%M:%S (%A) %Z")
                sender_name = m_copy.metadata.get("sender_name") or m_copy.sender
                header = f"[{sender_name} at {time_str}]:\n"
                m_copy.content = header + m_copy.content
                final_history.append(m_copy)
            else:
                final_history.append(m)

    return final_history
