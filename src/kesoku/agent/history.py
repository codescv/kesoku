"""History cleaning and optimization utilities for Kesoku."""

import datetime
import json
import logging
import os
from typing import Any, Literal

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


def messages_to_openlcm_dicts(history: list[Message]) -> list[dict[str, Any]]:
    """Convert Kesoku Message list to OpenLCM dictionary format.

    Groups consecutive assistant thoughts and tool calls into standard OpenAI
    assistant message dictionaries to prevent them from being sanitized out
    by OpenLCM tool-pair guardrails.

    Args:
        history: A list of Kesoku Message objects.

    Returns:
        A list of raw dictionaries matching the OpenAI/Anthropic format expected by OpenLCM.
    """
    lcm_msgs = []
    current_thought: str | None = None
    current_tool_calls: list[dict[str, Any]] = []

    def flush_assistant():
        nonlocal current_thought, current_tool_calls
        if current_thought is not None or current_tool_calls:
            d = {
                "role": "assistant",
                "content": current_thought,
            }
            if current_tool_calls:
                d["tool_calls"] = current_tool_calls
            lcm_msgs.append(d)
            current_thought = None
            current_tool_calls = []

    for msg in history:
        if msg.role == MessageRole.ASSISTANT and msg.type == MessageType.THOUGHT:
            flush_assistant()
            current_thought = msg.content
        elif msg.role == MessageRole.TOOL and msg.type == MessageType.TOOL_CALL:
            tool_call_id = msg.metadata.get("tool_call_id") or msg.id
            tool_name = msg.metadata.get("tool_name") or msg.sender
            tool_arguments = msg.metadata.get("tool_arguments") or {}
            if isinstance(tool_arguments, dict):
                tool_arguments_str = json.dumps(tool_arguments, ensure_ascii=False)
            else:
                tool_arguments_str = str(tool_arguments)

            tc_dict = {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": tool_arguments_str,
                },
            }
            current_tool_calls.append(tc_dict)
        elif msg.role == MessageRole.TOOL and msg.type == MessageType.TOOL_RESULT:
            flush_assistant()
            tool_call_id = msg.metadata.get("tool_call_id") or msg.parent_id or ""
            tool_name = msg.metadata.get("tool_name") or msg.sender
            tool_result_content = msg.metadata.get("tool_result") or msg.content
            lcm_msgs.append({
                "role": "tool",
                "content": tool_result_content,
                "tool_call_id": tool_call_id,
                "name": tool_name,
            })
        else:
            flush_assistant()
            role_val = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            lcm_msgs.append({
                "role": role_val,
                "content": msg.content,
            })

    flush_assistant()
    return lcm_msgs


def openlcm_dicts_to_messages(
    dicts: list[dict[str, Any]],
    session_id: str,
    chatbot_id: str,
    channel_id: str,
) -> list[Message]:
    """Reconstruct Kesoku Message list from OpenLCM dictionaries.

    Unpacks consolidated assistant message dictionaries (which might contain
    embedded tool_calls) back into separate assistant thought and tool call
    messages to match Kesoku's database schema and prompt builder formatting.

    Args:
        dicts: A list of dictionaries returned by OpenLCM.
        session_id: Target conversational session identifier.
        chatbot_id: Unique chatbot platform identifier.
        channel_id: External platform channel identifier.

    Returns:
        A reconstructed list of Kesoku Message objects.
    """
    msgs = []
    for d in dicts:
        role_str = d["role"]
        role = MessageRole(role_str) if role_str in MessageRole._value2member_map_ else MessageRole.USER
        content = d.get("content") or ""

        msg_type = MessageType.TEXT
        metadata = {}

        # Determine sender name
        if role == MessageRole.SYSTEM:
            sender = "System"
        elif role == MessageRole.USER and "[Note: This conversation uses Lossless Context" in content:
            sender = "System"  # Scaffold header
        elif role == MessageRole.ASSISTANT:
            sender = "Kesoku"
        else:
            sender = "User"

        if role == MessageRole.TOOL:
            msg_type = MessageType.TOOL_RESULT
            metadata["tool_name"] = d.get("name") or "unknown_tool"
            metadata["tool_call_id"] = d.get("tool_call_id") or ""
            msgs.append(
                Message(
                    session_id=session_id,
                    chatbot_id=chatbot_id,
                    channel_id=channel_id,
                    sender=sender,
                    role=role,
                    type=msg_type,
                    content=content,
                    metadata=metadata,
                    status=MessageStatus.RESPONDED,
                )
            )
        elif role == MessageRole.ASSISTANT:
            if d.get("tool_calls"):
                # Unpack consolidated assistant message into separate thought and tool calls!
                if content:
                    msgs.append(
                        Message(
                            session_id=session_id,
                            chatbot_id=chatbot_id,
                            channel_id=channel_id,
                            sender=sender,
                            role=MessageRole.ASSISTANT,
                            type=MessageType.THOUGHT,
                            content=content,
                            status=MessageStatus.RESPONDED,
                        )
                    )
                for tc in d["tool_calls"]:
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "unknown_tool")
                    fn_args = fn.get("arguments", {})
                    if isinstance(fn_args, str):
                        try:
                            fn_args = json.loads(fn_args)
                        except Exception:
                            pass

                    call_args_json = json.dumps(fn_args, indent=2, ensure_ascii=False)
                    msgs.append(
                        Message(
                            session_id=session_id,
                            chatbot_id=chatbot_id,
                            channel_id=channel_id,
                            sender=sender,
                            role=MessageRole.TOOL,
                            type=MessageType.TOOL_CALL,
                            content=f"Calling tool `{fn_name}` with arguments:\n```json\n{call_args_json}\n```",
                            metadata={
                                "tool_name": fn_name,
                                "tool_arguments": fn_args,
                                "tool_call_id": tc.get("id") or "",
                            },
                            status=MessageStatus.RESPONDED,
                        )
                    )
            else:
                msgs.append(
                    Message(
                        session_id=session_id,
                        chatbot_id=chatbot_id,
                        channel_id=channel_id,
                        sender=sender,
                        role=role,
                        type=MessageType.TEXT,
                        content=content,
                        metadata=metadata,
                        status=MessageStatus.RESPONDED,
                    )
                )
        else:
            msgs.append(
                Message(
                    session_id=session_id,
                    chatbot_id=chatbot_id,
                    channel_id=channel_id,
                    sender=sender,
                    role=role,
                    type=msg_type,
                    content=content,
                    metadata=metadata,
                    status=MessageStatus.RESPONDED,
                )
            )
    return msgs

