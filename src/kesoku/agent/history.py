"""History cleaning and optimization utilities for Kesoku."""

import datetime
import json
import logging
import os
import re
from typing import Any, Literal

from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.gateway.gateway import Gateway

logger = logging.getLogger(__name__)


async def build_history(
    gateway: Gateway,
    session_id: str,
    order: Literal["phased", "grouped"] = "phased",
    heal_orphans: bool = False,
) -> list[Message]:
    """Retrieve the raw conversational history from storage.

    Heals orphaned tool calls (if heal_orphans is True), loads the chronological
    history, and excludes internal notification messages. Returns the raw message list
    without stripping thoughts, attachments, or adding user formatting headers.
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

    return raw_history


def prepare_history_for_llm(history: list[Message]) -> list[Message]:
    """Clean up and format the conversational history specifically for the LLM.

    Groups messages into complete logical turns, strips thoughts from all completed
    turns, strips attachments from historical user prompts, adds headers to user
    messages, and returns the simplified history.
    """
    # 1. Groups messages into complete logical turns (User/System prompt -> ... -> before next user/system prompt).
    turns: list[list[Message]] = []
    current_turn: list[Message] = []
    for m in history:
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

    # 2. Clean each turn logically based on its completion status.
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
                            # Copy to avoid mutating shared state in-memory
                            m = m.model_copy()
                            m.content += placeholder
                            m.metadata = dict(m.metadata)
                            m.metadata.pop("attachments", None)

                clean_turn.append(m)
            cleaned_turns.append(clean_turn)

    # 3. Flatten turns and add user headers.
    final_history = []
    for turn in cleaned_turns:
        for m in turn:
            if m.role == MessageRole.USER:
                # Don't add header to the OpenLCM scaffold message
                if "[Note: This conversation uses Lossless Context" in m.content:
                    final_history.append(m)
                elif "<current_request" in m.content:
                    final_history.append(m)
                else:
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
    # 1. Group messages into turns and strip historical thoughts (Question 1)
    turns: list[list[Message]] = []
    current_turn: list[Message] = []
    for m in history:
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

    cleaned_history = []
    num_turns = len(turns)
    for idx, turn in enumerate(turns):
        is_latest = idx == num_turns - 1
        for m in turn:
            if not is_latest and m.role == MessageRole.ASSISTANT and m.type == MessageType.THOUGHT:
                continue
            cleaned_history.append(m)

    # 2. Sanitize paths and process messages (Question 4)
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

    for msg in cleaned_history:
        # Sanitize absolute paths in message content to reduce token count and prevent file lists (Question 4)
        msg_content = msg.content or ""
        if msg_content:
            # Replace absolute paths under sessions staging directory with $STAGING_DIR/filename.ext
            pattern = r'/[^\s\'\"\]]*/sessions/[^/]+/([^\s\'\"\]]*)'
            msg_content = re.sub(pattern, r'$STAGING_DIR/\1', msg_content)

        if msg.role == MessageRole.ASSISTANT and msg.type == MessageType.THOUGHT:
            flush_assistant()
            current_thought = msg_content
        elif msg.role == MessageRole.TOOL and msg.type == MessageType.TOOL_CALL:
            tool_call_id = msg.metadata.get("tool_call_id") or msg.id
            tool_name = msg.metadata.get("tool_name") or msg.sender
            tool_arguments = msg.metadata.get("tool_arguments") or {}
            if isinstance(tool_arguments, dict):
                # Also sanitize paths in tool arguments if they are strings
                sanitized_args = {}
                for k, v in tool_arguments.items():
                    if isinstance(v, str):
                        pattern = r'/[^\s\'\"\]]*/sessions/[^/]+/([^\s\'\"\]]*)'
                        sanitized_args[k] = re.sub(pattern, r'$STAGING_DIR/\1', v)
                    else:
                        sanitized_args[k] = v
                tool_arguments_str = json.dumps(sanitized_args, ensure_ascii=False)
            else:
                tool_arguments_str = str(tool_arguments)
                pattern = r'/[^\s\'\"\]]*/sessions/[^/]+/([^\s\'\"\]]*)'
                tool_arguments_str = re.sub(pattern, r'$STAGING_DIR/\1', tool_arguments_str)

            tc_dict = {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": tool_arguments_str,
                },
            }
            ts_hex = msg.metadata.get("thought_signature")
            if ts_hex:
                tc_dict["thought_signature"] = ts_hex
            current_tool_calls.append(tc_dict)
        elif msg.role == MessageRole.TOOL and msg.type == MessageType.TOOL_RESULT:
            flush_assistant()
            tool_call_id = msg.metadata.get("tool_call_id") or msg.parent_id or ""
            tool_name = msg.metadata.get("tool_name") or msg.sender
            tool_result_content = msg.metadata.get("tool_result") or msg_content
            if isinstance(tool_result_content, str):
                pattern = r'/[^\s\'\"\]]*/sessions/[^/]+/([^\s\'\"\]]*)'
                tool_result_content = re.sub(pattern, r'$STAGING_DIR/\1', tool_result_content)
            lcm_msgs.append({
                "role": "tool",
                "content": tool_result_content,
                "tool_call_id": tool_call_id,
                "name": tool_name,
            })
        elif msg.role == MessageRole.ASSISTANT and msg.type == MessageType.TEXT:
            if current_thought is not None:
                merged_content = f"<thought>{current_thought}</thought>\n\n{msg_content}"
                lcm_msgs.append({
                    "role": "assistant",
                    "content": merged_content,
                })
                current_thought = None
            else:
                lcm_msgs.append({
                    "role": "assistant",
                    "content": msg_content,
                })
        else:
            flush_assistant()
            role_val = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            lcm_msgs.append({
                "role": role_val,
                "content": msg_content,
            })

    flush_assistant()
    return lcm_msgs


def openlcm_dicts_to_messages(
    dicts: list[dict[str, Any]],
    session_id: str,
    chatbot_id: str,
    channel_id: str,
    workspace_name: str | None = None,
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
        workspace_name: Optional workspace name to restore $STAGING_DIR paths.

    Returns:
        A reconstructed list of Kesoku Message objects.
    """
    if workspace_name:
        from kesoku.config import get_config
        cfg = get_config()
        staging_dir = os.path.realpath(os.path.join(cfg.workspace.sessions_dir, workspace_name))

        restored_dicts = []
        for d in dicts:
            d_copy = dict(d)
            if "content" in d_copy and d_copy["content"]:
                d_copy["content"] = d_copy["content"].replace("$STAGING_DIR", staging_dir)
            if "tool_calls" in d_copy and d_copy["tool_calls"]:
                tcs_copy = []
                for tc in d_copy["tool_calls"]:
                    tc_copy = dict(tc)
                    if "function" in tc_copy and tc_copy["function"]:
                        fn_copy = dict(tc_copy["function"])
                        if "arguments" in fn_copy and fn_copy["arguments"]:
                            fn_copy["arguments"] = fn_copy["arguments"].replace("$STAGING_DIR", staging_dir)
                        tc_copy["function"] = fn_copy
                    tcs_copy.append(tc_copy)
                d_copy["tool_calls"] = tcs_copy
            restored_dicts.append(d_copy)
        dicts = restored_dicts

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
            tool_call_id = d.get("tool_call_id") or ""
            metadata["tool_name"] = d.get("name") or "unknown_tool"
            metadata["tool_call_id"] = tool_call_id
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
                    parent_id=tool_call_id or None,
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
                    tc_id = tc.get("id") or ""
                    ts_hex = tc.get("thought_signature")
                    msg_kwargs = {
                        "session_id": session_id,
                        "chatbot_id": chatbot_id,
                        "channel_id": channel_id,
                        "sender": sender,
                        "role": MessageRole.TOOL,
                        "type": MessageType.TOOL_CALL,
                        "content": f"Calling tool `{fn_name}` with arguments:\n```json\n{call_args_json}\n```",
                        "metadata": {
                            "tool_name": fn_name,
                            "tool_arguments": fn_args,
                            "tool_call_id": tc_id,
                        },
                        "status": MessageStatus.RESPONDED,
                    }
                    if ts_hex:
                        msg_kwargs["metadata"]["thought_signature"] = ts_hex
                    if tc_id:
                        msg_kwargs["id"] = tc_id
                    msgs.append(Message(**msg_kwargs))
            else:
                match = re.match(r"^<thought>(.*?)</thought>\s*(.*)$", content, re.DOTALL)
                if match:
                    thought_content = match.group(1)
                    reply_content = match.group(2)
                    msgs.append(
                        Message(
                            session_id=session_id,
                            chatbot_id=chatbot_id,
                            channel_id=channel_id,
                            sender=sender,
                            role=MessageRole.ASSISTANT,
                            type=MessageType.THOUGHT,
                            content=thought_content,
                            status=MessageStatus.RESPONDED,
                        )
                    )
                    if reply_content.strip():
                        msgs.append(
                            Message(
                                session_id=session_id,
                                chatbot_id=chatbot_id,
                                channel_id=channel_id,
                                sender=sender,
                                role=role,
                                type=MessageType.TEXT,
                                content=reply_content.strip(),
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

