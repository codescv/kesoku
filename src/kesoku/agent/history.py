"""History cleaning and optimization utilities for Kesoku."""

import datetime
import logging
import os
import re
import tempfile
import time
import uuid
from typing import Literal

import tzlocal

from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.gateway.gateway import Gateway
from kesoku.utils.path import PathResolver

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


def segment_logical_turns(history: list[Message]) -> list[list[Message]]:
    """Segment messages into logical turns starting with a USER or SYSTEM message.

    Internal notification assistant messages are excluded.
    """
    turns: list[list[Message]] = []
    current_turn: list[Message] = []
    for m in history:
        if m.role == MessageRole.ASSISTANT and m.sender == "Notification":
            continue
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
    return turns


def prepare_history_for_llm(
    history: list[Message],
    max_historical_tool_chars: int = 500,
    staging_dir: str | None = None,
) -> list[Message]:
    """Clean up and format the conversational history specifically for the LLM.

    Groups messages into complete logical turns, strips thoughts from all completed
    turns, truncates historical tool outputs exceeding max_historical_tool_chars to external files,
    strips attachments from historical user prompts, adds headers to user
    messages, and returns the simplified history.

    Args:
        history: Flat chronological list of database messages.
        max_historical_tool_chars: Maximum character limit for tool results in historical turns.
            Defaults to 500 characters.
        staging_dir: Optional session staging directory to write full outputs to.
    """
    # 1. Groups messages into complete logical turns (User/System prompt -> ... -> before next user/system prompt).
    turns = segment_logical_turns(history)

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
            # Completed turn: strip thoughts, signatures, truncate tool results, and clean historical user attachments
            clean_turn = []
            for m in turn:
                # Drop assistant thoughts in completed turns
                if m.role == MessageRole.ASSISTANT and m.type == MessageType.THOUGHT:
                    continue

                # Drop thought signature from historical tool call messages to optimize context size
                if m.role == MessageRole.TOOL and m.type == MessageType.TOOL_CALL:
                    if m.metadata and "thought_signature" in m.metadata:
                        # Copy to avoid mutating shared state in-memory
                        m = m.model_copy()
                        m.metadata = dict(m.metadata)
                        m.metadata.pop("thought_signature", None)

                # Truncate tool results in completed turns if they exceed max_historical_tool_chars (default 500)
                if m.role == MessageRole.TOOL and m.type == MessageType.TOOL_RESULT:
                    tool_name = (m.metadata.get("tool_name") if m.metadata else None) or m.sender or "tool"
                    tool_error = m.metadata.get("tool_error") if m.metadata else None
                    raw_res = (m.metadata.get("tool_result") if m.metadata else None) or m.content or ""

                    # Resolve target staging directory
                    target_dir = staging_dir
                    if not target_dir and m.session_id:
                        try:
                            target_dir = PathResolver.get_session_staging_dir(m.session_id)
                        except Exception:
                            target_dir = None
                    if not target_dir:
                        target_dir = os.path.join(tempfile.gettempdir(), "kesoku_staging")

                    if tool_error and len(tool_error) > max_historical_tool_chars:
                        os.makedirs(target_dir, exist_ok=True)
                        file_match = re.search(r"saved to(?: session workspace file)?: `([^`]+)`", tool_error)
                        if file_match and os.path.exists(file_match.group(1)):
                            err_filepath = file_match.group(1)
                        else:
                            timestamp = int(time.time())
                            err_filename = f"hist_err_{tool_name}_{timestamp}_{uuid.uuid4().hex[:8]}.txt"
                            err_filepath = os.path.join(target_dir, err_filename)
                            try:
                                with open(err_filepath, "w", encoding="utf-8") as f:
                                    f.write(tool_error)
                            except Exception as ex:
                                logger.warning(f"Failed to write historical tool error to {err_filepath}: {ex}")

                        err_preview = tool_error[-max_historical_tool_chars:]
                        truncated_err = (
                            f"Error truncated for history (total length {len(tool_error)} chars). "
                            f"Full error log saved to: `{err_filepath}`.\n\n"
                            f"Preview (last {max_historical_tool_chars} chars):\n{err_preview}"
                        )
                        m = m.model_copy()
                        m.metadata = dict(m.metadata) if m.metadata else {}
                        m.metadata["tool_error"] = truncated_err
                        m.metadata["output_filepath"] = err_filepath
                        m.content = f"Tool `{tool_name}` error:\n```\n{truncated_err}\n```"

                    elif not tool_error and len(raw_res) > max_historical_tool_chars:
                        os.makedirs(target_dir, exist_ok=True)
                        file_match = re.search(r"saved to(?: session workspace file)?: `([^`]+)`", raw_res)
                        if file_match and os.path.exists(file_match.group(1)):
                            filepath = file_match.group(1)
                        else:
                            timestamp = int(time.time())
                            filename = f"hist_tool_{tool_name}_{timestamp}_{uuid.uuid4().hex[:8]}.txt"
                            filepath = os.path.join(target_dir, filename)
                            try:
                                with open(filepath, "w", encoding="utf-8") as f:
                                    f.write(raw_res)
                            except Exception as ex:
                                logger.warning(f"Failed to write historical tool output to {filepath}: {ex}")

                        preview = raw_res[-max_historical_tool_chars:]
                        truncated_res = (
                            f"Output truncated for history (total length {len(raw_res)} chars). "
                            f"Full output saved to: `{filepath}`.\n\n"
                            f"Preview (last {max_historical_tool_chars} chars):\n{preview}"
                        )
                        m = m.model_copy()
                        m.metadata = dict(m.metadata) if m.metadata else {}
                        m.metadata["tool_result"] = truncated_res
                        m.metadata["output_filepath"] = filepath
                        m.content = f"Tool `{tool_name}` returned:\n```\n{truncated_res}\n```"

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
                # Don't add header to the custom scaffold message
                if "[Note: This conversation uses custom turn-based" in m.content:
                    final_history.append(m)
                elif "<current_message" in m.content:
                    final_history.append(m)
                else:
                    m_copy = m.model_copy()
                    msg_time = datetime.datetime.fromtimestamp(m_copy.timestamp).astimezone()
                    time_str = msg_time.strftime("%Y-%m-%d %H:%M:%S (%A) %Z")
                    sender_name = m_copy.metadata.get("sender_name") or m_copy.sender
                    if sender_name.lower() == "cronjob":
                        sender_name = "system"
                    try:
                        tz_name = tzlocal.get_localzone_name()
                    except Exception:
                        tz_name = msg_time.tzname() or "UTC"

                    m_copy.content = (
                        f'<history_message from="{sender_name}" time="{time_str}" timezone="{tz_name}">\n'
                        f"{m_copy.content}\n"
                        "</history_message>"
                    )
                    final_history.append(m_copy)
            else:
                final_history.append(m)

    return final_history
