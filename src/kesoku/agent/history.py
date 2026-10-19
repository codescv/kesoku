"""History cleaning and optimization utilities for Kesoku."""

import datetime
import logging
import os
import re
import tempfile
import uuid
from typing import Literal

import tzlocal

from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.gateway.gateway import Gateway
from kesoku.utils.path import PathResolver
from kesoku.utils.text import truncate_middle
from kesoku.utils.ui_markup import collapse_ui_blocks

logger = logging.getLogger(__name__)

# Tools whose (potentially long) file-content arguments are truncated once their turn is completed.
# The written content already lives on disk (and in session notes), so re-sending it every turn is waste.
# `update_session_notes` is no longer a tool, but legacy calls to it in stored history must still be truncated.
TRUNCATED_ARG_TOOLS = frozenset({"update_session_notes", "update_file"})
MAX_HISTORICAL_TOOL_ARG_CHARS = 300

# run_shell_command's own truncation banner ("Output truncated (total length N bytes). Full output saved to ...
# Preview (...):\n"). Stripped before building the history preview so its head budget shows real output.
_SHELL_TRUNCATION_BANNER_RE = re.compile(
    r"\AOutput truncated \(total length \d+ bytes\)\..*?Preview \([^)\n]*\):\n", re.DOTALL
)
_COMMAND_FAILED_PREFIX = "Command failed with exit code"
HISTORY_PREVIEW_PLACEHOLDER = "\n... <middle truncated> ...\n"


def strip_shell_truncation_banner(text: str) -> str:
    """Remove run_shell_command's truncation banner, keeping a leading `Command failed with exit code N.` line.

    Args:
        text: A tool result or tool error string.

    Returns:
        The text without the banner (unchanged if there is none).
    """
    if not text:
        return text
    prefix = ""
    body = text
    if body.startswith(_COMMAND_FAILED_PREFIX):
        first, sep, rest = body.partition("\n")
        prefix, body = first + sep, rest
    stripped = _SHELL_TRUNCATION_BANNER_RE.sub("", body, count=1)
    return prefix + stripped


def history_preview(text: str, max_chars: int) -> str:
    """Head + tail preview of a historical tool output (shell banner stripped, middle truncated)."""
    return truncate_middle(strip_shell_truncation_banner(text), max_chars, HISTORY_PREVIEW_PLACEHOLDER)


# Matches `[question: <stem> || <options>]` (or the legacy `<stem> | <opt> | ...` form) blocks.
# Nested brackets are not allowed inside question blocks, so `[^\]]` is a safe body matcher.
QUESTION_BLOCK_RE = re.compile(r"\[question:\s*[^\]]*\]")
QUESTION_PLACEHOLDER = "[question: ...]"


def strip_question_options(text: str) -> str:
    """Collapse every `[question: <stem> || <options>]` block in text into `[question: ...]`.

    Old question stems/options re-entering the context verbatim are a strong self-imitation source
    (near-identical stems and simile-laden options across dozens of turns), so they are elided from
    older completed turns while the turn structure (a question was asked) is preserved.
    """
    if not text or "[question:" not in text:
        return text
    return QUESTION_BLOCK_RE.sub(QUESTION_PLACEHOLDER, text)


def _hist_filename(kind: str, tool_name: str, msg: Message) -> str:
    """Stable staging filename for a truncated historical tool output, derived from the message id."""
    safe_tool = re.sub(r"[^A-Za-z0-9_.-]", "_", str(tool_name))
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(msg.id or uuid.uuid5(uuid.NAMESPACE_OID, msg.content or "")))
    return f"hist_{kind}_{safe_tool}_{safe_id}.txt"


def truncate_tool_call_arguments(tool_name: str | None, args: dict) -> dict:
    """Truncate long string arguments of historical file/notes tool calls.

    Returns the original dict unchanged when nothing needs truncation, otherwise a truncated copy.
    """
    if tool_name not in TRUNCATED_ARG_TOOLS:
        return args
    if not any(isinstance(v, str) and len(v) > MAX_HISTORICAL_TOOL_ARG_CHARS for v in args.values()):
        return args
    return {
        k: (
            v[:MAX_HISTORICAL_TOOL_ARG_CHARS] + f"…[truncated for history, {len(v)} chars total]"
            if isinstance(v, str) and len(v) > MAX_HISTORICAL_TOOL_ARG_CHARS
            else v
        )
        for k, v in args.items()
    }


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
    strip_old_questions: bool = True,
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
        strip_old_questions: When True, collapse `[question: ...]` and ```ui blocks (keeping the prompt and the
            user's answer when determinable) in assistant messages of
            completed turns older than the last completed turn. The last completed turn keeps its
            options so short/numeric replies ("选2") and clicked-option detection still work.
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
        # Turns strictly older than the last completed turn (idx == num_turns - 2) get their options elided.
        strip_questions_here = strip_old_questions and idx < num_turns - 2
        # The user's reply to this turn's question (first message of the next turn), used to keep the answer.
        next_turn_first = turns[idx + 1][0] if idx + 1 < num_turns and turns[idx + 1] else None
        next_user_text = None
        if next_turn_first is not None and next_turn_first.role == MessageRole.USER:
            next_user_text = next_turn_first.content

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

                # Elide question stems/options of older completed turns (anti self-imitation)
                if strip_questions_here and m.role == MessageRole.ASSISTANT and m.content:
                    stripped_content = collapse_ui_blocks(strip_question_options(m.content), next_user_text)
                    if stripped_content != m.content:
                        m = m.model_copy()
                        m.content = stripped_content

                # Drop thought signature from historical tool call messages to optimize context size,
                # and truncate long file-content arguments of file/notes tools.
                if m.role == MessageRole.TOOL and m.type == MessageType.TOOL_CALL:
                    if m.metadata and "thought_signature" in m.metadata:
                        # Copy to avoid mutating shared state in-memory
                        m = m.model_copy()
                        m.metadata = dict(m.metadata)
                        m.metadata.pop("thought_signature", None)
                    args = m.metadata.get("tool_arguments") if m.metadata else None
                    if isinstance(args, dict):
                        truncated_args = truncate_tool_call_arguments(m.metadata.get("tool_name"), args)
                        if truncated_args is not args:
                            m = m.model_copy()
                            m.metadata = dict(m.metadata)
                            m.metadata["tool_arguments"] = truncated_args

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
                            # Deterministic name (per message) keeps the truncated preview text byte-identical
                            # across calls, which implicit prefix caching depends on.
                            err_filepath = os.path.join(target_dir, _hist_filename("err", tool_name, m))
                            try:
                                with open(err_filepath, "w", encoding="utf-8") as f:
                                    f.write(tool_error)
                            except Exception as ex:
                                logger.warning(f"Failed to write historical tool error to {err_filepath}: {ex}")

                        err_preview = history_preview(tool_error, max_historical_tool_chars)
                        truncated_err = (
                            f"Error truncated for history (total length {len(tool_error)} chars). "
                            f"Full error log saved to: `{err_filepath}`.\n\n"
                            f"Preview (head and tail, {max_historical_tool_chars} chars):\n{err_preview}"
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
                            # Deterministic name (per message) keeps the truncated preview text byte-identical
                            # across calls, which implicit prefix caching depends on.
                            filepath = os.path.join(target_dir, _hist_filename("tool", tool_name, m))
                            try:
                                with open(filepath, "w", encoding="utf-8") as f:
                                    f.write(raw_res)
                            except Exception as ex:
                                logger.warning(f"Failed to write historical tool output to {filepath}: {ex}")

                        preview = history_preview(raw_res, max_historical_tool_chars)
                        truncated_res = (
                            f"Output truncated for history (total length {len(raw_res)} chars). "
                            f"Full output saved to: `{filepath}`.\n\n"
                            f"Preview (head and tail, {max_historical_tool_chars} chars):\n{preview}"
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
