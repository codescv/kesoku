"""History cleaning and optimization utilities for Kesoku."""

import logging
import os
import time
from typing import Any

from kesoku.config import get_config
from kesoku.constants import (
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL,
    ROLE_USER,
    STATUS_RESPONDED,
    TYPE_THOUGHT,
    TYPE_TOOL_CALL,
    TYPE_TOOL_RESULT,
)
from kesoku.db import Message
from kesoku.gateway.gateway import Gateway

logger = logging.getLogger(__name__)


def group_tool_results_by_llm_call(turn_msgs: list[Message]) -> list[list[Message]]:
    """Group tool result messages in the active turn by their corresponding LLM call."""
    # 1. Map each tool call id to its message
    tc_map = {m.id: m for m in turn_msgs if m.type == TYPE_TOOL_CALL}

    # 2. Get all tool results sorted by parent tool call timestamp
    tr_msgs = [m for m in turn_msgs if m.type == TYPE_TOOL_RESULT]
    tr_msgs.sort(key=lambda m: tc_map[m.parent_id].timestamp if m.parent_id in tc_map else m.timestamp)

    # 3. Group by parent tool call timestamp threshold (e.g., 0.5 seconds)
    batches: list[list[Message]] = []
    current_batch: list[Message] = []
    last_ts = None

    for tr in tr_msgs:
        parent_tc = tc_map.get(tr.parent_id)
        ts = parent_tc.timestamp if parent_tc else tr.timestamp

        if last_ts is None:
            current_batch.append(tr)
        elif ts - last_ts < 0.5:
            current_batch.append(tr)
        else:
            if current_batch:
                batches.append(current_batch)
            current_batch = [tr]
        last_ts = ts

    if current_batch:
        batches.append(current_batch)
    return batches


async def build_clean_history(
    gateway: Gateway,
    session_id: str,
    max_turns: int | None = None,
    pin_initial_turns: int | None = None,
    pin_recent_turns: int | None = None,
) -> list[Message]:
    """Retrieve, clean up, and format the conversational history for the LLM.

    Resolves orphaned tool calls, handles initial turns pinning, applies priority-based turn dropping,
    recovers loaded skills, and slides the turn window.

    Example Turn-Based Truncation for 100 Turns (max_turns=30, pin_initial_turns=3, pin_recent_turns=10):
    - System Prompt (kept at history[0])
    - Pinned initial Turns 1, 2, and 3 (retained in full)
    - Pinned recovered skill Turns (e.g., Turn 5 that loaded 'role-playing', recovered in full)
    - Candidate Turns 74 to 90 (stripped of thoughts and resolved tools, keeping only user/assistant text)
    - Candidate Turns 91 to 100 (kept in 100% full execution detail: prompts, thoughts, tool calls/results)

    Args:
        gateway: Gateway instance to interact with storage.
        session_id: Unique conversational session identifier.
        max_turns: Maximum logical turns allowed in context history. If None, uses config setting.
        pin_initial_turns: Number of initial turns to pin at the start. If None, uses config setting.
        pin_recent_turns: Number of latest turns to keep in full detail. If None, uses config setting.

    Returns:
        A list of cleanly structured, prioritized, and aligned Message objects for the LLM.
    """
    cfg = get_config().agent.history

    if max_turns is None:
        max_turns = cfg.max_turns
    if pin_initial_turns is None:
        pin_initial_turns = cfg.pin_initial_turns
    if pin_recent_turns is None:
        pin_recent_turns = cfg.pin_recent_turns

    # 1. Detects and heals orphaned tool calls by posting a synthesized interruption result.
    raw_history = await gateway.get_session_history(session_id, limit=0)
    tool_calls = [m for m in raw_history if m.type == TYPE_TOOL_CALL]
    tool_results_parent_ids = {m.parent_id for m in raw_history if m.type == TYPE_TOOL_RESULT and m.parent_id}

    healed = False
    for tc in tool_calls:
        if tc.id not in tool_results_parent_ids:
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
                role=ROLE_TOOL,
                type=TYPE_TOOL_RESULT,
                content=f"Tool `{tool_name}` execution was interrupted due to service restart.",
                status=STATUS_RESPONDED,
                parent_id=tc.id,
                metadata={
                    "tool_name": tool_name,
                    "tool_error": "Tool execution was interrupted due to service restart.",
                },
            )
            await gateway.post(interrupted_msg)
            healed = True

    if healed:
        raw_history = await gateway.get_session_history(session_id, limit=0)

    # 2. Always preserves the initial system message(s) at the start.
    system_msg = None
    for m in raw_history:
        if m.role == ROLE_SYSTEM:
            system_msg = m
            break

    conv_msgs = [m for m in raw_history if (not system_msg or m.id != system_msg.id)]

    # 3. Groups messages into complete logical turns (User prompt -> ... -> before next user prompt).
    turns: list[list[Message]] = []
    current_turn: list[Message] = []
    for m in conv_msgs:
        if m.role in (ROLE_USER, ROLE_SYSTEM):
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

    # 4. Clean each turn logically based on its position (pinned, latest, or in-between other turns).
    tc_to_tr = {}
    for m in raw_history:
        if m.type == TYPE_TOOL_RESULT and m.parent_id:
            tc_to_tr[m.parent_id] = m

    turn_to_tcs = {}
    for m in raw_history:
        if m.type == TYPE_TOOL_CALL and m.parent_id:
            turn_to_tcs.setdefault(m.parent_id, []).append(m)

    resolved_turns_without_skill = set()
    for user_msg_id, tcs in turn_to_tcs.items():
        is_resolved = all(tc.id in tc_to_tr for tc in tcs)
        has_skill = any(tc.metadata.get("tool_name") == "use_skill" for tc in tcs)
        if is_resolved and not has_skill:
            resolved_turns_without_skill.add(user_msg_id)

    cleaned_turns = []
    for idx, turn in enumerate(turns):
        is_pinned = (idx < pin_initial_turns)
        is_latest = (idx == len(turns) - 1)

        if is_latest:
            # Latest turn: full details
            cleaned_turns.append(turn)
            continue

        dropped_ids = set()
        if is_pinned:
            # pinned turns: only keep user, assistant and tool messages with skill.
            # thoughts must be dropped (thoughts: only in latest turns. drop in all others).
            for m in turn:
                if m.type == TYPE_THOUGHT:
                    dropped_ids.add(m.id)
                    continue
                if m.role in (ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM):
                    continue
                if m.role == ROLE_TOOL:
                    tool_name = m.metadata.get("tool_name")
                    if tool_name == "use_skill":
                        continue
                dropped_ids.add(m.id)
        else:
            # in-between other turns: drop thoughts, resolved intermediate tool calls/results
            user_prompt = next((m for m in turn if m.role in (ROLE_USER, ROLE_SYSTEM)), None)
            user_prompt_id = user_prompt.id if user_prompt else None

            for m in turn:
                if m.role == ROLE_ASSISTANT and m.type == TYPE_THOUGHT:
                    dropped_ids.add(m.id)
                    continue
                if m.type == TYPE_TOOL_CALL and user_prompt_id and user_prompt_id in resolved_turns_without_skill:
                    dropped_ids.add(m.id)
                    continue
                if m.type == TYPE_TOOL_RESULT and m.parent_id:
                    parent_tc = next((tc for tcs in turn_to_tcs.values() for tc in tcs if tc.id == m.parent_id), None)
                    if parent_tc and parent_tc.parent_id in resolved_turns_without_skill:
                        dropped_ids.add(m.id)
                        continue

        clean_turn = [m for m in turn if m.id not in dropped_ids]
        cleaned_turns.append(clean_turn)

    # 5. Partitions pinned turns vs candidate turns
    pinned_turns = cleaned_turns[:pin_initial_turns]

    # 6. Trims/aligns history strictly by turn count, naturally preserving user-message start.
    allowed_turns = max_turns - pin_initial_turns
    if allowed_turns <= 0:
        suffix_turns = []
        discarded_candidate_turns = cleaned_turns[pin_initial_turns:]
    else:
        suffix_idx = max(0, len(cleaned_turns) - pin_initial_turns - allowed_turns)
        suffix_turns = cleaned_turns[pin_initial_turns + suffix_idx:]
        discarded_candidate_turns = cleaned_turns[pin_initial_turns:pin_initial_turns + suffix_idx]

    # 7. Recovers any pinned skill use (use_skill) turns completely and atomically.
    recovered_turns = []
    for turn in discarded_candidate_turns:
        has_completed_skill = False
        for m in turn:
            if (
                m.type == TYPE_TOOL_RESULT
                and m.metadata.get("tool_name") == "use_skill"
                and "tool_error" not in m.metadata
            ):
                has_completed_skill = True
                break
        if has_completed_skill:
            recovered_turns.append(turn)

    # 8. Flatten turns and construct the final chronological context history.
    final_history = []
    if system_msg:
        final_history.append(system_msg)
    for turn in pinned_turns:
        final_history.extend(turn)
    for turn in recovered_turns:
        final_history.extend(turn)
    for turn in suffix_turns:
        final_history.extend(turn)

    # 9. Context Optimization: Serialize tool outputs to files and replace with pointer messages
    last_user_idx = -1
    for i, msg in enumerate(final_history):
        if msg.role == ROLE_USER:
            last_user_idx = i

    if last_user_idx != -1:
        session = await gateway.get_session(session_id)
        if session:
            app_cfg = get_config()
            staging_dir = os.path.realpath(
                os.path.join(app_cfg.workspace.sessions_dir, session.workspace_name)
            )
            os.makedirs(staging_dir, exist_ok=True)

            historical_msgs = final_history[:last_user_idx]
            active_msgs = final_history[last_user_idx:]

            # Optimize historical turns
            if cfg.serialize_historical_tool_results:
                for msg in historical_msgs:
                    if msg.type == TYPE_TOOL_RESULT:
                        if msg.metadata.get("tool_name") == "use_skill":
                            continue
                        raw_output = (
                            msg.metadata.get("tool_result")
                            or msg.metadata.get("tool_error")
                            or msg.content
                        )
                        if len(raw_output) > cfg.serialize_tool_results_threshold:
                            file_path = os.path.join(staging_dir, f"tool_output_{msg.id}.txt")
                            if not os.path.exists(file_path):
                                with open(file_path, "w", encoding="utf-8") as f:
                                    f.write(raw_output)

                            msg.content = f"tool output in {file_path}"
                            if "tool_result" in msg.metadata:
                                msg.metadata["tool_result"] = f"tool output in {file_path}"
                            if "tool_error" in msg.metadata:
                                msg.metadata["tool_error"] = f"tool output in {file_path}"

            # Optimize active turn
            if cfg.active_turn_keep_tool_results_for_k_recent_calls >= 0:
                batches = group_tool_results_by_llm_call(active_msgs)
                if len(batches) > cfg.active_turn_keep_tool_results_for_k_recent_calls:
                    if cfg.active_turn_keep_tool_results_for_k_recent_calls == 0:
                        batches_to_serialize = batches
                    else:
                        batches_to_serialize = batches[:-cfg.active_turn_keep_tool_results_for_k_recent_calls]
                    for batch in batches_to_serialize:
                        for msg in batch:
                            if msg.metadata.get("tool_name") == "use_skill":
                                continue
                            raw_output = (
                                msg.metadata.get("tool_result")
                                or msg.metadata.get("tool_error")
                                or msg.content
                            )
                            if len(raw_output) > cfg.serialize_tool_results_threshold:
                                file_path = os.path.join(staging_dir, f"tool_output_{msg.id}.txt")
                                if not os.path.exists(file_path):
                                    with open(file_path, "w", encoding="utf-8") as f:
                                        f.write(raw_output)

                                msg.content = f"tool output in {file_path}"
                                if "tool_result" in msg.metadata:
                                    msg.metadata["tool_result"] = f"tool output in {file_path}"
                                if "tool_error" in msg.metadata:
                                    msg.metadata["tool_error"] = f"tool output in {file_path}"

    return final_history
