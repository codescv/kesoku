"""History cleaning and optimization utilities for Kesoku."""

import logging
import os
from typing import Literal

from kesoku.config import get_config
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.gateway.gateway import Gateway

logger = logging.getLogger(__name__)


def group_tool_results_by_llm_call(turn_msgs: list[Message]) -> list[list[Message]]:
    """Group tool result messages in the active turn by their corresponding LLM call."""
    # 1. Map each tool call id to its message
    tc_map = {m.id: m for m in turn_msgs if m.type == MessageType.TOOL_CALL}

    # 2. Get all tool results sorted by parent tool call timestamp
    tr_msgs = [m for m in turn_msgs if m.type == MessageType.TOOL_RESULT]
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
    order: Literal["phased", "grouped"] = "phased",
) -> list[Message]:
    """Retrieve, clean up, and format the conversational history for the LLM.

    Resolves orphaned tool calls, deduplicates duplicate interrupted tool results, handles initial turns
    pinning, applies priority-based turn dropping, recovers loaded skills, and slides the turn window.

    Example Turn-Based Truncation for 100 Turns (max_turns=30, pin_initial_turns=3, pin_recent_turns=10):
    - System Prompt (kept at history[0])
    - Pinned initial Turns 1, 2, and 3 (retains user, assistant, system, and all tool
      calls/results; thoughts are dropped)
    - Pinned recovered skill Turns (e.g., Turn 5 that loaded 'role-playing', recovered in
      full, use_skill not serialized)
    - Candidate Turns 74 to 89 (drops thoughts, drops resolved intermediate tool
      calls/results; keeps user/assistant/system text)
    - Completed Recent Turns 90 to 98 (retains user, assistant, system, and all
      tool calls/results; thoughts are dropped)
    - Absolute Latest Turn 99 (kept in 100% full execution detail: prompts,
      thoughts, tool calls/results)

    Args:
        gateway: Gateway instance to interact with storage.
        session_id: Unique conversational session identifier.
        max_turns: Maximum logical turns allowed in context history. If None, uses config setting.
        pin_initial_turns: Number of initial turns to pin at the start. If None, uses config setting.
        pin_recent_turns: Number of latest turns to keep in full detail. If None, uses config setting.
        order: Sorting mechanism ("phased" or "grouped").

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

    # 1. Detects and heals orphaned tool calls using optimized database query.
    orphaned_calls = await gateway.get_orphaned_tool_calls(session_id)
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

    # 2. Retrieve chronological turn anchors and identify relevant turn segments
    anchors = await gateway.get_session_turn_anchors(session_id)
    system_anchors = [a for a in anchors if a["role"] == MessageRole.SYSTEM]
    user_anchors = [a for a in anchors if a["role"] == MessageRole.USER]

    # 3. Retrieve completed skill turn anchors
    skill_anchor_ids = set(await gateway.get_session_skill_anchor_ids(session_id))

    # 4. Determine the exact subset of anchors we want to load in full detail
    selected_anchor_ids = set()
    selected_anchor_ids.update(a["id"] for a in system_anchors)
    selected_anchor_ids.update(a["id"] for a in user_anchors[:pin_initial_turns])
    selected_anchor_ids.update(a["id"] for a in user_anchors if a["id"] in skill_anchor_ids)

    # Include suffix sliding window turns (up to max_turns - pin_initial_turns)
    suffix_limit = max(0, max_turns - pin_initial_turns)
    if suffix_limit > 0:
        selected_anchor_ids.update(a["id"] for a in user_anchors[-suffix_limit:])

    # 5. Construct ranges for the selected anchors
    ranges = []
    for idx, anchor in enumerate(anchors):
        if anchor["id"] in selected_anchor_ids:
            start = anchor["timestamp"]
            end = anchors[idx + 1]["timestamp"] if idx + 1 < len(anchors) else None
            ranges.append((start, end))

    # 6. Load messages only for the selected turn anchors using timestamp ranges
    raw_history = await gateway.get_session_history_by_ranges(session_id, ranges, order=order)

    # 6. Always preserves the initial system message(s) at the start.
    system_msg = None
    for m in raw_history:
        if m.role == MessageRole.SYSTEM:
            system_msg = m
            break

    conv_msgs = [m for m in raw_history if (not system_msg or m.id != system_msg.id)]

    # 7. Groups messages into complete logical turns (User prompt -> ... -> before next user prompt).
    turns: list[list[Message]] = []
    current_turn: list[Message] = []
    for m in conv_msgs:
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

    # 8. Resolve resolved intermediate tool call/result status for older turns
    tc_to_tr = {}
    for m in raw_history:
        if m.type == MessageType.TOOL_RESULT and m.parent_id:
            tc_to_tr[m.parent_id] = m

    turn_to_tcs = {}
    for m in raw_history:
        if m.type == MessageType.TOOL_CALL and m.parent_id:
            turn_to_tcs.setdefault(m.parent_id, []).append(m)

    resolved_turns_without_skill = set()
    for user_msg_id, tcs in turn_to_tcs.items():
        is_resolved = all(tc.id in tc_to_tr for tc in tcs)
        has_skill = any(tc.metadata.get("tool_name") == "use_skill" for tc in tcs)
        if is_resolved and not has_skill:
            resolved_turns_without_skill.add(user_msg_id)

    # 9. Clean each turn logically based on its position
    pinned_anchor_ids = {a["id"] for a in user_anchors[:pin_initial_turns]}
    recent_anchor_ids = {a["id"] for a in user_anchors[-pin_recent_turns:]}
    latest_anchor_id = user_anchors[-1]["id"] if user_anchors else None

    cleaned_turns = []
    for turn in turns:
        user_prompt = next((m for m in turn if m.role in (MessageRole.USER, MessageRole.SYSTEM)), None)
        user_prompt_id = user_prompt.id if user_prompt else None

        is_latest = (user_prompt_id == latest_anchor_id)
        is_pinned = (user_prompt_id in pinned_anchor_ids)
        is_recent = (user_prompt_id in recent_anchor_ids)

        dropped_ids = set()

        # Deduplicate duplicate/orphaned interrupted tool results for the same tool call
        tc_results = {}
        for m in turn:
            if m.type == MessageType.TOOL_RESULT and m.parent_id:
                tc_results.setdefault(m.parent_id, []).append(m)

        for parent_id, results in tc_results.items():
            if len(results) > 1:
                interrupted_msg = None
                has_valid_result = False
                for r in results:
                    if r.metadata.get("tool_error") == "Tool execution was interrupted due to service restart.":
                        interrupted_msg = r
                    else:
                        has_valid_result = True

                if interrupted_msg and has_valid_result:
                    dropped_ids.add(interrupted_msg.id)

        if is_latest:
            # Latest turn: keep full details (thoughts, tool calls/results)
            clean_turn = [m for m in turn if m.id not in dropped_ids]
            cleaned_turns.append(clean_turn)
        elif is_pinned or is_recent:
            # Pinned and recent turns: keep user, assistant, system, and all tool messages (serialize if needed).
            # Thoughts must be dropped.
            for m in turn:
                if m.type == MessageType.THOUGHT:
                    dropped_ids.add(m.id)
                    continue
                if m.role in (MessageRole.USER, MessageRole.ASSISTANT, MessageRole.SYSTEM, MessageRole.TOOL):
                    continue
                dropped_ids.add(m.id)
            clean_turn = [m for m in turn if m.id not in dropped_ids]
            cleaned_turns.append(clean_turn)
        else:
            # in-between other turns: drop thoughts, resolved intermediate tool calls/results
            for m in turn:
                if m.role == MessageRole.ASSISTANT and m.type == MessageType.THOUGHT:
                    dropped_ids.add(m.id)
                    continue
                if (
                    m.type == MessageType.TOOL_CALL
                    and user_prompt_id
                    and user_prompt_id in resolved_turns_without_skill
                ):
                    dropped_ids.add(m.id)
                    continue
                if m.type == MessageType.TOOL_RESULT and m.parent_id:
                    parent_tc = next((tc for tcs in turn_to_tcs.values() for tc in tcs if tc.id == m.parent_id), None)
                    if parent_tc and parent_tc.parent_id in resolved_turns_without_skill:
                        dropped_ids.add(m.id)
                        continue
            clean_turn = [m for m in turn if m.id not in dropped_ids]
            cleaned_turns.append(clean_turn)

        # Strip attachments for historical user turns to optimize context size
        if not is_latest:
            for m in cleaned_turns[-1]:
                if m.role == MessageRole.USER:
                    attachments = m.metadata.get("attachments")
                    if attachments:
                        filenames = [os.path.basename(att.get("path", "file")) for att in attachments]
                        placeholder = f"\n\n[Attachments stripped from history: {', '.join(filenames)}]"
                        if placeholder not in m.content:
                            m.content += placeholder
                        m.metadata.pop("attachments", None)


    # 9. Flatten turns and construct the final chronological context history.
    final_history = []
    if system_msg:
        final_history.append(system_msg)
    for turn in cleaned_turns:
        final_history.extend(turn)

    # 10. Context Optimization: Serialize tool outputs to files and replace with pointer messages
    last_user_idx = -1
    for i, msg in enumerate(final_history):
        if msg.role == MessageRole.USER:
            last_user_idx = i

    if last_user_idx != -1:
        session = await gateway.get_session(session_id)
        if session:
            app_cfg = get_config()
            staging_dir = os.path.realpath(  # noqa: ASYNC240
                os.path.join(app_cfg.workspace.sessions_dir, session.workspace_name)
            )
            os.makedirs(staging_dir, exist_ok=True)

            historical_msgs = final_history[:last_user_idx]
            active_msgs = final_history[last_user_idx:]

            # Optimize historical turns
            if cfg.serialize_historical_tool_results:
                for msg in historical_msgs:
                    if msg.type == MessageType.TOOL_RESULT:
                        if msg.metadata.get("tool_name") == "use_skill":
                            continue
                        raw_output = (
                            msg.metadata.get("tool_result")
                            or msg.metadata.get("tool_error")
                            or msg.content
                        )
                        if len(raw_output) > cfg.serialize_tool_results_threshold:
                            file_path = os.path.join(staging_dir, f"tool_output_{msg.id}.txt")
                            if not os.path.exists(file_path):  # noqa: ASYNC240
                                with open(file_path, "w", encoding="utf-8") as f:  # noqa: ASYNC230
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
                                if not os.path.exists(file_path):  # noqa: ASYNC240
                                    with open(file_path, "w", encoding="utf-8") as f:  # noqa: ASYNC230
                                        f.write(raw_output)

                                msg.content = f"tool output in {file_path}"
                                if "tool_result" in msg.metadata:
                                    msg.metadata["tool_result"] = f"tool output in {file_path}"
                                if "tool_error" in msg.metadata:
                                    msg.metadata["tool_error"] = f"tool output in {file_path}"

    return final_history

