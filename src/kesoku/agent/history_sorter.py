"""Logic for sorting session historical messages for AI agent turns."""

from typing import Literal

from kesoku.constants import MessageRole, MessageType
from kesoku.db.models import Message


def sort_session_messages(
    all_msgs: list[Message], order: Literal["phased", "grouped"]
) -> list[Message]:
    """Sort historical messages for a specific session ordered logically.

    Args:
        all_msgs: A list of Message objects to sort.
        order: Sorting mechanism ("phased" or "grouped").

    Returns:
        A list of logically sorted Message objects.
    """
    if not all_msgs:
        return []

    msg_map = {m.id: m for m in all_msgs}

    def get_root_timestamp(m: Message) -> float:
        curr = m
        while curr.parent_id and curr.parent_id in msg_map:
            curr = msg_map[curr.parent_id]
        return curr.timestamp

    def get_tool_group_timestamp(m: Message) -> float:
        if m.parent_id and m.parent_id in msg_map:
            parent_msg = msg_map[m.parent_id]
            if parent_msg.role == MessageRole.TOOL and parent_msg.type == MessageType.TOOL_CALL:
                return parent_msg.timestamp
        return m.timestamp

    if order == "grouped":
        # Grouped sorting simply sorts by root turn timestamp, then tool group timestamp, then actual timestamp
        return sorted(all_msgs, key=lambda m: (get_root_timestamp(m), get_tool_group_timestamp(m), m.timestamp))

    # Phased sorting logic (default for LLM inputs):
    # 1. Group messages by turn root timestamp
    turns: dict[float, list[Message]] = {}
    for msg in all_msgs:
        root_ts = get_root_timestamp(msg)
        turns.setdefault(root_ts, []).append(msg)

    # 2. Sort each turn individually
    for root_ts, turn_msgs in turns.items():
        # Identify all tool calls in the current turn
        tc_map = {m.id: m for m in turn_msgs if m.role == MessageRole.TOOL and m.type == MessageType.TOOL_CALL}

        # Collect and sort tool results by parent tool call timestamp
        tr_msgs = [m for m in turn_msgs if m.role == MessageRole.TOOL and m.type == MessageType.TOOL_RESULT]
        tr_msgs.sort(key=lambda m: tc_map[m.parent_id].timestamp if m.parent_id in tc_map else m.timestamp)

        # Group tool results into logical execution batches
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

        # Determine maximum timestamp boundary (cutoff) for each batch
        batch_cutoffs = [max(tr.timestamp for tr in batch) for batch in batches] if batches else []

        # Determine which iteration round a message belongs to based on these boundaries
        def get_iteration_index(m: Message) -> int:
            idx = 0
            for cutoff in batch_cutoffs:
                if m.timestamp > cutoff:
                    idx += 1
            return idx

        # Define the phase sorting inside a single iteration round
        def get_sorting_phase(m: Message) -> float:
            if m.role == MessageRole.TOOL and m.type == MessageType.TOOL_CALL:
                return 1.0
            elif m.role == MessageRole.TOOL and m.type == MessageType.TOOL_RESULT:
                return 2.0
            elif m.role == MessageRole.ASSISTANT and m.type == MessageType.THOUGHT:
                return 0.0
            elif m.role == MessageRole.ASSISTANT and m.type != MessageType.THOUGHT:
                return 3.0
            return 0.0

        # Sort turn messages in place
        turn_msgs.sort(
            key=lambda m: (
                get_iteration_index(m),
                m.timestamp,
                get_sorting_phase(m),
            )
        )

    # 3. Flatten all sorted turns chronologically
    sorted_msgs = []
    for r_ts in sorted(turns.keys()):
        sorted_msgs.extend(turns[r_ts])

    return sorted_msgs
