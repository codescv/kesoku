"""Custom context compression and hierarchical consolidation manager for Kesoku."""

import logging
import time
import uuid

from kesoku.agent.llm import BaseLLM
from kesoku.config import KesokuConfig
from kesoku.constants import MessageRole, MessageType
from kesoku.db import Message, SummaryNode
from kesoku.db.manager import AsyncDatabaseManager

logger = logging.getLogger(__name__)

SUMMARIZE_TURN_PROMPT = """You are an advanced agent context compiler.
Your task is to summarize the following segment of a conversation turn history into
a highly dense, factual, and chronologically accurate narrative summary.

Guidelines:
1. Summarize key decisions, lessons learned / pitfalls encountered, and any files
   created or modified **outside** the session's `$STAGING_DIR` (omit if none).
2. Do not omit any crucial context, but do not include conversational fluff.
3. Keep the summary structured and concise (< 1000 chars).

Conversation Segment to Summarize:
{segment}

Summary:"""

CONSOLIDATE_SUMMARIES_PROMPT = """You are an advanced agent context compiler.
Your task is to merge and consolidate the following chronological sequence of summaries
into a single, cohesive, higher-level summary.

Guidelines:
1. Resolve any overlapping narrative threads to ensure a smooth, logical progression.
2. If there are conflicting decisions or changes in approach, prioritize the latest decision
   and resolve contradictions in favor of the most recent events.
3. Maintain high density and clear structure (< 1000 chars).

Summaries to Merge (in chronological order):
{summaries}

Consolidated Summary:"""


class HistoryCompressor:
    """Manages custom turn-based context compression and consolidation for Kesoku."""

    def __init__(self, db: AsyncDatabaseManager) -> None:
        """Initialize the HistoryCompressor with database adapter."""
        self.db = db

    def segment_turns(self, messages: list[Message]) -> list[list[Message]]:
        """Segment messages into logical turns starting with a USER or SYSTEM message.

        Internal notifications are excluded from turns.
        """
        turns: list[list[Message]] = []
        current_turn: list[Message] = []
        for m in messages:
            if m.role == MessageRole.ASSISTANT and m.sender == "Notification":
                continue
            if m.role in (MessageRole.USER, MessageRole.SYSTEM) and current_turn:
                turns.append(current_turn)
                current_turn = []
            current_turn.append(m)
        if current_turn:
            turns.append(current_turn)
        return turns

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count based on character length fallback (approx 4 chars/token)."""
        if not text:
            return 0
        return len(text) // 4

    def format_turn_for_summary(self, turn: list[Message]) -> str:
        """Format a single turn into text for summarization, stripping thoughts."""
        lines = []
        for msg in turn:
            if msg.type == MessageType.THOUGHT:
                continue
            role_label = msg.role.upper()
            content = msg.content or ""
            lines.append(f"{role_label}: {content}")
        return "\n".join(lines)

    async def auto_compact_session(
        self,
        session_id: str,
        history: list[Message],
        llm: BaseLLM,
        config: KesokuConfig,
    ) -> bool:
        """Check context window usage and automatically compact history in-place.

        Returns:
            True if any compression/consolidation took place, False otherwise.
        """
        # 1. Segment history into turns
        turns = self.segment_turns(history)

        protect_front = config.agent.protect_front_turns
        protect_tail = config.agent.protect_tail_turns
        min_tokens = config.agent.base_node_min_tokens
        base_turns = config.agent.base_node_turns
        K = config.agent.context_consolidation_k

        if len(turns) <= protect_front + protect_tail:
            return False

        # Candidates for compression are the middle turns
        candidates = turns[protect_front : -protect_tail]

        # Filter to only turns that have not yet been compressed
        uncompressed_turns = []
        for turn in candidates:
            # If any message in this turn has summary_node_id set, the whole turn is considered compressed.
            if any(msg.summary_node_id is not None for msg in turn):
                continue
            uncompressed_turns.append(turn)

        if not uncompressed_turns:
            return False

        # Accumulator loop to generate Level-0 nodes
        compacted_occurred = False
        current_chunk: list[list[Message]] = []
        current_tokens = 0

        for turn in uncompressed_turns:
            current_chunk.append(turn)
            current_tokens += sum(self.estimate_tokens(msg.content) for msg in turn)

            if len(current_chunk) >= base_turns and current_tokens >= min_tokens:
                logger.info(
                    f"Compressing {len(current_chunk)} turns ({current_tokens} tokens) "
                    f"into Level-0 summary node for session {session_id}."
                )
                segment_text = ""
                for t in current_chunk:
                    segment_text += self.format_turn_for_summary(t) + "\n"

                prompt = SUMMARIZE_TURN_PROMPT.format(segment=segment_text)
                res = await llm.generate(prompt=prompt)
                summary_content = res.content.strip()

                start_ts = min(msg.timestamp for t in current_chunk for msg in t)
                end_ts = max(msg.timestamp for t in current_chunk for msg in t)

                node_id = str(uuid.uuid4())
                node = SummaryNode(
                    id=node_id,
                    session_id=session_id,
                    level=0,
                    summary=summary_content,
                    start_timestamp=start_ts,
                    end_timestamp=end_ts,
                    token_count=self.estimate_tokens(summary_content),
                    source_token_count=current_tokens,
                    parent_id=None,
                    created_at=time.time(),
                )

                # Save the summary node in DB
                await self.db.insert_summary_node(node)

                # Update the source messages in DB to reference this Level-0 node
                all_msg_ids = [msg.id for t in current_chunk for msg in t]
                await self.db.update_messages_summary_node(all_msg_ids, node_id)

                # Update in-memory message references to prevent buffer duplication
                for t in current_chunk:
                    for msg in t:
                        msg.summary_node_id = node_id

                compacted_occurred = True
                current_chunk.clear()
                current_tokens = 0

        # If any Level-0 nodes were created, trigger forest consolidation
        if compacted_occurred:
            await self.consolidate_forest(session_id, llm, K)

        return compacted_occurred

    async def consolidate_forest(self, session_id: str, llm: BaseLLM, K: int) -> None:
        """Consolidate root summary nodes hierarchically when they accumulate to 2K nodes.

        Only the oldest K nodes are merged.
        """
        level = 0
        while True:
            roots = await self.db.get_root_summary_nodes(session_id, level)

            # Check trigger condition
            if len(roots) < 2 * K:
                break

            logger.info(
                f"Consolidating Level-{level} root summary nodes for session {session_id} "
                f"({len(roots)} roots found, merging oldest {K} into Level-{level+1})."
            )

            # Merge the oldest K roots
            i = 0
            while len(roots) - i >= 2 * K:
                chunk = roots[i : i + K]
                summaries_text = ""
                for idx, nd in enumerate(chunk):
                    summaries_text += (
                        f"--- Summary {idx+1} (from ts {nd.start_timestamp} to {nd.end_timestamp}) ---\n"
                        f"{nd.summary}\n\n"
                    )

                prompt = CONSOLIDATE_SUMMARIES_PROMPT.format(summaries=summaries_text)
                res = await llm.generate(prompt=prompt)
                merged_summary = res.content.strip()

                parent_id = str(uuid.uuid4())
                parent_node = SummaryNode(
                    id=parent_id,
                    session_id=session_id,
                    level=level + 1,
                    summary=merged_summary,
                    start_timestamp=min(nd.start_timestamp for nd in chunk),
                    end_timestamp=max(nd.end_timestamp for nd in chunk),
                    token_count=self.estimate_tokens(merged_summary),
                    source_token_count=sum(nd.token_count for nd in chunk),
                    parent_id=None,
                    created_at=time.time(),
                )

                # Save parent summary node
                await self.db.insert_summary_node(parent_node)

                # Link children roots to the new parent
                for nd in chunk:
                    await self.db.update_summary_node_parent(nd.id, parent_id)

                i += K

            level += 1
