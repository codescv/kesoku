"""Custom context compression and hierarchical consolidation manager for Kesoku."""

import datetime
import json
import logging
import os
import re
import time
import uuid
from typing import Any

from kesoku.agent.history import segment_logical_turns
from kesoku.agent.llm import BaseLLM
from kesoku.config import KesokuConfig
from kesoku.constants import MessageType
from kesoku.db import Message, SummaryNode
from kesoku.db.manager import AsyncDatabaseManager

logger = logging.getLogger(__name__)

SUMMARIZE_TURN_PROMPT = """You are an advanced agent context compiler.
Your task is to summarize the following segment of a conversation turn history into
a highly dense, factual, and cohesive JSON object.
Keep the summary structured and concise (< 1200 chars total).

Date Context for this segment:
{date_context}

Guidelines:
1. timeline: Group nearby consecutive events into cohesive chronological narrative phases
   (e.g., "{example_start_time} - {example_end_time}: ..."). Do NOT produce granular per-turn line items,
   but also avoid collapsing the entire segment into a single sentence; capture distinct milestones, topics,
   or scene transitions as separate phase blocks (typically 2 to 5 blocks, maximum 5).
2. tools_and_skills: List distinct tools and skills invoked during this segment, accompanied by a brief
   one-sentence description of what each was used for (e.g., "run_shell_command (japanese-challenge skill):
   Quizzed grammar points and managed mistake notebook"). Return None if none were used.
3. learnings: Summarize practical takeaways:
   - Tool & skill execution insights: effective parameters, syntax fixes, and error recovery patterns.
   - User preferences & corrections: user preferences discovered, and behavioral rules learned when the
     user corrected the agent.
   Return None if none.

Output ONLY a valid JSON object with exact keys "timeline", "tools_and_skills", and "learnings":
{{
  "timeline": [
    "{example_start_time} - {example_end_time}: [Concise narrative of phase 1]",
    "{example_start_time} - {example_end_time}: [Concise narrative of phase 2]"
  ],
  "tools_and_skills": [
    "tool_name (skill_name if applicable): [One concise sentence explaining its purpose/usage]"
  ] or null,
  "learnings": "..." or null
}}

Conversation Segment to Summarize:
{segment}

JSON Summary:"""

CONSOLIDATE_SUMMARIES_PROMPT = """You are an advanced agent context compiler.
Your task is to merge and consolidate the following chronological sequence of summaries
into a single, cohesive, higher-level JSON object.
Maintain high density and clear structure (< 1200 chars total).

Date Context for these summaries:
{date_context}

Guidelines:
1. timeline: Merge and consolidate events chronologically into distinct macro-phase blocks
   (e.g., "{example_start_time} - {example_end_time}: ..."). Do not collapse the entire history into a single
   one-liner; preserve distinct narrative phases, key topics, or major activity milestones across the
   timeline (typically 2 to 5 blocks, maximum 5).
2. tools_and_skills: Deduplicate and consolidate all distinct tools and skills used across the summaries, each
   with a brief one-sentence description of its purpose/usage. Return None if none.
3. learnings: Merge, deduplicate, and synthesize all key learnings across the summaries:
   - Tool & skill execution insights: effective parameters, syntax fixes, and error recovery patterns.
   - User preferences & corrections: user preferences discovered, and behavioral rules learned when the
     user corrected the agent.
   Resolve any conflicting rules in favor of the most recent events. Return None if none.

Output ONLY a valid JSON object with exact keys "timeline", "tools_and_skills", and "learnings":
{{
  "timeline": [
    "{example_start_time} - {example_end_time}: [Concise narrative of merged macro phase 1]",
    "{example_start_time} - {example_end_time}: [Concise narrative of merged macro phase 2]"
  ],
  "tools_and_skills": [
    "tool_name (skill_name if applicable): [One concise sentence explaining its purpose/usage]"
  ] or null,
  "learnings": "..." or null
}}

Summaries to Merge (in chronological order):
{summaries}

Consolidated JSON Summary:"""

SUMMARY_TEMPLATE = """Timeline:
{timeline}

Tools & Skills:
{tools_and_skills}

Learnings:
{learnings}"""


class HistoryCompressor:
    """Manages custom turn-based context compression and consolidation for Kesoku."""

    def __init__(self, db: AsyncDatabaseManager) -> None:
        """Initialize the HistoryCompressor with database adapter."""
        self.db = db

    @classmethod
    def parse_summary_json(cls, raw_content: str) -> dict[str, Any]:
        """Parse JSON object from LLM response, handling markdown fenced blocks and extra text."""
        text = raw_content.strip()
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and start < end:
                json_str = text[start : end + 1]
            else:
                json_str = text
        try:
            data = json.loads(json_str)
            if isinstance(data, dict):
                return data
        except Exception as e:
            logger.warning(f"Failed to parse summary JSON from LLM output: {e}")
        return {}

    @classmethod
    def is_outside_staging_dir(cls, file_path: str, staging_dir: str | None) -> bool:
        """Check whether a file path is outside of the session STAGING_DIR."""
        if not file_path or not isinstance(file_path, str):
            return False
        cleaned = file_path.strip().strip("'\"")
        if not cleaned:
            return False
        if "STAGING_DIR" in cleaned:
            return False

        if not staging_dir:
            return True

        abs_staging = os.path.abspath(staging_dir)
        try:
            norm_staging = os.path.normpath(abs_staging)
            norm_file = os.path.normpath(os.path.abspath(cleaned))
            if os.path.commonpath([norm_staging, norm_file]) == norm_staging:
                return False

            session_folder_name = os.path.basename(norm_staging)
            if session_folder_name and session_folder_name != ".":
                if cleaned.startswith(f"{session_folder_name}/"):
                    return False
                if cleaned.startswith(f"sessions/{session_folder_name}/"):
                    return False
                if f"/sessions/{session_folder_name}/" in f"/{cleaned}":
                    return False

            candidate = os.path.join(norm_staging, cleaned)
            if os.path.exists(candidate):
                return False
        except ValueError:
            pass

        return True

    @classmethod
    def _format_field(cls, val: Any, default_none: bool = False) -> str:
        """Format a summary field value cleanly, handling None or empty cases."""
        if val is None:
            return "None" if default_none else ""
        if isinstance(val, list):
            items = [str(item).strip() for item in val if str(item).strip()]
            if not items:
                return "None" if default_none else ""
            if len(items) == 1 and items[0].lower() in ("none", "null", "n/a", "[]", ""):
                return "None" if default_none else items[0]
            return "\n".join(f"- {item}" if not item.startswith("-") else item for item in items)

        text = str(val).strip()
        if not text or text.lower() in ("none", "null", "n/a", "[]", ""):
            return "None" if default_none else text
        return text

    @classmethod
    def format_summary(cls, raw_content: str, staging_dir: str | None = None) -> str:
        """Parse LLM JSON output and format summary using template."""
        data = cls.parse_summary_json(raw_content)

        # 1. Timeline
        timeline_val = data.get("timeline") or data.get("timeline_events")
        timeline_str = cls._format_field(timeline_val, default_none=False)
        if not timeline_str and raw_content and not data:
            timeline_str = raw_content.strip()

        # 2. Tools and skills
        tools_val = data.get("tools_and_skills") or data.get("tools") or data.get("skills")
        tools_str = cls._format_field(tools_val, default_none=True)

        # 3. Learnings
        learnings_val = data.get("learnings") or data.get("learning")
        learnings_str = cls._format_field(learnings_val, default_none=True)

        return SUMMARY_TEMPLATE.format(
            timeline=timeline_str,
            tools_and_skills=tools_str,
            learnings=learnings_str,
        )

    @classmethod
    def format_ts(cls, ts: float | int | None) -> str:
        """Format UNIX timestamp into standard calendar date and time string with timezone."""
        if not ts or ts <= 0:
            return "Unknown Time"
        try:
            msg_time = datetime.datetime.fromtimestamp(ts).astimezone()
            return msg_time.strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            return "Unknown Time"

    def segment_turns(self, messages: list[Message]) -> list[list[Message]]:
        """Segment messages into logical turns starting with a USER or SYSTEM message.

        Internal notifications are excluded from turns.
        """
        return segment_logical_turns(messages)

    def format_turn_for_summary(self, turn: list[Message]) -> str:
        """Format a single turn into text for summarization, stripping thoughts and including timestamps."""
        lines = []
        for msg in turn:
            if msg.type == MessageType.THOUGHT:
                continue
            role_label = msg.role.upper()
            content = msg.content or ""
            ts_str = self.format_ts(msg.timestamp)
            lines.append(f"[{ts_str}] {role_label}: {content}")
        return "\n".join(lines)

    async def auto_compact_session(
        self,
        session_id: str,
        history: list[Message],
        llm: BaseLLM,
        config: KesokuConfig,
        staging_dir: str | None = None,
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
        candidates = turns[protect_front:-protect_tail]

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
            current_tokens += sum(llm.estimate_tokens_fallback(prompt=msg.content) for msg in turn)

            if len(current_chunk) >= base_turns and current_tokens >= min_tokens:
                logger.info(
                    f"Compressing {len(current_chunk)} turns ({current_tokens} tokens) "
                    f"into Level-0 summary node for session {session_id}."
                )
                segment_text = ""
                for t in current_chunk:
                    segment_text += self.format_turn_for_summary(t) + "\n"

                start_ts = min(msg.timestamp for t in current_chunk for msg in t)
                end_ts = max(msg.timestamp for t in current_chunk for msg in t)
                start_str = self.format_ts(start_ts)
                end_str = self.format_ts(end_ts)
                date_context = (
                    f"This conversation segment occurred from {start_str} to {end_str}. "
                    "Use these exact calendar dates for all timeline events."
                )

                start_dt = datetime.datetime.fromtimestamp(start_ts).astimezone()
                end_dt = datetime.datetime.fromtimestamp(end_ts).astimezone()
                example_start_time = start_dt.strftime("%Y-%m-%d %H:%M")
                example_end_time = end_dt.strftime("%Y-%m-%d %H:%M")

                prompt = SUMMARIZE_TURN_PROMPT.format(
                    date_context=date_context,
                    example_start_time=example_start_time,
                    example_end_time=example_end_time,
                    segment=segment_text,
                )
                res = await llm.generate(prompt=prompt)
                summary_content = self.format_summary(res.content, staging_dir=staging_dir)

                node_id = str(uuid.uuid4())
                node = SummaryNode(
                    id=node_id,
                    session_id=session_id,
                    level=0,
                    summary=summary_content,
                    start_timestamp=start_ts,
                    end_timestamp=end_ts,
                    token_count=llm.estimate_tokens_fallback(prompt=summary_content),
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
            await self.consolidate_forest(session_id, llm, K, staging_dir=staging_dir)

        return compacted_occurred

    async def consolidate_forest(
        self,
        session_id: str,
        llm: BaseLLM,
        K: int,
        staging_dir: str | None = None,
    ) -> None:
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
                f"({len(roots)} roots found, merging oldest {K} into Level-{level + 1})."
            )

            # Merge the oldest K roots
            i = 0
            while len(roots) - i >= 2 * K:
                chunk = roots[i : i + K]
                start_ts = min(nd.start_timestamp for nd in chunk)
                end_ts = max(nd.end_timestamp for nd in chunk)
                start_str = self.format_ts(start_ts)
                end_str = self.format_ts(end_ts)
                date_context = (
                    f"These summaries cover the period from {start_str} to {end_str}. "
                    "Use these exact calendar dates for all timeline events."
                )

                summaries_text = ""
                for idx, nd in enumerate(chunk):
                    s_str = self.format_ts(nd.start_timestamp)
                    e_str = self.format_ts(nd.end_timestamp)
                    summaries_text += (
                        f"--- Summary {idx + 1} (from {s_str} to {e_str}) ---\n"
                        f"{nd.summary}\n\n"
                    )

                start_dt = datetime.datetime.fromtimestamp(start_ts).astimezone()
                end_dt = datetime.datetime.fromtimestamp(end_ts).astimezone()
                example_start_time = start_dt.strftime("%Y-%m-%d %H:%M")
                example_end_time = end_dt.strftime("%Y-%m-%d %H:%M")

                prompt = CONSOLIDATE_SUMMARIES_PROMPT.format(
                    date_context=date_context,
                    example_start_time=example_start_time,
                    example_end_time=example_end_time,
                    summaries=summaries_text,
                )
                res = await llm.generate(prompt=prompt)
                merged_summary = self.format_summary(res.content, staging_dir=staging_dir)

                parent_id = str(uuid.uuid4())
                parent_node = SummaryNode(
                    id=parent_id,
                    session_id=session_id,
                    level=level + 1,
                    summary=merged_summary,
                    start_timestamp=min(nd.start_timestamp for nd in chunk),
                    end_timestamp=max(nd.end_timestamp for nd in chunk),
                    token_count=llm.estimate_tokens_fallback(prompt=merged_summary),
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
