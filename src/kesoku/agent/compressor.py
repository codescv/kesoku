"""Custom context compression and hierarchical consolidation manager for Kesoku."""

import datetime
import json
import logging
import os
import re
import time
import uuid
from typing import Any

from kesoku.agent.history import history_preview, segment_logical_turns, truncate_tool_call_arguments
from kesoku.agent.llm import BaseLLM
from kesoku.config import KesokuConfig
from kesoku.constants import MessageRole, MessageType
from kesoku.db import Message, SummaryNode
from kesoku.db.manager import AsyncDatabaseManager

logger = logging.getLogger(__name__)

SUMMARIZE_TURN_PROMPT = """You are an advanced agent context compiler.
Your task is to summarize the following segment of a conversation turn history into
a highly dense, factual, and cohesive JSON object.
Keep the summary structured and concise (< {max_chars} chars total).

Date Context for this segment:
{date_context}

Guidelines:
1. timeline: Group nearby consecutive events into cohesive chronological narrative phases
   (e.g., "{example_start_time} - {example_end_time}: ..."). Do NOT produce granular per-turn line items,
   but also avoid collapsing the entire segment into a single sentence; capture distinct milestones, topics,
   or scene transitions as separate phase blocks (typically 2 to 5 blocks, maximum 5).
2. learnings: Practical tool & skill execution insights only: effective parameters, syntax fixes, and error
   recovery patterns. Do NOT record user requirements, preferences, or behavioral rules here; those are
   maintained separately and would become stale. Return None if none.

Output ONLY a valid JSON object with exact keys "timeline" and "learnings":
{{
  "timeline": [
    "{example_start_time} - {example_end_time}: [Concise narrative of phase 1]",
    "{example_start_time} - {example_end_time}: [Concise narrative of phase 2]"
  ],
  "learnings": "..." or null
}}

Conversation Segment to Summarize:
{segment}

JSON Summary:"""

CONSOLIDATE_SUMMARIES_PROMPT = """You are an advanced agent context compiler.
Your task is to merge and consolidate the following chronological sequence of summaries
into a single, cohesive, higher-level JSON object.
Maintain high density and clear structure (< {max_chars} chars total).

Date Context for these summaries:
{date_context}

Guidelines:
1. timeline: Merge and consolidate events chronologically into distinct macro-phase blocks
   (e.g., "{example_start_time} - {example_end_time}: ..."). Do not collapse the entire history into a single
   one-liner; preserve distinct narrative phases, key topics, or major activity milestones across the
   timeline (typically 2 to 5 blocks, maximum 5).
2. learnings: Merge, deduplicate, and synthesize tool & skill execution insights only (effective parameters,
   syntax fixes, error recovery patterns). Drop any user requirements, preferences, or behavioral rules.
   Resolve conflicting insights in favor of the most recent events. Return None if none.

Output ONLY a valid JSON object with exact keys "timeline" and "learnings":
{{
  "timeline": [
    "{example_start_time} - {example_end_time}: [Concise narrative of merged macro phase 1]",
    "{example_start_time} - {example_end_time}: [Concise narrative of merged macro phase 2]"
  ],
  "learnings": "..." or null
}}

Summaries to Merge (in chronological order):
{summaries}

Consolidated JSON Summary:"""

SUMMARY_TEMPLATE = """Timeline:
{timeline}

Learnings:
{learnings}"""

# Default summary length budget (legacy chunk strategy and lower bound for watermark strategy).
DEFAULT_SUMMARY_MAX_CHARS = 1200

SCAFFOLD_HEADER = (
    "[Note: This conversation uses custom turn-based context management. Earlier turns have been compacted "
    "into the hierarchical summaries below. These summaries are HISTORICAL and may be outdated: whenever they "
    "conflict with <context name=\"notes\"> or with more recent messages, follow <context name=\"notes\"> and "
    "the most recent messages. You can search past messages and memories using standard search tools.]\n"
)

# Matches deprecated summary sections ("User Asks", "Tools & Skills") up to the next section header or end.
_LEGACY_SECTIONS_RE = re.compile(
    r"\n*(?:User Asks|Tools & Skills):\n.*?(?=\n\n(?:User Asks|Tools & Skills|Learnings):|\Z)", re.DOTALL
)

# Watermark strategy summary budgets (characters).
MAX_L0_SUMMARY_CHARS = 3000
MAX_MERGED_SUMMARY_CHARS = 4000
# Each verbatim user message in <recent_user_messages> is capped to this many characters.
RECENT_USER_MESSAGE_MAX_CHARS = 1000

_NON_ASCII_RE = re.compile(r"[^\x00-\x7f]")


def estimate_tokens_calibrated(text: str | None) -> int:
    """Estimate Gemini tokens from character classes.

    Calibrated against gemini-3.8-flash count_tokens on real Kesoku messages
    (≈0.24 tokens per ASCII char, ≈0.87 per non-ASCII char). The legacy fallback estimator counts
    non-ASCII characters as 2 tokens and overestimates CJK-heavy sessions by ~1.5x.
    """
    if not text:
        return 0
    non_ascii = len(_NON_ASCII_RE.findall(text))
    return int((len(text) - non_ascii) * 0.25 + non_ascii * 0.9)


def summary_max_chars(source_tokens: int) -> int:
    """Length budget for a Level-0 summary in the watermark strategy (~8% of source tokens, clamped)."""
    return min(MAX_L0_SUMMARY_CHARS, max(DEFAULT_SUMMARY_MAX_CHARS, int(source_tokens * 0.08)))


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
    def strip_legacy_sections(cls, summary: str) -> str:
        """Remove deprecated sections ("User Asks", "Tools & Skills") from a stored summary.

        Older summary nodes embedded a "User Asks" section whose requirements were frozen at
        summarization time and frequently contradicted newer instructions, plus a highly repetitive
        "Tools & Skills" section. Both are stripped whenever a stored summary is shown to the LLM
        (scaffold or consolidation input).
        """
        if not summary or ("User Asks:" not in summary and "Tools & Skills:" not in summary):
            return summary
        return _LEGACY_SECTIONS_RE.sub("", summary).strip()

    @classmethod
    def format_summary(cls, raw_content: str, staging_dir: str | None = None) -> str:
        """Parse LLM JSON output and format summary using template.

        Args:
            raw_content: Raw LLM output containing a JSON object.
            staging_dir: Unused; kept for backward compatibility of the call signature.
        """
        data = cls.parse_summary_json(raw_content)

        # 1. Timeline
        timeline_val = data.get("timeline") or data.get("timeline_events")
        timeline_str = cls._format_field(timeline_val, default_none=False)
        if not timeline_str and raw_content and not data:
            timeline_str = raw_content.strip()

        # 2. Learnings (tool execution insights only)
        learnings_val = data.get("learnings") or data.get("learning")
        learnings_str = cls._format_field(learnings_val, default_none=True)

        return SUMMARY_TEMPLATE.format(timeline=timeline_str, learnings=learnings_str)

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

    @staticmethod
    def _tokens(llm: BaseLLM, text: str, calibrated: bool) -> int:
        """Estimate tokens with either the legacy fallback or the calibrated estimator."""
        if calibrated:
            return estimate_tokens_calibrated(text)
        return llm.estimate_tokens_fallback(prompt=text)

    def estimate_prepared_turn_tokens(
        self,
        turn: list[Message],
        llm: BaseLLM,
        max_historical_tool_chars: int = 500,
        calibrated: bool = False,
    ) -> int:
        """Estimate token count for a completed turn as prepared for the LLM.

        Mirrors prepare_history_for_llm: strips assistant thoughts, truncates historical tool outputs
        to max_historical_tool_chars, and truncates long file-content arguments of historical
        file/notes tool calls.

        Args:
            turn: List of Message objects forming a single turn.
            llm: BaseLLM instance providing token estimation methods.
            max_historical_tool_chars: Maximum character limit for historical tool results.
            calibrated: Use the calibrated estimator (watermark strategy) instead of the legacy fallback.

        Returns:
            Estimated total tokens for the turn.
        """
        total_tokens = 0
        for msg in turn:
            # Drop assistant thoughts in completed turns
            if msg.type == MessageType.THOUGHT:
                continue

            content = msg.content or ""
            # Truncate historical tool outputs exceeding character limit
            if msg.role == MessageRole.TOOL and msg.type == MessageType.TOOL_RESULT:
                if len(content) > max_historical_tool_chars:
                    # Same head+tail preview prepare_history_for_llm builds (shell banner stripped).
                    meta = msg.metadata or {}
                    raw = meta.get("tool_error") or meta.get("tool_result") or content
                    content = history_preview(str(raw), max_historical_tool_chars)
            elif msg.role == MessageRole.TOOL and msg.type == MessageType.TOOL_CALL:
                args = (msg.metadata or {}).get("tool_arguments")
                if isinstance(args, dict):
                    tool_name = (msg.metadata or {}).get("tool_name")
                    content = json.dumps(truncate_tool_call_arguments(tool_name, args), ensure_ascii=False)

            total_tokens += self._tokens(llm, content, calibrated)
        return total_tokens

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

    @classmethod
    def build_recent_user_messages_block(cls, turns: list[list[Message]], limit: int) -> str:
        """Render verbatim user messages of already-compressed turns (most recent `limit`).

        User messages are tiny compared to agent output, so keeping them verbatim preserves the user's
        own wording of requirements after their turns have been summarized.
        """
        if limit <= 0:
            return ""
        user_msgs = [
            m
            for t in turns
            if any(x.summary_node_id is not None for x in t)
            for m in t
            if m.role == MessageRole.USER and (m.content or "").strip()
        ]
        if not user_msgs:
            return ""
        lines = []
        for m in user_msgs[-limit:]:
            ts = datetime.datetime.fromtimestamp(m.timestamp).astimezone().strftime("%Y-%m-%d %H:%M")
            text = (m.content or "").strip()
            if len(text) > RECENT_USER_MESSAGE_MAX_CHARS:
                text = text[:RECENT_USER_MESSAGE_MAX_CHARS] + "…"
            lines.append(f"[{ts}] {text}")
        return (
            "<recent_user_messages>\n"
            "<!-- Verbatim user messages from the compacted turns above, oldest first. Later messages override "
            "earlier ones. -->\n" + "\n".join(lines) + "\n</recent_user_messages>"
        )

    async def auto_compact_session(
        self,
        session_id: str,
        history: list[Message],
        llm: BaseLLM,
        config: KesokuConfig,
        staging_dir: str | None = None,
        overhead_tokens: int = 0,
    ) -> bool:
        """Check context usage and automatically compact history in-place using the configured strategy.

        Args:
            session_id: Session identifier.
            history: Raw session history (including the current turn).
            llm: LLM used for summarization.
            config: Kesoku configuration (agent.compression_strategy selects the algorithm).
            staging_dir: Optional session staging directory.
            overhead_tokens: Estimated fixed prompt overhead (system prompt + tool schemas); watermark only.

        Returns:
            True if any compression/consolidation took place, False otherwise.
        """
        if config.agent.compression_strategy == "watermark":
            return await self._compact_watermark(session_id, history, llm, config, overhead_tokens)
        return await self._compact_chunk(session_id, history, llm, config)

    async def _compact_chunk(
        self,
        session_id: str,
        history: list[Message],
        llm: BaseLLM,
        config: KesokuConfig,
    ) -> bool:
        """Legacy strategy: compress every qualifying chunk as soon as it leaves the protected tail."""
        turns = self.segment_turns(history)

        protect_front = config.agent.protect_front_turns
        protect_tail = config.agent.protect_tail_turns
        min_tokens = config.agent.base_node_min_tokens
        base_turns = config.agent.base_node_turns
        K = config.agent.context_consolidation_k

        if len(turns) <= protect_front + protect_tail:
            return False

        # Candidates for compression are the middle turns that have not yet been compressed.
        # If any message in a turn has summary_node_id set, the whole turn is considered compressed.
        uncompressed_turns = [
            t for t in turns[protect_front:-protect_tail] if not any(m.summary_node_id is not None for m in t)
        ]
        if not uncompressed_turns:
            return False

        # Accumulator loop to generate Level-0 nodes
        compacted_occurred = False
        current_chunk: list[list[Message]] = []
        current_tokens = 0
        for turn in uncompressed_turns:
            current_chunk.append(turn)
            current_tokens += self.estimate_prepared_turn_tokens(turn, llm)
            if len(current_chunk) >= base_turns and current_tokens >= min_tokens:
                await self._summarize_chunk(session_id, current_chunk, current_tokens, llm, DEFAULT_SUMMARY_MAX_CHARS)
                compacted_occurred = True
                current_chunk = []
                current_tokens = 0

        # If any Level-0 nodes were created, trigger forest consolidation
        if compacted_occurred:
            await self.consolidate_forest(session_id, llm, K)
        return compacted_occurred

    async def _compact_watermark(
        self,
        session_id: str,
        history: list[Message],
        llm: BaseLLM,
        config: KesokuConfig,
        overhead_tokens: int,
    ) -> bool:
        """Watermark strategy: compress only above the high watermark, down to the low watermark.

        The whole prompt (fixed overhead + scaffold + uncompressed history) is estimated with the calibrated
        estimator. When it exceeds `context_high_watermark`, the oldest unprotected turns are summarized in
        chunks of ~`watermark_chunk_tokens` until the estimate falls below `context_low_watermark`. The tail is
        protected by both a minimum turn count and a minimum token budget.
        """
        agent = config.agent
        turns = self.segment_turns(history)
        protect_front = agent.protect_front_turns
        if len(turns) <= protect_front + agent.protect_tail_turns:
            return False

        compressed = [any(m.summary_node_id is not None for m in t) for t in turns]
        turn_tokens = [
            0 if compressed[i] else self.estimate_prepared_turn_tokens(t, llm, calibrated=True)
            for i, t in enumerate(turns)
        ]
        roots = await self.db.get_root_summary_nodes(session_id)
        scaffold_tokens = sum(estimate_tokens_calibrated(self.strip_legacy_sections(n.summary)) for n in roots)
        scaffold_tokens += estimate_tokens_calibrated(
            self.build_recent_user_messages_block(turns, agent.recent_user_messages_limit)
        )
        total = overhead_tokens + scaffold_tokens + sum(turn_tokens)
        if total <= agent.context_high_watermark:
            return False

        # Protected tail: walk back from the current turn until both minimums are satisfied.
        tail_start = len(turns)
        tail_tokens = 0
        while tail_start > protect_front and (
            tail_tokens < agent.protect_tail_min_tokens or len(turns) - tail_start < agent.protect_tail_turns
        ):
            tail_start -= 1
            tail_tokens += turn_tokens[tail_start]

        candidates = [i for i in range(protect_front, tail_start) if not compressed[i]]
        # Guard against thrashing: never summarize less than one full chunk worth of turns.
        if sum(turn_tokens[i] for i in candidates) < agent.watermark_chunk_tokens:
            logger.info(
                f"Session {session_id}: prompt ~{total} tokens exceeds high watermark but only "
                f"{sum(turn_tokens[i] for i in candidates)} compressible tokens are outside the protected tail."
            )
            return False

        logger.info(
            f"Session {session_id}: prompt ~{total} tokens > high watermark {agent.context_high_watermark}; "
            f"compacting down to {agent.context_low_watermark}."
        )
        compacted_occurred = False
        while total > agent.context_low_watermark and candidates:
            chunk_idx: list[int] = []
            src = 0
            while candidates and (
                src < agent.watermark_chunk_tokens or len(chunk_idx) < agent.watermark_chunk_min_turns
            ):
                i = candidates.pop(0)
                chunk_idx.append(i)
                src += turn_tokens[i]
            node = await self._summarize_chunk(
                session_id, [turns[i] for i in chunk_idx], src, llm, summary_max_chars(src)
            )
            total += estimate_tokens_calibrated(node.summary) - src
            compacted_occurred = True

        if compacted_occurred:
            await self.consolidate_forest(session_id, llm, agent.context_consolidation_k, proportional=True)
        return compacted_occurred

    async def _summarize_chunk(
        self,
        session_id: str,
        chunk: list[list[Message]],
        source_tokens: int,
        llm: BaseLLM,
        max_chars: int,
    ) -> SummaryNode:
        """Summarize a chunk of turns into a Level-0 node, persist it, and link the source messages."""
        logger.info(
            f"Compressing {len(chunk)} turns ({source_tokens} prepared tokens) "
            f"into Level-0 summary node for session {session_id}."
        )
        segment_text = "".join(self.format_turn_for_summary(t) + "\n" for t in chunk)

        start_ts = min(msg.timestamp for t in chunk for msg in t)
        end_ts = max(msg.timestamp for t in chunk for msg in t)
        date_context = (
            f"This conversation segment occurred from {self.format_ts(start_ts)} to {self.format_ts(end_ts)}. "
            "Use these exact calendar dates for all timeline events."
        )
        start_dt = datetime.datetime.fromtimestamp(start_ts).astimezone()
        end_dt = datetime.datetime.fromtimestamp(end_ts).astimezone()

        prompt = SUMMARIZE_TURN_PROMPT.format(
            max_chars=max_chars,
            date_context=date_context,
            example_start_time=start_dt.strftime("%Y-%m-%d %H:%M"),
            example_end_time=end_dt.strftime("%Y-%m-%d %H:%M"),
            segment=segment_text,
        )
        res = await llm.generate(prompt=prompt)
        summary_content = self.format_summary(res.content)

        node = SummaryNode(
            id=str(uuid.uuid4()),
            session_id=session_id,
            level=0,
            summary=summary_content,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            token_count=llm.estimate_tokens_fallback(prompt=summary_content),
            source_token_count=source_tokens,
            parent_id=None,
            created_at=time.time(),
        )
        # Save the summary node in DB and point the source messages at it
        await self.db.insert_summary_node(node)
        await self.db.update_messages_summary_node([msg.id for t in chunk for msg in t], node.id)

        # Update in-memory message references to prevent buffer duplication
        for t in chunk:
            for msg in t:
                msg.summary_node_id = node.id
        return node

    async def consolidate_forest(
        self,
        session_id: str,
        llm: BaseLLM,
        K: int,
        staging_dir: str | None = None,
        proportional: bool = False,
    ) -> None:
        """Consolidate root summary nodes hierarchically when they accumulate to 2K nodes.

        Only the oldest K nodes are merged. With `proportional`, the merged summary's length budget scales
        with the children's total length instead of the fixed legacy budget.
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
                date_context = (
                    f"These summaries cover the period from {self.format_ts(start_ts)} to {self.format_ts(end_ts)}. "
                    "Use these exact calendar dates for all timeline events."
                )

                children = [self.strip_legacy_sections(nd.summary) for nd in chunk]
                summaries_text = ""
                for idx, (nd, text) in enumerate(zip(chunk, children)):
                    s_str = self.format_ts(nd.start_timestamp)
                    e_str = self.format_ts(nd.end_timestamp)
                    summaries_text += f"--- Summary {idx + 1} (from {s_str} to {e_str}) ---\n{text}\n\n"

                max_chars = DEFAULT_SUMMARY_MAX_CHARS
                if proportional:
                    max_chars = min(
                        MAX_MERGED_SUMMARY_CHARS,
                        max(DEFAULT_SUMMARY_MAX_CHARS, sum(len(c) for c in children) // 2),
                    )

                start_dt = datetime.datetime.fromtimestamp(start_ts).astimezone()
                end_dt = datetime.datetime.fromtimestamp(end_ts).astimezone()
                prompt = CONSOLIDATE_SUMMARIES_PROMPT.format(
                    max_chars=max_chars,
                    date_context=date_context,
                    example_start_time=start_dt.strftime("%Y-%m-%d %H:%M"),
                    example_end_time=end_dt.strftime("%Y-%m-%d %H:%M"),
                    summaries=summaries_text,
                )
                res = await llm.generate(prompt=prompt)
                merged_summary = self.format_summary(res.content)

                parent_id = str(uuid.uuid4())
                parent_node = SummaryNode(
                    id=parent_id,
                    session_id=session_id,
                    level=level + 1,
                    summary=merged_summary,
                    start_timestamp=start_ts,
                    end_timestamp=end_ts,
                    token_count=llm.estimate_tokens_fallback(prompt=merged_summary),
                    source_token_count=sum(nd.token_count for nd in chunk),
                    parent_id=None,
                    created_at=time.time(),
                )

                # Save parent summary node and link children roots to it
                await self.db.insert_summary_node(parent_node)
                for nd in chunk:
                    await self.db.update_summary_node_parent(nd.id, parent_id)

                i += K

            level += 1
