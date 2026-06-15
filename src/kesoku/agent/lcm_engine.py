"""Custom LCMEngine subclass for Kesoku to support true incremental compression."""

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any

from openlcm import LCMEngine
from openlcm.core.dag import SummaryNode
from openlcm.core.message_content import normalize_content_value
from openlcm.core.tokens import count_messages_tokens, count_tokens

logger = logging.getLogger(__name__)


class KesokuLCMEngine(LCMEngine):
    """Subclass of LCMEngine that overrides compaction and assembly to ensure incremental behavior."""

    def _filter_already_compacted(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter out messages that have already been compacted (store_id <= _last_compacted_store_id)."""
        if not self._session_id or self._last_compacted_store_id <= 0:
            return list(messages)

        try:
            db_msgs = self._store.get_session_messages(self._session_id)
        except Exception as exc:
            logger.warning("Failed to load session messages for filtering: %s", exc)
            return list(messages)

        filtered = []
        db_idx = 0
        for msg in messages:
            role = msg.get("role")
            content = normalize_content_value(msg.get("content")) or ""

            # Find matching message in db_msgs starting from db_idx
            matched_store_id = None
            while db_idx < len(db_msgs):
                db_msg = db_msgs[db_idx]
                db_role = db_msg.get("role")
                db_content = normalize_content_value(db_msg.get("content")) or ""

                if db_role == role and (db_content == content or db_content.startswith(content[:80])):
                    matched_store_id = db_msg.get("store_id")
                    db_idx += 1  # Move past this matched message
                    break
                db_idx += 1

            if matched_store_id is not None and matched_store_id <= self._last_compacted_store_id:
                # Already compacted. We keep system message as it is handled as anchor separately,
                # but if we filter it here we might lose it if _assemble_context expects it.
                # Actually, _assemble_context takes system_message as a separate argument and removes it
                # from remaining_messages if present at index 0.
                # If we filter it here, remaining_messages won't have it, which is fine.
                # But base_sys is extracted before filtering in compress().
                # So keeping system message here is safer to avoid breaking base_sys extraction if we filter earlier.
                # Actually, we filter AFTER extracting base_sys in our new compress flow.
                if role == "system":
                    filtered.append(msg)
            else:
                filtered.append(msg)

        return filtered

    async def compress(
        self,
        messages: list[dict[str, Any]],
        current_tokens: int | None = None,
        focus_topic: str = "",
    ) -> list[dict[str, Any]]:
        """Override compress to filter already compacted messages and always assemble context."""
        if not messages:
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = "empty message list"
            return messages

        if not self._session_id:
            raise RuntimeError(
                "KesokuLCMEngine.compress() called before bind_session(). "
                "Call engine.bind_session('my-session', context_length=N) first."
            )

        if self._session_ignored or self._session_stateless:
            return messages

        observed_prompt_tokens = (
            current_tokens or self.last_prompt_tokens or count_messages_tokens(messages)
        )
        tokens_before = observed_prompt_tokens

        # Determine overflow state before ingesting (signals are pre-ingest)
        force_overflow = self._should_force_overflow_recovery(
            observed_tokens=observed_prompt_tokens,
            messages=messages,
        )
        recovery_assembly_cap = (
            self._overflow_recovery_assembly_cap(
                observed_tokens=observed_prompt_tokens,
                messages=messages,
            )
            if force_overflow
            else None
        )

        self._emit("compaction_start", {
            "session_id": self._session_id,
            "messages_count": len(messages),
            "prompt_tokens": tokens_before,
        })
        self._last_compression_status = "running"

        # Compute memory injection once from original messages (before any compression)
        _memory_injection: str | None = (
            self._build_memory_injection(messages) if self._config.auto_inject_memory else None
        )

        # Step 1: Ingest new messages
        working_messages = self._ingest_messages(messages)

        # Extract system message before filtering
        leading_anchor_count = self._leading_anchor_count(working_messages)
        base_sys = working_messages[0] if leading_anchor_count else None

        # Filter out already compacted messages from the working set
        working_messages = self._filter_already_compacted(working_messages)

        # Overflow recovery: forced convergence when context already exceeds cap
        if force_overflow:
            # Re-evaluate leading anchor on filtered messages (should be just system if kept, or 0)
            filtered_leading = self._leading_anchor_count(working_messages)
            compressed = self._assemble_overflow_recovery_context(
                self._apply_memory_injection(base_sys, _memory_injection),
                working_messages[filtered_leading:],
                assembly_cap_override=recovery_assembly_cap,
            )
            return self._finalize_forced_overflow_result(
                working_messages,  # Note: this might mismatch in logs, but functional flow is correct
                compressed,
                assembly_cap_override=recovery_assembly_cap,
            )

        # Deferred maintenance: run extra leaf passes when backlog debt is recorded
        critical_budget_pressure = self._critical_budget_pressure_reached(
            observed_tokens=observed_prompt_tokens,
            messages=working_messages,
        )
        deferred_maintenance_active = self._should_run_deferred_maintenance(
            working_messages,
            observed_tokens=observed_prompt_tokens,
        )

        leaf_compacted = False
        leaf_passes = 0
        if deferred_maintenance_active:
            max_leaf_passes = max(1, self._config.deferred_maintenance_max_passes)
        elif self._config.dynamic_leaf_chunk_enabled:
            max_leaf_passes = 4
        else:
            max_leaf_passes = 1
        noop_reason = "no eligible raw backlog outside fresh tail"

        # Step 2-5: Leaf compaction loop
        while leaf_passes < max_leaf_passes:
            n = len(working_messages)
            fresh_tail_start = max(0, n - self._config.fresh_tail_count)
            filtered_leading = self._leading_anchor_count(working_messages)

            if fresh_tail_start <= filtered_leading:
                noop_reason = "no eligible raw backlog outside fresh tail"
                break

            candidate_raw = working_messages[filtered_leading:fresh_tail_start]
            if not candidate_raw:
                noop_reason = "no eligible raw backlog outside fresh tail"
                break

            raw_tokens = count_messages_tokens(candidate_raw)
            working_chunk_tokens = self._working_leaf_chunk_tokens(raw_tokens)

            if raw_tokens < working_chunk_tokens:
                noop_reason = "raw backlog below leaf chunk threshold"
                # Deferred maintenance under critical pressure pushes through anyway
                if not (deferred_maintenance_active and critical_budget_pressure):
                    break

            to_compact = (
                candidate_raw if not self._config.dynamic_leaf_chunk_enabled
                else self._select_oldest_leaf_chunk(candidate_raw, working_chunk_tokens)
            )
            if not to_compact:
                noop_reason = "no eligible leaf chunk selected"
                break

            # Pre-compaction extraction (best-effort)
            if self._config.extraction_enabled:
                self._run_pre_compaction_extraction(to_compact)

            # Summarize with rescue
            compacted_chunk, source_tokens, summary_text, _level, _attempts = \
                await self._summarize_leaf_chunk_with_rescue(to_compact, focus_topic=focus_topic)

            source_store_ids = self._get_store_ids_for_messages(compacted_chunk)
            earliest_at, latest_at = self._store.get_time_bounds(source_store_ids)
            summary_tokens = count_tokens(summary_text)

            node = SummaryNode(
                session_id=self._session_id,
                depth=0,
                summary=summary_text,
                token_count=summary_tokens,
                source_token_count=source_tokens,
                source_ids=source_store_ids,
                source_type="messages",
                created_at=time.time(),
                earliest_at=earliest_at,
                latest_at=latest_at,
                expand_hint=self._extract_expand_hint(summary_text),
            )
            self._dag.add_node(node)
            self._last_compacted_store_id = max(source_store_ids) if source_store_ids else 0
            self._persist_frontier_marker()

            self._emit("node_added", {
                "node_id": node.node_id,
                "depth": 0,
                "token_count": summary_tokens,
                "source_token_count": source_tokens,
                "source_ids_count": len(source_store_ids),
            })

            # Auto-extraction: populate fact store from summary (fire-and-forget)
            if self._config.extraction_to_facts_enabled:
                try:
                    asyncio.ensure_future(self._extract_facts_from_summary(summary_text))
                except RuntimeError:
                    pass

            # Semantic embedding of new node (fire-and-forget)
            if self._embeddings.enabled and node.node_id:
                try:
                    asyncio.ensure_future(self._embeddings.embed("node", node.node_id, summary_text))
                except RuntimeError:
                    pass

            # Trim compacted messages from working set
            remaining = working_messages[filtered_leading + len(compacted_chunk):]
            working_messages = working_messages[:filtered_leading] + remaining
            leaf_compacted = True
            leaf_passes += 1

            if not self._config.dynamic_leaf_chunk_enabled and not deferred_maintenance_active:
                break

        # Persist or clear backlog debt based on remaining work
        self._refresh_raw_backlog_debt(
            working_messages,
            observed_tokens=observed_prompt_tokens,
        )

        # Step 6: Condense DAG nodes
        if leaf_compacted:
            await self._maybe_condense(
                focus_topic=focus_topic,
                leaf_compacted_this_turn=True,
                force_overflow=False,
                critical_budget_pressure=critical_budget_pressure,
            )

        # Step 7: Assemble new active context (Always run if we have active nodes, even if noop)
        filtered_leading = self._leading_anchor_count(working_messages)
        compressed = self._assemble_context(
            self._apply_memory_injection(base_sys, _memory_injection),
            working_messages[filtered_leading:],
            assembly_cap_override=recovery_assembly_cap,
        )

        if leaf_compacted:
            self.compression_count += 1
            self._last_compression_status = "compacted"
        else:
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = noop_reason

        self._ingest_cursor = len(compressed)

        tokens_after = count_messages_tokens(compressed)
        logger.info(
            "LCM compaction #%d (KesokuLCMEngine): %d msgs → %d, %d→%d tokens, %d leaf pass(es), %d DAG nodes",
            self.compression_count,
            len(messages),
            len(compressed),
            tokens_before,
            tokens_after,
            leaf_passes,
            len(self._dag.get_session_nodes(self._session_id)),
        )
        self._emit("compaction_end", {
            "session_id": self._session_id,
            "messages_before": len(messages),
            "messages_after": len(compressed),
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "compression_count": self.compression_count,
            "dag_nodes": len(self._dag.get_session_nodes(self._session_id)),
            "leaf_passes": leaf_passes,
        })
        return compressed

    def _assemble_context(
        self,
        system_message: dict[str, Any] | None,
        remaining_messages: list[dict[str, Any]],
        *,
        assembly_cap_override: int | None = None,
    ) -> list[dict[str, Any]]:
        """Override to include all remaining_messages instead of just fresh_tail."""
        session_id = self._session_id

        # Determine effective assembly cap
        cap = assembly_cap_override if assembly_cap_override is not None else self._effective_assembly_token_cap()

        dag_nodes = self._dag.get_active_nodes(session_id)

        if not dag_nodes:
            result: list[dict[str, Any]] = []
            if system_message:
                result.append(system_message)
            result.extend(remaining_messages)
            result = self._sanitize_active_context_messages(result)
            if cap is not None:
                result = self._trim_to_cap(result, cap, system_message)
            return result

        # Group nodes by depth, highest first
        by_depth: dict[int, list[SummaryNode]] = defaultdict(list)
        for node in dag_nodes:
            by_depth[node.depth].append(node)

        # Build LCM scaffold message
        scaffold_parts: list[str] = [
            "[Note: This conversation uses Lossless Context Management (LCM). "
            "Earlier turns have been compacted into hierarchical summaries below. "
            "Use lcm_grep, lcm_expand, or lcm_expand_query to recall specifics.]\n"
        ]
        max_dag_depth = max(by_depth.keys())
        for depth in range(max_dag_depth, -1, -1):
            nodes_at_depth = sorted(by_depth.get(depth, []), key=lambda nd: nd.created_at)
            depth_label = {0: "Recent", 1: "Session Arc", 2: "Durable"}.get(depth, f"Depth-{depth}")
            for node in nodes_at_depth:
                scaffold_parts.append(
                    f"\n[{depth_label} Summary (d{depth}, node {node.node_id})]"
                    f"\n{node.summary}"
                    f"\n[{node.expand_hint or 'Expand for details'}]"
                )
        scaffold_content = "\n".join(scaffold_parts)

        # We keep ALL remaining_messages raw, as they are already filtered to remove compacted ones.
        # This prevents losing "middle" messages that were not compacted and not in the protected tail.
        fresh_tail = remaining_messages

        result = []
        if system_message:
            result.append(system_message)
        result.append({"role": "user", "content": scaffold_content})
        result.append({
            "role": "assistant",
            "content": "Understood. I have access to the full conversation history through LCM tools.",
        })
        result.extend(fresh_tail)

        result = self._sanitize_active_context_messages(result)

        if cap is not None:
            result = self._trim_to_cap(result, cap, system_message)

        return result
