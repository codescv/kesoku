"""Lossless Context Management (LCM) search, view, and status query tools for Kesoku AI Agent."""

import asyncio
import json
import logging
from typing import Any

from openlcm.core.tools import (
    lcm_describe as _lcm_describe,
)
from openlcm.core.tools import (
    lcm_expand as _lcm_expand,
)
from openlcm.core.tools import (
    lcm_expand_query as _lcm_expand_query,
)
from openlcm.core.tools import (
    lcm_grep as _lcm_grep,
)
from openlcm.core.tools import (
    lcm_status as _lcm_status,
)

from kesoku.agent.tools.memory import _resolve_memory_role
from kesoku.agent.tools.registry import ToolContext, default_registry

logger = logging.getLogger(__name__)






@default_registry.register
async def lcm_grep(
    query: str,
    limit: int = 10,
    source: str | None = None,
    role: str | None = None,
    time_from: Any = None,
    time_to: Any = None,
    sort: str | None = None,
    context: ToolContext | None = None,
) -> str:
    """Search raw history messages and summaries across all sessions of the current role.

    Args:
        query: Search query keywords (supports exact phrase quoting).
        limit: Maximum result hits to return.
        source: Filter by message sender or file/tool source name.
        role: Filter by message sender role (system, user, assistant, tool).
        time_from: Filter messages after this Unix timestamp or timezone-aware ISO 8601 time.
        time_to: Filter messages before this Unix timestamp or timezone-aware ISO 8601 time.
        sort: Result sorting preference: 'recency', 'relevance', 'hybrid'.
        context: Injected tool execution context (automatically resolved).
    """
    if not context or not context.gateway:
        return "Error: ToolContext is missing."

    # Resolve active role to restrict session access to current persona memory
    active_role = await _resolve_memory_role(category="user_preferences", role_param=None, context=context)
    allowed_sessions = await context.gateway.db.get_role_session_ids(active_role)
    allowed_sessions_set = set(allowed_sessions)

    lcm_engine = context.lcm_engine
    args = {
        "query": query,
        "limit": limit,
        "session_scope": "all",
        "session_id": None,
        "source": source,
        "role": role,
        "time_from": time_from,
        "time_to": time_to,
        "sort": sort,
    }
    raw_response = await asyncio.to_thread(_lcm_grep, args, engine=lcm_engine)

    try:
        data = json.loads(raw_response)
        if "results" in data:
            data["results"] = data["results"][:limit]
            data["total_results"] = len(data["results"])
            return json.dumps(data)
    except Exception as e:
        logger.error(f"Failed to post-filter lcm_grep results: {e}")

    return raw_response


@default_registry.register
async def lcm_expand(
    node_id: int | None = None,
    store_id: int | None = None,
    externalized_ref: str | None = None,
    max_tokens: int = 4000,
    source_offset: int = 0,
    content_offset: int = 0,
    context: ToolContext | None = None,
) -> str:
    """Expand a summary node, externalized payload, or raw message by its ID to read its full, uncompacted text.

    Args:
        node_id: Expand a summary node to its child sources (current session only).
        store_id: Fetch a single raw message by its unique database store ID. Works across sessions.
        externalized_ref: Expand an externalized tool result file by its reference string.
        max_tokens: Token budget for the returned content slice.
        source_offset: Offset cursor for paginated list index results.
        content_offset: Character offset cursor for paginated text slices.
        context: Injected tool execution context (automatically resolved).
    """
    if not context or not context.gateway:
        return "Error: ToolContext is missing."
    lcm_engine = context.lcm_engine
    args = {
        "node_id": node_id,
        "store_id": store_id,
        "externalized_ref": externalized_ref,
        "max_tokens": max_tokens,
        "source_offset": source_offset,
        "content_offset": content_offset,
    }
    return await asyncio.to_thread(_lcm_expand, args, engine=lcm_engine)


@default_registry.register
async def lcm_expand_query(
    prompt: str,
    query: str | None = None,
    node_ids: list[int] | None = None,
    max_tokens: int = 2000,
    context_max_tokens: int | None = None,
    max_results: int = 5,
    context: ToolContext | None = None,
) -> str:
    """Answer a specific question by searching, expanding, and synthesizing compacted summary nodes.

    Args:
        prompt: The target question or instruction to solve.
        query: Search query keywords to find relevant compacted summary nodes to expand.
        node_ids: Explicit list of summary node IDs to expand.
        max_tokens: Token budget for the generated answer.
        context_max_tokens: Maximum total tokens of uncompacted history to feed into the query synthesis model.
        max_results: Maximum number of compacted summary nodes to expand.
        context: Injected tool execution context (automatically resolved).
    """
    if not context or not context.gateway:
        return "Error: ToolContext is missing."
    lcm_engine = context.lcm_engine
    args = {
        "prompt": prompt,
        "query": query,
        "node_ids": node_ids,
        "max_tokens": max_tokens,
        "context_max_tokens": context_max_tokens,
        "max_results": max_results,
    }
    return await asyncio.to_thread(_lcm_expand_query, args, engine=lcm_engine)


@default_registry.register
async def lcm_describe(
    node_id: int | None = None,
    externalized_ref: str | None = None,
    context: ToolContext | None = None,
) -> str:
    """Retrieve structural overview of the session memory hierarchy or inspect a node's immediate subtree.

    Args:
        node_id: Return children summaries and token statistics of this summary node.
        externalized_ref: Preview file details of an externalized payload.
        context: Injected tool execution context (automatically resolved).
    """
    if not context or not context.gateway:
        return "Error: ToolContext is missing."
    lcm_engine = context.lcm_engine
    args = {
        "node_id": node_id,
        "externalized_ref": externalized_ref,
    }
    return await asyncio.to_thread(_lcm_describe, args, engine=lcm_engine)


@default_registry.register
async def lcm_status(context: ToolContext | None = None) -> str:
    """Get a quick health overview of the active session, compaction metrics, and token usage statistics.

    Args:
        context: Injected tool execution context (automatically resolved).
    """
    if not context or not context.gateway:
        return "Error: ToolContext is missing."
    lcm_engine = context.lcm_engine
    return await asyncio.to_thread(_lcm_status, {}, engine=lcm_engine)


async def _ensure_embeddings_indexed(engine: Any) -> None:
    es = getattr(engine, "_embeddings", None)
    if not es or not es.enabled or not getattr(es, "_conn", None):
        return

    conn = es._conn
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS lcm_embeddings (
                emb_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                content_type  TEXT NOT NULL,
                content_id    INTEGER NOT NULL,
                model         TEXT NOT NULL DEFAULT '',
                embedding     BLOB NOT NULL,
                created_at    REAL NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_emb_content
                ON lcm_embeddings(content_type, content_id, model);
        """)
    except Exception as e:
        logger.error(f"Failed to ensure lcm_embeddings table: {e}")
        return

    model_name = es.embedding_model
    try:
        nodes = conn.execute(
            """
            SELECT node_id, summary FROM summary_nodes
            WHERE node_id NOT IN (
                SELECT content_id FROM lcm_embeddings
                WHERE content_type = 'node' AND model = ?
            )
            """,
            (model_name,),
        ).fetchall()
        for node_id, summary in nodes:
            if summary and summary.strip():
                await es.embed("node", node_id, summary)
    except Exception as e:
        logger.error(f"Failed to backfill node embeddings: {e}")

    try:
        facts = conn.execute(
            """
            SELECT fact_id, value FROM lcm_facts
            WHERE fact_id NOT IN (
                SELECT content_id FROM lcm_embeddings
                WHERE content_type = 'fact' AND model = ?
            )
            """,
            (model_name,),
        ).fetchall()
        for fact_id, value in facts:
            if value and value.strip():
                await es.embed("fact", fact_id, value)
    except Exception as e:
        logger.error(f"Failed to backfill fact embeddings: {e}")


@default_registry.register
async def lcm_semantic_search(
    query: str,
    limit: int = 10,
    content_type: str = "all",
    context: ToolContext | None = None,
) -> str:
    """Semantic similarity search across LCM nodes and facts using embeddings.

    Args:
        query: Natural language search query.
        limit: Maximum results to return.
        content_type: Content type to search: 'all', 'node', 'fact'.
        context: Injected tool execution context (automatically resolved).
    """
    if not context or not context.gateway:
        return "Error: ToolContext is missing."

    active_role = await _resolve_memory_role(category="user_preferences", role_param=None, context=context)
    allowed_sessions = await context.gateway.db.get_role_session_ids(active_role)
    allowed_sessions_set = set(allowed_sessions)

    lcm_engine = context.lcm_engine
    es = getattr(lcm_engine, "_embeddings", None)
    if not es:
        return json.dumps({"results": [], "total": 0})

    if not es.enabled:
        es._init_db()
    await _ensure_embeddings_indexed(lcm_engine)

    try:
        target_ct = content_type if content_type in ("node", "fact") else None
        hits = await es.search(query, content_type=target_ct, limit=limit * 3)
        filtered = []
        for hit in hits:
            ct = hit.get("content_type")
            cid = hit.get("content_id")
            if ct == "node":
                node = lcm_engine._dag.get_node(cid) if hasattr(lcm_engine._dag, "get_node") else None
                if node and node.session_id in allowed_sessions_set:
                    hit["session_id"] = node.session_id
                    hit["summary_preview"] = node.summary[:300] if node.summary else ""
                    hit["depth"] = node.depth
                    hit["created_at"] = node.created_at
                    filtered.append(hit)
            elif ct == "fact":
                row = es._conn.execute(
                    "SELECT scope, key, value, created_at, source_session_id FROM lcm_facts WHERE fact_id = ?",
                    (cid,),
                ).fetchone()
                if row:
                    scope, key, value, created_at, source_sess = row
                    if (
                        scope in allowed_sessions_set
                        or (source_sess and source_sess in allowed_sessions_set)
                        or scope == "global"
                    ):
                        hit["scope"] = scope
                        hit["key"] = key
                        hit["value"] = value
                        hit["created_at"] = created_at
                        filtered.append(hit)

        results = filtered[:limit]
        return json.dumps({
            "query": query,
            "content_type": content_type,
            "limit": limit,
            "total": len(results),
            "results": results,
            "model": es.embedding_model,
        })
    except Exception as exc:
        logger.error(f"Semantic search error: {exc}")
        return json.dumps({"error": str(exc), "results": []})
