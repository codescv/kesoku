"""Lossless Context Management (LCM) expand tool for Kesoku AI Agent."""

import asyncio
import logging

from openlcm.core.tools import (
    lcm_expand as _lcm_expand,
)

from kesoku.agent.tools.registry import ToolContext, default_registry

logger = logging.getLogger(__name__)


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
