"""`memory.proposal.*` requests and `memory.proposal` / `memory.proposal.resolved` events for the app channel."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from kesoku.agent.memory_proposals import (
    PROPOSAL_TEXT_MAX_CHARS,
    ProposalError,
    ProposalStore,
)
from kesoku.gateway.chatbot.app.host import RequestContext
from kesoku.gateway.chatbot.app.protocol import PROPOSAL_STATUSES, ProtocolError, bounded_int
from kesoku.logger import setup_logger

logger = setup_logger(__name__)


def proposal_ops_for_wire(ops: Any) -> list[dict[str, Any]]:
    """Project stored ops to the client shape `{file, section, text, tags[]}` (+ `pinned` when set)."""
    out: list[dict[str, Any]] = []
    for op in ops or []:
        if not isinstance(op, dict):
            continue
        item: dict[str, Any] = {
            "file": str(op.get("file") or ""),
            "section": str(op.get("section") or ""),
            "text": str(op.get("text") or "")[:PROPOSAL_TEXT_MAX_CHARS],
            "tags": [str(t) for t in (op.get("tags") or []) if isinstance(t, str)],
        }
        if op.get("pinned"):
            item["pinned"] = True
        out.append(item)
    return out


def proposal_for_wire(p: dict[str, Any]) -> dict[str, Any]:
    """Serialize a stored proposal for clients."""
    status = p.get("status")
    out: dict[str, Any] = {
        "proposal_id": p["proposal_id"],
        "session_id": p["session_id"],
        "message_id": p.get("message_id"),
        "role": p.get("role"),
        "source_mode": p.get("source_mode"),
        "ops": proposal_ops_for_wire(p.get("ops")),
        "status": status if status in PROPOSAL_STATUSES else "pending",
        "created_at": p.get("created_at"),
        "expires_at": p.get("expires_at"),
    }
    if p.get("resolved_at") is not None:
        out["resolved_at"] = p["resolved_at"]
    if p.get("applied_ops"):
        out["applied_ops"] = proposal_ops_for_wire(p.get("applied_ops"))
    return out


class ProposalService:
    """Bridges `ProposalStore` to the wire protocol."""

    def __init__(
        self,
        store: ProposalStore,
        emit_session_event: Callable[[str, str, dict[str, Any]], Awaitable[Any]],
    ) -> None:
        """Initialize the service.

        Args:
            store: Proposal store (shares its listener list with the memory consolidator).
            emit_session_event: Async `(session_id, type, payload)` emitting a sequenced, broadcast event.
        """
        self.store = store
        self._emit = emit_session_event

    async def on_staged(self, proposal: dict[str, Any]) -> None:
        """Listener for newly staged proposals: emit a sequenced `memory.proposal` event."""
        payload = proposal_for_wire(proposal)
        payload.pop("status", None)
        # `message_id` is the triggering user message, i.e. the turn id clients key assistant replies by.
        if payload.get("message_id"):
            payload["turn_id"] = payload["message_id"]
        await self._emit(proposal["session_id"], "memory.proposal", payload)

    async def handle_list(self, ctx: RequestContext, payload: dict[str, Any]) -> dict[str, Any]:
        """`memory.proposal.list{status?, limit}` -> `{proposals:[...]}`."""
        status = payload.get("status")
        if status is not None and status not in PROPOSAL_STATUSES:
            raise ProtocolError("BAD_REQUEST", "invalid status filter")
        limit = bounded_int(payload, "limit", 50, 1, 200)
        rows = await asyncio.to_thread(self.store.list_sync, status, limit)
        return {"proposals": [proposal_for_wire(p) for p in rows]}

    async def handle_resolve(self, ctx: RequestContext, payload: dict[str, Any]) -> dict[str, Any]:
        """`memory.proposal.resolve{proposal_id, action, edited_ops?}` — the only `confirm_only` write entry."""
        proposal_id = payload.get("proposal_id")
        action = payload.get("action")
        if not isinstance(proposal_id, str) or not proposal_id or len(proposal_id) > 64:
            raise ProtocolError("BAD_REQUEST", "proposal_id required")
        if action not in ("confirm", "dismiss"):
            raise ProtocolError("BAD_REQUEST", "action must be confirm or dismiss")
        edited = payload.get("edited_ops")
        try:
            if action == "confirm":
                result = await asyncio.to_thread(self.store.confirm_sync, proposal_id, edited, ctx.device_id)
            else:
                result = await asyncio.to_thread(self.store.dismiss_sync, proposal_id, ctx.device_id)
        except ProposalError as e:
            raise ProtocolError(e.code, str(e)) from e
        event: dict[str, Any] = {"proposal_id": proposal_id, "status": result["status"]}
        if result.get("applied_files") is not None:
            event["applied_files"] = result["applied_files"]
        await self._emit(result["session_id"], "memory.proposal.resolved", event)
        return {"ok": True, "proposal_id": proposal_id, "status": result["status"], **(
            {"applied_files": result["applied_files"]} if "applied_files" in result else {}
        )}

    async def expire_due(self) -> int:
        """Expire overdue pending proposals and broadcast `memory.proposal.resolved{status:"expired"}`."""
        expired = await asyncio.to_thread(self.store.expire_sync)
        for p in expired:
            try:
                await self._emit(p["session_id"], "memory.proposal.resolved", {
                    "proposal_id": p["proposal_id"], "status": "expired"
                })
            except Exception as e:
                logger.warning(f"Failed to broadcast proposal expiry {p.get('proposal_id')}: {e}")
        return len(expired)
