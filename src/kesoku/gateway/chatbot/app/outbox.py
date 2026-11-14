"""Per-session sequenced outbox (`app_outbox`) with offline replay.

Only the app adapter allocates `seq`. Every persisted session event keeps its original envelope id, type and
timestamp so replays are byte-for-byte idempotent on clients (deduplicated by `id`).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from kesoku.db.manager import DatabaseManager
from kesoku.gateway.chatbot.app.protocol import SEQ_EVENT_TYPES, make_envelope, new_id
from kesoku.logger import setup_logger

logger = setup_logger(__name__)


class Outbox:
    """Allocates per-session `seq` and replays persisted session events."""

    def __init__(self, db: DatabaseManager, max_per_session: int = 5000, retention_days: int = 30) -> None:
        """Initialize the outbox.

        Args:
            db: Synchronous database manager.
            max_per_session: Events retained per session when pruning.
            retention_days: Age limit in days when pruning (each session's head event is always kept).
        """
        self.db = db
        self.max_per_session = max(1, int(max_per_session))
        self.retention_days = max(1, int(retention_days))

    async def append(self, session_id: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Persist a sequenced session event and return its full envelope (with `seq`)."""
        if event_type not in SEQ_EVENT_TYPES:
            raise ValueError(f"Event type '{event_type}' is not a sequenced session event")
        event_id = new_id("evt")
        ts = time.time()
        seq = await asyncio.to_thread(self.db.append_app_outbox, session_id, event_type, payload, event_id, ts)
        return make_envelope(event_type, payload, session_id=session_id, seq=seq, event_id=event_id, ts=ts)

    async def head(self, session_id: str) -> int:
        """Return the newest `seq` of a session (0 when empty)."""
        _, hi = await asyncio.to_thread(self.db.get_app_outbox_bounds, session_id)
        return hi

    async def heads(self, session_ids: list[str]) -> dict[str, int]:
        """Return newest `seq` per session."""
        return await asyncio.to_thread(self.db.get_app_outbox_heads, session_ids)

    async def replay(self, session_id: str, after_seq: int, limit: int = 500) -> dict[str, Any]:
        """Return replay frames after `after_seq`.

        Returns:
            `{"reset": bool, "events": [envelope...], "head_seq": int, "replayed_through_seq": int,
            "has_more": bool}`. `reset` is True when `after_seq` predates the retained window (or is ahead
            of the head, e.g. after a database reset) and the client must reload a full snapshot.
        """
        after = max(0, int(after_seq))
        lo, hi = await asyncio.to_thread(self.db.get_app_outbox_bounds, session_id)
        if hi == 0:
            return {"reset": after > 0, "events": [], "head_seq": 0, "replayed_through_seq": 0, "has_more": False}
        if after > hi or (after + 1 < lo):
            return {"reset": True, "events": [], "head_seq": hi, "replayed_through_seq": after, "has_more": False}
        rows = await asyncio.to_thread(self.db.get_app_outbox_after, session_id, after, max(1, min(limit, 500)))
        events = [
            make_envelope(
                r["type"],
                r["payload"],
                session_id=session_id,
                seq=r["seq"],
                event_id=r["event_id"] or new_id("evt"),
                ts=r["ts"],
            )
            for r in rows
        ]
        through = events[-1]["seq"] if events else after
        return {
            "reset": False,
            "events": events,
            "head_seq": hi,
            "replayed_through_seq": through,
            "has_more": through < hi,
        }

    async def prune(self) -> int:
        """Apply retention (count + age), always keeping each session's head event."""
        cutoff = time.time() - self.retention_days * 86400.0
        try:
            return await asyncio.to_thread(self.db.prune_app_outbox, self.max_per_session, cutoff)
        except Exception as e:
            logger.warning(f"App outbox pruning failed: {e}")
            return 0
