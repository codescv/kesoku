"""Per-mode, user-configurable memory policy (authoritative override) for the app channel.

The current mode `M` of a session is read generically from `$STAGING_DIR/game/games.json` (the active
top-level instance's `game` id, or the default chat mode when nothing is active). `app_mode_memory_prefs[M]`
is then returned as the authoritative memory policy, overriding static frontmatter / glob defaults.
Unknown modes (not configured) or missing rows return None so the static resolution applies.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
from typing import Any

from kesoku.config import APP_MEMORY_POLICIES, DEFAULT_APP_CHAT_MODE_ID, AppConfig
from kesoku.db.manager import DatabaseManager
from kesoku.gateway.chatbot.app.host import Reply, RequestContext
from kesoku.gateway.chatbot.app.protocol import ProtocolError
from kesoku.logger import setup_logger

logger = setup_logger(__name__)

GAMES_STATE_REL = os.path.join("game", "games.json")
GAMES_STATE_MAX_BYTES = 512 * 1024


def read_active_mode(staging_dir: str | None, default_mode: str = DEFAULT_APP_CHAT_MODE_ID) -> str:
    """Return the session's current mode id from `game/games.json` (default mode when none is active).

    Accepts either a list of instances or `{"instances": [...]}`. Active entries are those with
    `active` true (default) and not `archived`. The last active top-level entry (no `parent`) wins; nested
    sub-instances resolve to their outermost parent's id.
    """
    if not staging_dir:
        return default_mode
    path = os.path.join(staging_dir, GAMES_STATE_REL)
    try:
        if not os.path.isfile(path) or os.path.getsize(path) > GAMES_STATE_MAX_BYTES:
            return default_mode
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return default_mode
    instances = raw.get("instances") if isinstance(raw, dict) else raw
    if not isinstance(instances, list):
        return default_mode
    active = [
        i
        for i in instances
        if isinstance(i, dict)
        and i.get("active", True) is not False
        and not i.get("archived")
        and isinstance(i.get("game"), str)
        and i.get("game")
    ]
    if not active:
        return default_mode
    top_level = [i for i in active if not i.get("parent")]
    chosen = top_level[-1] if top_level else active[0]
    return str(chosen["game"])


class MemoryPolicyService:
    """Serves `memory.policy.*` requests and the authoritative override provider."""

    def __init__(
        self,
        db: DatabaseManager,
        app_config: AppConfig,
        staging_dir_for_sync: Callable[[str], str | None],
        broadcast: Callable[[str, dict[str, Any]], Any],
    ) -> None:
        """Initialize the service.

        Args:
            db: Synchronous database manager.
            app_config: App channel configuration (mode list + factory defaults).
            staging_dir_for_sync: Synchronous resolver `session_id -> staging dir`.
            broadcast: Async callable `(event_type, payload)` sending to every device.
        """
        self.db = db
        self.app_config = app_config
        self._staging_dir_for = staging_dir_for_sync
        self._broadcast = broadcast

    def seed_defaults_sync(self) -> int:
        """`INSERT OR IGNORE` factory defaults for every configured mode (never overrides user rows)."""
        return self.db.seed_app_mode_memory_prefs(self.app_config.effective_memory_policy_defaults())

    # ---------------------------------------------------------------- provider (sync; runs on worker threads)
    def current_mode(self, session_id: str) -> str:
        """Return the session's current mode id."""
        return read_active_mode(self._staging_dir_for(session_id))

    def resolve_override(self, session_id: str) -> str | None:
        """Return the user-configured policy for the session's current mode, or None to keep static rules."""
        mode_id = self.current_mode(session_id)
        if mode_id not in self.app_config.mode_ids:
            return None
        policy = self.db.get_app_mode_memory_policy(mode_id)
        return policy if policy in APP_MEMORY_POLICIES else None

    # ---------------------------------------------------------------- requests
    def _policies_sync(self) -> tuple[dict[str, str], dict[str, str], float]:
        defaults = self.app_config.effective_memory_policy_defaults()
        rows = self.db.list_app_mode_memory_prefs()
        policies: dict[str, str] = {}
        updated_at = 0.0
        for mode_id in self.app_config.mode_ids:
            row = rows.get(mode_id)
            if row is not None:
                policies[mode_id] = row["memory_policy"]
                updated_at = max(updated_at, float(row["updated_at"]))
            else:
                policies[mode_id] = defaults.get(mode_id, "confirm_only")
        return policies, {m: defaults[m] for m in self.app_config.mode_ids if m in defaults}, updated_at

    async def handle_list(self, ctx: RequestContext, payload: dict[str, Any]) -> dict[str, Any] | Reply:
        """`memory.policy.list` -> `memory.policy.list.result{policies, defaults, updated_at}`."""
        policies, defaults, updated_at = await asyncio.to_thread(self._policies_sync)
        return {"policies": policies, "defaults": defaults, "updated_at": updated_at}

    async def handle_set(self, ctx: RequestContext, payload: dict[str, Any]) -> dict[str, Any]:
        """`memory.policy.set{mode_id, memory_policy}`: UPSERT + broadcast `memory.policy.updated` (no LLM turn)."""
        mode_id = payload.get("mode_id")
        policy = payload.get("memory_policy")
        if not isinstance(mode_id, str) or mode_id not in self.app_config.mode_ids:
            raise ProtocolError("BAD_REQUEST", "unknown mode_id")
        if policy not in APP_MEMORY_POLICIES:
            raise ProtocolError("BAD_REQUEST", "memory_policy must be one of full / confirm_only / isolated")
        stored_at = await asyncio.to_thread(self.db.set_app_mode_memory_policy, mode_id, policy, time.time())
        policies, _, updated_at = await asyncio.to_thread(self._policies_sync)
        await self._broadcast(
            "memory.policy.updated",
            {"policies": policies, "changed": [mode_id], "updated_at": max(updated_at, stored_at)},
        )
        return {"mode_id": mode_id, "memory_policy": policy, "updated_at": stored_at}
