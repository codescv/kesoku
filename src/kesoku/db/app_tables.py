"""SQLite schema (migration v4) and data-access helpers for the app channel and memory proposals.

Tables:
  * `app_devices` / `app_pairings`: paired client devices and short-lived pairing offers.
  * `app_outbox`: per-session monotonically sequenced outbound events (offline replay cursor).
  * `app_session_prefs`: per-session client preferences (LLM override, mode prefs, origin channel).
  * `app_mode_memory_prefs`: user-configurable per-mode memory policy (authoritative override).
  * `memory_proposals` / `memory_proposal_blocklist`: `confirm_only` memory candidates awaiting the user.
  * `avatar_jobs`: asynchronous avatar restyle jobs.

Every statement is parameterized. The mixin is combined into `DatabaseManager` and relies on its
`connection_provider`.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from kesoku.logger import setup_logger

logger = setup_logger(__name__)

APP_SCHEMA_VERSION = 4

APP_MEMORY_POLICY_VALUES = ("full", "confirm_only", "isolated")
PROPOSAL_STATUSES = ("pending", "applied", "dismissed", "expired")

_V4_STATEMENTS: tuple[tuple[str, str], ...] = (
    (
        "app_devices",
        """
        CREATE TABLE IF NOT EXISTS app_devices (
            device_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            platform TEXT,
            created_at REAL NOT NULL,
            last_seen_at REAL,
            revoked_at REAL
        )
        """,
    ),
    (
        "app_pairings",
        """
        CREATE TABLE IF NOT EXISTS app_pairings (
            code_hash TEXT PRIMARY KEY,
            salt TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL,
            redeemed_device_id TEXT
        )
        """,
    ),
    (
        "app_outbox",
        """
        CREATE TABLE IF NOT EXISTS app_outbox (
            session_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            type TEXT NOT NULL,
            payload TEXT NOT NULL,
            ts REAL NOT NULL,
            event_id TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (session_id, seq)
        )
        """,
    ),
    (
        "app_session_prefs",
        """
        CREATE TABLE IF NOT EXISTS app_session_prefs (
            session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
            llm_override TEXT,
            mode_prefs TEXT,
            origin_channel TEXT,
            created_by_device TEXT,
            updated_at REAL NOT NULL
        )
        """,
    ),
    (
        "app_mode_memory_prefs",
        """
        CREATE TABLE IF NOT EXISTS app_mode_memory_prefs (
            mode_id TEXT PRIMARY KEY,
            memory_policy TEXT NOT NULL CHECK (memory_policy IN ('full', 'confirm_only', 'isolated')),
            updated_at REAL NOT NULL
        )
        """,
    ),
    (
        "memory_proposals",
        """
        CREATE TABLE IF NOT EXISTS memory_proposals (
            proposal_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            message_id TEXT,
            role TEXT NOT NULL,
            source_mode TEXT,
            ops TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL,
            resolved_at REAL,
            resolved_by_device TEXT,
            applied_ops TEXT
        )
        """,
    ),
    (
        "idx_memory_proposals_status",
        "CREATE INDEX IF NOT EXISTS idx_memory_proposals_status ON memory_proposals(status, created_at)",
    ),
    (
        "memory_proposal_blocklist",
        """
        CREATE TABLE IF NOT EXISTS memory_proposal_blocklist (
            fingerprint TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            dismissed_at REAL NOT NULL
        )
        """,
    ),
    (
        "avatar_jobs",
        """
        CREATE TABLE IF NOT EXISTS avatar_jobs (
            job_id TEXT PRIMARY KEY,
            device_id TEXT NOT NULL,
            target_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            style_preset TEXT NOT NULL,
            prompt_hint TEXT,
            source_path TEXT NOT NULL,
            candidates TEXT,
            status TEXT NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL
        )
        """,
    ),
    (
        "seed_chat_policy",
        # Only the generic default chat mode is seeded here; configured modes are seeded by the app
        # adapter at startup via `seed_app_mode_memory_prefs` (OR IGNORE never overrides user rows).
        "INSERT OR IGNORE INTO app_mode_memory_prefs (mode_id, memory_policy, updated_at) "
        "VALUES ('chat', 'full', CAST(strftime('%s','now') AS REAL))",
    ),
)


def _json_or_none(value: Any) -> str | None:
    """Serialize `value` to compact JSON, or None when value is None."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads(raw: Any, default: Any = None) -> Any:
    """Deserialize a JSON column defensively."""
    if raw is None or raw == "":
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


class AppTablesMixin:
    """Data-access helpers for app channel and memory proposal tables (mixed into `DatabaseManager`)."""

    connection_provider: Any

    # ------------------------------------------------------------------ migration
    def _apply_app_v4_migration(self, conn: sqlite3.Connection) -> bool:
        """Create app/proposal tables idempotently and bump `PRAGMA user_version` to 4 when all succeed."""
        ok = True
        for label, stmt in _V4_STATEMENTS:
            try:
                with conn:
                    conn.execute(stmt)
            except Exception as e:
                ok = False
                logger.error(f"Migration v4 ({label}) failed: {e}")
        if ok:
            try:
                current = int(conn.execute("PRAGMA user_version").fetchone()[0])
                if 3 <= current < APP_SCHEMA_VERSION:
                    with conn:
                        conn.execute(f"PRAGMA user_version = {APP_SCHEMA_VERSION}")
            except Exception as e:
                logger.warning(f"Failed setting PRAGMA user_version = {APP_SCHEMA_VERSION}: {e}")
        return ok

    # ------------------------------------------------------------------ sessions / history
    def list_sessions_by_chatbots(
        self,
        chatbot_ids: Sequence[str],
        role: str | None = None,
        cursor: tuple[float, str] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List sessions bound to any of `chatbot_ids`, newest first, with keyset pagination.

        Args:
            chatbot_ids: Chatbot identifiers whose channel bindings qualify a session.
            role: Optional persona role filter.
            cursor: Optional `(updated_at, session_id)` keyset; only strictly older sessions are returned.
            limit: Maximum number of sessions.

        Returns:
            Dicts with session columns plus `chatbot_ids` (all bindings of the session, sorted).
        """
        ids = [c for c in chatbot_ids if c]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        clauses = [f"s.id IN (SELECT session_id FROM channel_sessions WHERE chatbot_id IN ({placeholders}))"]
        params: list[Any] = list(ids)
        if role:
            clauses.append("COALESCE(s.role_name, 'default') = ?")
            params.append(role)
        if cursor is not None:
            clauses.append("(s.updated_at < ? OR (s.updated_at = ? AND s.id < ?))")
            params.extend([float(cursor[0]), float(cursor[0]), str(cursor[1])])
        params.append(max(1, min(int(limit), 500)))
        sql = (
            "SELECT s.id, s.title, s.created_at, s.updated_at, COALESCE(s.role_name, 'default') AS role_name, "
            "(SELECT GROUP_CONCAT(chatbot_id, ',') FROM channel_sessions cs WHERE cs.session_id = s.id) "
            "AS bindings FROM sessions s WHERE "
            + " AND ".join(clauses)
            + " ORDER BY s.updated_at DESC, s.id DESC LIMIT ?"
        )
        with self.connection_provider.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            bindings = sorted({b for b in str(r["bindings"] or "").split(",") if b})
            out.append(
                {
                    "session_id": r["id"],
                    "title": r["title"],
                    "created_at": float(r["created_at"]),
                    "updated_at": float(r["updated_at"]),
                    "role": r["role_name"],
                    "chatbot_ids": bindings,
                }
            )
        return out

    def get_messages_page(
        self,
        session_id: str,
        before_ts: float | None = None,
        limit: int = 50,
        visible_only: bool = True,
    ) -> list[Any]:
        """Return up to `limit` messages strictly older than `before_ts`, in chronological order.

        Args:
            session_id: Session identifier.
            before_ts: Exclusive upper timestamp bound (None = newest).
            limit: Maximum number of messages.
            visible_only: Only user / assistant text messages (excludes thoughts, tools, notifications).

        Returns:
            Message models oldest to newest.
        """
        clauses = ["session_id = ?"]
        params: list[Any] = [session_id]
        if before_ts is not None:
            clauses.append("timestamp < ?")
            params.append(float(before_ts))
        if visible_only:
            clauses.append(
                "role IN ('user', 'assistant') AND type = 'text' AND sender <> 'Notification' "
                "AND trim(content) <> 'Processed request successfully.'"
            )
        params.append(max(1, min(int(limit), 500)))
        sql = "SELECT * FROM messages WHERE " + " AND ".join(clauses) + " ORDER BY timestamp DESC, id DESC LIMIT ?"
        with self.connection_provider.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        msgs = [self._row_to_message(r) for r in rows]  # type: ignore[attr-defined]
        msgs.reverse()
        return msgs

    def list_channel_bindings(self, session_id: str) -> list[tuple[str, str]]:
        """Return all `(chatbot_id, channel_id)` bindings for `session_id`."""
        with self.connection_provider.connection() as conn:
            rows = conn.execute(
                "SELECT chatbot_id, channel_id FROM channel_sessions WHERE session_id = ? ORDER BY chatbot_id",
                (session_id,),
            ).fetchall()
        return [(str(r["chatbot_id"]), str(r["channel_id"])) for r in rows]

    def bind_channel_session_if_absent(self, chatbot_id: str, channel_id: str, session_id: str) -> bool:
        """Insert a `channel_sessions` binding without touching existing rows; returns True when inserted."""
        with self.connection_provider.connection() as conn:
            with conn:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO channel_sessions (chatbot_id, channel_id, session_id) VALUES (?, ?, ?)",
                    (chatbot_id, channel_id, session_id),
                )
                return cur.rowcount > 0

    def update_session_role(self, session_id: str, role: str) -> None:
        """Set `sessions.role_name` for one session."""
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute("UPDATE sessions SET role_name = ? WHERE id = ?", (role, session_id))

    def reassign_role_references(self, old_role: str, new_role: str) -> int:
        """Point `channel_roles` / `sessions.role_name` referencing `old_role` to `new_role`; returns rows."""
        with self.connection_provider.connection() as conn:
            with conn:
                c1 = conn.execute("UPDATE channel_roles SET role = ? WHERE role = ?", (new_role, old_role))
                c2 = conn.execute("UPDATE sessions SET role_name = ? WHERE role_name = ?", (new_role, old_role))
                return int(c1.rowcount) + int(c2.rowcount)

    # ------------------------------------------------------------------ devices
    def upsert_app_device(
        self,
        device_id: str,
        name: str,
        kind: str,
        platform: str | None = None,
        created_at: float | None = None,
    ) -> None:
        """Insert a device, or refresh its name/platform when it already exists (revocation is kept)."""
        now = float(created_at if created_at is not None else time.time())
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO app_devices (device_id, name, kind, platform, created_at, last_seen_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(device_id) DO UPDATE SET
                        name = excluded.name,
                        platform = COALESCE(excluded.platform, app_devices.platform)
                    """,
                    (device_id, name[:80], kind, platform, now, now),
                )

    def get_app_device(self, device_id: str) -> dict[str, Any] | None:
        """Return a device row as a dict, or None."""
        with self.connection_provider.connection() as conn:
            row = conn.execute("SELECT * FROM app_devices WHERE device_id = ?", (device_id,)).fetchone()
        return dict(row) if row else None

    def list_app_devices(self, include_revoked: bool = True) -> list[dict[str, Any]]:
        """List devices ordered by creation time (newest first)."""
        sql = "SELECT * FROM app_devices"
        if not include_revoked:
            sql += " WHERE revoked_at IS NULL"
        sql += " ORDER BY created_at DESC"
        with self.connection_provider.connection() as conn:
            return [dict(r) for r in conn.execute(sql).fetchall()]

    def revoke_app_device(self, device_id: str, ts: float | None = None) -> bool:
        """Mark a device revoked; returns True when a non-revoked device was revoked."""
        with self.connection_provider.connection() as conn:
            with conn:
                cur = conn.execute(
                    "UPDATE app_devices SET revoked_at = ? WHERE device_id = ? AND revoked_at IS NULL",
                    (float(ts if ts is not None else time.time()), device_id),
                )
                return cur.rowcount > 0

    def touch_app_device(self, device_id: str, ts: float | None = None) -> None:
        """Update `last_seen_at` for a device."""
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute(
                    "UPDATE app_devices SET last_seen_at = ? WHERE device_id = ?",
                    (float(ts if ts is not None else time.time()), device_id),
                )

    def list_revoked_app_device_ids(self) -> list[str]:
        """Return identifiers of all revoked devices."""
        with self.connection_provider.connection() as conn:
            rows = conn.execute("SELECT device_id FROM app_devices WHERE revoked_at IS NOT NULL").fetchall()
        return [str(r["device_id"]) for r in rows]

    def revoke_inactive_app_devices(self, cutoff_ts: float, now: float | None = None) -> list[str]:
        """Revoke devices whose `last_seen_at` (or creation) is older than `cutoff_ts`; returns their ids."""
        ts = float(now if now is not None else time.time())
        with self.connection_provider.connection() as conn:
            with conn:
                rows = conn.execute(
                    "SELECT device_id FROM app_devices WHERE revoked_at IS NULL "
                    "AND COALESCE(last_seen_at, created_at) < ?",
                    (float(cutoff_ts),),
                ).fetchall()
                ids = [str(r["device_id"]) for r in rows]
                for dev in ids:
                    conn.execute("UPDATE app_devices SET revoked_at = ? WHERE device_id = ?", (ts, dev))
        return ids

    # ------------------------------------------------------------------ pairings
    def insert_app_pairing(self, code_hash: str, salt: str, name: str, created_at: float, expires_at: float) -> None:
        """Store a pairing offer; older unredeemed offers are dropped (only one active offer at a time)."""
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute("DELETE FROM app_pairings WHERE redeemed_device_id IS NULL")
                conn.execute(
                    "INSERT INTO app_pairings (code_hash, salt, name, created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
                    (code_hash, salt, name[:80], float(created_at), float(expires_at)),
                )

    def get_active_app_pairing(self, now: float | None = None) -> dict[str, Any] | None:
        """Return the newest unredeemed, unexpired pairing offer."""
        ts = float(now if now is not None else time.time())
        with self.connection_provider.connection() as conn:
            row = conn.execute(
                "SELECT * FROM app_pairings WHERE redeemed_device_id IS NULL AND expires_at > ? "
                "ORDER BY created_at DESC LIMIT 1",
                (ts,),
            ).fetchone()
        return dict(row) if row else None

    def get_app_pairing(self, code_hash: str) -> dict[str, Any] | None:
        """Return one pairing row by hash."""
        with self.connection_provider.connection() as conn:
            row = conn.execute("SELECT * FROM app_pairings WHERE code_hash = ?", (code_hash,)).fetchone()
        return dict(row) if row else None

    def mark_app_pairing_redeemed(self, code_hash: str, device_id: str) -> bool:
        """Record which device redeemed an offer; returns True on success."""
        with self.connection_provider.connection() as conn:
            with conn:
                cur = conn.execute(
                    "UPDATE app_pairings SET redeemed_device_id = ? WHERE code_hash = ? AND redeemed_device_id IS NULL",
                    (device_id, code_hash),
                )
                return cur.rowcount > 0

    def delete_app_pairing(self, code_hash: str) -> None:
        """Delete a pairing offer (e.g. burned by failed attempts)."""
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute("DELETE FROM app_pairings WHERE code_hash = ?", (code_hash,))

    def delete_expired_app_pairings(self, now: float | None = None) -> int:
        """Delete expired, unredeemed offers; returns deleted row count."""
        ts = float(now if now is not None else time.time())
        with self.connection_provider.connection() as conn:
            with conn:
                cur = conn.execute(
                    "DELETE FROM app_pairings WHERE redeemed_device_id IS NULL AND expires_at <= ?", (ts,)
                )
                return int(cur.rowcount)

    # ------------------------------------------------------------------ outbox
    def append_app_outbox(self, session_id: str, event_type: str, payload: Mapping[str, Any], event_id: str,
                          ts: float | None = None) -> int:
        """Atomically allocate the next per-session `seq` and persist the event; returns the seq."""
        now = float(ts if ts is not None else time.time())
        body = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"))
        with self.connection_provider.connection() as conn:
            conn.isolation_level = None
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) AS head FROM app_outbox WHERE session_id = ?", (session_id,)
                ).fetchone()
                seq = int(row["head"]) + 1
                conn.execute(
                    "INSERT INTO app_outbox (session_id, seq, type, payload, ts, event_id) VALUES (?, ?, ?, ?, ?, ?)",
                    (session_id, seq, event_type, body, now, event_id),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return seq

    def get_app_outbox_after(self, session_id: str, after_seq: int, limit: int = 500) -> list[dict[str, Any]]:
        """Return outbox events with `seq > after_seq` in ascending order."""
        with self.connection_provider.connection() as conn:
            rows = conn.execute(
                "SELECT session_id, seq, type, payload, ts, event_id FROM app_outbox "
                "WHERE session_id = ? AND seq > ? ORDER BY seq ASC LIMIT ?",
                (session_id, int(after_seq), max(1, min(int(limit), 1000))),
            ).fetchall()
        return [
            {
                "session_id": r["session_id"],
                "seq": int(r["seq"]),
                "type": r["type"],
                "payload": _loads(r["payload"], {}),
                "ts": float(r["ts"]),
                "event_id": r["event_id"],
            }
            for r in rows
        ]

    def get_app_outbox_bounds(self, session_id: str) -> tuple[int, int]:
        """Return `(min_seq, head_seq)` for a session (both 0 when empty)."""
        with self.connection_provider.connection() as conn:
            row = conn.execute(
                "SELECT COALESCE(MIN(seq), 0) AS lo, COALESCE(MAX(seq), 0) AS hi FROM app_outbox WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["lo"]), int(row["hi"])

    def get_app_outbox_heads(self, session_ids: Iterable[str]) -> dict[str, int]:
        """Return head seq for each session id (0 when empty)."""
        ids = [s for s in session_ids if s]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self.connection_provider.connection() as conn:
            rows = conn.execute(
                f"SELECT session_id, MAX(seq) AS hi FROM app_outbox WHERE session_id IN ({placeholders}) "
                "GROUP BY session_id",
                ids,
            ).fetchall()
        heads = {s: 0 for s in ids}
        for r in rows:
            heads[str(r["session_id"])] = int(r["hi"] or 0)
        return heads

    def prune_app_outbox(self, max_per_session: int, cutoff_ts: float) -> int:
        """Prune events beyond `max_per_session` or older than `cutoff_ts`, always keeping each head row.

        Keeping the newest row guarantees `seq` never restarts after pruning.
        """
        with self.connection_provider.connection() as conn:
            with conn:
                c1 = conn.execute(
                    """
                    DELETE FROM app_outbox
                    WHERE seq <= (SELECT MAX(o2.seq) FROM app_outbox o2 WHERE o2.session_id = app_outbox.session_id) - ?
                    """,
                    (int(max_per_session),),
                )
                c2 = conn.execute(
                    """
                    DELETE FROM app_outbox
                    WHERE ts < ?
                      AND seq < (SELECT MAX(o2.seq) FROM app_outbox o2 WHERE o2.session_id = app_outbox.session_id)
                    """,
                    (float(cutoff_ts),),
                )
                return int(c1.rowcount) + int(c2.rowcount)

    def delete_app_outbox_session(self, session_id: str) -> None:
        """Drop all outbox events of a deleted session."""
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute("DELETE FROM app_outbox WHERE session_id = ?", (session_id,))

    # ------------------------------------------------------------------ session prefs
    def get_app_session_prefs(self, session_id: str) -> dict[str, Any] | None:
        """Return decoded per-session app preferences, or None."""
        with self.connection_provider.connection() as conn:
            row = conn.execute("SELECT * FROM app_session_prefs WHERE session_id = ?", (session_id,)).fetchone()
        if not row:
            return None
        return {
            "session_id": row["session_id"],
            "llm_override": _loads(row["llm_override"]),
            "mode_prefs": _loads(row["mode_prefs"]),
            "origin_channel": row["origin_channel"],
            "created_by_device": row["created_by_device"],
            "updated_at": float(row["updated_at"]),
        }

    def upsert_app_session_prefs(self, session_id: str, **fields: Any) -> None:
        """Insert or update selected per-session preference fields.

        Accepted keys: `llm_override` (dict | None), `mode_prefs` (dict | None), `origin_channel`,
        `created_by_device`. Keys not provided are left unchanged.
        """
        allowed = ("llm_override", "mode_prefs", "origin_channel", "created_by_device")
        unknown = set(fields) - set(allowed)
        if unknown:
            raise ValueError(f"Unknown app_session_prefs fields: {sorted(unknown)}")
        now = time.time()
        encoded = {
            k: (_json_or_none(v) if k in ("llm_override", "mode_prefs") else v) for k, v in fields.items()
        }
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute(
                    "INSERT OR IGNORE INTO app_session_prefs (session_id, updated_at) VALUES (?, ?)",
                    (session_id, now),
                )
                for key, value in encoded.items():
                    # Column names come from the fixed `allowed` whitelist above; values are bound.
                    conn.execute(
                        f"UPDATE app_session_prefs SET {key} = ?, updated_at = ? WHERE session_id = ?",
                        (value, now, session_id),
                    )

    # ------------------------------------------------------------------ mode memory prefs
    def seed_app_mode_memory_prefs(self, defaults: Mapping[str, str], ts: float | None = None) -> int:
        """`INSERT OR IGNORE` factory defaults (never overrides rows the user already changed)."""
        now = float(ts if ts is not None else time.time())
        inserted = 0
        with self.connection_provider.connection() as conn:
            with conn:
                for mode_id, policy in defaults.items():
                    if policy not in APP_MEMORY_POLICY_VALUES:
                        continue
                    cur = conn.execute(
                        "INSERT OR IGNORE INTO app_mode_memory_prefs (mode_id, memory_policy, updated_at) "
                        "VALUES (?, ?, ?)",
                        (mode_id, policy, now),
                    )
                    inserted += int(cur.rowcount)
        return inserted

    def get_app_mode_memory_policy(self, mode_id: str) -> str | None:
        """Return the user-configured memory policy for `mode_id`, or None."""
        with self.connection_provider.connection() as conn:
            row = conn.execute(
                "SELECT memory_policy FROM app_mode_memory_prefs WHERE mode_id = ?", (mode_id,)
            ).fetchone()
        return str(row["memory_policy"]) if row else None

    def list_app_mode_memory_prefs(self) -> dict[str, dict[str, Any]]:
        """Return `{mode_id: {memory_policy, updated_at}}` for all rows."""
        with self.connection_provider.connection() as conn:
            rows = conn.execute("SELECT mode_id, memory_policy, updated_at FROM app_mode_memory_prefs").fetchall()
        return {
            str(r["mode_id"]): {"memory_policy": str(r["memory_policy"]), "updated_at": float(r["updated_at"])}
            for r in rows
        }

    def set_app_mode_memory_policy(self, mode_id: str, policy: str, ts: float | None = None) -> float:
        """UPSERT a mode's memory policy; returns the stored `updated_at`."""
        if policy not in APP_MEMORY_POLICY_VALUES:
            raise ValueError(f"Invalid memory policy '{policy}'")
        now = float(ts if ts is not None else time.time())
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO app_mode_memory_prefs (mode_id, memory_policy, updated_at) VALUES (?, ?, ?)
                    ON CONFLICT(mode_id) DO UPDATE SET memory_policy = excluded.memory_policy,
                        updated_at = excluded.updated_at
                    """,
                    (mode_id, policy, now),
                )
        return now

    # ------------------------------------------------------------------ memory proposals
    @staticmethod
    def _proposal_row(row: sqlite3.Row) -> dict[str, Any]:
        """Decode a `memory_proposals` row."""
        return {
            "proposal_id": row["proposal_id"],
            "session_id": row["session_id"],
            "message_id": row["message_id"],
            "role": row["role"],
            "source_mode": row["source_mode"],
            "ops": _loads(row["ops"], []),
            "fingerprint": row["fingerprint"],
            "status": row["status"],
            "created_at": float(row["created_at"]),
            "expires_at": float(row["expires_at"]),
            "resolved_at": float(row["resolved_at"]) if row["resolved_at"] is not None else None,
            "resolved_by_device": row["resolved_by_device"],
            "applied_ops": _loads(row["applied_ops"]),
        }

    def insert_memory_proposal(
        self,
        *,
        proposal_id: str,
        session_id: str,
        message_id: str | None,
        role: str,
        source_mode: str | None,
        ops: Sequence[Mapping[str, Any]],
        fingerprint: str,
        created_at: float,
        expires_at: float,
    ) -> None:
        """Insert a pending memory proposal."""
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO memory_proposals
                        (proposal_id, session_id, message_id, role, source_mode, ops, fingerprint, status,
                         created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                    """,
                    (
                        proposal_id,
                        session_id,
                        message_id,
                        role,
                        source_mode,
                        json.dumps([dict(o) for o in ops], ensure_ascii=False),
                        fingerprint,
                        float(created_at),
                        float(expires_at),
                    ),
                )

    def get_memory_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        """Return one decoded proposal, or None."""
        with self.connection_provider.connection() as conn:
            row = conn.execute("SELECT * FROM memory_proposals WHERE proposal_id = ?", (proposal_id,)).fetchone()
        return self._proposal_row(row) if row else None

    def list_memory_proposals(self, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """List proposals (optionally by status), newest first."""
        params: list[Any] = []
        sql = "SELECT * FROM memory_proposals"
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, min(int(limit), 500)))
        with self.connection_provider.connection() as conn:
            return [self._proposal_row(r) for r in conn.execute(sql, params).fetchall()]

    def resolve_memory_proposal(
        self,
        proposal_id: str,
        status: str,
        *,
        ts: float | None = None,
        device_id: str | None = None,
        applied_ops: Sequence[Mapping[str, Any]] | None = None,
    ) -> bool:
        """Atomically transition a *pending* proposal to `status`; returns False when it was not pending."""
        if status not in PROPOSAL_STATUSES or status == "pending":
            raise ValueError(f"Invalid proposal resolution status '{status}'")
        with self.connection_provider.connection() as conn:
            with conn:
                cur = conn.execute(
                    """
                    UPDATE memory_proposals
                    SET status = ?, resolved_at = ?, resolved_by_device = ?, applied_ops = ?
                    WHERE proposal_id = ? AND status = 'pending'
                    """,
                    (
                        status,
                        float(ts if ts is not None else time.time()),
                        device_id,
                        _json_or_none([dict(o) for o in applied_ops] if applied_ops is not None else None),
                        proposal_id,
                    ),
                )
                return cur.rowcount > 0

    def claim_memory_proposal(self, proposal_id: str) -> bool:
        """Mark a pending proposal as being applied (`status='applying'`) so concurrent confirms cannot race."""
        with self.connection_provider.connection() as conn:
            with conn:
                cur = conn.execute(
                    "UPDATE memory_proposals SET status = 'applying' WHERE proposal_id = ? AND status = 'pending'",
                    (proposal_id,),
                )
                return cur.rowcount > 0

    def finish_memory_proposal(
        self,
        proposal_id: str,
        status: str,
        *,
        ts: float | None = None,
        device_id: str | None = None,
        applied_ops: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        """Finalize a claimed (`applying`) proposal as `status` (`applied`, or back to `pending` on failure)."""
        with self.connection_provider.connection() as conn:
            with conn:
                if status == "pending":
                    conn.execute(
                        "UPDATE memory_proposals SET status = 'pending' WHERE proposal_id = ? AND status = 'applying'",
                        (proposal_id,),
                    )
                    return
                conn.execute(
                    """
                    UPDATE memory_proposals
                    SET status = ?, resolved_at = ?, resolved_by_device = ?, applied_ops = ?
                    WHERE proposal_id = ? AND status = 'applying'
                    """,
                    (
                        status,
                        float(ts if ts is not None else time.time()),
                        device_id,
                        _json_or_none([dict(o) for o in applied_ops] if applied_ops is not None else None),
                        proposal_id,
                    ),
                )

    def expire_memory_proposals(self, now: float | None = None) -> list[str]:
        """Expire pending proposals past `expires_at`; returns their ids."""
        ts = float(now if now is not None else time.time())
        with self.connection_provider.connection() as conn:
            with conn:
                rows = conn.execute(
                    "SELECT proposal_id FROM memory_proposals WHERE status = 'pending' AND expires_at <= ?", (ts,)
                ).fetchall()
                ids = [str(r["proposal_id"]) for r in rows]
                for pid in ids:
                    conn.execute(
                        "UPDATE memory_proposals SET status = 'expired', resolved_at = ? "
                        "WHERE proposal_id = ? AND status = 'pending'",
                        (ts, pid),
                    )
        return ids

    def add_memory_proposal_blocklist(self, fingerprints: Iterable[str], role: str, ts: float | None = None) -> int:
        """Insert fingerprints into the blocklist (idempotent); returns inserted count."""
        now = float(ts if ts is not None else time.time())
        n = 0
        with self.connection_provider.connection() as conn:
            with conn:
                for fp in fingerprints:
                    if not fp:
                        continue
                    cur = conn.execute(
                        "INSERT OR IGNORE INTO memory_proposal_blocklist (fingerprint, role, dismissed_at) "
                        "VALUES (?, ?, ?)",
                        (fp, role, now),
                    )
                    n += int(cur.rowcount)
        return n

    def get_blocklisted_fingerprints(self, fingerprints: Iterable[str]) -> set[str]:
        """Return the subset of `fingerprints` present in the blocklist."""
        fps = [f for f in fingerprints if f]
        if not fps:
            return set()
        placeholders = ",".join("?" for _ in fps)
        with self.connection_provider.connection() as conn:
            rows = conn.execute(
                f"SELECT fingerprint FROM memory_proposal_blocklist WHERE fingerprint IN ({placeholders})", fps
            ).fetchall()
        return {str(r["fingerprint"]) for r in rows}

    def clear_memory_proposal_blocklist(self, role: str | None = None) -> int:
        """Clear the blocklist (optionally only for one role); returns deleted count."""
        with self.connection_provider.connection() as conn:
            with conn:
                if role:
                    cur = conn.execute("DELETE FROM memory_proposal_blocklist WHERE role = ?", (role,))
                else:
                    cur = conn.execute("DELETE FROM memory_proposal_blocklist")
                return int(cur.rowcount)

    # ------------------------------------------------------------------ avatar jobs
    @staticmethod
    def _avatar_job_row(row: sqlite3.Row) -> dict[str, Any]:
        """Decode an `avatar_jobs` row."""
        data = dict(row)
        data["candidates"] = _loads(row["candidates"], [])
        return data

    def insert_avatar_job(
        self,
        *,
        job_id: str,
        device_id: str,
        target_type: str,
        target_id: str,
        style_preset: str,
        prompt_hint: str | None,
        source_path: str,
        candidates: Sequence[Mapping[str, Any]] = (),
        status: str = "running",
        created_at: float | None = None,
        expires_at: float | None = None,
    ) -> None:
        """Insert an avatar restyle job."""
        now = float(created_at if created_at is not None else time.time())
        exp = float(expires_at if expires_at is not None else now + 86400.0)
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO avatar_jobs (job_id, device_id, target_type, target_id, style_preset, prompt_hint,
                        source_path, candidates, status, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        device_id,
                        target_type,
                        target_id,
                        style_preset,
                        prompt_hint,
                        source_path,
                        json.dumps([dict(c) for c in candidates], ensure_ascii=False),
                        status,
                        now,
                        exp,
                    ),
                )

    def get_avatar_job(self, job_id: str) -> dict[str, Any] | None:
        """Return one decoded avatar job, or None."""
        with self.connection_provider.connection() as conn:
            row = conn.execute("SELECT * FROM avatar_jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._avatar_job_row(row) if row else None

    def update_avatar_job(
        self,
        job_id: str,
        *,
        candidates: Sequence[Mapping[str, Any]] | None = None,
        status: str | None = None,
    ) -> None:
        """Update candidates and/or status of an avatar job."""
        with self.connection_provider.connection() as conn:
            with conn:
                if candidates is not None:
                    conn.execute(
                        "UPDATE avatar_jobs SET candidates = ? WHERE job_id = ?",
                        (json.dumps([dict(c) for c in candidates], ensure_ascii=False), job_id),
                    )
                if status is not None:
                    conn.execute("UPDATE avatar_jobs SET status = ? WHERE job_id = ?", (status, job_id))

    def count_recent_avatar_jobs(self, device_id: str, since_ts: float) -> int:
        """Count jobs a device started since `since_ts` (rate limiting)."""
        with self.connection_provider.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM avatar_jobs WHERE device_id = ? AND created_at >= ?",
                (device_id, float(since_ts)),
            ).fetchone()
        return int(row["n"])

    def list_expired_avatar_jobs(self, now: float | None = None) -> list[dict[str, Any]]:
        """Return jobs past `expires_at`."""
        ts = float(now if now is not None else time.time())
        with self.connection_provider.connection() as conn:
            rows = conn.execute("SELECT * FROM avatar_jobs WHERE expires_at <= ?", (ts,)).fetchall()
        return [self._avatar_job_row(r) for r in rows]

    def delete_avatar_job(self, job_id: str) -> None:
        """Delete an avatar job row."""
        with self.connection_provider.connection() as conn:
            with conn:
                conn.execute("DELETE FROM avatar_jobs WHERE job_id = ?", (job_id,))
