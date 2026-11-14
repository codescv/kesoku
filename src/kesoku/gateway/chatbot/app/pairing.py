"""Device pairing (6-digit codes) and device lifecycle for the app channel.

`kesoku app pair` (CLI process) writes a short-lived offer into `app_pairings`; the running adapter polls the
table and forwards only `code_hash + salt` to the relay (the code itself never leaves the host terminal).
The relay redeems codes and notifies the engine (`pair.redeem`, `pair.burned`, `device.registered`).
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import time
from collections.abc import Awaitable, Callable
from typing import Any

from kesoku.db.manager import DatabaseManager
from kesoku.logger import setup_logger

logger = setup_logger(__name__)

PAIRING_TTL_SECONDS = 120
DEVICE_INACTIVITY_REVOKE_SECONDS = 180 * 86400
DEVICE_NAME_MAX = 80
TOUCH_INTERVAL_SECONDS = 60.0


def hash_pairing_code(code: str, salt: str) -> str:
    """Return `hex(sha256(utf8(code + salt)))` as agreed with the relay."""
    return hashlib.sha256(f"{code}{salt}".encode()).hexdigest()


def generate_pairing_code() -> str:
    """Return a CSPRNG 6-digit numeric code."""
    return f"{secrets.randbelow(1_000_000):06d}"


def create_pairing_offer(
    db: DatabaseManager, name: str, ttl_seconds: int = PAIRING_TTL_SECONDS, now: float | None = None
) -> dict[str, Any]:
    """Create (and persist) a new pairing offer, replacing any older unredeemed offer.

    Returns:
        `{code, code_hash, salt, name, expires_at}` — `code` must only be shown on the local terminal.
    """
    ts = float(now if now is not None else time.time())
    code = generate_pairing_code()
    salt = secrets.token_hex(16)
    code_hash = hash_pairing_code(code, salt)
    clean_name = (name or "device").strip()[:DEVICE_NAME_MAX] or "device"
    db.insert_app_pairing(code_hash, salt, clean_name, ts, ts + ttl_seconds)
    return {"code": code, "code_hash": code_hash, "salt": salt, "name": clean_name, "expires_at": ts + ttl_seconds}


class PairingService:
    """Engine-side pairing/device bookkeeping (offer forwarding, redemption, revocation sync)."""

    def __init__(
        self,
        db: DatabaseManager,
        send: Callable[[dict[str, Any]], Awaitable[bool]],
        make_frame: Callable[[str, dict[str, Any]], dict[str, Any]],
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Initialize the service.

        Args:
            db: Synchronous database manager.
            send: Async sender of engine->relay frames (returns False when offline).
            make_frame: Envelope factory `(type, payload) -> envelope`.
            clock: Injectable time source.
        """
        self.db = db
        self._send = send
        self._make = make_frame
        self._clock = clock
        self._offered_hash: str | None = None
        self._sent_revoked: set[str] = set()
        self._last_touch: dict[str, float] = {}

    def reset_link_state(self) -> None:
        """Forget what was sent on the previous relay connection (re-offer / re-sync on reconnect)."""
        self._offered_hash = None
        self._sent_revoked = set()

    async def revoked_device_ids(self) -> list[str]:
        """Return all revoked device ids (for `engine.hello`)."""
        ids = await asyncio.to_thread(self.db.list_revoked_app_device_ids)
        self._sent_revoked = set(ids)
        return ids

    async def poll_once(self) -> None:
        """Forward a new pairing offer and newly revoked devices to the relay (called ~1 s)."""
        now = self._clock()
        await asyncio.to_thread(self.db.delete_expired_app_pairings, now)
        offer = await asyncio.to_thread(self.db.get_active_app_pairing, now)
        if offer is not None and offer["code_hash"] != self._offered_hash:
            frame = self._make(
                "pair.offer",
                {
                    "code_hash": offer["code_hash"],
                    "salt": offer["salt"],
                    "expires_at": float(offer["expires_at"]),
                    "name": offer["name"],
                },
            )
            if await self._send(frame):
                self._offered_hash = offer["code_hash"]
                logger.info("Forwarded pairing offer to relay")
        revoked = set(await asyncio.to_thread(self.db.list_revoked_app_device_ids))
        new_revoked = sorted(revoked - self._sent_revoked)
        if new_revoked and await self._send(self._make("device.revoked", {"device_ids": new_revoked})):
            self._sent_revoked |= set(new_revoked)

    async def handle_redeem(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Register a device redeemed by the relay and mark the active offer redeemed."""
        device_id = payload.get("device_id")
        if not isinstance(device_id, str) or not device_id or len(device_id) > 64:
            logger.warning("Ignoring pair.redeem without a valid device_id")
            return None
        offer = await asyncio.to_thread(self.db.get_active_app_pairing, self._clock())
        name = payload.get("name") if isinstance(payload.get("name"), str) and payload.get("name") else None
        name = (name or (offer["name"] if offer else "device"))[:DEVICE_NAME_MAX]
        platform = payload.get("platform") if isinstance(payload.get("platform"), str) else None
        await asyncio.to_thread(self.db.upsert_app_device, device_id, name, "pair", platform, self._clock())
        if offer is not None:
            await asyncio.to_thread(self.db.mark_app_pairing_redeemed, offer["code_hash"], device_id)
        logger.info(f"Paired new app device {device_id} ({name})")
        return await asyncio.to_thread(self.db.get_app_device, device_id)

    async def handle_burned(self, payload: dict[str, Any]) -> None:
        """Drop an offer burned by too many failed attempts and warn on the host."""
        code_hash = payload.get("code_hash")
        if isinstance(code_hash, str) and code_hash:
            await asyncio.to_thread(self.db.delete_app_pairing, code_hash)
        else:
            offer = await asyncio.to_thread(self.db.get_active_app_pairing, self._clock())
            if offer is not None:
                await asyncio.to_thread(self.db.delete_app_pairing, offer["code_hash"])
        self._offered_hash = None
        logger.warning(
            f"Pairing offer burned after {payload.get('attempts', 'too many')} failed attempts; "
            "run `kesoku app pair` again to create a new code."
        )

    async def handle_registered(self, payload: dict[str, Any]) -> None:
        """Register a device created by the relay's web sign-in flow."""
        device_id = payload.get("device_id")
        if not isinstance(device_id, str) or not device_id or len(device_id) > 64:
            return
        name = payload.get("email") or payload.get("name") or "web"
        platform = payload.get("platform") if isinstance(payload.get("platform"), str) else "web"
        await asyncio.to_thread(self.db.upsert_app_device, device_id, str(name), "google", platform, self._clock())

    async def ensure_device(self, device_id: str) -> dict[str, Any] | None:
        """Return the device row, auto-registering unknown (relay-authenticated) ids; None when revoked."""
        row = await asyncio.to_thread(self.db.get_app_device, device_id)
        if row is None:
            await asyncio.to_thread(self.db.upsert_app_device, device_id, device_id[:24], "google", None, self._clock())
            row = await asyncio.to_thread(self.db.get_app_device, device_id)
        if row is None or row.get("revoked_at") is not None:
            return None
        now = self._clock()
        if now - self._last_touch.get(device_id, 0.0) >= TOUCH_INTERVAL_SECONDS:
            self._last_touch[device_id] = now
            await asyncio.to_thread(self.db.touch_app_device, device_id, now)
        return row

    async def revoke(self, device_id: str) -> bool:
        """Revoke a device and push the revocation to the relay."""
        changed = await asyncio.to_thread(self.db.revoke_app_device, device_id, self._clock())
        if changed and await self._send(self._make("device.revoked", {"device_ids": [device_id]})):
            self._sent_revoked.add(device_id)
        return changed

    async def revoke_inactive(self) -> list[str]:
        """Auto-revoke devices inactive for 180 days."""
        now = self._clock()
        ids = await asyncio.to_thread(self.db.revoke_inactive_app_devices, now - DEVICE_INACTIVITY_REVOKE_SECONDS, now)
        if ids:
            logger.info(f"Auto-revoked {len(ids)} inactive app device(s)")
            if await self._send(self._make("device.revoked", {"device_ids": ids})):
                self._sent_revoked |= set(ids)
        return ids
