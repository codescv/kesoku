"""WeChat listener for handling long-polling updates from iLink Bot API."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from .client import (
    LONG_POLL_TIMEOUT_MS,
    ILinkClient,
)

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_FAILURES = 3
RETRY_DELAY_SECONDS = 2
BACKOFF_DELAY_SECONDS = 30
SESSION_EXPIRED_ERRCODE = -14
RATE_LIMIT_ERRCODE = -2


def _is_stale_session_ret(ret: int | None, errcode: int | None, errmsg: str | None) -> bool:
    if ret != RATE_LIMIT_ERRCODE and errcode != RATE_LIMIT_ERRCODE:
        return False
    return (errmsg or "").lower() == "unknown error"


class WeChatListener:
    """Manages the long-polling event loop for WeChat iLink Bot API."""

    def __init__(
        self,
        client: ILinkClient,
        message_callback: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Initialize WeChatListener."""
        self.client = client
        self.message_callback = message_callback
        self._running = False
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        """Start the long-polling loop in a background task."""
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())

    def stop(self) -> None:
        """Stop the long-polling loop and cancel the background task."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def _poll_loop(self) -> None:
        sync_buf = ""
        timeout_ms = LONG_POLL_TIMEOUT_MS
        consecutive_failures = 0

        while self._running:
            try:
                response = await self.client.get_updates(sync_buf, timeout_ms=timeout_ms)
                suggested_timeout = response.get("longpolling_timeout_ms")
                if isinstance(suggested_timeout, int) and suggested_timeout > 0:
                    timeout_ms = suggested_timeout

                ret = response.get("ret", 0)
                errcode = response.get("errcode", 0)
                if ret not in {0, None} or errcode not in {0, None}:
                    if (
                        ret == SESSION_EXPIRED_ERRCODE
                        or errcode == SESSION_EXPIRED_ERRCODE
                        or _is_stale_session_ret(ret, errcode, response.get("errmsg"))
                    ):
                        logger.error("WeChat: Session expired; pausing for 10 minutes")
                        await asyncio.sleep(600)
                        consecutive_failures = 0
                        continue

                    consecutive_failures += 1
                    logger.warning(
                        "WeChat: getUpdates failed ret=%s errcode=%s errmsg=%s (%d/%d)",
                        ret,
                        errcode,
                        response.get("errmsg", ""),
                        consecutive_failures,
                        MAX_CONSECUTIVE_FAILURES,
                    )
                    await asyncio.sleep(
                        BACKOFF_DELAY_SECONDS
                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES
                        else RETRY_DELAY_SECONDS
                    )
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        consecutive_failures = 0
                    continue

                consecutive_failures = 0
                new_sync_buf = str(response.get("get_updates_buf") or "")
                if new_sync_buf:
                    sync_buf = new_sync_buf

                for msg in response.get("msgs") or []:
                    asyncio.create_task(self.message_callback(msg))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                consecutive_failures += 1
                logger.error(
                    "WeChat: poll error (%d/%d): %s",
                    consecutive_failures,
                    MAX_CONSECUTIVE_FAILURES,
                    exc,
                )
                await asyncio.sleep(
                    BACKOFF_DELAY_SECONDS if consecutive_failures >= MAX_CONSECUTIVE_FAILURES else RETRY_DELAY_SECONDS
                )
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    consecutive_failures = 0
