"""iLink Bot API client for WeChat integration."""

import base64
import json
import logging
import secrets
import struct
from typing import Any
from urllib.parse import quote

import aiohttp

logger = logging.getLogger(__name__)

ILINK_BASE_URL = "https://ilinkai.weixin.qq.com"
WEIXIN_CDN_BASE_URL = "https://novac2c.cdn.weixin.qq.com/c2c"
ILINK_APP_ID = "bot"
CHANNEL_VERSION = "2.2.0"
ILINK_APP_CLIENT_VERSION = (2 << 16) | (2 << 8) | 0

EP_GET_UPDATES = "ilink/bot/getupdates"
EP_SEND_MESSAGE = "ilink/bot/sendmessage"
EP_SEND_TYPING = "ilink/bot/sendtyping"
EP_GET_CONFIG = "ilink/bot/getconfig"
EP_GET_UPLOAD_URL = "ilink/bot/getuploadurl"
EP_GET_BOT_QR = "ilink/bot/get_bot_qrcode"
EP_GET_QR_STATUS = "ilink/bot/get_qrcode_status"

LONG_POLL_TIMEOUT_MS = 35_000
API_TIMEOUT_MS = 15_000
CONFIG_TIMEOUT_MS = 10_000
QR_TIMEOUT_MS = 35_000

MEDIA_IMAGE = 1
MEDIA_VIDEO = 2
MEDIA_FILE = 3
MEDIA_VOICE = 4

ITEM_TEXT = 1
ITEM_IMAGE = 2
ITEM_VOICE = 3
ITEM_FILE = 4
ITEM_VIDEO = 5

MSG_TYPE_BOT = 2
MSG_STATE_FINISH = 2
TYPING_START = 1
TYPING_STOP = 2


def _random_wechat_uin() -> str:
    value = struct.unpack(">I", secrets.token_bytes(4))[0]
    return base64.b64encode(str(value).encode("utf-8")).decode("ascii")


def _base_info() -> dict[str, Any]:
    return {"channel_version": CHANNEL_VERSION}


def _headers(token: str | None, body: str) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "AuthorizationType": "ilink_bot_token",
        "Content-Length": str(len(body.encode("utf-8"))),
        "X-WECHAT-UIN": _random_wechat_uin(),
        "iLink-App-Id": ILINK_APP_ID,
        "iLink-App-ClientVersion": str(ILINK_APP_CLIENT_VERSION),
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def cdn_download_url(cdn_base_url: str, encrypted_query_param: str) -> str:
    """Generate WeChat CDN download URL."""
    return f"{cdn_base_url.rstrip('/')}/download?encrypted_query_param={quote(encrypted_query_param, safe='')}"


def cdn_upload_url(cdn_base_url: str, upload_param: str, filekey: str) -> str:
    """Generate WeChat CDN upload URL."""
    return (
        f"{cdn_base_url.rstrip('/')}/upload"
        f"?encrypted_query_param={quote(upload_param, safe='')}"
        f"&filekey={quote(filekey, safe='')}"
    )


class ILinkClient:
    """Client for Tencent iLink Bot API."""

    def __init__(self, session: aiohttp.ClientSession, base_url: str, token: str) -> None:
        """Initialize ILinkClient."""
        self.session = session
        self.base_url = base_url.strip().rstrip("/")
        self.token = token.strip()

    async def _post(self, endpoint: str, payload: dict[str, Any], timeout_ms: int) -> dict[str, Any]:
        body = _json_dumps({**payload, "base_info": _base_info()})
        url = f"{self.base_url}/{endpoint}"
        timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000)
        async with self.session.post(
            url, data=body, headers=_headers(self.token, body), timeout=timeout
        ) as response:
            raw = await response.text()
            if not response.ok:
                raise RuntimeError(f"iLink POST {endpoint} HTTP {response.status}: {raw[:200]}")
            return json.loads(raw)

    async def _get(self, endpoint: str, timeout_ms: int) -> dict[str, Any]:
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "iLink-App-Id": ILINK_APP_ID,
            "iLink-App-ClientVersion": str(ILINK_APP_CLIENT_VERSION),
        }
        timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000)
        async with self.session.get(url, headers=headers, timeout=timeout) as response:
            raw = await response.text()
            if not response.ok:
                raise RuntimeError(f"iLink GET {endpoint} HTTP {response.status}: {raw[:200]}")
            return json.loads(raw)

    async def get_updates(self, sync_buf: str, timeout_ms: int = LONG_POLL_TIMEOUT_MS) -> dict[str, Any]:
        """Poll for new messages/events from WeChat."""
        payload = {
            "get_updates_buf": sync_buf,
            "longpolling_timeout_ms": timeout_ms,
        }
        return await self._post(EP_GET_UPDATES, payload, timeout_ms=timeout_ms + 5000)

    async def get_bot_qrcode(self, bot_type: int = 3) -> dict[str, Any]:
        """Fetch pairing QR code."""
        return await self._get(f"{EP_GET_BOT_QR}?bot_type={bot_type}", timeout_ms=QR_TIMEOUT_MS)

    async def get_qrcode_status(self, qrcode: str) -> dict[str, Any]:
        """Poll QR code scan status."""
        return await self._get(f"{EP_GET_QR_STATUS}?qrcode={qrcode}", timeout_ms=QR_TIMEOUT_MS)

    async def send_message(
        self,
        to: str,
        text: str,
        context_token: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        """Send a text message to a WeChat user or room."""
        if not text or not text.strip():
            raise ValueError("send_message: text must not be empty")
        message: dict[str, Any] = {
            "from_user_id": "",
            "to_user_id": to,
            "client_id": client_id or "",
            "message_type": MSG_TYPE_BOT,
            "message_state": MSG_STATE_FINISH,
            "item_list": [{"type": ITEM_TEXT, "text_item": {"text": text}}],
        }
        if context_token:
            message["context_token"] = context_token
        return await self._post(EP_SEND_MESSAGE, {"msg": message}, timeout_ms=API_TIMEOUT_MS)

    async def send_raw_message(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Send a raw message payload to WeChat."""
        return await self._post(EP_SEND_MESSAGE, {"msg": msg}, timeout_ms=API_TIMEOUT_MS)

    async def send_typing(self, to: str, typing_status: int, ticket: str) -> dict[str, Any]:
        """Send typing indicator status."""
        payload = {
            "ilink_user_id": to,
            "typing_ticket": ticket,
            "status": typing_status,
        }
        return await self._post(EP_SEND_TYPING, payload, timeout_ms=CONFIG_TIMEOUT_MS)

    async def get_config(self, user_id: str, context_token: str | None = None) -> dict[str, Any]:
        """Fetch config (including typing ticket) for a user."""
        payload: dict[str, Any] = {"ilink_user_id": user_id}
        if context_token:
            payload["context_token"] = context_token
        return await self._post(EP_GET_CONFIG, payload, timeout_ms=CONFIG_TIMEOUT_MS)

    async def get_upload_url(
        self,
        to_user_id: str,
        media_type: int,
        filekey: str,
        rawsize: int,
        rawfilemd5: str,
        filesize: int,
        aeskey_hex: str,
    ) -> dict[str, Any]:
        """Request a media upload URL from WeChat."""
        payload = {
            "to_user_id": to_user_id,
            "media_type": media_type,
            "filekey": filekey,
            "rawsize": rawsize,
            "rawfilemd5": rawfilemd5,
            "filesize": filesize,
            "aeskey": aeskey_hex,
        }
        return await self._post(EP_GET_UPLOAD_URL, payload, timeout_ms=API_TIMEOUT_MS)

    async def upload_ciphertext(self, ciphertext: bytes, upload_url: str) -> str:
        """Upload encrypted media bytes to WeChat CDN."""
        timeout = aiohttp.ClientTimeout(total=120)  # Large timeout for media upload
        async with self.session.post(upload_url, data=ciphertext, timeout=timeout) as response:
            raw = await response.text()
            if not response.ok:
                raise RuntimeError(f"WeChat: CDN upload failed HTTP {response.status}: {raw[:200]}")
            try:
                res_data = json.loads(raw)
                return str(res_data.get("encrypted_query_param") or "")
            except Exception:
                # Fallback if response is not JSON
                return raw.strip()

    async def download_bytes(self, url: str, timeout_seconds: float = 60.0) -> bytes:
        """Download raw bytes from a URL (usually WeChat CDN)."""
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        async with self.session.get(url, timeout=timeout) as response:
            if not response.ok:
                raise RuntimeError(f"WeChat: CDN download failed HTTP {response.status}")
            return await response.read()
