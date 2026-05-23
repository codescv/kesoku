"""WeChat chatbot adapter for Kesoku AI Agent framework.

Connects Kesoku to WeChat personal accounts via Tencent's iLink Bot API.
Provides command-line pairing via barcode and supports text and media messages.
"""

import asyncio
import base64
import hashlib
import json
import mimetypes
import os
import re
import secrets
import struct
import textwrap
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import aiohttp
import qrcode
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from kesoku.config import get_config
from kesoku.constants import (
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL,
    ROLE_USER,
    STATUS_DELIVERED,
    STATUS_PENDING_AGENT,
    TYPE_TEXT,
    TYPE_THOUGHT,
    TYPE_TOOL_CALL,
)
from kesoku.db import Message
from kesoku.gateway.chatbot.base import Chatbot, parse_message_content
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger

logger = setup_logger(__name__)

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

MAX_CONSECUTIVE_FAILURES = 3
RETRY_DELAY_SECONDS = 2
BACKOFF_DELAY_SECONDS = 30
SESSION_EXPIRED_ERRCODE = -14
RATE_LIMIT_ERRCODE = -2
MESSAGE_DEDUP_TTL_SECONDS = 300
WEIXIN_COPY_LINE_WIDTH = 120

MEDIA_IMAGE = 1
MEDIA_VIDEO = 2
MEDIA_FILE = 3
MEDIA_VOICE = 4

ITEM_TEXT = 1
ITEM_IMAGE = 2
ITEM_VOICE = 3
ITEM_FILE = 4
ITEM_VIDEO = 5

MSG_TYPE_USER = 1
MSG_TYPE_BOT = 2
MSG_STATE_FINISH = 2

TYPING_START = 1
TYPING_STOP = 2

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TABLE_RULE_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")
_FENCE_RE = re.compile(r"^```([^\n`]*)\s*$")


def _safe_id(value: str | None, keep: int = 8) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "?"
    if len(raw) <= keep:
        return raw
    return raw[:keep]


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _pkcs7_pad(data: bytes, block_size: int = 16) -> bytes:
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len] * pad_len)


def _aes128_ecb_encrypt(plaintext: bytes, key: bytes) -> bytes:
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    encryptor = cipher.encryptor()
    return encryptor.update(_pkcs7_pad(plaintext)) + encryptor.finalize()


def _aes128_ecb_decrypt(ciphertext: bytes, key: bytes) -> bytes:
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    decryptor = cipher.decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()
    if not padded:
        return padded
    pad_len = padded[-1]
    if 1 <= pad_len <= 16 and padded.endswith(bytes([pad_len]) * pad_len):
        return padded[:-pad_len]
    return padded


def _aes_padded_size(size: int) -> int:
    return ((size + 1 + 15) // 16) * 16


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


def _is_stale_session_ret(
    ret: int | None, errcode: int | None, errmsg: str | None
) -> bool:
    if ret != RATE_LIMIT_ERRCODE and errcode != RATE_LIMIT_ERRCODE:
        return False
    return (errmsg or "").lower() == "unknown error"


class ContextTokenStore:
    """Persistent or in-memory cache for WeChat ``context_token`` keyed by account + peer."""

    def __init__(self, persist_path: str | None = None) -> None:
        """Initialize ContextTokenStore cache."""
        self._cache: dict[str, str] = {}
        self._persist_path = persist_path
        self._load()

    def _key(self, account_id: str, user_id: str) -> str:
        return f"{account_id}:{user_id}"

    def _load(self) -> None:
        if not self._persist_path or not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
        except Exception as e:
            logger.warning("WeChat: Failed to load persistent context tokens: %s", e)

    def _save(self) -> None:
        if not self._persist_path:
            return
        try:
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("WeChat: Failed to save persistent context tokens: %s", e)

    def get(self, account_id: str, user_id: str) -> str | None:
        """Get context token for the given user ID."""
        return self._cache.get(self._key(account_id, user_id))

    def set(self, account_id: str, user_id: str, token: str) -> None:
        """Store context token for the given user ID."""
        self._cache[self._key(account_id, user_id)] = token
        self._save()



class TypingTicketCache:
    """Short-lived typing ticket cache from ``getconfig``."""

    def __init__(self, ttl_seconds: float = 600.0):
        """Initialize TypingTicketCache."""
        self._ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[str, float]] = {}

    def get(self, user_id: str) -> str | None:
        """Get typing ticket for the given user ID."""
        entry = self._cache.get(user_id)
        if not entry:
            return None
        if time.time() - entry[1] >= self._ttl_seconds:
            self._cache.pop(user_id, None)
            return None
        return entry[0]

    def set(self, user_id: str, ticket: str) -> None:
        """Store typing ticket for the given user ID."""
        self._cache[user_id] = (ticket, time.time())


class MessageDeduplicator:
    """Deduplicates incoming messages using a TTL cache of message IDs."""

    def __init__(self, ttl_seconds: float = MESSAGE_DEDUP_TTL_SECONDS):
        """Initialize MessageDeduplicator."""
        self._ttl_seconds = ttl_seconds
        self._cache: dict[str, float] = {}

    def is_duplicate(self, msg_id: str) -> bool:
        """Check if a message ID has been seen recently."""
        now = time.time()
        # Clean expired entries
        expired = [k for k, v in self._cache.items() if now - v > self._ttl_seconds]
        for k in expired:
            self._cache.pop(k, None)

        if msg_id in self._cache:
            return True
        self._cache[msg_id] = now
        return False


def _cdn_download_url(cdn_base_url: str, encrypted_query_param: str) -> str:
    return f"{cdn_base_url.rstrip('/')}/download?encrypted_query_param={quote(encrypted_query_param, safe='')}"


def _cdn_upload_url(cdn_base_url: str, upload_param: str, filekey: str) -> str:
    return (
        f"{cdn_base_url.rstrip('/')}/upload"
        f"?encrypted_query_param={quote(upload_param, safe='')}"
        f"&filekey={quote(filekey, safe='')}"
    )


def _parse_aes_key(aes_key_b64: str) -> bytes:
    decoded = base64.b64decode(aes_key_b64)
    if len(decoded) == 16:
        return decoded
    if len(decoded) == 32:
        text = decoded.decode("ascii", errors="ignore")
        if text and all(ch in "0123456789abcdefABCDEF" for ch in text):
            return bytes.fromhex(text)
    raise ValueError(f"unexpected aes_key format ({len(decoded)} decoded bytes)")


def _guess_chat_type(message: dict[str, Any], account_id: str) -> tuple[str, str]:
    room_id = str(message.get("room_id") or message.get("chat_room_id") or "").strip()
    to_user_id = str(message.get("to_user_id") or "").strip()
    is_group = bool(room_id) or (
        to_user_id
        and account_id
        and to_user_id != account_id
        and message.get("msg_type") == 1
    )
    if is_group:
        return "group", room_id or to_user_id or str(message.get("from_user_id") or "")
    return "dm", str(message.get("from_user_id") or "")


async def _api_post(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    token: str | None,
    timeout_ms: int,
) -> dict[str, Any]:
    body = _json_dumps({**payload, "base_info": _base_info()})
    url = f"{base_url.rstrip('/')}/{endpoint}"
    timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000)
    async with session.post(url, data=body, headers=_headers(token, body), timeout=timeout) as response:
        raw = await response.text()
        if not response.ok:
            raise RuntimeError(f"iLink POST {endpoint} HTTP {response.status}: {raw[:200]}")
        return json.loads(raw)


async def _api_get(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    endpoint: str,
    timeout_ms: int,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/{endpoint}"
    headers = {
        "iLink-App-Id": ILINK_APP_ID,
        "iLink-App-ClientVersion": str(ILINK_APP_CLIENT_VERSION),
    }
    timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000)
    async with session.get(url, headers=headers, timeout=timeout) as response:
        raw = await response.text()
        if not response.ok:
            raise RuntimeError(f"iLink GET {endpoint} HTTP {response.status}: {raw[:200]}")
        return json.loads(raw)


async def _get_updates(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    token: str,
    sync_buf: str,
    timeout_ms: int,
) -> dict[str, Any]:
    try:
        return await _api_post(
            session,
            base_url=base_url,
            endpoint=EP_GET_UPDATES,
            payload={"get_updates_buf": sync_buf},
            token=token,
            timeout_ms=timeout_ms,
        )
    except TimeoutError:
        return {"ret": 0, "msgs": [], "get_updates_buf": sync_buf}


async def _send_message(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    token: str,
    to: str,
    text: str,
    context_token: str | None,
    client_id: str,
) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("_send_message: text must not be empty")
    message: dict[str, Any] = {
        "from_user_id": "",
        "to_user_id": to,
        "client_id": client_id,
        "message_type": MSG_TYPE_BOT,
        "message_state": MSG_STATE_FINISH,
        "item_list": [{"type": ITEM_TEXT, "text_item": {"text": text}}],
    }
    if context_token:
        message["context_token"] = context_token
    return await _api_post(
        session,
        base_url=base_url,
        endpoint=EP_SEND_MESSAGE,
        payload={"msg": message},
        token=token,
        timeout_ms=API_TIMEOUT_MS,
    )


async def _send_typing(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    token: str,
    to_user_id: str,
    typing_ticket: str,
    status: int,
) -> None:
    await _api_post(
        session,
        base_url=base_url,
        endpoint=EP_SEND_TYPING,
        payload={
            "ilink_user_id": to_user_id,
            "typing_ticket": typing_ticket,
            "status": status,
        },
        token=token,
        timeout_ms=CONFIG_TIMEOUT_MS,
    )


async def _get_config(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    token: str,
    user_id: str,
    context_token: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"ilink_user_id": user_id}
    if context_token:
        payload["context_token"] = context_token
    return await _api_post(
        session,
        base_url=base_url,
        endpoint=EP_GET_CONFIG,
        payload=payload,
        token=token,
        timeout_ms=CONFIG_TIMEOUT_MS,
    )


async def _get_upload_url(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    token: str,
    to_user_id: str,
    media_type: int,
    filekey: str,
    rawsize: int,
    rawfilemd5: str,
    filesize: int,
    aeskey_hex: str,
) -> dict[str, Any]:
    return await _api_post(
        session,
        base_url=base_url,
        endpoint=EP_GET_UPLOAD_URL,
        payload={
            "filekey": filekey,
            "media_type": media_type,
            "to_user_id": to_user_id,
            "rawsize": rawsize,
            "rawfilemd5": rawfilemd5,
            "filesize": filesize,
            "no_need_thumb": True,
            "aeskey": aeskey_hex,
        },
        token=token,
        timeout_ms=API_TIMEOUT_MS,
    )


async def _upload_ciphertext(
    session: aiohttp.ClientSession,
    *,
    ciphertext: bytes,
    upload_url: str,
) -> str:
    async def _do_upload() -> str:
        headers = {"Content-Type": "application/octet-stream"}
        async with session.post(
            upload_url, data=ciphertext, headers=headers
        ) as response:
            if response.status == 200:
                encrypted_param = response.headers.get("x-encrypted-param")
                if encrypted_param:
                    await response.read()
                    return encrypted_param
                raw = await response.text()
                raise RuntimeError(
                    f"CDN upload missing x-encrypted-param header: {raw[:200]}"
                )
            raw = await response.text()
            raise RuntimeError(f"CDN upload HTTP {response.status}: {raw[:200]}")

    return await asyncio.wait_for(_do_upload(), timeout=120)


async def _download_bytes(
    session: aiohttp.ClientSession,
    *,
    url: str,
    timeout_seconds: float = 60.0,
) -> bytes:
    async def _do_download() -> bytes:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.read()

    return await asyncio.wait_for(_do_download(), timeout=timeout_seconds)


_WEIXIN_CDN_ALLOWLIST: frozenset[str] = frozenset(
    {
        "novac2c.cdn.weixin.qq.com",
        "ilinkai.weixin.qq.com",
        "wx.qlogo.cn",
        "thirdwx.qlogo.cn",
        "res.wx.qq.com",
        "mmbiz.qpic.cn",
        "mmbiz.qlogo.cn",
    }
)


def _assert_weixin_cdn_url(url: str) -> None:
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        host = parsed.hostname or ""
    except Exception as exc:
        raise ValueError(f"Unparseable media URL: {url!r}") from exc

    if scheme not in {"http", "https"}:
        raise ValueError(
            f"Media URL has disallowed scheme {scheme!r}; only http/https are permitted."
        )
    if host not in _WEIXIN_CDN_ALLOWLIST:
        raise ValueError(
            f"Media URL host {host!r} is not in the WeChat CDN allowlist. Refusing to fetch."
        )


def _media_reference(item: dict[str, Any], key: str) -> dict[str, Any]:
    return (item.get(key) or {}).get("media") or {}


async def _download_and_decrypt_media(
    session: aiohttp.ClientSession,
    *,
    cdn_base_url: str,
    encrypted_query_param: str | None,
    aes_key_b64: str | None,
    full_url: str | None,
    timeout_seconds: float,
) -> bytes:
    if encrypted_query_param:
        raw = await _download_bytes(
            session,
            url=_cdn_download_url(cdn_base_url, encrypted_query_param),
            timeout_seconds=timeout_seconds,
        )
    elif full_url:
        _assert_weixin_cdn_url(full_url)
        raw = await _download_bytes(
            session, url=full_url, timeout_seconds=timeout_seconds
        )
    else:
        raise RuntimeError("media item had neither encrypt_query_param nor full_url")
    if aes_key_b64:
        raw = _aes128_ecb_decrypt(raw, _parse_aes_key(aes_key_b64))
    return raw


def _mime_from_filename(filename: str) -> str:
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


def _split_table_row(line: str) -> list[str]:
    row = line.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [cell.strip() for cell in row.split("|")]


def _rewrite_headers_for_weixin(line: str) -> str:
    match = _HEADER_RE.match(line)
    if not match:
        return line.rstrip()
    level = len(match.group(1))
    title = match.group(2).strip()
    if level == 1:
        return f"【{title}】"
    return f"**{title}**"


def _rewrite_table_block_for_weixin(lines: list[str]) -> str:
    if len(lines) < 2:
        return "\n".join(lines)
    headers = _split_table_row(lines[0])
    body_rows = [_split_table_row(line) for line in lines[2:] if line.strip()]
    if not headers or not body_rows:
        return "\n".join(lines)

    formatted_rows: list[str] = []
    for row in body_rows:
        pairs = []
        for idx, header in enumerate(headers):
            if idx >= len(row):
                break
            label = header or f"Column {idx + 1}"
            value = row[idx].strip()
            if value:
                pairs.append((label, value))
        if not pairs:
            continue
        if len(pairs) == 1:
            label, value = pairs[0]
            formatted_rows.append(f"- {label}: {value}")
            continue
        if len(pairs) == 2:
            label, value = pairs[0]
            other_label, other_value = pairs[1]
            formatted_rows.append(f"- {label}: {value}")
            formatted_rows.append(f"  {other_label}: {other_value}")
            continue
        summary = " | ".join(f"{label}: {value}" for label, value in pairs)
        formatted_rows.append(f"- {summary}")
    return "\n".join(formatted_rows) if formatted_rows else "\n".join(lines)


def _normalize_markdown_blocks(content: str) -> str:
    lines = content.splitlines()
    result: list[str] = []
    in_code_block = False
    blank_run = 0

    for raw_line in lines:
        line = raw_line.rstrip()
        if _FENCE_RE.match(line.strip()):
            in_code_block = not in_code_block
            result.append(line)
            blank_run = 0
            continue

        if in_code_block:
            result.append(line)
            continue

        if not line.strip():
            blank_run += 1
            if blank_run <= 1:
                result.append("")
            continue

        blank_run = 0
        result.append(_rewrite_headers_for_weixin(line))

    return "\n".join(result).strip()


def _wrap_copy_friendly_lines_for_weixin(content: str) -> str:
    if not content:
        return content

    wrapped: list[str] = []
    in_code_block = False

    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if _FENCE_RE.match(stripped):
            in_code_block = not in_code_block
            wrapped.append(line)
            continue

        if (
            in_code_block
            or len(line) <= WEIXIN_COPY_LINE_WIDTH
            or not stripped
            or stripped.startswith("|")
            or _TABLE_RULE_RE.match(stripped)
        ):
            wrapped.append(line)
            continue

        wrapped_lines = textwrap.wrap(
            line,
            width=WEIXIN_COPY_LINE_WIDTH,
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=True,
        )
        wrapped.extend(wrapped_lines or [line])

    return "\n".join(wrapped).strip()


def _split_markdown_blocks(content: str) -> list[str]:
    if not content:
        return []

    blocks: list[str] = []
    lines = content.splitlines()
    current: list[str] = []
    in_code_block = False

    for raw_line in lines:
        line = raw_line.rstrip()
        if _FENCE_RE.match(line.strip()):
            if not in_code_block and current:
                blocks.append("\n".join(current).strip())
                current = []
            current.append(line)
            in_code_block = not in_code_block
            if not in_code_block:
                blocks.append("\n".join(current).strip())
                current = []
            continue

        if in_code_block:
            current.append(line)
            continue

        if not line.strip():
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)

    if current:
        blocks.append("\n".join(current).strip())
    return [block for block in blocks if block]


def _split_delivery_units_for_weixin(content: str) -> list[str]:
    units: list[str] = []

    for block in _split_markdown_blocks(content):
        if _FENCE_RE.match(block.splitlines()[0].strip()):
            units.append(block)
            continue

        current: list[str] = []
        for raw_line in block.splitlines():
            line = raw_line.rstrip()
            if not line.strip():
                if current:
                    units.append("\n".join(current).strip())
                    current = []
                continue

            is_continuation = bool(current) and raw_line.startswith((" ", "\t"))
            if is_continuation:
                current.append(line)
                continue

            if current:
                units.append("\n".join(current).strip())
            current = [line]

        if current:
            units.append("\n".join(current).strip())

    return [unit for unit in units if unit]


def _looks_like_chatty_line_for_weixin(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 48:
        return False
    if line.startswith((" ", "\t")):
        return False
    if stripped.startswith((">", "-", "*", "【", "#", "|")):
        return False
    if _TABLE_RULE_RE.match(stripped):
        return False
    if re.match(r"^\*\*[^*]+\*\*$", stripped):
        return False
    if re.match(r"^\d+\.\s", stripped):
        return False
    return True


def _looks_like_heading_line_for_weixin(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _HEADER_RE.match(stripped):
        return True
    return len(stripped) <= 24 and stripped.endswith((":", "："))


def _should_split_short_chat_block_for_weixin(block: str) -> bool:
    lines = [line for line in block.splitlines() if line.strip()]
    if not 2 <= len(lines) <= 6:
        return False
    if _looks_like_heading_line_for_weixin(lines[0]):
        return False
    return all(_looks_like_chatty_line_for_weixin(line) for line in lines)


def _truncate_message(text: str, max_length: int) -> list[str]:
    if len(text) <= max_length:
        return [text]
    chunks = []
    for i in range(0, len(text), max_length):
        chunks.append(text[i : i + max_length])
    return chunks


def _pack_markdown_blocks_for_weixin(content: str, max_length: int) -> list[str]:
    if len(content) <= max_length:
        return [content]

    packed: list[str] = []
    current = ""
    for block in _split_markdown_blocks(content):
        candidate = block if not current else f"{current}\n\n{block}"
        if len(candidate) <= max_length:
            current = candidate
            continue
        if current:
            packed.append(current)
            current = ""
        if len(block) <= max_length:
            current = block
            continue
        packed.extend(_truncate_message(block, max_length))
    if current:
        packed.append(current)
    return packed


def _split_text_for_weixin_delivery(
    content: str, max_length: int, split_per_line: bool = False
) -> list[str]:
    if not content:
        return []
    if split_per_line:
        if len(content) <= max_length and "\n" not in content:
            return [content]
        chunks: list[str] = []
        for unit in _split_delivery_units_for_weixin(content):
            if len(unit) <= max_length:
                chunks.append(unit)
                continue
            chunks.extend(_pack_markdown_blocks_for_weixin(unit, max_length))
        return [c for c in chunks if c] or [content]

    if len(content) <= max_length:
        return (
            [u for u in _split_delivery_units_for_weixin(content) if u]
            if _should_split_short_chat_block_for_weixin(content)
            else [content]
        )
    return _pack_markdown_blocks_for_weixin(content, max_length) or [content]


def _extract_text(item_list: list[dict[str, Any]]) -> str:
    for item in item_list:
        if item.get("type") == ITEM_TEXT:
            text = str((item.get("text_item") or {}).get("text") or "")
            ref = item.get("ref_msg") or {}
            ref_item = ref.get("message_item") or {}
            ref_type = ref_item.get("type")
            if ref_type in {ITEM_IMAGE, ITEM_VIDEO, ITEM_FILE, ITEM_VOICE}:
                title = ref.get("title") or ""
                prefix = f"[引用媒体: {title}]\n" if title else "[引用媒体]\n"
                return f"{prefix}{text}".strip()
            if ref_item:
                parts: list[str] = []
                if ref.get("title"):
                    parts.append(str(ref["title"]))
                ref_text = _extract_text([ref_item])
                if ref_text:
                    parts.append(ref_text)
                if parts:
                    return f"[引用: {' | '.join(parts)}]\n{text}".strip()
            return text
    for item in item_list:
        if item.get("type") == ITEM_VOICE:
            voice_text = str((item.get("voice_item") or {}).get("text") or "")
            if voice_text:
                return voice_text
    return ""


async def qr_login(
    timeout_seconds: int = 480,
) -> dict[str, str] | None:
    """Run the interactive iLink QR login flow in the terminal."""
    ssl_ctx = None
    try:
        import ssl

        import certifi

        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass

    connector = aiohttp.TCPConnector(ssl=ssl_ctx) if ssl_ctx else None

    async with aiohttp.ClientSession(trust_env=True, connector=connector) as session:
        try:
            qr_resp = await _api_get(
                session,
                base_url=ILINK_BASE_URL,
                endpoint=f"{EP_GET_BOT_QR}?bot_type=3",
                timeout_ms=QR_TIMEOUT_MS,
            )
        except Exception as exc:
            logger.error("weixin: failed to fetch QR code: %s", exc)
            return None

        qrcode_value = str(qr_resp.get("qrcode") or "")
        qrcode_url = str(qr_resp.get("qrcode_img_content") or "")
        if not qrcode_value:
            logger.error("weixin: QR response missing qrcode")
            return None

        qr_scan_data = qrcode_url if qrcode_url else qrcode_value

        print("\n请使用微信扫描以下二维码：")
        if qrcode_url:
            print(qrcode_url)
        try:
            qr = qrcode.QRCode()
            qr.add_data(qr_scan_data)
            qr.make(fit=True)
            qr.print_ascii(invert=True)
        except Exception as _qr_exc:
            print(f"（终端二维码渲染失败: {_qr_exc}，请直接打开上面的二维码链接）")

        deadline = time.monotonic() + timeout_seconds
        current_base_url = ILINK_BASE_URL
        refresh_count = 0

        while time.monotonic() < deadline:
            try:
                status_resp = await _api_get(
                    session,
                    base_url=current_base_url,
                    endpoint=f"{EP_GET_QR_STATUS}?qrcode={qrcode_value}",
                    timeout_ms=QR_TIMEOUT_MS,
                )
            except TimeoutError:
                await asyncio.sleep(1)
                continue
            except Exception as exc:
                logger.warning("weixin: QR poll error: %s", exc)
                await asyncio.sleep(1)
                continue

            status = str(status_resp.get("status") or "wait")
            if status == "wait":
                print(".", end="", flush=True)
            elif status == "scaned":
                print("\n已扫码，请在微信里确认...")
            elif status == "scaned_but_redirect":
                redirect_host = str(status_resp.get("redirect_host") or "")
                if redirect_host:
                    current_base_url = f"https://{redirect_host}"
            elif status == "expired":
                refresh_count += 1
                if refresh_count > 3:
                    print("\n二维码多次过期，请重新执行登录。")
                    return None
                print(f"\n二维码已过期，正在刷新... ({refresh_count}/3)")
                try:
                    qr_resp = await _api_get(
                        session,
                        base_url=ILINK_BASE_URL,
                        endpoint=f"{EP_GET_BOT_QR}?bot_type=3",
                        timeout_ms=QR_TIMEOUT_MS,
                    )
                    qrcode_value = str(qr_resp.get("qrcode") or "")
                    qrcode_url = str(qr_resp.get("qrcode_img_content") or "")
                    qr_scan_data = qrcode_url if qrcode_url else qrcode_value
                    if qrcode_url:
                        print(qrcode_url)
                    try:
                        qr = qrcode.QRCode()
                        qr.add_data(qr_scan_data)
                        qr.make(fit=True)
                        qr.print_ascii(invert=True)
                    except Exception:
                        pass
                except Exception as exc:
                    logger.error("weixin: QR refresh failed: %s", exc)
                    return None
            elif status == "confirmed":
                account_id = str(status_resp.get("ilink_bot_id") or "")
                token = str(status_resp.get("bot_token") or "")
                baseurl = str(status_resp.get("baseurl") or ILINK_BASE_URL)
                user_id = str(status_resp.get("ilink_user_id") or "")
                if not account_id or not token:
                    logger.error(
                        "weixin: QR confirmed but credential payload incomplete"
                    )
                    return None
                print(f"\n微信连接成功，account_id={account_id}")
                return {
                    "account_id": account_id,
                    "token": token,
                    "base_url": baseurl,
                    "user_id": user_id,
                }
            await asyncio.sleep(1)

        print("\n微信登录超时。")
        return None


class WechatChatbot(Chatbot):
    """WeChat chatbot adapter for Kesoku AI Agent framework."""

    def __init__(self, chatbot_id: str, gateway: Gateway) -> None:
        """Initialize WechatChatbot adapter."""
        super().__init__(chatbot_id, gateway)
        cfg = get_config()
        self.config = cfg.wechat

        if not self.config.enabled:
            raise ValueError("WeChat chatbot is disabled in configuration.")

        if not self.config.account_id or not self.config.token:
            raise ValueError(
                "WeChat chatbot is enabled but account_id or token is missing. "
                "Please run 'kesoku wechat pair' to initialize credentials."
            )

        self._running = False
        self._poll_task: asyncio.Task | None = None
        self._poll_session: aiohttp.ClientSession | None = None
        self._send_session: aiohttp.ClientSession | None = None

        persist_path = None
        if cfg.agent_working_dir:
            persist_path = os.path.join(cfg.agent_working_dir, ".wechat_context_tokens.json")
        self._token_store = ContextTokenStore(persist_path)
        self._typing_cache = TypingTicketCache()
        self._dedup = MessageDeduplicator()

        self._account_id = self.config.account_id.strip()
        self._token = self.config.token.strip()
        self._base_url = self.config.base_url.strip().rstrip("/")
        self._cdn_base_url = WEIXIN_CDN_BASE_URL

    def _build_wechat_custom_prompt(self, chat_id: str, chat_type: str) -> str:
        return f"""
# WeChat Platforms Instructions
You are interacting with the user via WeChat (Weixin).
- Channel/Chat Type: {chat_type}
- Chat ID: {chat_id}
- Wechat is for human interaction. Talk like a human. Don't use markdown formats, bullet points, etc.
- Keep your messages relatively concise since WeChat clients have character and readability limitations.
"""

    async def start(self) -> None:
        """Start the WeChat bot and Gateway listener subscriber loop."""
        self._running = True
        self._listener_task = asyncio.create_task(super().start())

        ssl_ctx = None
        try:
            import ssl

            import certifi

            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            pass

        connector = aiohttp.TCPConnector(ssl=ssl_ctx) if ssl_ctx else None
        self._poll_session = aiohttp.ClientSession(trust_env=True, connector=connector)
        self._send_session = aiohttp.ClientSession(
            trust_env=True,
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=None),
        )

        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info(f"WeChat bot '{self.chatbot_id}' successfully connected.")
        await asyncio.gather(self._listener_task, self._poll_task)

    def stop(self) -> None:
        """Stop the WeChat bot and disconnect sessions."""
        self._running = False
        super().stop()
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
        if self._poll_session and not self._poll_session.closed:
            asyncio.create_task(self._poll_session.close())
        if self._send_session and not self._send_session.closed:
            asyncio.create_task(self._send_session.close())

    async def _poll_loop(self) -> None:
        sync_buf = ""
        timeout_ms = LONG_POLL_TIMEOUT_MS
        consecutive_failures = 0

        while self._running:
            try:
                response = await _get_updates(
                    self._poll_session,
                    base_url=self._base_url,
                    token=self._token,
                    sync_buf=sync_buf,
                    timeout_ms=timeout_ms,
                )
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
                    asyncio.create_task(self._process_message_safe(msg))
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
                    BACKOFF_DELAY_SECONDS
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES
                    else RETRY_DELAY_SECONDS
                )
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    consecutive_failures = 0

    async def _handle_slash_command(self, chat_id: str, cmd_text: str, sender_id: str) -> None:
        """Process text-based slash commands for WeChat."""
        parts = cmd_text.strip().split()
        command = parts[0].lower()
        context_token = self._token_store.get(self._account_id, sender_id)
        client_id = f"kesoku-wechat-cmd-{uuid.uuid4().hex}"

        if command == "/restart":
            logger.info(f"WeChat: Received /restart command from {sender_id} in {chat_id}")
            try:
                await _send_message(
                    self._send_session,
                    base_url=self._base_url,
                    token=self._token,
                    to=chat_id,
                    text="🔄 Restarting service...",
                    context_token=context_token,
                    client_id=client_id,
                )
            except Exception as e:
                logger.error(f"WeChat: failed to send restart confirmation: {e}")

            await asyncio.sleep(0.5)
            self.stop()

            # Resolve kesoku binary path
            import shutil
            import sys
            executable_dir = os.path.dirname(sys.executable)
            kesoku_bin = os.path.join(executable_dir, "kesoku")
            if not os.path.exists(kesoku_bin):  # noqa: ASYNC240
                kesoku_bin = shutil.which("kesoku") or "kesoku"

            import subprocess
            cmd = [kesoku_bin, "service", "restart"]
            service_user = os.environ.get("KESOKU_SERVICE_USER", "true") == "true"
            if service_user:
                cmd.append("--user")
            else:
                cmd.append("--system")

            instance_name = os.environ.get("KESOKU_SERVICE_INSTANCE_NAME")
            if instance_name:
                cmd.extend(["--name", instance_name])

            logger.info(f"WeChat: Launching restart command: {' '.join(cmd)}")
            subprocess.Popen(cmd, start_new_session=True)  # noqa: ASYNC220

        elif command in ("/clear", "/reset"):
            logger.info(f"WeChat: Received {command} command from {sender_id} in {chat_id}")
            session = await self.gateway.get_session_by_channel(self.chatbot_id, chat_id)
            if session:
                agent = self.gateway.agent
                if agent:
                    worker = agent.workers.get(session.id)
                    if worker:
                        worker.stop()
                        agent.workers.pop(session.id, None)
                await self.gateway.delete_session(session.id)
                reply = "♻️ Session successfully cleared. The next message will initiate a new session."
            else:
                reply = "⚠️ No active session found for this chat."

            try:
                await _send_message(
                    self._send_session,
                    base_url=self._base_url,
                    token=self._token,
                    to=chat_id,
                    text=reply,
                    context_token=context_token,
                    client_id=client_id,
                )
            except Exception as e:
                logger.error(f"WeChat: failed to send reset reply: {e}")

        elif command == "/status":
            logger.info(f"WeChat: Received /status command from {sender_id} in {chat_id}")
            session = await self.gateway.get_session_by_channel(self.chatbot_id, chat_id)
            if session:
                history = await self.gateway.get_session_history(session.id, limit=100)
                metrics = None
                for msg in reversed(history):
                    if msg.role == ROLE_ASSISTANT and msg.metadata and msg.metadata.get("turn_metrics"):
                        metrics = msg.metadata.get("turn_metrics")
                        break

                session_turns = len([m for m in history if m.role == ROLE_USER])
                context_tokens = metrics.get("context_tokens", 0) if metrics else 0
                turn_tool_calls = metrics.get("turn_tool_calls", 0) if metrics else len(
                    [m for m in history if m.role == ROLE_TOOL and m.type == TYPE_TOOL_CALL]
                )
                turn_tokens = metrics.get("turn_tokens", 0) if metrics else 0
                turn_time = metrics.get("turn_time", 0.0) if metrics else 0.0

                context_k = f"{round(context_tokens / 1000)}K" if context_tokens else "0K"
                turn_k = f"{round(turn_tokens / 1000)}K" if turn_tokens else "0K"

                reply = (
                    f"【Current Stats】\n"
                    f"⚡ Session: {session_turns} turns\n"
                    f"📖 Context: {context_k} tokens\n"
                    f"⏱️ Last Turn:\n"
                    f"  - Tool Calls: {turn_tool_calls}\n"
                    f"  - Tokens: {turn_k}\n"
                    f"  - Time: {turn_time:.1f}s"
                )
            else:
                reply = "⚠️ No active session found for this chat."

            try:
                await _send_message(
                    self._send_session,
                    base_url=self._base_url,
                    token=self._token,
                    to=chat_id,
                    text=reply,
                    context_token=context_token,
                    client_id=client_id,
                )
            except Exception as e:
                logger.error(f"WeChat: failed to send status reply: {e}")
        else:
            reply = f"⚠️ Unrecognized command: {command}"
            try:
                await _send_message(
                    self._send_session,
                    base_url=self._base_url,
                    token=self._token,
                    to=chat_id,
                    text=reply,
                    context_token=context_token,
                    client_id=client_id,
                )
            except Exception as e:
                logger.error(f"WeChat: failed to send error reply: {e}")

    async def trigger_cronjob(
        self,
        channel_id: str,
        prompt_content: str,
        mention_user_id: str | None = None,
    ) -> None:
        """Trigger a scheduled cronjob in the specified WeChat chat/room."""
        session = await self.gateway.get_session_by_channel(self.chatbot_id, channel_id)
        if not session:
            title = f"WeChat Scheduled Job {channel_id}"
            chat_type = "group" if channel_id.endswith("@chatroom") else "dm"
            custom_prompt = self._build_wechat_custom_prompt(channel_id, chat_type)

            # Read custom configurable system prompt file if present
            if self.config.sys_prompt_file:
                sys_file = self.config.sys_prompt_file
                cfg = get_config()
                if not os.path.isabs(sys_file) and cfg.agent_working_dir:
                    sys_file = os.path.join(cfg.agent_working_dir, sys_file)  # noqa: ASYNC240
                if os.path.exists(sys_file):  # noqa: ASYNC240
                    try:
                        with open(sys_file, encoding="utf-8") as f:  # noqa: ASYNC230
                            custom_sys_prompt = f.read().strip()
                        if custom_sys_prompt:
                            custom_prompt = f"{custom_prompt}\n\n{custom_sys_prompt}"
                    except Exception as e:
                        logger.error(f"WeChat: Failed to read system prompt file {sys_file}: {e}")

            session = await self.gateway.create_session(
                session_id=None,
                title=title,
                custom_prompt=custom_prompt,
            )
        else:
            await self.gateway.update_session_updated_at(session.id)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg_content = f"Scheduled job starting at `{now_str}`:\n{prompt_content}"
        if mention_user_id:
            msg_content = f"@{mention_user_id} {msg_content}"

        user_msg = Message(
            session_id=session.id,
            chatbot_id=self.chatbot_id,
            channel_id=channel_id,
            sender="System Scheduler",
            role=ROLE_USER,
            type=TYPE_TEXT,
            content=msg_content,
            timestamp=time.time(),
            status=STATUS_PENDING_AGENT,
            metadata={
                "cronjob": True,
            },
        )
        await self.gateway.post(user_msg)

        # Start typing indicator
        typing_ticket = self._typing_cache.get(channel_id)
        if typing_ticket:
            try:
                await _send_typing(
                    self._send_session,
                    base_url=self._base_url,
                    token=self._token,
                    to_user_id=channel_id,
                    typing_ticket=typing_ticket,
                    status=TYPING_START,
                )
            except Exception as e:
                logger.debug("WeChat: typing start failed: %s", e)

    async def _process_message_safe(self, message: dict[str, Any]) -> None:
        try:
            await self._process_message(message)
        except Exception as exc:
            logger.error(
                "WeChat: unhandled inbound error: %s",
                exc,
                exc_info=True,
            )

    async def _process_message(self, message: dict[str, Any]) -> None:
        sender_id = str(message.get("from_user_id") or "").strip()
        if not sender_id or sender_id == self._account_id:
            return

        message_id = str(message.get("message_id") or "").strip()
        if message_id and self._dedup.is_duplicate(message_id):
            return

        item_list = message.get("item_list") or []
        text = _extract_text(item_list)

        chat_type, effective_chat_id = _guess_chat_type(message, self._account_id)

        if text.startswith("/"):
            await self._handle_slash_command(effective_chat_id, text, sender_id)
            return

        context_token = str(message.get("context_token") or "").strip()
        if context_token:
            self._token_store.set(self._account_id, effective_chat_id, context_token)
        asyncio.create_task(
            self._maybe_fetch_typing_ticket(sender_id, context_token or None)
        )

        # Resolve or create Kesoku session mapped to the thread/space context
        channel_id = effective_chat_id
        session = await self.gateway.get_session_by_channel(
            self.chatbot_id, channel_id
        )
        if not session:
            title = f"WeChat Session: {text[:30]}"
            custom_prompt = self._build_wechat_custom_prompt(channel_id, chat_type)

            # Read custom configurable system prompt file if present
            if self.config.sys_prompt_file:
                sys_file = self.config.sys_prompt_file
                cfg = get_config()
                if not os.path.isabs(sys_file) and cfg.agent_working_dir:
                    sys_file = os.path.join(cfg.agent_working_dir, sys_file)  # noqa: ASYNC240
                if os.path.exists(sys_file):  # noqa: ASYNC240
                    try:
                        with open(sys_file, encoding="utf-8") as f:  # noqa: ASYNC230
                            custom_sys_prompt = f.read().strip()
                        if custom_sys_prompt:
                            custom_prompt = (
                                f"{custom_prompt}\n\n{custom_sys_prompt}"
                            )
                    except Exception as e:
                        logger.error(
                            f"WeChat: Failed to read system prompt file {sys_file}: {e}"
                        )

            session = await self.gateway.create_session(
                session_id=None,
                title=title,
                custom_prompt=custom_prompt,
            )
        else:
            await self.gateway.update_session_updated_at(session.id)

        attachments_metadata = []
        session_staging_dir = os.path.realpath(  # noqa: ASYNC240
            os.path.join(
                get_config().workspace.sessions_dir, session.workspace_name
            )
        )
        os.makedirs(session_staging_dir, exist_ok=True)

        # Download and decrypt media assets
        for item in item_list:
            item_type = item.get("type")
            filename = None
            data = None
            mime = "application/octet-stream"

            if item_type == ITEM_IMAGE:
                media = _media_reference(item, "image_item")
                aeskey = (item.get("image_item") or {}).get("aeskey")
                aes_key_b64 = (
                    base64.b64encode(bytes.fromhex(str(aeskey))).decode("ascii")
                    if aeskey
                    else media.get("aes_key")
                )
                try:
                    data = await _download_and_decrypt_media(
                        self._poll_session,
                        cdn_base_url=self._cdn_base_url,
                        encrypted_query_param=media.get("encrypt_query_param"),
                        aes_key_b64=aes_key_b64,
                        full_url=media.get("full_url"),
                        timeout_seconds=30.0,
                    )
                    filename = f"wechat_image_{secrets.token_hex(4)}.jpg"
                    mime = "image/jpeg"
                except Exception as e:
                    logger.warning("WeChat: image download failed: %s", e)

            elif item_type == ITEM_VIDEO:
                media = _media_reference(item, "video_item")
                try:
                    data = await _download_and_decrypt_media(
                        self._poll_session,
                        cdn_base_url=self._cdn_base_url,
                        encrypted_query_param=media.get("encrypt_query_param"),
                        aes_key_b64=media.get("aes_key"),
                        full_url=media.get("full_url"),
                        timeout_seconds=120.0,
                    )
                    filename = f"wechat_video_{secrets.token_hex(4)}.mp4"
                    mime = "video/mp4"
                except Exception as e:
                    logger.warning("WeChat: video download failed: %s", e)

            elif item_type == ITEM_FILE:
                file_item = item.get("file_item") or {}
                media = file_item.get("media") or {}
                filename = str(file_item.get("file_name") or "document.bin")
                mime = _mime_from_filename(filename)
                try:
                    data = await _download_and_decrypt_media(
                        self._poll_session,
                        cdn_base_url=self._cdn_base_url,
                        encrypted_query_param=media.get("encrypt_query_param"),
                        aes_key_b64=media.get("aes_key"),
                        full_url=media.get("full_url"),
                        timeout_seconds=60.0,
                    )
                except Exception as e:
                    logger.warning("WeChat: file download failed: %s", e)

            elif item_type == ITEM_VOICE:
                voice_item = item.get("voice_item") or {}
                media = voice_item.get("media") or {}
                if not voice_item.get("text"):
                    try:
                        data = await _download_and_decrypt_media(
                            self._poll_session,
                            cdn_base_url=self._cdn_base_url,
                            encrypted_query_param=media.get("encrypt_query_param"),
                            aes_key_b64=media.get("aes_key"),
                            full_url=media.get("full_url"),
                            timeout_seconds=60.0,
                        )
                        filename = f"wechat_voice_{secrets.token_hex(4)}.silk"
                        mime = "audio/silk"
                    except Exception as e:
                        logger.warning("WeChat: voice download failed: %s", e)

            if data and filename:
                # Sanitize filename
                safe_filename = "".join(
                    c for c in filename if c.isalnum() or c in "._-"
                )
                filepath = os.path.join(session_staging_dir, safe_filename)
                with open(filepath, "wb") as f:  # noqa: ASYNC230
                    f.write(data)
                attachments_metadata.append({
                    "path": filepath,
                    "mime_type": mime,
                    "filename": filename,
                })

        if not text and not attachments_metadata:
            return

        msg_content = text
        if attachments_metadata:
            files_str = "\n".join(
                f"[Attachment: {a['filename']} ({a['mime_type']}) saved at {a['path']}]"
                for a in attachments_metadata
            )
            if msg_content:
                msg_content += f"\n\nAttachments:\n{files_str}"
            else:
                msg_content = f"Attachments:\n{files_str}"

        user_msg = Message(
            session_id=session.id,
            chatbot_id=self.chatbot_id,
            channel_id=channel_id,
            sender=sender_id,
            role=ROLE_USER,
            type=TYPE_TEXT,
            content=msg_content,
            timestamp=time.time(),
            status=STATUS_PENDING_AGENT,
            metadata={
                "wechat_message_id": message_id,
                "attachments": attachments_metadata,
                "chat_type": chat_type,
            },
        )
        await self.gateway.post(user_msg)

        # Start typing indicator
        typing_ticket = self._typing_cache.get(channel_id)
        if typing_ticket:
            try:
                await _send_typing(
                    self._send_session,
                    base_url=self._base_url,
                    token=self._token,
                    to_user_id=channel_id,
                    typing_ticket=typing_ticket,
                    status=TYPING_START,
                )
            except Exception as e:
                logger.debug("WeChat: typing start failed: %s", e)

    async def _maybe_fetch_typing_ticket(
        self, user_id: str, context_token: str | None
    ) -> None:
        if not self._poll_session or not self._token:
            return
        if self._typing_cache.get(user_id):
            return
        try:
            response = await _get_config(
                self._poll_session,
                base_url=self._base_url,
                token=self._token,
                user_id=user_id,
                context_token=context_token,
            )
            typing_ticket = str(response.get("typing_ticket") or "")
            if typing_ticket:
                self._typing_cache.set(user_id, typing_ticket)
        except Exception as e:
            logger.debug("WeChat: getConfig failed: %s", e)

    async def handle_message(self, message: Message) -> None:
        """Process and send outgoing assistant messages to WeChat API."""
        if message.chatbot_id != self.chatbot_id:
            return

        chat_id = message.channel_id
        context_token = self._token_store.get(self._account_id, chat_id)

        is_intermediate = (
            (message.role == ROLE_ASSISTANT and message.type == TYPE_THOUGHT)
            or (message.role == ROLE_TOOL)
            or (message.role == ROLE_SYSTEM)
        )
        if is_intermediate:
            return

        # Stop typing status
        typing_ticket = self._typing_cache.get(chat_id)
        if typing_ticket:
            try:
                await _send_typing(
                    self._send_session,
                    base_url=self._base_url,
                    token=self._token,
                    to_user_id=chat_id,
                    typing_ticket=typing_ticket,
                    status=TYPING_STOP,
                )
            except Exception as e:
                logger.debug("WeChat: typing stop failed: %s", e)

        segments = parse_message_content(message.content)

        for segment in segments:
            if segment["type"] == "text":
                text_content = segment["content"]
                if text_content.strip():
                    # Normalize markdown and lines for WeChat
                    normalized_text = _wrap_copy_friendly_lines_for_weixin(
                        _normalize_markdown_blocks(text_content)
                    )
                    chunks = _split_text_for_weixin_delivery(
                        normalized_text, max_length=2000
                    )
                    for chunk in chunks:
                        if chunk.strip():
                            client_id = f"kesoku-wechat-{uuid.uuid4().hex}"
                            try:
                                await _send_message(
                                    self._send_session,
                                    base_url=self._base_url,
                                    token=self._token,
                                    to=chat_id,
                                    text=chunk,
                                    context_token=context_token,
                                    client_id=client_id,
                                )
                            except Exception as e:
                                logger.error("WeChat: failed to send text: %s", e)

            elif segment["type"] in {"file", "voice"}:
                file_path = segment["path"]
                if not os.path.exists(file_path):  # noqa: ASYNC240
                    logger.error("WeChat: Outbound file not found: %s", file_path)
                    continue

                # Upload and send file
                try:
                    await self._send_file(
                        chat_id=chat_id,
                        path=file_path,
                        context_token=context_token,
                    )
                except Exception as e:
                    logger.error("WeChat: failed to send outbound file: %s", e)

        await self.gateway.update_message_status(message.id, STATUS_DELIVERED)

    async def _send_file(
        self,
        chat_id: str,
        path: str,
        context_token: str | None,
    ) -> str:
        plaintext = Path(path).read_bytes()  # noqa: ASYNC240
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"

        # Determine media type
        if mime.startswith("image/"):
            media_type = MEDIA_IMAGE
        elif mime.startswith("video/"):
            media_type = MEDIA_VIDEO
        elif path.endswith(".silk"):
            media_type = MEDIA_VOICE
        else:
            media_type = MEDIA_FILE

        filekey = secrets.token_hex(16)
        aes_key = secrets.token_bytes(16)
        rawsize = len(plaintext)
        rawfilemd5 = hashlib.md5(plaintext).hexdigest()

        upload_response = await _get_upload_url(
            self._send_session,
            base_url=self._base_url,
            token=self._token,
            to_user_id=chat_id,
            media_type=media_type,
            filekey=filekey,
            rawsize=rawsize,
            rawfilemd5=rawfilemd5,
            filesize=_aes_padded_size(rawsize),
            aeskey_hex=aes_key.hex(),
        )

        upload_param = str(upload_response.get("upload_param") or "")
        upload_full_url = str(upload_response.get("upload_full_url") or "")
        ciphertext = _aes128_ecb_encrypt(plaintext, aes_key)

        if upload_full_url:
            upload_url = upload_full_url
        elif upload_param:
            upload_url = _cdn_upload_url(
                WEIXIN_CDN_BASE_URL, upload_param, filekey
            )
        else:
            raise RuntimeError(
                f"WeChat: upload URL unavailable: {upload_response}"
            )

        encrypted_query_param = await _upload_ciphertext(
            self._send_session,
            ciphertext=ciphertext,
            upload_url=upload_url,
        )

        aes_key_for_api = base64.b64encode(
            aes_key.hex().encode("ascii")
        ).decode("ascii")

        media_item = {
            "media": {
                "encrypt_query_param": encrypted_query_param,
                "aes_key": aes_key_for_api,
                "encrypt_type": 1,
            }
        }

        if media_type == MEDIA_IMAGE:
            media_payload = {
                "type": ITEM_IMAGE,
                "image_item": {
                    "media": media_item["media"],
                    "mid_size": len(ciphertext),
                },
            }
        elif media_type == MEDIA_VIDEO:
            media_payload = {
                "type": ITEM_VIDEO,
                "video_item": {
                    "media": media_item["media"],
                    "video_size": len(ciphertext),
                    "video_md5": rawfilemd5,
                },
            }
        elif media_type == MEDIA_VOICE:
            media_payload = {
                "type": ITEM_VOICE,
                "voice_item": {
                    "media": media_item["media"],
                    "encode_type": 6,
                    "sample_rate": 24000,
                    "bits_per_sample": 16,
                },
            }
        else:
            media_payload = {
                "type": ITEM_FILE,
                "file_item": {
                    "media": media_item["media"],
                    "file_name": Path(path).name,
                    "len": str(rawsize),
                },
            }

        last_message_id = f"kesoku-wechat-{uuid.uuid4().hex}"
        await _api_post(
            self._send_session,
            base_url=self._base_url,
            endpoint=EP_SEND_MESSAGE,
            payload={
                "msg": {
                    "from_user_id": "",
                    "to_user_id": chat_id,
                    "client_id": last_message_id,
                    "message_type": MSG_TYPE_BOT,
                    "message_state": MSG_STATE_FINISH,
                    "item_list": [media_payload],
                    **({"context_token": context_token} if context_token else {}),
                }
            },
            token=self._token,
            timeout_ms=API_TIMEOUT_MS,
        )
        return last_message_id
