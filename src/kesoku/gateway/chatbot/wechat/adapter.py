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
import ssl
import textwrap
import time
import uuid
from pathlib import Path
from typing import Any

import aiohttp
import certifi
import qrcode

from kesoku.config import get_config
from kesoku.constants import MessageRole, MessageType
from kesoku.db import Message
from kesoku.gateway.attachment_manager import AttachmentManager
from kesoku.gateway.chatbot.base import Chatbot, InboundMessageAttachment, InboundMessageDTO
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger
from kesoku.utils.async_fs import (
    async_exists,
    async_read_bytes,
    async_read_text_file,
)
from kesoku.utils.crypto import (
    aes_padded_size as _aes_padded_size,
)
from kesoku.utils.image import (
    compress_image as _compress_image,
)
from kesoku.utils.image import (
    detect_image_mime_type as _detect_image_mime_type,
)

from .client import (
    ILINK_BASE_URL,
    ITEM_FILE,
    ITEM_IMAGE,
    ITEM_TEXT,
    ITEM_VIDEO,
    ITEM_VOICE,
    MEDIA_FILE,
    MEDIA_IMAGE,
    MEDIA_VIDEO,
    MEDIA_VOICE,
    MSG_STATE_FINISH,
    MSG_TYPE_BOT,
    TYPING_START,
    TYPING_STOP,
    WEIXIN_CDN_BASE_URL,
    ILinkClient,
)
from .listener import WeChatListener
from .media import WeChatMediaManager

logger = setup_logger(__name__)

MESSAGE_DEDUP_TTL_SECONDS = 300
WEIXIN_COPY_LINE_WIDTH = 120

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
            with open(self._persist_path, encoding="utf-8") as f:
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

    def get_all_channels(self, account_id: str) -> list[str]:
        """Get all channel IDs (user IDs) associated with this account ID."""
        prefix = f"{account_id}:"
        return [k[len(prefix) :] for k in self._cache.keys() if k.startswith(prefix)]


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





def _guess_chat_type(message: dict[str, Any], account_id: str) -> tuple[str, str]:
    room_id = str(message.get("room_id") or message.get("chat_room_id") or "").strip()
    to_user_id = str(message.get("to_user_id") or "").strip()
    is_group = bool(room_id) or (
        to_user_id and account_id and to_user_id != account_id and message.get("msg_type") == 1
    )
    if is_group:
        return "group", room_id or to_user_id or str(message.get("from_user_id") or "")
    return "dm", str(message.get("from_user_id") or "")











def _media_reference(item: dict[str, Any], key: str) -> dict[str, Any]:
    return (item.get(key) or {}).get("media") or {}





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


def _split_text_for_weixin_delivery(content: str, max_length: int, split_per_line: bool = False) -> list[str]:
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
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass

    connector = aiohttp.TCPConnector(ssl=ssl_ctx) if ssl_ctx else None

    async with aiohttp.ClientSession(trust_env=True, connector=connector) as session:
        client = ILinkClient(session, base_url=ILINK_BASE_URL, token="")
        try:
            qr_resp = await client.get_bot_qrcode(bot_type=3)
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
        refresh_count = 0

        while time.monotonic() < deadline:
            try:
                status_resp = await client.get_qrcode_status(qrcode=qrcode_value)
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
                    client.base_url = f"https://{redirect_host}"
            elif status == "expired":
                refresh_count += 1
                if refresh_count > 3:
                    print("\n二维码多次过期，请重新执行登录。")
                    return None
                print(f"\n二维码已过期，正在刷新... ({refresh_count}/3)")
                try:
                    client.base_url = ILINK_BASE_URL
                    qr_resp = await client.get_bot_qrcode(bot_type=3)
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
                    logger.error("weixin: QR confirmed but credential payload incomplete")
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
        self.attachment_manager = AttachmentManager()
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
        self._poll_session_val: aiohttp.ClientSession | None = None
        self._send_session_val: aiohttp.ClientSession | None = None

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

        self.poll_client: ILinkClient | None = None
        self.send_client: ILinkClient | None = None
        self.media_manager: WeChatMediaManager | None = None
        self.listener: WeChatListener | None = None

    @property
    def _poll_session(self) -> aiohttp.ClientSession | None:
        return self._poll_session_val

    @_poll_session.setter
    def _poll_session(self, session: aiohttp.ClientSession | None) -> None:
        self._poll_session_val = session
        if session:
            self.poll_client = ILinkClient(session, base_url=self._base_url, token=self._token)
            self.listener = WeChatListener(self.poll_client, self._process_message_safe)
        else:
            self.poll_client = None
            self.listener = None

    @property
    def _send_session(self) -> aiohttp.ClientSession | None:
        return self._send_session_val

    @_send_session.setter
    def _send_session(self, session: aiohttp.ClientSession | None) -> None:
        self._send_session_val = session
        if session:
            self.send_client = ILinkClient(session, base_url=self._base_url, token=self._token)
            self.media_manager = WeChatMediaManager(self.send_client)
        else:
            self.send_client = None
            self.media_manager = None

    def _build_wechat_custom_prompt(self, chat_id: str, chat_type: str) -> str:
        return f"""
# WeChat Platforms Instructions
You are interacting with the user via WeChat (Weixin).
- Channel/Chat Type: {chat_type}
- Chat ID: {chat_id}
- Wechat is for human interaction. Talk like a human. Don't use markdown formats, bullet points, etc.
- Keep your messages relatively concise since WeChat clients have character and readability limitations.
"""

    async def _compile_custom_prompt(self, channel_id: str, chat_type: str) -> str:
        custom_prompt = self._build_wechat_custom_prompt(channel_id, chat_type)
        if self.config.sys_prompt_file:
            sys_file = self.config.sys_prompt_file
            cfg = get_config()
            if not os.path.isabs(sys_file) and cfg.agent_working_dir:
                sys_file = os.path.join(cfg.agent_working_dir, sys_file)
            if await async_exists(sys_file):
                try:
                    custom_sys_prompt = await async_read_text_file(sys_file)
                    if custom_sys_prompt:
                        custom_prompt = f"{custom_prompt}\n\n{custom_sys_prompt}"
                except Exception as e:
                    logger.error(f"WeChat: Failed to read system prompt file {sys_file}: {e}")
        return custom_prompt

    async def start(self) -> None:
        """Start the WeChat bot and Gateway listener subscriber loop."""
        self._running = True
        self._listener_task = asyncio.create_task(super().start())

        ssl_ctx = None
        try:
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

        self.listener.start()
        logger.info(f"WeChat bot '{self.chatbot_id}' successfully connected.")
        await self._listener_task

    def stop(self) -> None:
        """Stop the WeChat bot and disconnect sessions."""
        self._running = False
        super().stop()
        if self.listener:
            self.listener.stop()
        if self._poll_session and not self._poll_session.closed:
            asyncio.create_task(self._poll_session.close())
        if self._send_session and not self._send_session.closed:
            asyncio.create_task(self._send_session.close())

    async def _handle_slash_command(self, chat_id: str, cmd_text: str, sender_id: str) -> None:
        """Process text-based slash commands for WeChat."""
        context_token = self._token_store.get(self._account_id, sender_id)
        client_id = f"kesoku-wechat-cmd-{uuid.uuid4().hex}"

        async def reply_func(text: str) -> None:
            try:
                await self.send_client.send_message(
                    to=chat_id,
                    text=text,
                    context_token=context_token,
                    client_id=client_id,
                )
            except Exception as e:
                logger.error(f"WeChat: failed to send command reply: {e}")

        await self.execute_command_from_text(cmd_text, reply_func, channel_id=chat_id)

    async def trigger_cronjob(
        self,
        channel_id: str | None,
        prompt_content: str,
        mention_user_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Trigger a scheduled cronjob in the specified WeChat chat/room."""
        min_idle = kwargs.get("min_idle_time") or kwargs.get("min_idle_time_seconds")
        max_messages = kwargs.get("max_messages")

        async def should_trigger_for_channel(chan: str) -> bool:
            # 1. Idle check
            if min_idle is not None:
                last_msg_ts = await self.gateway.db.get_last_message_timestamp(self.chatbot_id, chan)
                if last_msg_ts is not None:
                    idle_time = time.time() - last_msg_ts
                    if idle_time < min_idle:
                        logger.info(
                            f"WeChat: Skip channel {chan} because it has been idle for only {idle_time:.1f}s "
                            f"(required {min_idle}s)."
                        )
                        return False

            # 2. Max Messages check
            if max_messages is not None:
                count, _ = await self.gateway.db.get_cronjob_sent_stats_today(self.chatbot_id, chan)
                if count >= max_messages:
                    logger.info(
                        f"WeChat: Skip channel {chan} because daily max messages limit of {max_messages} "
                        f"has already been reached today ({count} sent)."
                    )
                    return False
            return True

        if not channel_id:
            channels = self._token_store.get_all_channels(self._account_id)
            if not channels:
                logger.warning(
                    "WeChat: Cannot trigger cronjob because no active channel/user is saved in the context file."
                )
                return
            for chan in channels:
                if not await should_trigger_for_channel(chan):
                    continue
                await self._trigger_cronjob_for_channel(
                    channel_id=chan,
                    prompt_content=prompt_content,
                    mention_user_id=mention_user_id,
                )
        else:
            if not await should_trigger_for_channel(channel_id):
                return
            await self._trigger_cronjob_for_channel(
                channel_id=channel_id,
                prompt_content=prompt_content,
                mention_user_id=mention_user_id,
            )

    async def _trigger_cronjob_for_channel(
        self,
        channel_id: str,
        prompt_content: str,
        mention_user_id: str | None = None,
    ) -> None:
        """Helper to trigger cronjob in a single specific channel."""
        chat_type = "group" if channel_id.endswith("@chatroom") else "dm"
        custom_prompt = await self._compile_custom_prompt(channel_id, chat_type)

        msg_content = prompt_content
        if mention_user_id:
            msg_content = f"@{mention_user_id} {msg_content}"

        await self.trigger_cronjob_message(
            channel_id=channel_id,
            prompt_content=msg_content,
            sender_name="Cronjob",
            custom_prompt=custom_prompt,
            metadata={"wechat_cronjob": True},
            title=f"WeChat Scheduled Job {channel_id}",
        )

        # Start typing indicator
        typing_ticket = self._typing_cache.get(channel_id)
        if typing_ticket:
            try:
                await self.send_client.send_typing(
                    to=channel_id,
                    typing_status=TYPING_START,
                    ticket=typing_ticket,
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

    async def pre_ingest_hook(self, dto: InboundMessageDTO) -> None:
        """Hook executed before session resolution or creation."""
        context_token = dto.raw_metadata.get("context_token")
        if context_token:
            self._token_store.set(self._account_id, dto.channel_id, context_token)
        asyncio.create_task(self._maybe_fetch_typing_ticket(dto.sender_id, context_token or None))

    async def process_attachments_hook(
        self, session: Any, dto: InboundMessageDTO, raw_message: Any
    ) -> list[InboundMessageAttachment]:
        """Hook to process and save attachments using the resolved session workspace."""
        item_list = raw_message.get("item_list") or []
        attachments_metadata = []

        for item in item_list:
            item_type = item.get("type")
            filename = None
            data = None
            mime = "application/octet-stream"

            if item_type == ITEM_IMAGE:
                media = _media_reference(item, "image_item")
                aeskey = (item.get("image_item") or {}).get("aeskey")
                aes_key_b64 = (
                    base64.b64encode(bytes.fromhex(str(aeskey))).decode("ascii") if aeskey else media.get("aes_key")
                )
                try:
                    data = await self.media_manager.download_and_decrypt(
                        encrypted_query_param=media.get("encrypt_query_param"),
                        aes_key_b64=aes_key_b64,
                        full_url=media.get("full_url"),
                        timeout_seconds=30.0,
                    )
                    mime, ext = _detect_image_mime_type(data, fallback_mime="image/jpeg")
                    filename = f"wechat_image_{secrets.token_hex(4)}{ext}"
                except Exception as e:
                    logger.warning("WeChat: image download failed: %s", e)

            elif item_type == ITEM_VIDEO:
                media = _media_reference(item, "video_item")
                try:
                    data = await self.media_manager.download_and_decrypt(
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
                    data = await self.media_manager.download_and_decrypt(
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
                        data = await self.media_manager.download_and_decrypt(
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
                saved = await self.attachment_manager.save_attachment(
                    filename=filename,
                    workspace_name=session.workspace_name,
                    data=data,
                    collision_id=secrets.token_hex(4),
                )
                attachments_metadata.append(
                    InboundMessageAttachment(
                        path=saved["path"],
                        mime_type=mime,
                        filename=filename,
                    )
                )
        return attachments_metadata

    async def post_ingest_hook(self, session: Any, message: Message, dto: InboundMessageDTO) -> None:
        """Hook executed after the message is successfully posted to the gateway."""
        # Start typing indicator
        typing_ticket = self._typing_cache.get(dto.channel_id)
        if typing_ticket:
            try:
                await self.send_client.send_typing(
                    to=dto.channel_id,
                    typing_status=TYPING_START,
                    ticket=typing_ticket,
                )
            except Exception as e:
                logger.debug("WeChat: typing start failed: %s", e)

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
        channel_id = effective_chat_id

        # Compile prompt
        custom_prompt = await self._compile_custom_prompt(channel_id, chat_type)

        context_token = str(message.get("context_token") or "").strip()

        dto = InboundMessageDTO(
            sender_id=sender_id,
            channel_id=channel_id,
            text=text,
            message_id=message_id,
            timestamp=time.time(),
            raw_metadata={
                "wechat_message_id": message_id,
                "chat_type": chat_type,
                "context_token": context_token,
            },
            session_title=f"WeChat Session: {text[:30]}",
            custom_prompt=custom_prompt,
        )

        async def reply_func(reply_text: str) -> None:
            msg_context_token = context_token or self._token_store.get(self._account_id, sender_id)
            client_id = f"kesoku-wechat-cmd-{uuid.uuid4().hex}"
            try:
                await self.send_client.send_message(
                    to=channel_id,
                    text=reply_text,
                    context_token=msg_context_token,
                    client_id=client_id,
                )
            except Exception as e:
                logger.error(f"WeChat: failed to send command reply: {e}")

        await self.ingest_message(dto, raw_message=message, reply_callback=reply_func)

    async def _maybe_fetch_typing_ticket(self, user_id: str, context_token: str | None) -> None:
        if not self._poll_session or not self._token:
            return
        if self._typing_cache.get(user_id):
            return
        try:
            response = await self.poll_client.get_config(
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
        await self.render_outgoing_message(message)

    def format_text(self, text: str) -> str:
        """Clean, normalize, and wrap markdown blocks for WeChat compatibility."""
        return _wrap_copy_friendly_lines_for_weixin(_normalize_markdown_blocks(text))

    def split_text_into_chunks(self, text: str, max_length: int) -> list[str]:
        """Split text blocks into chunks appropriate for WeChat character limits."""
        return _split_text_for_weixin_delivery(text, max_length)

    async def send_text_chunks(self, channel_id: str, chunks: list[str], message: Message) -> None:
        """Deliver formatted text chunks to WeChat API room or user context."""
        context_token = self._token_store.get(self._account_id, channel_id)
        for chunk in chunks:
            if chunk.strip():
                client_id = f"kesoku-wechat-{uuid.uuid4().hex}"
                try:
                    await self.send_client.send_message(
                        to=channel_id,
                        text=chunk,
                        context_token=context_token,
                        client_id=client_id,
                    )
                except Exception as e:
                    logger.error("WeChat: failed to send text: %s", e)

    async def send_file_segment(
        self,
        channel_id: str,
        file_path: str,
        message: Message,
    ) -> None:
        """Deliver a file attachment segment via WeChat media API with retry mechanism and logging."""
        if not await async_exists(file_path):
            logger.error("WeChat: Outbound file not found: %s", file_path)
            return

        context_token = self._token_store.get(self._account_id, channel_id)
        logger.info("WeChat: Starting file transmission for %s to channel %s...", Path(file_path).name, channel_id)

        max_attempts = 4  # 1 original attempt + 3 retries
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            if attempt > 1:
                logger.info(
                    "WeChat: Retrying file transmission for %s [Attempt %d/%d]...",
                    Path(file_path).name,
                    attempt,
                    max_attempts,
                )
            try:
                await self._send_file(
                    chat_id=channel_id,
                    path=file_path,
                    context_token=context_token,
                )
                logger.info(
                    "WeChat: File transmission completed successfully for %s to channel %s on attempt %d/%d.",
                    Path(file_path).name,
                    channel_id,
                    attempt,
                    max_attempts,
                )
                return
            except Exception as e:
                err_msg = "timeout" if isinstance(e, asyncio.TimeoutError) else str(e)
                if attempt < max_attempts:
                    logger.warning(
                        "WeChat: File transmission failed on attempt %d/%d due to %s. Retrying in 2 seconds...",
                        attempt,
                        max_attempts,
                        err_msg,
                    )
                    await asyncio.sleep(2)
                else:
                    logger.error(
                        "WeChat: File transmission failed completely for %s to channel %s after %d attempts: %s",
                        Path(file_path).name,
                        channel_id,
                        max_attempts,
                        e,
                        exc_info=True,
                    )

    async def send_voice_segment(self, channel_id: str, file_path: str, message: Message) -> None:
        """Deliver a voice segment, routing it directly as a generic file attachment."""
        await self.send_file_segment(channel_id, file_path, message)

    async def send_question_segment(self, channel_id: str, question: str, choices: list[str], message: Message) -> None:
        """Log warning since WeChat doesn't support native multiple-choice question UI."""
        logger.warning("WeChat chatbot does not support multiple-choice question UI components.")

    async def on_message_delivered(self, message: Message) -> None:
        """Lifecycle hook: stop the typing indicator spinner when the final text is delivered."""
        if message.role == MessageRole.ASSISTANT and message.type == MessageType.TEXT:
            typing_ticket = self._typing_cache.get(message.channel_id)
            if typing_ticket:
                try:
                    await self.send_client.send_typing(
                        to=message.channel_id,
                        typing_status=TYPING_STOP,
                        ticket=typing_ticket,
                    )
                except Exception as e:
                    logger.debug("WeChat: typing stop failed: %s", e)

    async def _send_file(
        self,
        chat_id: str,
        path: str,
        context_token: str | None,
        playtime_sec: int | None = None,
    ) -> str:
        plaintext = await async_read_bytes(path)
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"

        # Compress outbound images if they are too large to prevent CDN upload 500 errors
        if mime.startswith("image/") and len(plaintext) > 1024 * 1024:
            logger.info(
                "WeChat: Outbound image %s is large (%d bytes), compressing...",
                Path(path).name,
                len(plaintext),
            )
            plaintext = _compress_image(plaintext)
            mime = "image/jpeg"
            logger.info("WeChat: Compressed image size: %d bytes", len(plaintext))

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

        logger.info("WeChat: Initiating send_file for path=%s, mime=%s, size=%d", path, mime, rawsize)

        encrypted_query_param = await self.media_manager.encrypt_and_upload(
            plaintext=plaintext,
            to_user_id=chat_id,
            media_type=media_type,
            filekey=filekey,
            aes_key=aes_key,
        )

        aes_key_for_api = base64.b64encode(aes_key.hex().encode("ascii")).decode("ascii")

        media_item = {
            "media": {
                "encrypt_query_param": encrypted_query_param,
                "aes_key": aes_key_for_api,
                "encrypt_type": 1,
            }
        }

        padded_size = _aes_padded_size(rawsize)

        if media_type == MEDIA_IMAGE:
            media_payload = {
                "type": ITEM_IMAGE,
                "image_item": {
                    "media": media_item["media"],
                    "mid_size": padded_size,
                },
            }
        elif media_type == MEDIA_VIDEO:
            rawfilemd5 = hashlib.md5(plaintext).hexdigest()
            media_payload = {
                "type": ITEM_VIDEO,
                "video_item": {
                    "media": media_item["media"],
                    "video_size": padded_size,
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
                    **({"playtime": playtime_sec} if playtime_sec is not None else {}),
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
        await self.send_client.send_raw_message({
            "from_user_id": "",
            "to_user_id": chat_id,
            "client_id": last_message_id,
            "message_type": MSG_TYPE_BOT,
            "message_state": MSG_STATE_FINISH,
            "item_list": [media_payload],
            **({"context_token": context_token} if context_token else {}),
        })
        return last_message_id
