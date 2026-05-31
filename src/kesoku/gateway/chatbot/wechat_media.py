"""Media manager for WeChat integration, handling encryption, decryption, and CDN transfer."""

import hashlib
import logging
from urllib.parse import urlparse

from kesoku.gateway.chatbot.wechat_client import (
    WEIXIN_CDN_BASE_URL,
    ILinkClient,
    cdn_download_url,
    cdn_upload_url,
)
from kesoku.utils.crypto import (
    aes128_ecb_decrypt as _aes128_ecb_decrypt,
)
from kesoku.utils.crypto import (
    aes128_ecb_encrypt as _aes128_ecb_encrypt,
)
from kesoku.utils.crypto import (
    aes_padded_size as _aes_padded_size,
)
from kesoku.utils.crypto import (
    parse_aes_key as _parse_aes_key,
)

logger = logging.getLogger(__name__)

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
        raise ValueError(f"Media URL has disallowed scheme {scheme!r}; only http/https are permitted.")
    if host not in _WEIXIN_CDN_ALLOWLIST:
        raise ValueError(f"Media URL host {host!r} is not in the WeChat CDN allowlist. Refusing to fetch.")


class WeChatMediaManager:
    """Handles media download, decryption, encryption, and upload orchestration."""

    def __init__(self, client: ILinkClient, cdn_base_url: str = WEIXIN_CDN_BASE_URL) -> None:
        """Initialize WeChatMediaManager."""
        self.client = client
        self.cdn_base_url = cdn_base_url

    async def download_and_decrypt(
        self,
        *,
        encrypted_query_param: str | None,
        aes_key_b64: str | None,
        full_url: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> bytes:
        """Download media from CDN and decrypt it if an AES key is provided."""
        if full_url:
            _assert_weixin_cdn_url(full_url)
            url = full_url
        elif encrypted_query_param:
            url = cdn_download_url(self.cdn_base_url, encrypted_query_param)
        else:
            raise ValueError("Either 'full_url' or 'encrypted_query_param' must be provided.")

        ciphertext = await self.client.download_bytes(url, timeout_seconds=timeout_seconds)
        if not aes_key_b64:
            return ciphertext

        key = _parse_aes_key(aes_key_b64)
        return _aes128_ecb_decrypt(ciphertext, key)

    async def encrypt_and_upload(
        self,
        plaintext: bytes,
        to_user_id: str,
        media_type: int,
        filekey: str,
        aes_key: bytes,
    ) -> str:
        """Encrypt media and upload it to WeChat CDN, returning the encrypted query param."""
        rawsize = len(plaintext)
        rawfilemd5 = hashlib.md5(plaintext).hexdigest()

        upload_response = await self.client.get_upload_url(
            to_user_id=to_user_id,
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
            _assert_weixin_cdn_url(upload_full_url)
            upload_url = upload_full_url
        elif upload_param:
            upload_url = cdn_upload_url(self.cdn_base_url, upload_param, filekey)
        else:
            raise RuntimeError(f"WeChat: upload URL unavailable: {upload_response}")

        return await self.client.upload_ciphertext(ciphertext, upload_url)
