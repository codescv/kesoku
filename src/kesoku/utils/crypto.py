"""Cryptographic utility functions for Kesoku AI Agent framework, specifically AES ECB."""

import base64

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def pkcs7_pad(data: bytes, block_size: int = 16) -> bytes:
    """Apply PKCS7 padding to data.

    Args:
        data: The raw bytes to pad.
        block_size: The block size in bytes (default 16).

    Returns:
        The padded bytes.
    """
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len] * pad_len)


def aes128_ecb_encrypt(plaintext: bytes, key: bytes) -> bytes:
    """Encrypt plaintext using AES-128 in ECB mode with PKCS7 padding.

    Args:
        plaintext: Raw bytes to encrypt.
        key: The 16-byte AES key.

    Returns:
        The encrypted ciphertext bytes.
    """
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    encryptor = cipher.encryptor()
    return encryptor.update(pkcs7_pad(plaintext)) + encryptor.finalize()


def aes128_ecb_decrypt(ciphertext: bytes, key: bytes) -> bytes:
    """Decrypt ciphertext using AES-128 in ECB mode, stripping PKCS7 padding.

    Args:
        ciphertext: Padded encrypted bytes to decrypt.
        key: The 16-byte AES key.

    Returns:
        The decrypted plaintext bytes.
    """
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    decryptor = cipher.decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()
    if not padded:
        return padded
    pad_len = padded[-1]
    if 1 <= pad_len <= 16 and padded.endswith(bytes([pad_len]) * pad_len):
        return padded[:-pad_len]
    return padded


def aes_padded_size(size: int) -> int:
    """Calculate the padded size of data after PKCS7 padding for AES (block size 16)."""
    return ((size + 1 + 15) // 16) * 16


def parse_aes_key(aes_key_b64: str) -> bytes:
    """Decode a base64 encoded AES key, resolving hex strings if needed.

    Supports 16-byte raw base64 decoded keys or 32-character hex string keys
    encoded in base64.

    Args:
        aes_key_b64: Base64 encoded key string.

    Returns:
        The decoded 16-byte key in bytes.
    """
    decoded = base64.b64decode(aes_key_b64)
    if len(decoded) == 16:
        return decoded
    if len(decoded) == 32:
        text = decoded.decode("ascii", errors="ignore")
        if text and all(ch in "0123456789abcdefABCDEF" for ch in text):
            return bytes.fromhex(text)
    raise ValueError(f"unexpected aes_key format ({len(decoded)} decoded bytes)")
