"""Unit tests for the cryptographic utility functions."""

import base64

import pytest

from kesoku.utils.crypto import (
    aes128_ecb_decrypt,
    aes128_ecb_encrypt,
    aes_padded_size,
    parse_aes_key,
    pkcs7_pad,
)


def test_pkcs7_pad() -> None:
    """Test PKCS7 padding for various input sizes."""
    assert pkcs7_pad(b"", block_size=16) == b"\x10" * 16
    assert pkcs7_pad(b"A", block_size=16) == b"A" + b"\x0f" * 15
    assert pkcs7_pad(b"123456789012345", block_size=16) == b"123456789012345\x01"
    assert pkcs7_pad(b"1234567890123456", block_size=16) == b"1234567890123456" + b"\x10" * 16


def test_aes128_ecb_encrypt_decrypt() -> None:
    """Test roundtrip AES-128 ECB encryption and decryption."""
    key = b"1234567890123456"  # 16 bytes key
    plaintext = b"Hello, Kesoku AI!"

    ciphertext = aes128_ecb_encrypt(plaintext, key)
    assert ciphertext != plaintext
    assert len(ciphertext) % 16 == 0

    decrypted = aes128_ecb_decrypt(ciphertext, key)
    assert decrypted == plaintext


def test_aes_padded_size() -> None:
    """Test calculation of padded sizes."""
    assert aes_padded_size(0) == 16
    assert aes_padded_size(1) == 16
    assert aes_padded_size(15) == 16
    assert aes_padded_size(16) == 32
    assert aes_padded_size(31) == 32


def test_parse_aes_key() -> None:
    """Test parsing of base64 encoded keys in different formats."""
    # Case 1: 16-byte raw key base64 encoded
    raw_key = b"1234567890123456"
    b64_raw = base64.b64encode(raw_key).decode("ascii")
    assert parse_aes_key(b64_raw) == raw_key

    # Case 2: 32-character hex key base64 encoded
    # "1234567890123456" in hex is "31323334353637383930313233343536" (32 chars)
    hex_key_str = b"31323334353637383930313233343536"
    b64_hex = base64.b64encode(hex_key_str).decode("ascii")
    assert parse_aes_key(b64_hex) == raw_key

    # Case 3: Invalid key sizes/formats
    with pytest.raises(ValueError):
        parse_aes_key(base64.b64encode(b"too_short").decode("ascii"))
    with pytest.raises(ValueError):
        parse_aes_key(base64.b64encode(b"12345678901234567").decode("ascii"))
