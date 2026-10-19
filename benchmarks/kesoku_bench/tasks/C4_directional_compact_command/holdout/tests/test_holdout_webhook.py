"""Held-out F2P verification suite for C4_directional_compact_command."""

from __future__ import annotations

import ast
import hmac
from pathlib import Path
from unittest.mock import patch

from svc.subsystem_y.webhook import compute_signature, set_clock, verify

REPO_ROOT = Path(__file__).resolve().parents[1]
WEBHOOK_PATH = REPO_ROOT / "svc" / "subsystem_y" / "webhook.py"


def test_tolerance_boundary_inclusive() -> None:
    """verify() must accept timestamps on the exact boundary abs(now - ts) == tolerance."""
    base_now = 1_700_000_000
    set_clock(lambda: float(base_now))
    body = b'{"event":"ping"}'

    # Exact lower boundary: now - ts == 300
    ts_past_edge = base_now - 300
    sig_past = compute_signature(body, ts_past_edge)
    assert verify(sig_past, body, ts_past_edge, tolerance=300) is True

    # Exact upper boundary: ts - now == 300
    ts_future_edge = base_now + 300
    sig_future = compute_signature(body, ts_future_edge)
    assert verify(sig_future, body, ts_future_edge, tolerance=300) is True

    # Zero tolerance when ts == now
    sig_zero = compute_signature(body, base_now)
    assert verify(sig_zero, body, base_now, tolerance=0) is True

    # 1 second outside boundary must still be rejected
    ts_outside = base_now - 301
    sig_outside = compute_signature(body, ts_outside)
    assert verify(sig_outside, body, ts_outside, tolerance=300) is False


def test_empty_body_accepted_with_valid_signature() -> None:
    """Empty bytes (b'') and empty str ('') payloads must verify when signed."""
    base_now = 1_700_000_000
    set_clock(lambda: float(base_now))

    sig_empty_bytes = compute_signature(b"", base_now)
    assert verify(sig_empty_bytes, b"", base_now, tolerance=300) is True
    assert verify(sig_empty_bytes, "", base_now, tolerance=300) is True

    # Wrong signature on empty body must still return False
    assert verify("sha256=" + ("0" * 64), b"", base_now, tolerance=300) is False


def test_non_utf8_bytes_signature_returns_false() -> None:
    """Non-UTF8 bytes or non-ASCII signature headers must return False, never raise."""
    base_now = 1_700_000_000
    set_clock(lambda: float(base_now))
    body = b'{"event":"webhook"}'

    assert verify(b"\xff\xfe\xfd\x80invalid", body, base_now, tolerance=300) is False
    assert verify(b"sha256=\xff\xfe\x00\x81", body, base_now, tolerance=300) is False
    assert verify("sha256=こんにちは", body, base_now, tolerance=300) is False


def test_constant_time_compare_digest_still_used() -> None:
    """webhook.py must continue using hmac.compare_digest and pass boundary cases."""
    source = WEBHOOK_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(WEBHOOK_PATH))
    uses_compare_digest = any(
        isinstance(node, ast.Attribute) and node.attr == "compare_digest"
        for node in ast.walk(tree)
    )
    assert uses_compare_digest, "hmac.compare_digest is no longer referenced in webhook.py"

    base_now = 1_700_000_000
    set_clock(lambda: float(base_now))
    sig = compute_signature(b"", base_now)
    with patch.object(hmac, "compare_digest", wraps=hmac.compare_digest) as spy:
        assert verify(sig, b"", base_now, tolerance=0) is True
        assert spy.called, "hmac.compare_digest was not called during verify()"
