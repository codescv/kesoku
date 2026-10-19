"""Visible unit tests for Subsystem Y (webhook HMAC verifier)."""

from __future__ import annotations

from svc.subsystem_y.webhook import WebhookVerifier, compute_signature, set_clock, verify


def test_valid_webhook_verification() -> None:
    """Valid signature well inside tolerance window returns True."""
    set_clock(lambda: 1_700_000_000.0)
    body = b'{"event":"invoice.paid","id":"evt_01"}'
    ts = 1_700_000_000 - 45
    sig = compute_signature(body, ts)

    assert verify(sig, body, ts, tolerance=300) is True
    assert verify(sig.encode("utf-8"), body, ts, tolerance=300) is True


def test_tampered_body_and_expired_timestamp_rejected() -> None:
    """Tampered payloads or timestamps beyond tolerance return False."""
    set_clock(lambda: 1_700_000_000.0)
    body = b'{"event":"user.created"}'
    ts = 1_700_000_000 - 10
    sig = compute_signature(body, ts)

    assert verify(sig, b'{"event":"user.deleted"}', ts, tolerance=300) is False
    assert verify(sig, body, ts - 600, tolerance=300) is False

    verifier = WebhookVerifier(secret=b"custom_secret", clock=lambda: 1_700_000_000.0)
    custom_sig = verifier.sign(body, ts)
    assert verifier.verify(custom_sig, body, ts) is True
    assert verifier.verify(custom_sig, body, ts, now=1_700_000_999.0) is False
