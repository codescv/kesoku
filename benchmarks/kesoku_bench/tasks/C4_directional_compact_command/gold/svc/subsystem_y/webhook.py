"""Webhook HMAC-SHA256 signature calculation and verification (fixed)."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable

DEFAULT_SECRET: bytes = b"whsec_live_9f8e7d6c5b4a3210"


def _default_clock() -> float:
    """Return the default deterministic clock reading."""
    return 1_700_000_000.0


_CLOCK_FN: Callable[[], float] = _default_clock


def set_clock(clock_fn: Callable[[], float]) -> None:
    """Override the module-level clock used by ``verify()`` when ``now`` is omitted."""
    global _CLOCK_FN
    _CLOCK_FN = clock_fn


def get_now() -> float:
    """Return the current timestamp from the module-level clock."""
    return float(_CLOCK_FN())


def _normalize_secret(secret: bytes | str) -> bytes:
    """Convert secret to UTF-8 bytes."""
    if isinstance(secret, bytes):
        return secret
    return secret.encode("utf-8")


def _normalize_body(body: bytes | str) -> bytes:
    """Convert request body to UTF-8 bytes."""
    if isinstance(body, bytes):
        return body
    if isinstance(body, str):
        return body.encode("utf-8")
    raise TypeError("body must be bytes or str")


def compute_signature(
    body: bytes | str,
    ts: int | float,
    *,
    secret: bytes | str = DEFAULT_SECRET,
    prefix: str = "sha256=",
) -> str:
    """Compute HMAC-SHA256 webhook signature over ``f'{int(ts)}.' + body_bytes``."""
    key_bytes = _normalize_secret(secret)
    body_bytes = _normalize_body(body)
    signed_payload = f"{int(ts)}.".encode("ascii") + body_bytes
    digest = hmac.new(key_bytes, signed_payload, hashlib.sha256).hexdigest()
    return f"{prefix}{digest}" if prefix else digest


def verify(
    sig: str | bytes,
    body: bytes | str,
    ts: int | float,
    *,
    tolerance: int = 300,
    secret: bytes | str = DEFAULT_SECRET,
    now: int | float | None = None,
) -> bool:
    """Verify an incoming webhook HMAC-SHA256 signature and timestamp window.

    Args:
        sig: Signature header (``str`` or ``bytes``), optionally prefixed with
            ``"sha256="`` or ``"v1="``.
        body: Raw request payload (``bytes`` or ``str``; empty payload allowed).
        ts: Webhook unix timestamp in seconds.
        tolerance: Maximum allowed clock drift in seconds, inclusive (default 300).
        secret: Shared HMAC secret (defaults to ``DEFAULT_SECRET``).
        now: Optional current unix timestamp override (defaults to ``get_now()``).

    Returns:
        ``True`` if the signature is valid and within ``[0, tolerance]`` inclusive,
        ``False`` otherwise.
    """
    if tolerance < 0 or body is None or not isinstance(body, (bytes, str)):
        return False

    if isinstance(sig, bytes):
        try:
            sig_str = sig.decode("utf-8")
        except UnicodeDecodeError:
            return False
    elif isinstance(sig, str):
        sig_str = sig
    else:
        return False

    sig_str = sig_str.strip()
    if not sig_str or not sig_str.isascii():
        return False

    for scheme in ("sha256=", "v1="):
        if sig_str.startswith(scheme):
            sig_str = sig_str[len(scheme) :]
            break

    if not sig_str:
        return False

    current_ts = float(get_now() if now is None else now)
    drift = abs(current_ts - float(ts))
    if drift > tolerance:
        return False

    expected = compute_signature(body, ts, secret=secret, prefix="")
    try:
        return hmac.compare_digest(sig_str.lower(), expected.lower())
    except TypeError:
        return False


class WebhookVerifier:
    """Stateful wrapper around ``verify`` and ``compute_signature``."""

    def __init__(
        self,
        secret: bytes | str = DEFAULT_SECRET,
        clock: Callable[[], float] | None = None,
        default_tolerance: int = 300,
    ) -> None:
        """Initialize the verifier with a shared secret and clock."""
        self.secret = secret
        self.clock = clock
        self.default_tolerance = default_tolerance

    def sign(self, body: bytes | str, ts: int | float) -> str:
        """Compute a ``sha256=<hex>`` signature for ``(body, ts)``."""
        return compute_signature(body, ts, secret=self.secret)

    def verify(
        self,
        sig: str | bytes,
        body: bytes | str,
        ts: int | float,
        *,
        tolerance: int | None = None,
        now: int | float | None = None,
    ) -> bool:
        """Verify ``sig`` against ``(body, ts)`` using this verifier's secret and clock."""
        tol = self.default_tolerance if tolerance is None else tolerance
        effective_now = (
            now
            if now is not None
            else (self.clock() if self.clock is not None else None)
        )
        return verify(
            sig,
            body,
            ts,
            tolerance=tol,
            secret=self.secret,
            now=effective_now,
        )
