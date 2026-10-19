"""Replay client: re-sends a captured session to the recorder.

The client must use the session parameters the recorder negotiated, otherwise
the recorder rejects the HELLO (codec / token mismatch) or truncates frames
(``MAX_FRAME`` mismatch).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .dump import read_dump

MAX_FRAME = 4096
CODEC = "zstd-v1"
SESSION_TOKEN = "000000"

PROTOCOL = "netdiag/2.3"


def hello() -> dict[str, object]:
    """Return the HELLO message sent when the replay session opens."""
    return {"proto": PROTOCOL, "codec": CODEC, "max_frame": MAX_FRAME, "token": SESSION_TOKEN}


def hello_line() -> str:
    """Return the HELLO message in its wire (text) form."""
    msg = hello()
    return f"HELLO {msg['proto']} codec={msg['codec']} max_frame={msg['max_frame']} token={msg['token']}"


def frames(payload: bytes, max_frame: int | None = None) -> list[bytes]:
    """Split ``payload`` into frames of at most ``max_frame`` bytes (default ``MAX_FRAME``)."""
    size = MAX_FRAME if max_frame is None else max_frame
    if size <= 0:
        raise ValueError("max_frame must be positive")
    return [payload[i : i + size] for i in range(0, len(payload), size)] or [b""]


@dataclass(frozen=True)
class ReplayPlan:
    """What a replay of a dump would send.

    Attributes:
        hello: The HELLO wire line.
        packets: Number of packets that will be re-sent (dropped ones are skipped).
        frames: Number of frames needed for all re-sent packets.
        bytes: Total payload bytes.
    """

    hello: str
    packets: int
    frames: int
    bytes: int


def plan(dump_path: str | Path) -> ReplayPlan:
    """Compute the replay plan for a saved dump."""
    sent = [p for p in read_dump(dump_path) if p.status != "drop"]
    n_frames = sum(len(frames(b"\0" * p.length)) for p in sent)
    return ReplayPlan(hello=hello_line(), packets=len(sent), frames=n_frames, bytes=sum(p.length for p in sent))
