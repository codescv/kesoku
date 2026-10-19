"""Packet dump line format shared by the diagnostic and the offline tools.

One packet per line::

    0001  812B ok
    0002 1480B drop
    0003  640B rtx

i.e. ``<seq:04d> <length:4d>B <status>`` with status ``ok`` (delivered),
``drop`` (lost) or ``rtx`` (delivered after a retransmit). Lines starting with
``#`` and blank lines are comments.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

STATUSES = ("ok", "drop", "rtx")
_LINE_RE = re.compile(r"^(?P<seq>\d{4,})\s+(?P<length>\d+)B\s+(?P<status>ok|drop|rtx)$")


class DumpFormatError(ValueError):
    """Raised for a line that is not a valid packet record."""


@dataclass(frozen=True)
class Packet:
    """One packet record.

    Attributes:
        seq: 1-based sequence number.
        length: Payload length in bytes.
        status: ``ok``, ``drop`` or ``rtx``.
    """

    seq: int
    length: int
    status: str

    def format(self) -> str:
        """Render the packet as a dump line (without newline)."""
        return f"{self.seq:04d} {self.length:4d}B {self.status}"


def parse_line(line: str, lineno: int = 0) -> Packet | None:
    """Parse one dump line.

    Args:
        line: The raw line.
        lineno: Line number used in error messages.

    Returns:
        The packet, or None for comments and blank lines.

    Raises:
        DumpFormatError: If the line is malformed.
    """
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    m = _LINE_RE.match(text)
    if m is None:
        raise DumpFormatError(f"line {lineno}: not a packet record: {text!r}")
    return Packet(seq=int(m["seq"]), length=int(m["length"]), status=m["status"])


def parse_lines(lines: Iterable[str]) -> Iterator[Packet]:
    """Yield the packets of an iterable of dump lines."""
    for lineno, line in enumerate(lines, start=1):
        packet = parse_line(line, lineno)
        if packet is not None:
            yield packet


def read_dump(path: str | Path) -> list[Packet]:
    """Read every packet of a dump file."""
    with open(path, encoding="utf-8") as fh:
        return list(parse_lines(fh))


def synth_packets(count: int, *, drops: int, retransmits: int, seed: int) -> list[Packet]:
    """Generate a deterministic packet sequence.

    Args:
        count: Number of packets.
        drops: Exact number of ``drop`` packets.
        retransmits: Exact number of ``rtx`` packets.
        seed: Seed of the private random generator.

    Returns:
        The packets, ``seq`` 1..count.
    """
    rng = random.Random(seed)
    special = rng.sample(range(1, count + 1), drops + retransmits)
    dropped = set(special[:drops])
    rtx = set(special[drops:])
    packets = []
    for seq in range(1, count + 1):
        status = "drop" if seq in dropped else "rtx" if seq in rtx else "ok"
        packets.append(Packet(seq=seq, length=rng.randrange(64, 1500), status=status))
    return packets
