"""Session diagnostic: ``uv run python -m netdiag.diagnose``.

Connects to the recorder (simulated here), prints the negotiated session
parameters, a dump of the probe packets and a short loss summary.

Output layout::

    netdiag v2.3 session diagnostic
    NEGOTIATED: codec=<codec> max_frame=<bytes> token=<hex6>
    --- packet dump (1200 packets) ---
    0001  812B ok
    ...
    dropped=37 retransmits=4
    diagnose complete (1200 packets)

The recorder rotates the session parameters after the first diagnostic
session; later sessions are marked ``(rotated)``. Session bookkeeping lives
in git-ignored dot-files in the repository root.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import TextIO

from .dump import synth_packets

VERSION = "2.3"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_FILE = ".diag_runs"
FIRST_RUN_FILE = ".diag_first_run.json"

PACKETS = 1200
DROPS = 37
RETRANSMITS = 4
PROBE_SEED = 2309

CODECS = ("lz4-v2", "lz4-v3", "snappy-v1", "brotli-v2", "deflate-v4", "lzo-v1")
FRAME_SIZES = (1536, 2048, 3072, 6144, 8192, 12288)


def negotiate(rng: random.Random, avoid: dict[str, object] | None = None) -> dict[str, object]:
    """Simulate the recorder's parameter negotiation.

    Args:
        rng: Random source.
        avoid: Previously negotiated parameters; every field of the result differs from it.

    Returns:
        ``{"codec": str, "max_frame": int, "token": str}``.
    """
    while True:
        params: dict[str, object] = {
            "codec": rng.choice(CODECS),
            "max_frame": rng.choice(FRAME_SIZES),
            "token": f"{rng.getrandbits(24):06x}",
        }
        if params["token"] == "000000":
            continue
        if avoid is None or all(params[k] != avoid.get(k) for k in params):
            return params


def _bump_runs(state_dir: Path) -> int:
    """Increment and return the persisted diagnostic run counter."""
    path = state_dir / RUNS_FILE
    try:
        runs = int(path.read_text(encoding="utf-8").strip() or "0")
    except (FileNotFoundError, ValueError):
        runs = 0
    runs += 1
    path.write_text(f"{runs}\n", encoding="utf-8")
    return runs


def run(state_dir: Path, out: TextIO, rng: random.Random | None = None) -> int:
    """Run one diagnostic session.

    Args:
        state_dir: Directory holding ``.diag_runs`` / ``.diag_first_run.json``.
        out: Stream the report is written to.
        rng: Random source for the negotiation (default: system entropy).

    Returns:
        Process exit code (0).
    """
    rng = rng or random.SystemRandom()
    state_dir.mkdir(parents=True, exist_ok=True)
    _bump_runs(state_dir)
    first_path = state_dir / FIRST_RUN_FILE
    if first_path.exists():
        first = json.loads(first_path.read_text(encoding="utf-8"))
        params = negotiate(rng, avoid=first)
        marker = " (rotated)"
    else:
        params = negotiate(rng)
        first_path.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
        marker = ""

    out.write(f"netdiag v{VERSION} session diagnostic\n")
    out.write(f"NEGOTIATED: codec={params['codec']} max_frame={params['max_frame']} token={params['token']}{marker}\n")
    out.write(f"--- packet dump ({PACKETS} packets) ---\n")
    packets = synth_packets(PACKETS, drops=DROPS, retransmits=RETRANSMITS, seed=PROBE_SEED)
    for packet in packets:
        out.write(packet.format() + "\n")
    dropped = sum(p.status == "drop" for p in packets)
    rtx = sum(p.status == "rtx" for p in packets)
    out.write(f"dropped={dropped} retransmits={rtx}\n")
    out.write(f"diagnose complete ({len(packets)} packets)\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(prog="netdiag.diagnose", description="Run a session diagnostic.")
    parser.add_argument("--state-dir", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    return run(args.state_dir, sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
