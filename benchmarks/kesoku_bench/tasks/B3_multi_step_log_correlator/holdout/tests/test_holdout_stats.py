"""Hidden test for B3 T2: ``netdiag.cli stats <dumpfile> --json``."""

import json
import subprocess
import sys
from pathlib import Path

from netdiag.stats import compute_stats

REPO = Path(__file__).resolve().parents[1]
FIXTURE = "fixtures/capture_01.dump"


def _stats_json(*args: str) -> dict:
    """Run the CLI in a subprocess and parse stdout as a JSON object."""
    proc = subprocess.run(
        [sys.executable, "-m", "netdiag.cli", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert isinstance(data, dict), data
    return data


def test_stats_json():
    """--json prints the same statistics as the text output, as one JSON object."""
    expected = compute_stats(REPO / FIXTURE)
    for args in (("stats", FIXTURE, "--json"), ("stats", "--json", FIXTURE)):
        data = _stats_json(*args)
        for key in ("packets", "delivered", "dropped", "retransmits", "bytes"):
            assert data[key] == expected[key], (key, data)
        for key in ("loss_pct", "mean_len"):
            assert abs(float(data[key]) - float(expected[key])) < 0.051, (key, data)
    text = subprocess.run(
        [sys.executable, "-m", "netdiag.cli", "stats", FIXTURE],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout
    assert text.splitlines()[0] == "packets: 600"
