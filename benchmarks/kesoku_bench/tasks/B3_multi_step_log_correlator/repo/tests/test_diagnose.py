import io
import json
import random
import re

from netdiag import diagnose

HEADER_RE = re.compile(r"^NEGOTIATED: codec=(\S+) max_frame=(\d+) token=([0-9a-f]{6})( \(rotated\))?$")


def _run(state_dir, seed):
    """Run one diagnostic session into a string."""
    buf = io.StringIO()
    assert diagnose.run(state_dir, buf, rng=random.Random(seed)) == 0
    return buf.getvalue()


def test_report_layout(tmp_path):
    """Header, 1200 packet lines, two-line tail."""
    out = _run(tmp_path, 1)
    lines = out.splitlines()
    assert lines[0] == "netdiag v2.3 session diagnostic"
    assert HEADER_RE.match(lines[1])
    assert lines[2] == "--- packet dump (1200 packets) ---"
    assert len(out.split("--- packet dump")[0]) < 150
    assert len(lines) == 3 + 1200 + 2
    assert lines[-2] == "dropped=37 retransmits=4"
    assert lines[-1] == "diagnose complete (1200 packets)"


def test_first_run_state_is_persisted(tmp_path):
    """The first session records its parameters and the run counter."""
    out = _run(tmp_path, 2)
    codec, max_frame, token, rotated = HEADER_RE.match(out.splitlines()[1]).groups()
    assert rotated is None
    saved = json.loads((tmp_path / diagnose.FIRST_RUN_FILE).read_text())
    assert saved == {"codec": codec, "max_frame": int(max_frame), "token": token}
    assert (tmp_path / diagnose.RUNS_FILE).read_text().strip() == "1"


def test_later_runs_rotate(tmp_path):
    """Second and later sessions get different parameters, marked (rotated)."""
    first = HEADER_RE.match(_run(tmp_path, 3).splitlines()[1]).groups()
    saved = (tmp_path / diagnose.FIRST_RUN_FILE).read_text()
    for seed in (4, 5):
        later = HEADER_RE.match(_run(tmp_path, seed).splitlines()[1]).groups()
        assert later[3] == " (rotated)"
        assert all(a != b for a, b in zip(first[:3], later[:3], strict=True))
    assert (tmp_path / diagnose.FIRST_RUN_FILE).read_text() == saved
    assert (tmp_path / diagnose.RUNS_FILE).read_text().strip() == "3"


def test_packet_dump_is_deterministic(tmp_path):
    """The probe dump does not depend on the negotiation."""
    a = _run(tmp_path / "a", 10).splitlines()[3:]
    b = _run(tmp_path / "b", 11).splitlines()[3:]
    assert a == b
