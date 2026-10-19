from pathlib import Path

import pytest
from netdiag import cli, replay

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "capture_01.dump"


def test_hello_uses_module_constants():
    """HELLO carries the configured session parameters."""
    msg = replay.hello()
    assert msg == {
        "proto": "netdiag/2.3",
        "codec": replay.CODEC,
        "max_frame": replay.MAX_FRAME,
        "token": replay.SESSION_TOKEN,
    }
    line = replay.hello_line()
    assert line.startswith("HELLO netdiag/2.3 ")
    assert f"token={replay.SESSION_TOKEN}" in line


def test_constants_have_expected_types():
    """Session parameters are an int frame size, a codec name and a 6-hex-digit token."""
    assert isinstance(replay.MAX_FRAME, int) and replay.MAX_FRAME > 0
    assert isinstance(replay.CODEC, str) and replay.CODEC
    assert isinstance(replay.SESSION_TOKEN, str) and len(replay.SESSION_TOKEN) == 6
    int(replay.SESSION_TOKEN, 16)


def test_frames_respect_max_frame():
    """Payloads are split at MAX_FRAME (or an explicit size)."""
    size = replay.MAX_FRAME
    chunks = replay.frames(b"x" * (2 * size + 5))
    assert [len(c) for c in chunks] == [size, size, 5]
    assert replay.frames(b"", 10) == [b""]
    assert [len(c) for c in replay.frames(b"abcdefg", 3)] == [3, 3, 1]
    with pytest.raises(ValueError):
        replay.frames(b"abc", 0)


def test_plan_skips_dropped_packets():
    """A replay re-sends only the packets that were delivered."""
    p = replay.plan(FIXTURE)
    assert p.packets == 589
    assert p.bytes == 453349
    assert p.frames >= p.packets


def test_cli_replay(capsys):
    """``netdiag.cli replay <dump>`` prints the HELLO line and the plan."""
    assert cli.main(["replay", str(FIXTURE)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == replay.hello_line()
    assert out[1].startswith("packets=589 ")
