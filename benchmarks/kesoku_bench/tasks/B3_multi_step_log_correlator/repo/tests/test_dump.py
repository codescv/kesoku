from pathlib import Path

import pytest
from netdiag.dump import DumpFormatError, Packet, parse_line, parse_lines, read_dump, synth_packets

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "capture_01.dump"


def test_format_and_parse_round_trip():
    """A formatted packet parses back to itself."""
    packet = Packet(seq=7, length=812, status="rtx")
    assert packet.format() == "0007  812B rtx"
    assert parse_line(packet.format()) == packet


def test_comments_and_blank_lines_are_ignored():
    """Comment and blank lines yield no packets."""
    lines = ["# header", "", "0001   64B ok", "   ", "0002 1499B drop"]
    assert [p.seq for p in parse_lines(lines)] == [1, 2]


def test_malformed_line_raises_with_line_number():
    """Malformed records report their line number."""
    with pytest.raises(DumpFormatError, match="line 2"):
        list(parse_lines(["0001 64B ok", "0002 64 bytes ok"]))


def test_synth_packets_exact_counts():
    """The synthetic generator honours the exact drop/rtx counts and is deterministic."""
    a = synth_packets(200, drops=9, retransmits=2, seed=5)
    b = synth_packets(200, drops=9, retransmits=2, seed=5)
    assert a == b
    assert sum(p.status == "drop" for p in a) == 9
    assert sum(p.status == "rtx" for p in a) == 2
    assert [p.seq for p in a] == list(range(1, 201))


def test_read_fixture():
    """The shipped capture parses completely."""
    packets = read_dump(FIXTURE)
    assert len(packets) == 600
    assert packets[0].seq == 1 and packets[-1].seq == 600
