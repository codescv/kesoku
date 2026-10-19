from pathlib import Path

from netdiag import cli
from netdiag.stats import STAT_KEYS, compute_stats, format_stats, summarise

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "capture_01.dump"

EXPECTED = {
    "packets": 600,
    "delivered": 589,
    "dropped": 11,
    "retransmits": 3,
    "bytes": 453349,
    "loss_pct": 1.83,
    "mean_len": 769.7,
}


def test_compute_stats_fixture():
    """Statistics of the shipped capture."""
    assert compute_stats(FIXTURE) == EXPECTED


def test_summarise_empty():
    """An empty dump has zero counts and no division by zero."""
    stats = summarise([])
    assert stats["packets"] == 0 and stats["loss_pct"] == 0.0 and stats["mean_len"] == 0.0
    assert tuple(stats) == STAT_KEYS


def test_format_stats_lines():
    """Text rendering is one ``key: value`` line per statistic, in STAT_KEYS order."""
    text = format_stats(EXPECTED)
    assert text.splitlines()[0] == "packets: 600"
    assert [line.split(":")[0] for line in text.splitlines()] == list(STAT_KEYS)


def test_cli_stats_text(capsys):
    """``netdiag.cli stats <dump>`` prints the text summary."""
    assert cli.main(["stats", str(FIXTURE)]) == 0
    out = capsys.readouterr().out
    assert out.strip() == format_stats(EXPECTED)


def test_cli_stats_missing_file(capsys, tmp_path):
    """A missing dump file is reported with exit code 1."""
    assert cli.main(["stats", str(tmp_path / "nope.dump")]) == 1
    assert "error:" in capsys.readouterr().err
