import json
from pathlib import Path

from netdiag import cli
from netdiag.stats import compute_stats

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "capture_01.dump"


def test_cli_stats_json(capsys):
    """``stats --json`` prints compute_stats() as a JSON object."""
    assert cli.main(["stats", str(FIXTURE), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == compute_stats(FIXTURE)
