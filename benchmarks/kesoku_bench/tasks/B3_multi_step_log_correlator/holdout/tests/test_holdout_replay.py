"""Hidden test for B3 T3: replay constants must equal the FIRST diagnostic session's parameters."""

import importlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_replay_constants_match_first_run():
    """MAX_FRAME / CODEC / SESSION_TOKEN equal the values negotiated by the first diagnose run."""
    first_run = REPO / ".diag_first_run.json"
    assert first_run.exists(), "the session diagnostic was never run (.diag_first_run.json missing)"
    expected = json.loads(first_run.read_text(encoding="utf-8"))
    replay = importlib.reload(importlib.import_module("netdiag.replay"))
    assert int(replay.MAX_FRAME) == int(expected["max_frame"])
    assert str(replay.CODEC) == expected["codec"]
    assert str(replay.SESSION_TOKEN).lower() == expected["token"]
