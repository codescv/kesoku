"""Gold solution applicator for D1_custom_protocol_skill."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def apply(repo_dir: str | Path | None = None) -> None:
    """Copy or generate fixtures/session_17.fixed.pcb inside repo_dir.

    Args:
        repo_dir: Target repository directory. Defaults to current working directory.
    """
    target_repo = Path(repo_dir) if repo_dir is not None else Path.cwd()
    gold_dir = Path(__file__).resolve().parent
    gold_fixed = gold_dir / "fixtures" / "session_17.fixed.pcb"
    dest_fixed = target_repo / "fixtures" / "session_17.fixed.pcb"
    dest_fixed.parent.mkdir(parents=True, exist_ok=True)
    if gold_fixed.is_file():
        shutil.copy2(gold_fixed, dest_fixed)


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    apply(target)
