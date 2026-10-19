"""Gold post-overlay step for B3 (stdlib only).

Invoked by verify-gold as ``[sys.executable, <task>/gold/apply.py, <repo_dir>]``
with ``cwd=<repo_dir>`` after ``gold/`` has been overlaid onto the repo.

Simulates turns T1 and T3 of the reference solution:

1. T1: if the session diagnostic has never run (``.diag_first_run.json`` is
   missing), run it once, exactly as the agent would
   (``uv run python -m netdiag.diagnose``).
2. T3: rewrite the ``MAX_FRAME`` / ``CODEC`` / ``SESSION_TOKEN`` constants in
   ``netdiag/replay.py`` to the first session's negotiated values.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def run_first_diagnostic(repo: Path) -> None:
    """Run the session diagnostic once if it has never been run in ``repo``."""
    if (repo / ".diag_first_run.json").exists():
        return
    uv = shutil.which("uv")
    cmd = (
        [uv, "run", "--quiet", "python", "-m", "netdiag.diagnose"] if uv else [sys.executable, "-m", "netdiag.diagnose"]
    )
    proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=120, check=True)
    header = proc.stdout[:150]
    if "NEGOTIATED:" not in header or "(rotated)" in header:
        raise SystemExit(f"unexpected diagnostic header: {header!r}")


def set_constant(source: str, name: str, value: str) -> str:
    """Replace the module-level assignment ``NAME = ...`` in ``source``.

    Args:
        source: Python source text.
        name: Constant name.
        value: Replacement right-hand side (a Python literal).

    Returns:
        The updated source.
    """
    new, count = re.subn(rf"(?m)^{name}\s*=.*$", f"{name} = {value}", source)
    if count != 1:
        raise SystemExit(f"expected exactly one assignment to {name}, found {count}")
    return new


def main(argv: list[str]) -> int:
    """Entry point: ``apply.py <repo_dir>``."""
    repo = Path(argv[1]).resolve() if len(argv) > 1 else Path.cwd()
    run_first_diagnostic(repo)
    params = json.loads((repo / ".diag_first_run.json").read_text(encoding="utf-8"))
    replay = repo / "netdiag" / "replay.py"
    source = replay.read_text(encoding="utf-8")
    source = set_constant(source, "MAX_FRAME", str(int(params["max_frame"])))
    source = set_constant(source, "CODEC", json.dumps(params["codec"]))
    source = set_constant(source, "SESSION_TOKEN", json.dumps(params["token"]))
    replay.write_text(source, encoding="utf-8")
    print(f"replay.py <- codec={params['codec']} max_frame={params['max_frame']} token={params['token']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
