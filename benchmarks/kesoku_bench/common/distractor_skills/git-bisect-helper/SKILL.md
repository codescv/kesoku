---
name: git-bisect-helper
description: "Automate `git bisect` with a test command to find the commit that introduced a regression."
version: "1.0.4"
---

# git-bisect-helper

Drives `git bisect run` for you: give it a known-good ref, a known-bad ref and a command that exits 0 on
good commits.

## Usage

```bash
uv run python $SKILL_DIR/scripts/bisect_run.py --good v1.8.0 --bad HEAD -- uv run pytest tests/test_api.py -x
```

- Exit code 125 from the command marks a commit as untestable (skipped).
- The helper always runs `git bisect reset` at the end, even on Ctrl-C.

Prints the first bad commit and its diffstat.
