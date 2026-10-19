---
name: changelog-gen
description: "Generate a Keep-a-Changelog section from Conventional Commit messages since the last tag."
version: "1.0.0"
---

# changelog-gen

Builds the next `CHANGELOG.md` section from commit subjects that follow Conventional Commits
(`feat:`, `fix:`, `perf:`, `docs:` ...), grouped into Added / Fixed / Changed.

## Usage

```bash
uv run python $SKILL_DIR/scripts/changelog.py --since v2.3.0 --version 2.4.0 >> CHANGELOG.md
```

- Commits with `!` or a `BREAKING CHANGE:` footer go to a "Breaking" group at the top.
- `--dry-run` prints without touching any file.

Edit the generated text by hand before releasing.
