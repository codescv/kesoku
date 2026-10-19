"""Gold apply helper for C3_multi_file_migration_resume."""

from __future__ import annotations

import sys
from pathlib import Path

CHECKLIST_CONTENT = """# Migration Checklist

- [x] cfgkit/loader.py (migrated to ConfigV2)
- [x] cfgkit/merge.py (migrated to ConfigV2)
- [ ] cfgkit/env.py (skipped per plan change — keep OldConfig and legacy.py)
- [x] cfgkit/export.py (migrated to ConfigV2, preserving legacy key order)
"""


def main() -> None:
    """Materialise reference migration_checklist.md for verify-gold."""
    repo_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
    (repo_dir / ".migration_checklist.md").write_text(CHECKLIST_CONTENT, encoding="utf-8")
    sessions_gold = repo_dir.parent / "sessions" / "gold"
    try:
        sessions_gold.mkdir(parents=True, exist_ok=True)
        (sessions_gold / "migration_checklist.md").write_text(
            CHECKLIST_CONTENT, encoding="utf-8"
        )
    except OSError:
        pass


if __name__ == "__main__":
    main()
