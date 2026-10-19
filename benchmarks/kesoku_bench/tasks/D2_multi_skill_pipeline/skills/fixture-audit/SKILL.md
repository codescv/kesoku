---
name: fixture-audit
description: Audit CSV data fixtures against declarative rules in rules.yaml and quarantine violating rows into data/quarantine/<file>.csv.
version: "1.1.0"
required_permissions:
  - run_shell_command
metadata:
  tags:
    - audit
    - csv
    - quarantine
    - datapipe
    - rules
  related_skills:
    - schema-gen
---

# Fixture-Audit (CSV Rule Auditor & Quarantine Tool)

This skill audits CSV input tables in `data/` against the declarative constraints in `rules.yaml` and moves violating rows to `data/quarantine/`.

## Team Conventions

1. **Quarantine, Never Delete or Edit In-Place**: Rows that violate `rules.yaml` must **never** be deleted or modified in place. They must be moved verbatim (preserving the CSV header and original cell values) into `data/quarantine/<file>.csv`, while keeping all valid rows unchanged in `data/<file>.csv`.
2. **Row Conservation**: The sum of rows in `data/<table>.csv` and `data/quarantine/<table>.csv` must equal `expected_total_rows` declared in `rules.yaml`.

## CLI Usage

Run from the repository root (`repo/`):

```bash
uv run python $SKILL_DIR/scripts/audit.py --rules rules.yaml
```

Optional flags:
- `--check-only`: Report violations without modifying files (exits non-zero if violations exist in `data/`).
