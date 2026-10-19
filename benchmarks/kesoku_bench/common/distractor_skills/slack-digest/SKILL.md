---
name: slack-digest
description: "Summarise a day of Slack channel export JSON into a short digest of decisions and action items."
version: "1.1.0"
---

# slack-digest

Reads a Slack channel export (one JSON file per day) and produces a digest: decisions, open questions,
action items with owners.

## Usage

```bash
uv run python $SKILL_DIR/scripts/digest.py exports/eng-oncall/2024-05-02.json --max-items 10
```

- Messages from bots are ignored unless `--include-bots` is given.
- Threads are folded into their parent message.

The digest is Markdown; paste it into the weekly status doc.
