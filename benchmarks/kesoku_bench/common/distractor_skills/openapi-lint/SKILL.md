---
name: openapi-lint
description: "Lint an OpenAPI 3 spec (YAML or JSON) for naming, missing descriptions and inconsistent error responses."
version: "0.8.5"
---

# openapi-lint

Checks an OpenAPI 3.x document against the API style guide: kebab-case paths, camelCase properties,
every operation has `operationId`, `summary` and a documented `4xx` response.

## Usage

```bash
uv run python $SKILL_DIR/scripts/oas_lint.py api/openapi.json --fail-on warning
```

- `--rules naming,errors` limits the rule set.
- Output format: `path:line rule message`, one finding per line.

Run it before opening an API review; reviewers expect a clean report.
