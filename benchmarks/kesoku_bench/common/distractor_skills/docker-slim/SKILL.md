---
name: docker-slim
description: "Shrink a Docker image: multi-stage rewrite suggestions, layer size report and .dockerignore hints."
version: "1.5.0"
---

# docker-slim

Analyses a `Dockerfile` and reports how to make the image smaller: multi-stage builds, combining `RUN`
layers, cleaning package caches, a better base image, and missing `.dockerignore` entries.

## Usage

```bash
uv run python $SKILL_DIR/scripts/slim.py Dockerfile --report
```

- `--rewrite out.Dockerfile` writes a suggested multi-stage version (review before use).
- Needs no Docker daemon; it only reads files.

Typical saving is 40-70% for Python images built on full Debian bases.
