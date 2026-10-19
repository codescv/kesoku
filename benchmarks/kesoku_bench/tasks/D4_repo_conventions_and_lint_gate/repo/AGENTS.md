# Repository Engineering Conventions (`AGENTS.md`)

All contributors and coding agents working in this repository **must** follow these four mandatory rules on every change:

1. **Google-Style Docstrings (`Args:` / `Returns:`)**:
   Every public function (name not starting with `_`) in any library module you touch under `textutils/` must have a Google-style docstring that explicitly includes both `Args:` and `Returns:` sections.
2. **No `print()` in Library Code**:
   Never use `print()` anywhere inside `textutils/`. Use the standard library `logging` module (`logger = logging.getLogger(__name__)`) instead.
3. **Issue Regression Test File (`tests/regression/test_<issue>.py`)**:
   Every bug fix for an issue `issues/<issue>.md` must add a dedicated regression test module at `tests/regression/test_<issue>.py` (for example, fixing `issues/142.md` requires creating `tests/regression/test_142.py` with `test_*` functions covering the issue).
4. **Mandatory Conventions Lint Gate**:
   Before finishing your task, you **must** run:
   ```bash
   uv run python scripts/lint_conventions.py
   ```
   and ensure it exits with code `0` (in addition to `uv run pytest`). Note that `scripts/lint_conventions.py` is a standalone check and is not invoked by `pytest`.
