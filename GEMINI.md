# Code Style
- Put import statements at the top of the module. Avoid putting them at the function level unless there is a strong reason.
- Use type annotations for function arguments and return types.
- Add docstrings for all functions more than 5 lines of code.
- Add inline comments for complex logic.

# Documentation
- Use `docs/DESIGN.md` as your design doc and update it as your project evolves.
- Use `README.md` as the introduction to the end user. It should include:
  - The introduction and main features
  - Basic installation and configuration

# Workflow
- Use `uv` to manage dependencies. Always use `uv run`, `uv add`, `uv pytest` etc. Never run `python`, `pip` or `pytest` directly.

## Unit Tests
- Add extensive unit tests for every module you write. Always keep unit tests up to date and run them after any changes.

## Code Style
- Run `uv run ruff check` after your changes and make sure there are no lint errors and fix them if there are any.

## Keep documentation up to date
- After you make major changes, make sure the documentation under `docs/` are still relevant.