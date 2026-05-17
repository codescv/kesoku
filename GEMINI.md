# Technical Requirements
- Use `Python` as the main language. 
- Use `uv` to manage dependencies. Always use `uv run`, `uv add`, `uv pytest` etc. Never run `python`, `pip` or `pytest` directly.
- Unless there is a strong reason, add import statements at the top of your module.
- Use type annotations for function arguments and return types.
- Add docstrings for all functions more than 5 lines of code.
- Add inline comments for complex logic.
- Add extensive unit tests for every module you write.
- Use `docs/DESIGN.md` as your design doc and update it as your project evolves.
- Use `README.md` as the introduction to the end user. It should include:
  - The introduction and main features
  - Basic installation and configuration

# Workflow
- Use `private/` as the workspace for integration testing. You need to call the model to test your changes. You are free to reinitialize / delete the db file in this directory.
- When you plan to make changes, plan ahead how you will automatically test it.
- After you make the changes, test it, both unit test and CLI command with real model.
- Summarize your test result and send it to the user for every change.