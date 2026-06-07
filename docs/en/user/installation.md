# Installation Guide

This guide will walk you through setting up Kesoku on your local machine or server.

---

## 📋 Prerequisites

Before installing Kesoku, ensure your environment meets the following requirements:

*   **Python**: Version 3.12 or newer.
*   **uv**: A fast Python package manager and resolver.
    *   If you don't have `uv` installed, run:
        ```bash
        curl -LsSf https://astral-sh.uv.run/install.sh | sh
        ```
*   **Git**: Required if installing from the repository.

---

## 📦 Setting Up Kesoku (Recommended)

The easiest way to install and run Kesoku as a global command-line tool is using `uv tool install`. This automatically manages the virtual environment for you and places the `kesoku` executable in your system path.

### Option A: Install directly from GitHub
Run the following command to install the latest version from the main branch:

```bash
uv tool install git+https://github.com/codescv/kesoku.git
```

### Option B: Install from a local clone
If you have cloned the repository locally:

```bash
git clone https://github.com/codescv/kesoku.git
cd kesoku
uv tool install .
```

### Verify the Installation
Verify that the `kesoku` executable is available:

```bash
kesoku --help
```
You should see output displaying the CLI subcommands (`chat`, `start`, `service`, `init`, `wechat`, `memory`, etc.).

---

## 🛠️ Developer Installation

If you plan to contribute to Kesoku, run tests, or modify its codebase:

1. Clone the repository and install it as an **editable tool**, allowing any local code changes to reflect immediately:
   ```bash
   git clone https://github.com/codescv/kesoku.git
   cd kesoku
   uv tool install -e .
   ```
2. To install all development-group dependencies (for testing and docs compilation) inside a local project virtual environment:
   ```bash
   uv sync --all-groups
   ```
3. Run tests using `uv run pytest` or compile documentation with `uv run mkdocs build`.
