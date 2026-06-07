# Installation Guide

This guide will walk you through setting up Kesoku on your local machine or server.

---

## 📋 Prerequisites

Before installing Kesoku, ensure your environment meets the following requirements:

*   **Python**: Version 3.12 or newer.
*   **uv**: A fast Python package installer and resolver.
    *   If you don't have `uv` installed, follow the [uv installation guide](https://github.com/astral-sh/uv#installation). Typically, you can install it via curl:
        ```bash
        curl -LsSf https://astral-sh.uv.run/install.sh | sh
        ```
*   **Git**: Required for cloning the repository.

---

## 📦 Setting Up Kesoku

Follow these steps to download and set up Kesoku:

### Step 1: Clone the Repository
Clone the repository to your target workspace:

```bash
git clone https://github.com/codescv/kesoku.git
cd kesoku
```

### Step 2: Synchronize Dependencies
Kesoku uses `uv` to lock and manage virtual environment packages. To create the virtual environment and install all necessary dependencies (including the standard runtime ones), run:

```bash
uv sync
```
This command creates a local virtual environment under `.venv/` and synchronizes all packages defined in `uv.lock`.

### Step 3: Verify the CLI Installation
Verify that the `kesoku` command-line executable works inside the `uv` environment:

```bash
uv run kesoku --help
```
You should see output displaying the CLI subcommands (`chat`, `start`, `service`, `init`, `wechat`, etc.).

---

## 🛠️ Developer Installation (Optional)

If you plan to contribute to Kesoku, run tests, or build this documentation site, make sure to install development-group dependencies:

```bash
# Sync all dependencies including ruff, pytest, and mkdocs
uv sync --all-groups
```
