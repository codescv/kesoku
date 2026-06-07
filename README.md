# Kesoku ⚡️

Kesoku (name inspired by `結束/Kessoku`) is a minimal, readable, and modular autonomous AI agent designed for seamless integration with chat platforms (like Discord and one-shot CLI chat) powered by SQLite message persistence, session management, and extensible tool calling.

---

## 📖 Documentation

For detailed guides on advanced configuration, platform bots (Discord, WeChat, Google Chat), cron jobs, and internal architecture, visit the official documentation:
👉 **[https://codescv.github.io/kesoku/](https://codescv.github.io/kesoku/)**

---

## 📦 Quick Start Installation

Install Kesoku globally as a CLI tool using `uv`:

```bash
uv tool install git+https://github.com/codescv/kesoku.git
```

To initialize a workspace and configuration template:
```bash
kesoku init -c private/config.toml
```

---

## 💬 CLI Chat Usage

Start an interactive CLI session with your agent using the `chat` subcommand:

```bash
# Start a new chat session
kesoku chat -c private/config.toml "What is 25 + 15?"

# Resume the latest active chat session
kesoku chat -c private/config.toml -z "And multiply that by 2."

# List all persistent chat sessions
kesoku chat -c private/config.toml -l
```
