# Kesoku ⚡️

Kesoku (name inspired by `結束/Kessoku`) is a minimal, readable, and modular autonomous AI agent designed for seamless integration with chat platforms (like Discord and one-shot CLI chat) powered by SQLite message persistence, session management, and extensible tool calling.

## Features 🌟

- **Structured TOML Config**: Manage workspaces, LLMs (API key or Vertex AI), and chatbot tokens centrally in `config.toml`.
- **Gateway Architecture**: Decouples Chatbots from the Agent loop. Messages are buffered and routed reliably.
- **Session Management**: Maintain multiple persistent chat sessions in SQLite. List, resume, and view formatted chat histories instantly with `rich`.
- **Separate Execution Modes**: Run `kesoku chat` for one-shot session-based CLI interactions or `kesoku start` to run background daemons (like Discord).
- **Extensible Tooling & Skills**: Simple decorator-based or function registry system allowing the agent to execute tools.

## Installation 📦

Kesoku uses `uv` for lightning-fast dependency management and packaging.

```bash
git clone <repository_url> kesoku
cd kesoku
uv sync
```

## Configuration ⚙️

Initialize a workspace and `config.toml` in any directory (e.g. `private/`):
```bash
uv run kesoku -c private/config.toml init
```
This creates `private/config.toml`, `private/kesoku.db`, and `private/skills/`.

Sample `config.toml`:
```toml
[workspace]
db_path = "kesoku.db"
skills_dir = "skills"

[gemini]
model_name = "gemini-2.5-flash"
auth_mode = "api_key" # Use "vertex" for Google Cloud Vertex AI
api_key = "your_api_key"

[discord]
enabled = false
bot_token = "your_discord_bot_token"
```

## Usage 🚀

### Session-Based CLI Chat (`chat`)

Start a new chat session:
```bash
uv run kesoku -c private/config.toml chat "What is 25 + 15?"
```

List all current chat sessions:
```bash
uv run kesoku -c private/config.toml chat -l
```

Resume a specific chat session by its ID:
```bash
uv run kesoku -c private/config.toml chat -r abc12345 "And multiply that by 2."
```

Resume the latest active chat session:
```bash
uv run kesoku -c private/config.toml chat -z "And subtract 10."
```

Show full formatted chat history of a session:
```bash
uv run kesoku -c private/config.toml chat --show-history abc12345
```

### Daemon Mode (`start`)

Run background daemons (Discord bot):
```bash
uv run kesoku -c private/config.toml start
```
