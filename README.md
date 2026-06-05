# Kesoku ⚡️

Kesoku (name inspired by `結束/Kessoku`) is a minimal, readable, and modular autonomous AI agent designed for seamless integration with chat platforms (like Discord and one-shot CLI chat) powered by SQLite message persistence, session management, and extensible tool calling.

## Features 🌟

- **Structured TOML Config**: Manage workspaces, LLMs (API key or Vertex AI), and chatbot tokens centrally in `config.toml`.
- **Gateway Architecture**: Decouples Chatbots from the Agent loop. Messages are buffered and routed reliably.
- **Session Management**: Maintain multiple persistent chat sessions in SQLite. List, resume, and view formatted chat histories instantly with `rich`.
- **Separate Execution Modes**: Run `kesoku chat` for one-shot session-based CLI interactions or `kesoku start` to run background daemons (supporting both Discord and Google Chat).
- **Extensible Tooling & Skills**: Simple decorator-based or function registry system allowing the agent to execute tools.

## Installation 📦

Kesoku uses `uv` for lightning-fast dependency management and packaging.

```bash
git clone <repository_url> kesoku
# Navigate to project
cd kesoku
# Synchronize dependencies
uv sync
```

## Configuration ⚙️

Initialize a workspace and `config.toml` in any directory (e.g. `private/`):
```bash
uv run kesoku init -c private/config.toml
```
This creates `private/config.toml`, `private/kesoku.db`, and `private/skills/`.

Sample `config.toml`:
```toml
[workspace]
db_path = "kesoku.db"
skills_dir = "skills"

[agent]
llm = "gemini"

[gemini]
model_name = "gemini-2.5-flash"
auth_mode = "api_key" # Use "vertex" for Google Cloud Vertex AI
api_key = "your_api_key"
thinking_level = "high"

[claude]
model_name = "claude-3-5-sonnet@20241022"
project_id = "your-gcp-project"
location = "us-east5"

[discord]
enabled = false
bot_token = "your_discord_bot_token" # Optional if DISCORD_TOKEN environment variable is set

# Channel-specific overrides
[[discord.channels]]
channels = ["1234567890", "announcements"]
llm = "claude"
auto_thread = false

[google_chat]
enabled = false
chatbot_id = "google_chat"
project_id = "your-gcp-project"
topic_id = "kesoku-chat-events"
subscription_id = "kesoku-chat-sub"
credentials_json = "" # Optional path to JSON key file. If empty, uses ADC.
impersonate_service_account = "" # Optional target service account email to impersonate (key-less)
# reaction_emoji = "👀" # Optional: Emoji to react with when receiving a user message

# For step-by-step instructions on setting up Google Cloud Platform (GCP) components,
# see: docs/GOOGLE_CHAT_SETUP.md
```

## Usage 🚀

### Session-Based CLI Chat (`chat`)

Start a new chat session:
```bash
uv run kesoku chat -c private/config.toml "What is 25 + 15?"
```

List all current chat sessions:
```bash
uv run kesoku chat -c private/config.toml -l
```

Resume a specific chat session by its ID:
```bash
uv run kesoku chat -c private/config.toml -r abc12345 "And multiply that by 2."
```

Resume the latest active chat session:
```bash
uv run kesoku chat -c private/config.toml -z "And subtract 10."
```

Show full formatted chat history of a session:
```bash
uv run kesoku chat -c private/config.toml --show-history abc12345
```

### Daemon Mode (`start`)

Run background daemons (Discord bot):
```bash
uv run kesoku start -c private/config.toml
```

### Running as a Background Service (Linux/macOS)

To run Kesoku as a persistent background service (using `systemd` on Linux or `launchd` on macOS), use the `service` command group:

#### 1. Install the Service
Generates and installs the service file (`.service` unit on Linux, `.plist` configuration on macOS). By default, the installation automatically inherits environment variables matching `PATH`, `HTTP_PROXY`, `HTTPS_PROXY`, `GOOGLE_API_KEY`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `GOOGLE_GENAI_USE_VERTEXAI`, and `DISCORD_TOKEN` from the current shell. You can override or add custom environment variables using the `-e` / `--env` options.

- **User-Level Service (Default, recommended - no root required)**:
  ```bash
  uv run kesoku service install -c private/config.toml -e GEMINI_API_KEY=your-api-key
  ```
- **System-Level Service (Global, requires root)**:
  ```bash
  sudo uv run kesoku service install --system -c private/config.toml -e GEMINI_API_KEY=your-api-key
  ```
- **Dry-Run (prints service file configuration to stdout)**:
  ```bash
  uv run kesoku service install --dry-run -c private/config.toml -e GEMINI_API_KEY=your-api-key
  ```

#### 2. Manage the Service
You can start, stop, restart, and uninstall the service using `kesoku service` subcommand wrappers:

- **Start the service**:
  ```bash
  uv run kesoku service start
  ```
- **Stop the service**:
  ```bash
  uv run kesoku service stop
  ```
- **Restart the service**:
  ```bash
  uv run kesoku service restart
  ```
- **Uninstall the service** (stops, disables, and deletes the configuration file):
  ```bash
  uv run kesoku service uninstall
  ```

*Note: Add the `--system` flag to any of the management subcommands if you installed the service in system-level mode (e.g. `uv run kesoku service start --system`).*

#### 3. Check Status and Logs
- **Check service status**:
  ```bash
  uv run kesoku service status
  ```
- **Follow real-time background logs**:
  ```bash
  uv run kesoku service logs -f
  ```

