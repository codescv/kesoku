# Configuration Guide

Kesoku is centrally configured via a structured TOML file (usually `config.toml`). At startup, this configuration is parsed into Pydantic models and managed as a global singleton.

---

## 🚀 Initialization

To generate a default configuration file, navigate to your workspace and run:

```bash
kesoku init -c config.toml
```

This creates a default configuration file named `config.toml` in your current directory.

---

## ⚙️ Config file Schema

Here is a comprehensive breakdown of all settings blocks in `config.toml`:

### 1. `[workspace]`
Manages paths for the SQLite database, logs, and custom capability folders.

*   **`db_path`** (string, default: `"kesoku.db"`): Relative or absolute path to the SQLite persistence file.
*   **`skills_dir`** (string, default: `"skills"`): Folder where custom skills (`SKILL.md`) are placed.
*   **`sessions_dir`** (string, default: `"sessions"`): Workspace directory where per-session raw logs, trajectories, and attachments are staged.

### 2. `[agent]`
Defines active agent features.

*   **`llm`** (string, default: `"gemini"`): The active LLM engine provider. Supported values are `"gemini"` and `"claude"`.

### 3. `[gemini]`
Configures Google GenAI/Gemini integrations.

*   **`model_name`** (string, default: `"gemini-2.5-flash"`): The model ID.
*   **`auth_mode`** (string, default: `"api_key"`): Authentication method. Use `"api_key"` for raw API Keys, or `"vertex"` to run via Google Cloud Vertex AI (which uses Application Default Credentials).
*   **`api_key`** (string): The API key (if `auth_mode = "api_key"`). If empty, falls back to the `GEMINI_API_KEY` environment variable.
*   **`project_id`** (string): Google Cloud Project ID (required for `"vertex"` mode).
*   **`location`** (string, default: `"us-central1"`): GCP region for Vertex AI endpoints.
*   **`thinking_level`** (string, default: `"high"`): The thinking/reasoning budget. Supported: `"minimal"`, `"low"`, `"medium"`, `"high"`.

### 4. `[claude]`
Configures Anthropic's Claude models hosted on Google Cloud Vertex AI.

*   **`model_name`** (string, default: `"claude-3-5-sonnet@20241022"`): Vertex model ID.
*   **`project_id`** (string): Google Cloud Project ID.
*   **`location`** (string, default: `"us-east5"`): Vertex AI region.

### 5. `[shell]`
Defines safety configurations for the shell command tool runner execution.

*   **`enabled`** (boolean, default: `true`): Whether the agent is allowed to execute shell commands on the host machine.
*   **`mode`** (string, default: `"blocklist"`): Safe patterns evaluation strategy. Set to `"blocklist"` or `"allowlist"`.
*   **`allowlist_patterns`** (list of regex, default: echo/pwd/git/uv/etc.): Regexes matching commands that are allowed to run.
*   **`blocklist_patterns`** (list of regex, default: rm/sudo/shutdown/etc.): Regexes matching dangerous commands that will be rejected.

---

## 💬 Chatbot Platform Settings

For setups relating to Discord, Google Chat, and WeChat, refer to the [Platforms configuration guides](platforms/discord.md).
