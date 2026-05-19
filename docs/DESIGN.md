# Kesoku System Design Document

## Executive Summary
Kesoku is a lightweight, highly readable, and robust autonomous AI agent framework. Designed around a decoupled gateway architecture with a Pure Broker pub/sub pattern and structured TOML configuration, Kesoku enables asynchronous interaction between various chat interfaces (such as Discord and one-shot CLI sessions) and a powerful autonomous agent loop equipped with per-session background concurrency and tool execution capabilities.

## Architectural Overview

```
+------------------------------------------------------------------------+
|                        Configuration Layer                             |
|                `config.toml` loaded via `src/kesoku/config.py`         |
+------------------------------------------------------------------------+
                                   |
              +--------------------+--------------------+
              | (`kesoku start`)                        | (`kesoku chat`)
              v                                         v
+------------------------------------+    +------------------------------------+
|  Foreground Service Mode (Start)   |    |    Session CLI Mode (Chat)         |
| - Launches all background bots     |    | - Launches local CLIChatbot        |
|   (Discord, etc.) from config      |    | - Buffers one-shot session events  |
+------------------------------------+    +------------------------------------+
              |                                         |
              +--------------------+--------------------+
                                   v
+------------------------------------------------------------------------+
|                           Kesoku Gateway                               |
| - Stateless Pub/Sub Broker (`post(msg)` & `listen(**filters)`)         |
| - Unified message ingestion, routing, and persistence via `post`       |
| - Manages persistent conversational sessions in SQLite                 |
+------------------------------------------------------------------------+
                                   |
                       SQLite Persistence Layer
     (Database tables: `messages` & `sessions` at configured `db_path`)
                                   ^
                                   |
+------------------------------------------------------------------------+
|                     Kesoku Agent Dispatcher Loop                       |
| - Asynchronously listens for `role="user"` messages                    |
| - Dispatches per-session background tasks (`SessionWorker`)            |
|                                                                        |
|       +--------------------------------------------------------+       |
|       |                    SessionWorker                       |       |
|       | - Pulls user messages from session queue               |       |
|       | - Checks for thought interruptions before/after steps  |       |
|       | - Invokes LLM (Gemini API) & executes atomic tools     |       |
|       +--------------------------------------------------------+       |
+------------------------------------------------------------------------+
```

## Concurrency, Anti-Stall Mechanism & Thread Sorting (V4)
To handle multiple users and user interruptions gracefully, Kesoku implements an advanced concurrency model:
1. **Stateless Pub/Sub Hub**: The Gateway provides `post(message)` to save messages and broadcast them to active in-memory `listen(**filters)` async generators.
2. **Agent Dispatcher**: The master Agent runs `listen(role="user")`. Upon receiving a message, it checks if a `SessionWorker` task exists for that `session_id`. If not, it spawns one; otherwise, it pushes the message into the worker's queue.
3. **Session Worker & Interruption Policy**:
   - Each worker processes its queue atomic step by atomic step (LLM inference or Tool execution).
   - **Never Kill Mid-Tool**: For safety, tool executions are atomic. The worker waits for the tool to complete before checking for new user input.
   - **Thought Interruption**: If a new user message arrives in the queue while the LLM is generating or before a tool is invoked, the worker pivots immediately to the new message, updating previous pending actions as `interrupted`.
4. **Turn-Based Thread Sorting**: To perfectly preserve interrupted branches and asynchronous tool outputs without temporal interleaving, session history is ordered logically by turn root timestamp: `(root_message_timestamp, message_timestamp)`.

## Message Data Model & Native Tool Calling
All message ingestion and routing is unified through `Gateway.post()`. Every message in Kesoku follows strict role, type, status, and sender conventions as detailed in [Message and Lifecycle Specification](MESSAGE_AND_LIFECYCLE.md):
- **Roles**: `user`, `assistant`, `tool`, `system`
- **Types**: `text`, `thought`, `tool_call`, `tool_result`
- **Sender Rules**:
  - User input: External username or `User`
  - Assistant thought / response text: `Kesoku`
  - Tool call: `Kesoku`
  - Tool output: The specific tool name (e.g., `calculator`)
  - System notifications: `System`
- **Native Function Calling, Thought Signatures & Parallel Batching**: Tool requests and execution results store structured dictionaries in message `metadata` (`{"tool_name": ..., "tool_arguments": ..., "thought_signature": ...}`). To strictly comply with the Gemini API specification for parallel function calling, all `tool_call` messages for a given model turn are batched and posted before executing the tools concurrently. Their corresponding `tool_result` messages are then posted together, guaranteeing that multiple parallel calls and responses are grouped correctly into consecutive parts without interleaving.


## Discord Chatbot Adapter Architecture
Kesoku includes a fully functional Discord chatbot adapter (`DiscordChatbot`) connecting external Discord servers with the internal Gateway broker:
- **Allowlist Filtering & Token Fallback**: Configured via `user_allowlist`. If populated, unlisted users only receive replies if explicitly mentioning the bot. Messages explicitly mentioning third parties are ignored. The chatbot supports fallback token lookup via the `DISCORD_TOKEN` environment variable when `bot_token` is not specified in the configuration file.
- **Thread-Based Context Separation & Direct Channel Interaction**: To prevent multi-user context collisions, conversations are normally isolated inside Discord threads, where the thread ID maps to Kesoku's external `channel_id`. To support direct channel interaction without thread creation, a configuration option `no_auto_thread_channels` can be populated with channel IDs or names. For designated channels in this list, the bot interacts directly in the main channel without automatically spawning new threads. If a user manually starts a thread within a direct channel, the adapter detects the thread context and interacts within the thread naturally. Session creation timestamps are synchronized with initial Discord message timestamps to ensure correct chronological ordering. If multiple bots run in the same channel, thread creation race conditions (`discord.HTTPException`) are gracefully handled by discovering and joining the thread created by peer bots.
- **Special Context Prompts**: Server and channel metadata, along with active member lists and Discord user IDs, are dynamically injected into the session's system prompt.
- **Newline Chunking, File-Sending Syntax & Message Splitting**: The bot renders all assistant responses, thoughts (`💭`), tool calls (`🛠️`), and tool results (`📥`) to Discord. Output exceeding Discord's 2000-character limit is cleanly chunked at newline boundaries. In addition, Kesoku supports a standardized file-sending syntax `[file: /abs/path/to/file]`. When the Discord chatbot encounters this syntax in any outgoing message content, it automatically splits the content around the file blocks, uploads existing files as attachments via `discord.File`, and filters out empty or whitespace-only message segments to strictly conform to Discord API constraints. If a specified file does not exist on disk, a user-friendly warning notification is gracefully dispatched to the thread.
- **Asynchronous Typing Status Indicator**: While the agent is thinking, running tools, or generating responses, `DiscordChatbot` continuously displays a typing status in the corresponding thread or channel. The typing indicator starts when an incoming user message is received, and is gracefully cancelled upon successful delivery of the final assistant text response. To prevent infinite typing during an unexpected error or network drop, each typing task is guarded by a robust 10-minute safety timeout.


## Systemd Service Integration
To support running Kesoku as a continuous background daemon in production environments on Linux, the CLI provides a `service` command group implemented modularly inside `src/kesoku/cli_service.py`. This command group automates generating, registering, running, and removing the systemd unit file (`kesoku.service`).

### Service Subcommands Design
- **Main Command**: `kesoku service` (mounted as a sub-Typer application group)
- **Subcommands**:
  - `install`: Generates and writes the systemd unit file, runs systemd `daemon-reload`, and automatically enables (`systemctl enable`) the service to configure boot-time auto-start (without starting the active service process immediately).
    - Options:
      - `-c / --config <path>`: Custom configuration file path (default: `config.toml`). Resolves to an absolute path and sets `WorkingDirectory` and `ExecStart` automatically.
      - `-e / --env KEY=VALUE`: Environment variables injected into the systemd unit definition (can be specified multiple times).
      - `--user / --system`: Configures as a user-level unit (default, target path: `~/.config/systemd/user/kesoku.service`) or system-level unit (target path: `/etc/systemd/system/kesoku.service`).
      - `--dry-run`: Prints the generated service unit file directly to stdout without writing.
    - Environment Variable Inheritance:
      - By default, the service installer inherits specific environment variables from the shell of execution if present: `PATH`, `HTTP_PROXY`, `HTTPS_PROXY`, `GOOGLE_API_KEY`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `GOOGLE_GENAI_USE_VERTEXAI`, and `DISCORD_TOKEN`. Users can override or append custom environment variables using the `-e` / `--env` options.
    - Service Unit Optimizations:
      - `Restart=always`: Ensures the background daemon is always restarted by systemd upon clean exits, unclean crashes, or signal termination.
      - `RestartSec=5`: Introduces a safe 5-second restart delay to prevent tight crash loops.
      - `TimeoutStopSec=210`: Provides a generous 3.5-minute shutdown grace period, giving the autonomous agent ample time to complete active LLM iterations or atomic tool executions cleanly and persist its state.
      - `StandardOutput=journal` and `StandardError=journal`: Forces all stdout/stderr log streams to standard systemd `journald` infrastructure.
  - `uninstall`: Stops and disables the service, removes the unit file from disk, and reloads systemd daemon.
  - `start`: Starts the background systemd service via `systemctl [--user] start kesoku`.
  - `stop`: Stops the background systemd service via `systemctl [--user] stop kesoku`.
  - `restart`: Restarts the background systemd service via `systemctl [--user] restart kesoku`.
  - `status`: Queries and displays the active runtime status of the background service via `systemctl [--user] status kesoku`.
  - `logs`: Displays or streams log output directly from `journald` using `journalctl`.
    - Options:
      - `-f / --follow`: Stream/follow live log output.
      - `-n / --lines <int>`: Show specified number of recent log lines (default: 50).



## Configuration Schema (`config.toml`)
Kesoku is centrally configured via a structured TOML file managed by Pydantic models in `src/kesoku/config.py`. Once loaded at CLI startup, the configuration acts as a global singleton accessible from any module via `get_config()`, avoiding the need to pass configuration objects across components.

```toml
[workspace]
db_path = "kesoku.db"
skills_dir = "skills"
sessions_dir = "sessions"

[agent]
llm = "gemini"

[gemini]
model_name = "gemini-2.5-flash"
auth_mode = "api_key" # or "vertex"
api_key = "your-api-key" # optional if GEMINI_API_KEY env var is set
project_id = "gcp-project-id" # for vertex mode
location = "us-central1" # for vertex mode
thinking_level = "high" # thinking level for reasoning ('minimal', 'low', 'medium', 'high')

[discord]
enabled = false
bot_token = "discord-bot-token"
chatbot_id = "discord"
user_allowlist = ["allowed_username"]
no_auto_thread_channels = ["no-thread-channel-name-or-id"]

[shell]
enabled = true
use_shell = true
mode = "blocklist"
allowlist_patterns = ["^(echo|ls|pwd|cat|git|uv|grep|find|python|sed|awk)(\\s|$)"]
blocklist_patterns = ["(\\b|^)(rm|sudo|shutdown|reboot|mkfs|dd|chmod|chown)(\\b|\\s|$)"]
```

## Autonomous Skill System
Kesoku features an autonomous skill system that allows agents to dynamically discover and adopt specialized domain capabilities and prompt instructions during conversational sessions.

### Skill Architecture & Self-Contained Manifests
Skills are organized in subdirectories inside the configured `workspace.skills_dir` (default `skills/`). A skill is entirely defined by a self-contained `SKILL.md` file (or any `.md` file in the skill's root folder) containing YAML frontmatter enclosed in `---`.

```yaml
---
name: ai-image
description: AI image generation and editing.
metadata:
  tags: [aigc, gcp]
  platforms: [linux, darwin]
---
```

- **Platform Filtering**: If `platforms` is specified under `metadata`, `list_skills` evaluates the host OS (`platform.system().lower()`). If the current platform is not in the list, the skill is excluded. If `platforms` is omitted (`None`), the skill is treated as cross-platform and listed on all operating systems. If explicitly defined as empty (`[]`), the skill is excluded from all platforms.

### Native Tool Integration & Script Robustness
The skill manager exposes two native tools to the LLM:
1. `list_skills()`: Scans `skills_dir`, parses YAML frontmatter, filters by OS platform, and returns a summary of available skills.
2. `use_skill(skill_name: str)`: Returns the complete markdown instructions from the skill's `SKILL.md`. To ensure robust execution when a skill includes scripts or tools (e.g., `uv run scripts/script.py`), `use_skill` automatically prepends a prominent header with the exact absolute path to the skill directory, instructing the LLM to use absolute paths for all CLI command invocations.
