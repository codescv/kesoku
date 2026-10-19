# Kesoku ⚡️

Welcome to the official documentation for **Kesoku** (结束 / Kessoku).

Kesoku is a minimal, readable, and highly modular autonomous AI agent framework. Designed around a decoupled gateway architecture with a pure pub/sub broker pattern, Kesoku enables robust, asynchronous communication between multiple chat frontends (such as Discord and Google Chat) and an autonomous reasoning loop featuring concurrency, thought interruption, and tool execution.

---

## 🌟 Core Features

*   **Decoupled Gateway Broker**: Adapters (Discord, Google Chat, WeChat) and backend agents are decoupled, communicating purely via state-driven message queues.
*   **Asynchronous Concurrency**: Dispatcher spawns a dedicated task loop (`SessionWorker`) for each session to prevent multi-user collisions.
*   **Anti-Stall & Interruption**: Agent workers support thought interruption when a new user prompt is received mid-turn, without killing active safe tools.
*   **Structured TOML Config**: Manage models, database settings, and credentials centrally in `config.toml`.
*   **Coding & Compaction Suite**: Dedicated `read_file` / diagnostic `update_file` tools, head+tail middle-out output truncation, continuation-oriented hierarchical compaction (`Dead Ends`, `Pending State`, water-fill `<recent_user_messages>`), per-turn `<running_jobs>` / `<staging_artifacts>` / `<repo_rules>` state snapshots, `manage_job`, `delegate_subagent`, and configurable post-tool/stop hooks.
*   **Extensible Skills & Static Prompt Index**: Discover and apply domain-specific instruction manuals (`SKILL.md`) with a zero-turn static `# Available Skills` index in the system prompt.
*   **OS Daemon Integration**: Register and manage the agent easily as a `systemd` or `launchd` service.

---

## 🗺️ Documentation Roadmap

To get started, choose one of the guides below:

### 👥 User Guides
*   [**Installation**](user/installation.md): Set up the Python environment using `uv`.
*   [**Configuration**](user/configuration.md): Learn about the keys and schemas in `config.toml`.
*   [**Platforms**](user/platforms/discord.md): Connect the agent to Discord, Google Chat, or WeChat.
*   [**Service Management**](user/service.md): Set up Kesoku to run permanently in the background.

### 💻 Developer Guides
*   [**Architecture & Design**](dev/architecture.md): Deep dive into the Broker gateway, workers, and concurrency models.
*   [**Agent Loop Cycle**](dev/agent-cycle.md): Step-by-step walkthrough of how prompt events flow through the system.
*   [**Custom Skills**](dev/skills.md): Teach the agent new capabilities by creating your own skill bundles.
