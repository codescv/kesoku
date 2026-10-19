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
Defines active agent features, context compaction, repository rules, and sub-agent delegation.

*   **`llm`** (string, default: `"gemini"`): The active LLM engine provider. Supported values are `"gemini"` and `"claude"`.
*   **`protect_front_turns`** / **`protect_tail_turns`** (int, defaults: `1` / `3`): Number of initial and most-recent logical turns protected from automatic compaction.
*   **`base_node_min_tokens`** (int, default: `25000`): Token watermark above which eligible buffer turns are compacted into Level-0 summary nodes.
*   **`recent_user_messages_limit`** / **`recent_user_messages_total_chars`** / **`recent_user_message_ceiling_chars`** (ints, defaults: `100` / `12000` / `6000`): Water-filling budget for verbatim user messages retained in `<recent_user_messages>` across compacted turns.
*   **`repo_rules_files`** (list of strings, default: `["AGENTS.md", "GEMINI.md"]`): Filenames discovered automatically along touched directory paths up to `$AWD` and injected as `<context name="repo_rules">`.
*   **`repo_rules_max_files`** / **`repo_rules_max_chars`** (ints, defaults: `3` / `4000`): Cap on discovered repository rule files and characters per file.
*   **`repo_rules_stop_reminder`** (bool, default: `true`): When enabled, reminds the agent once before ending a turn that modified files inside a discovered rules subtree to run required verification/lint checks.
*   **`subagent_allowed_tools`** (list of strings, default: `["read_file", "run_shell_command", "list_skills", "use_skill"]`): Tools available to `delegate_subagent` sub-agents.
*   **`subagent_allow_write`** (bool, default: `false`): Whether sub-agents may use `update_file`.
*   **`subagent_max_steps`** / **`subagent_max_concurrent`** / **`subagent_max_per_turn`** (ints, defaults: `12` / `1` / `3`): Step and concurrency limits for `delegate_subagent`.
*   **`subagent_shell_timeout_seconds`** (int, default: `120`): Foreground timeout for sub-agent shell commands (sub-agent commands never background).
*   **`subagent_model_name`** (string, default: `""`): Optional model override for sub-agent runs.

### 3. `[gemini]`
Configures Google GenAI/Gemini integrations.

*   **`model_name`** (string, default: `"gemini-2.5-flash"`): The model ID.
*   **`auth_mode`** (string, default: `"api_key"`): Authentication method. Use `"api_key"` for raw API Keys, or `"vertex"` to run via Google Cloud Vertex AI (which uses Application Default Credentials).
*   **`api_key`** (string): The API key (if `auth_mode = "api_key"`). If empty, falls back to the `GEMINI_API_KEY` environment variable.
*   **`project_id`** (string): Google Cloud Project ID (required for `"vertex"` mode).
*   **`location`** (string, default: `"us-central1"`): GCP region for Vertex AI endpoints.
*   **`thinking_level`** (string, default: `"high"`): The thinking/reasoning budget. Supported: `"minimal"`, `"low"`, `"medium"`, `"high"`.
*   **`safety_threshold`** (string, default: `"off"`): Harm block threshold applied to every configurable safety category (`"off"`, `"block_none"`, `"block_low_and_above"`, `"block_medium_and_above"`, `"block_only_high"`). Covers harassment, hate speech, sexually explicit, dangerous content and civic integrity, plus the Vertex-only image (hate / dangerous / harassment / sexually explicit) and jailbreak categories when `auth_mode = "vertex"`. The provider defaults intermittently blank out long roleplay replies (the response comes back with `finish_reason=SAFETY` and no content at all), so filtering is disabled by default. Set to `null` to fall back to the provider defaults.

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
*   **`forbidden_patterns`** (dictionary/table of regex to string): Regexes matching commands mapped to custom error messages. If a command matches, it is rejected and the custom error message is returned.

### 6. `[hooks]`
Configures optional `[[hooks.post_tool_use]]` and `[[hooks.stop]]` commands executed automatically during turns:

*   **`[[hooks.post_tool_use]]`**: Runs after matching `tools` (e.g. `["update_file"]`) and optional `file_pattern` regex (e.g. `"\\.py$"`), substituting `{file}`, `{awd}`, and `{staging_dir}` in `command`. Appends `[hook <name> exit <code>]` to the tool result when `append_on` is `"nonzero"` or `"always"`.
*   **`[[hooks.stop]]`**: Runs before the agent finishes a turn when any tool in `when_tools_used` ran during the turn. Non-zero exit codes re-prompt the agent up to `max_retries` times.

---

## 💬 Chatbot Platform Settings

For setups relating to Discord, Google Chat, and WeChat, refer to the [Platforms configuration guides](platforms/discord.md).
