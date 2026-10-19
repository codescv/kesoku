# Kesoku ⚡️

Kesoku (name inspired by `結束/Kessoku`) is a minimal, readable, and modular autonomous AI coding and chat agent designed for seamless integration with chat platforms (Discord, Google Chat, WeChat, and CLI chat) powered by SQLite message persistence, hierarchical context compaction, and extensible tool calling.

---

## ✨ Main Features

- **Decoupled Broker Gateway & Per-Session Workers**: Asynchronous pub/sub message routing with isolated per-session workers, atomic tool execution, and mid-turn thought interruption.
- **Coding-First Tool Suite (P0)**:
  - **Head+Tail Middle-Out Truncation (`truncate_middle`)**: Preserves both the command/header at the top and tracebacks/summaries at the bottom of long shell and historical tool outputs.
  - **Dedicated `read_file` & Diagnostic `update_file`**: Line-numbered file reads (`offset`/`limit`), binary & directory guards, `$STAGING_DIR`/`$AWD` variable expansion, line-number-prefix auto-stripping, fuzzy-match line hints, and compact unified diffs on every edit.
  - **Static `# Available Skills` Prompt Index**: Injects a compact `[name]: description` skill catalog directly into the system prompt so the agent calls `use_skill(name)` directly without redundant discovery turns.
- **Continuation-Oriented Context Compaction & State Snapshots (P1)**:
  - **Structured Hierarchical Compaction**: Summarizes older turns into Level-0 / Level-1+ nodes with explicit `Dead Ends (Do Not Repeat)` and `Pending State` sections so post-compaction turns never repeat failed approaches or lose active PIDs/ports/branches.
  - **Water-Fill `<recent_user_messages>`**: Retains verbatim user messages from compacted turns using a fair-share water-filling character pool (`12,000` chars total, `6,000` ceiling).
  - **Per-Turn State Snapshots**: Automatically injects `<context name="running_jobs">` and `<context name="staging_artifacts">` (frozen once per turn to preserve prompt prefix caching).
  - **User-Steered `/compact [instructions]`**: Triggers immediate compaction from chat or CLI with optional focus instructions.
- **Job Lifecycle, Repo Rules & Sub-Agent Delegation (P2)**:
  - **`manage_job` Tool**: Structured `list`, `status`, and process-group `kill` (`SIGTERM` -> `SIGKILL`) for background shell jobs with wake-up alert suppression on intentional kills.
  - **Touch-Driven `AGENTS.md` / `GEMINI.md` Discovery & Hooks**: Automatically discovers repository convention files along touched paths into `<context name="repo_rules">`, nudges the agent with a built-in stop reminder after code writes, and supports configurable `[[hooks.post_tool_use]]` and `[[hooks.stop]]` commands.
  - **`delegate_subagent` Tool**: Spawns an isolated, bounded read-only sub-agent for noisy investigations (log/codebase searches) and returns only a concise `<= 1,500`-char report to the main conversation.
- **Reproducible Evaluation Suite (`kesoku-bench`)**: 16-task end-to-end benchmark (`benchmarks/kesoku_bench/`) covering coding accuracy, compaction survival, token efficiency, and autonomous operations.

---

## 📖 Documentation

For detailed guides on advanced configuration, platform bots (Discord, WeChat, Google Chat), cron jobs, and internal architecture, visit the official documentation:
👉 **[https://codescv.github.io/kesoku/](https://codescv.github.io/kesoku/)** (or see [`docs/DESIGN.md`](docs/DESIGN.md) and [`docs/en/`](docs/en/index.md)).

---

## 📦 Quick Start Installation & Configuration

Install Kesoku globally as a CLI tool using `uv`:

```bash
uv tool install git+https://github.com/codescv/kesoku.git
```

For local development from a checkout:

```bash
uv tool install -e .
```

To initialize a workspace and configuration template (`config.toml`):
```bash
kesoku init -c private/config.toml
```

Edit `private/config.toml` to configure your `[workspace]`, `[agent]`, `[gemini]` / `[claude]` credentials, `[shell]` rules, and optional `[hooks]`.

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
