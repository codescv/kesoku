# System Prompt Compilation Principles

To ensure consistency, safety, and capability awareness, Kesoku dynamically compiles a unified system prompt at the beginning of each conversational turn. This guide details how prompts are assembled from configuration files, roles directories, and hardcoded instructions.

---

## 🛠️ Composition Flow

The prompt is constructed dynamically by `build_sys_prompt()` inside `src/kesoku/agent/prompt.py`. The compiled string is joined using double newlines (`\n\n`) and consists of the following sections in sequence:

```text
┌────────────────────────────────────────────────────────┐
│ 1. Active Persona: roles/{role}/intro.md               │
├────────────────────────────────────────────────────────┤
│ 2. Agent Working Directory (AWD)                       │
├────────────────────────────────────────────────────────┤
│ 3. Session Staging Directory (STAGING_DIR)             │
├────────────────────────────────────────────────────────┤
│ 4. Built-in Capability Instructions (Skills, Files,...)│
├────────────────────────────────────────────────────────┤
│ 5. Configured User Prompts (config.agent.user_prompts) │
├────────────────────────────────────────────────────────┤
│ 6. Adapter Custom Prompts (Discord Member metadata)    │
└────────────────────────────────────────────────────────┘
```

---

## 🔍 Detailed Sections Breakdown

### 1. Active Persona (Role Profile)
*   **Resolution**: Resolves the channel's bound role name from the database (defaults to `"default"`).
*   **File Loader**: Reads the contents of `roles/{role_name}/intro.md` inside your roles folder.
*   **Purpose**: Sets the personality, response style, and character boundaries of the assistant.

### 2. Agent Working Directory (AWD)
*   **Instruction**: Tells the agent its execution root on the host system:
    ```markdown
    # Agent Working Directory
    > AWD='{cfg.agent_working_dir}'
    You are working in the agent working directory (AWD)...
    ```
*   **Purpose**: Restricts the agent from searching or reading files outside this folder unless explicitly asked.

### 3. Session Staging Directory (STAGING_DIR)
*   **Instruction**: Sets the dedicated session staging workspace path:
    ```markdown
    # Session Staging Directory
    > STAGING_DIR='{sessions_dir}/{session_workspace_name}'
    - This is where you are supposed to save your output files...
    ```
*   **Purpose**: Instructs the agent to save all generated images, downloaded files, and execution outputs inside this isolated folder to prevent workspace pollution.

### 3b. Available Skills Index
*   **Instruction**: `build_skills_section()` renders a deterministic `# Available Skills` section right after the
    directory blocks: one `- <name>: <description>` line per installed skill (global + role-scoped, role skills win
    on name collisions), sorted by name, descriptions collapsed to one line and cut at 200 chars, at most 60 entries
    and 6000 chars (the rest is summarised with a pointer to `list_skills()`).
*   **Purpose**: The agent calls `use_skill(skill_name=...)` directly instead of spending a `list_skills()` call.
*   **Skipped** when `use_skill` is in `agent.disabled_tools`, when no skills are installed, or when listing fails.
*   **Caveat**: the system prompt is persisted per session at creation (and on role switch/resume), so skills added
    later only appear in new sessions; `list_skills()` remains the live listing.

### 4. Built-in Instructions
These are static, hardcoded instructions inside `prompt.py` that train the agent on how to use Kesoku's features:

*   **`SKILLS_INSTRUCTIONS`**: Guides the agent on how to discover custom instruction files by calling `list_skills()` and `use_skill(name)`.
*   **`FILE_SENDING_INSTRUCTIONS`**: Teaches the agent the file-sending markup syntax: `[file: /absolute/path/to/file]`. If this string is matched in outgoing text, chatbot adapters will upload the file as a native attachment.
*   **Choice questions** (in `OUTPUT_FORMATTING_INSTRUCTIONS`): a fenced ```` ```ui ```` YAML block with `type: select | multiselect`, `prompt`, `options` and optional `max`. At most one per reply; clicked options come back as the option text, multiselect answers as `选择：A；B`. Meaningless "manual input / I'll type it myself / other" options are forbidden (users can always type in chat); when offering candidate choices, include a "再帮我想 X 个不同的选项" ("Give me X different options") option to generate a fresh batch. The legacy `[question: ...]` tag is still parsed but no longer taught.
*   **Collapsible sections**: `<details><summary>Title</summary>Body</details>` (Discord: `▸ Title` + 展开 button; other platforms: bold title followed by the body).
*   **Hidden text**: `<spoiler>…</spoiler>` marks inline text the user must click to reveal. Each adapter's `format_text` renders it: Discord emits native `||…||`; platforms without hidden-text support (CLI, WeChat, Google Chat) strip the tags and show the content. See `kesoku.utils.text.render_spoilers`.
*   **`MEMORY_AND_HISTORY_INSTRUCTIONS`**: Guides the agent on how to access and modify structured memories in SQLite.
*   **`SESSION_NOTES_INSTRUCTIONS`**: Explains the per-turn `<context name="...">` blocks (see section 7), how to maintain `$STAGING_DIR/notes.md` with `update_file`, and that files under `$STAGING_DIR/context/` are tool-maintained and must never be edited with `update_file`.
*   **`FILE_TOOL_INSTRUCTIONS`**: Read files with `read_file(file_name, start_line, end_line)` instead of `cat`/`head`/`sed -n`; create files with `update_file(file_name, new_content=...)` (omit `old_content`); edit with a short unique block copied from `read_file` output without the `N: ` prefixes; page truncated shell outputs with `read_file`.
*   **`BACKGROUND_EXECUTION_INSTRUCTIONS`**: Instructs the agent on how to handle long-running background shell executions and yield the conversational turn.

### 5. Configured User Prompts
*   **Resolution**: Iterates over the file paths array defined in `config.toml` under `[agent].user_prompts`.
*   **Formatting**: Each file's content is wrapped in boundaries:
    ```markdown
    === BEGIN {filename} ===
    {content}
    === END {filename} ===
    ```
*   **Purpose**: Allows administrators to inject global custom instruction guides (e.g. coding standards, API references).

### 6. Adapter Custom Prompts
*   **Resolution**: Chatbot adapters can pass a `custom_prompt` parameter to `build_sys_prompt()`.
*   **Purpose**: Injects real-time workspace metadata. For example, the Discord chatbot fetches the list of server channel members, active user IDs, and channel names, and appends it to the prompt so the agent is aware of who it is talking to.

---

## 🧩 7. Per-Turn Injected Context (not part of the system prompt)

Besides the cached system prompt, `TurnExecutor` prepends per-turn blocks to the latest user message (so the system prompt stays stable for prefix caching). The order is:

1.  `<instructions>`: the active role's `roles/{role}/preferences.md` (every turn).
2.  `<context name="...">` blocks (every turn, see `kesoku.agent.context_dir`):
    *   `<context name="notes">`: the session's pinned notes, read from `$STAGING_DIR/notes.md` (editable by the agent via `update_file`).
    *   One block per `$STAGING_DIR/context/*.md` file, sorted by filename. The block name is the filename without a leading numeric prefix (`20-`, `05_`) and without `.md`, e.g. `20-scene.md` → `<context name="scene">`. These files are written by tools/skills; the agent must not edit them. The name `notes` is reserved for the pinned notes: a `context/notes.md` (or `NN-notes.md`) is rendered as `<context name="notes-1">` with a warning.
    *   Budget: each file is capped at 6000 characters (tail truncated, `[truncated]` appended); the total is capped by `agent.context_dir_max_chars` (default 20000). When exceeded, files are dropped from the end of the sorted list and a warning is logged.
3.  `<facts>`: the role's `facts.md` (first message of the day only).
4.  Bootstrap guidelines / auto-retrieved history, then `<current_message ...>`.

The `/context` command shows the same `<context>` blocks prepended to the latest user message of the fresh tail.
