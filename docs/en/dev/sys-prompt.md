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

### 4. Built-in Instructions
These are static, hardcoded instructions inside `prompt.py` that train the agent on how to use Kesoku's features:
*   **`SKILLS_INSTRUCTIONS`**: Guides the agent on how to discover custom instruction files by calling `list_skills()` and `use_skill(name)`.
*   **`FILE_SENDING_INSTRUCTIONS`**: Teaches the agent the file-sending markup syntax: `[file: /absolute/path/to/file]`. If this string is matched in outgoing text, chatbot adapters will upload the file as a native attachment.
*   **`QUESTION_INSTRUCTION`**: Teaches the agent the multiple-choice button markup syntax: `[question: <title> | choice1 | choice2]`.
*   **`MEMORY_AND_HISTORY_INSTRUCTIONS`**: Guides the agent on how to access and modify structured memories in SQLite.
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
