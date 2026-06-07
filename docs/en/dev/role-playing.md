# Role Playing Principles & Implementation

This document details how Kesoku manages, stores, and dynamically resolves character personas (roles) per chat channel/thread.

---

## 💾 Database Schema (`channel_roles` table)

The roleplay mappings are stored inside the SQLite database inside the `channel_roles` table:

```sql
CREATE TABLE IF NOT EXISTS channel_roles (
    chatbot_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    role TEXT NOT NULL,
    PRIMARY KEY(chatbot_id, channel_id)
);
```

Whenever a user requests a role switch or the agent executes the `play_role` tool:

1.  The mapping is upserted into `channel_roles`:
    ```sql
    INSERT OR REPLACE INTO channel_roles (chatbot_id, channel_id, role)
    VALUES (?, ?, ?)
    ```
2.  The mapping remains persistent across restarts.

---

## 🔄 Persona Resolution Inheritance Model

When a new conversational turn starts or when compiling system prompts, the active role name is resolved using the method:
`get_channel_role_with_inheritance(chatbot_id, channel_id, session_id)`

The resolution chain follows a strict inheritance order to ensure threads inherit context correctly:

```text
┌────────────────────────────────────────────────────────┐
│ 1. Direct Lookup                                       │
│    Is there a role bound to current channel_id?        │
└─────────────────────────┬──────────────────────────────┘
                          │ (If No)
                          ▼
┌────────────────────────────────────────────────────────┐
│ 2. Thread Inheritance Fallback (Discord only)          │
│    Is it a thread? Query parent_channel_id from        │
│    user message metadata, check parent role mapping.   │
└─────────────────────────┬──────────────────────────────┘
                          │ (If No)
                          ▼
┌────────────────────────────────────────────────────────┐
│ 3. Global Default                                      │
│    Return "default" persona scope.                      │
└────────────────────────────────────────────────────────┘
```

### 1. Direct Lookup
Queries the `channel_roles` table directly for matching `(chatbot_id, channel_id)`. If found, returns that role immediately.

### 2. Thread Inheritance (Discord)
For Discord bot sessions:

*   A new conversational thread is created automatically by the chatbot adapter.
*   This thread has its own unique Discord ID (`channel_id` in database).
*   Instead of requiring the user to set the role for every single thread, the database checks the last user message metadata in the active session for `parent_channel_id`.
*   If the parent text channel (e.g. `#general`) has a bound role (e.g. `"coder"`), the thread automatically adopts the `"coder"` role persona.

### 3. Root Fallback
If no channel or parent channel is bound, the executor falls back to `"default"`.

---

## 🔄 System Prompt Rebuilding Sequence

When the agent executes the `play_role(role="coder")` tool call mid-session:

1.  The database helper updates the channel role binding:
    ```python
    await db.set_channel_role(chatbot_id, channel_id, "coder")
    ```
2.  The tool triggers a prompt rebuild for the active session:
    ```python
    # Rebuild system prompt dynamically
    new_sys_prompt = build_sys_prompt(session=session)
    ```
3.  The compiled system prompt is saved back into the `sessions` table in SQLite:
    ```python
    await db.update_session_system_prompt(session_id, new_sys_prompt)
    ```
4.  The LLM backend loads the updated system prompt from the session database on its next inference cycle.

---

## 🎨 Persona Creation & Scaffolding Mechanism (`role-creator` Skill)

To facilitate creating custom character personas, Kesoku bundles the `role-creator` skill. It relies on a pure agent-based scaffolding workflow instead of a dedicated script, meaning the agent interactively designs and generates files based on templates.

### 1. Standard Persona Directory Structure

A complete character persona lives under `${AWD}/roles/{name}/` and adheres to the following layout and design specification:

*   **`intro.md`**: Main profile file. Contains the name, traits, speech patterns, and catchphrases, alongside TTS and Image script instructions that guide the LLM to run custom scripts when vocal or visual output is requested.
*   **`images/`**: Reference avatar or portrait image (if provided), loaded as `--image` when calling the `ai-image` skill to achieve visual consistency (Image-to-Image).
*   **`audio/`**: Reference WAV recording of the target voice (if provided), used as the base reference audio for TTS voice cloning.
*   **`scripts/`**: Houses generated wrapper shell scripts:
    *   `{name}-tts.sh`: Executable script for vocal cloning text-to-speech.
    *   `{name}-image.sh`: Executable script for rendering character illustrations.

### 2. Scaffolding Workflow by the Agent

When the `role-creator` skill is triggered, the agent executes the following steps within the AWD (Agent Working Directory):

1.  **Interact & Gather**: Ask the user for character parameters (name, core traits, references).
2.  **Scaffolding Folders**: Create the `roles/{name}/` directory and its subfolders `images/`, `audio/`, and `scripts/`.
3.  **Process References**:
    *   Copy and rename the avatar image to `roles/{name}/images/`.
    *   Copy and rename the WAV reference voice recording to `roles/{name}/audio/`.
4.  **Write Character Profile (`intro.md`)**:
    *   Compile the interactive choices into an instruction block and write it to `roles/{name}/intro.md`, highlighting speech and rendering rules.
5.  **Render Shell Scripts**:
    *   **TTS Script**: Reference the template `${SKILL_DIR}/template/scripts/asuka-tts.sh`. The agent replaces `asuka` (both lowercase and capitalized) with `{name}`, points `REF_AUDIO` to the copied WAV file, writes the matching `REF_TEXT` transcript, and saves the output to `roles/{name}/scripts/{name}-tts.sh`.
    *   **Image Script**: Reference the template `${SKILL_DIR}/template/scripts/asuka-image.sh`. The agent replaces name placeholders, points `REF_IMAGE` to the copied avatar file, and saves the output to `roles/{name}/scripts/{name}-image.sh`.
6.  **Executable Permissions**:
    *   After writing the shell scripts, the agent must run a system command (e.g. `chmod +x`) to set execute permissions (`0o755`) so they can be executed by the LLM during chat sessions.

