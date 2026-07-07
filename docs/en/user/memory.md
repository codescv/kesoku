# Memory & Sessions User Guide

Kesoku stores all chat histories, settings, and agent memories inside a localized SQLite database (`kesoku.db`). This page explains how to manage active chat sessions and manipulate long-term agent memory entries using the CLI.

---

## 💬 Session Management via CLI

Every interaction is isolated within a unique session (identified by a `session_id`). In daemon mode, these IDs are bound automatically to Discord threads or Google Chat rooms.

You can manage these sessions using the `kesoku chat` command group:

### 1. List All Active Sessions
To inspect all recorded sessions in the SQLite database, along with their creation time, character role, and the number of messages:

```bash
kesoku chat -c config.toml -l
```

### 2. Show Session Chat History
To print the full, beautiful, and colorized conversational history of a specific session to your terminal (utilizing `rich` formatting):

```bash
kesoku chat -c config.toml --show-history <session_id>
```

### 3. Resume a Session
To carry out a command-line chat turn inside an existing session:

```bash
kesoku chat -c config.toml -r <session_id> "What was the previous number?"
```
To resume the **most recent** active session instantly:

```bash
kesoku chat -c config.toml -z "Continue the task."
```

---

## 🧠 Managing Long-Term Agent Memories (`memory`)

Kesoku implements an agent memory system allowing the agent to store structured knowledge (e.g. user preferences, key milestones, or configurations) in SQLite. These memories are scoped by **Category** and **Role Persona**.

Administrators can inspect, edit, or migrate these memories using the `kesoku memory` command group:

### 1. List Memories
List all stored memories in a specific category (and optionally filter by character role):

```bash
# List all memories in the 'user_preference' category for the default role
kesoku memory list --category user_preference --role default
```

### 2. View Specific Memory Content
Show detailed content of a single memory key:

```bash
kesoku memory view --category user_preference --key user_timezone --role default
```

### 3. Update or Add Memory
Manually update or insert a memory record:

```bash
kesoku memory update --category user_preference --key user_timezone --title "User Timezone" --content "Asia/Tokyo" --role default
```

### 4. Delete Memory
Delete a memory record:

```bash
kesoku memory delete --category user_preference --key user_timezone --role default
```

### 5. Backup & Migration (Export / Import)
To backup all memories across all roles, or migrate them to another database:

*   **Export to JSON**:
    ```bash
    kesoku memory export -o memories_backup.json
    ```
*   **Import from JSON**:
    ```bash
    kesoku memory import -i memories_backup.json
    ```

---

### 6. Semantic Search & Vector Index Rebuilding (`rebuild-index`)

Kesoku supports multilingual semantic search utilizing a lightweight, local Embedding model.

*   **Automatic Indexing**:
    *   During active chat conversations, all **explicitly saved Memories** and standard chat dialogue (**textual User queries & Assistant responses**) are automatically vectorized and stored (via the `embedding` column) on SQLite write.
    *   To keep the index clean, internal Assistant Thoughts and external Tool calls/results are excluded from vector indexing.
*   **Manual Index Rebuilding**:
    *   If you import historical data or want to reset/update vector coordinates, you can run the `rebuild-index` command to incrementally or fully regenerate embeddings for all records:
    ```bash
    # Index all memories and messages that lack an embedding
    kesoku memory rebuild-index -c config.toml

    # Force clear all current embeddings and fully rebuild the index
    kesoku memory rebuild-index -c config.toml --force
    ```

