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
*   If the parent text channel (e.g. `#general`) has a bound role (e.g. `"tifa"`), the thread automatically adopts the `"tifa"` role persona.

### 3. Root Fallback
If no channel or parent channel is bound, the executor falls back to `"default"`.

---

## 🔄 System Prompt Rebuilding Sequence

When the agent executes the `play_role(role="tifa")` tool call mid-session:

1.  The database helper updates the channel role binding:
    ```python
    await db.set_channel_role(chatbot_id, channel_id, "tifa")
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
