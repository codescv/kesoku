# Chatbot Slash Commands

Kesoku chatbot adapters support standard slash commands (e.g. on Discord) or text command prefixes (e.g. `/command` on WeChat or Console) to control agent execution, manage context compaction, switch personas, and monitor system performance.

---

## 🚀 Available Commands

### 1. Persona Management
*   **`/role`**: Displays the active persona name and lists all available roles in the AWD directory.
*   **`/role {name}`**: Binds and switches the current channel or thread to the specified persona (e.g., `/role coder`).

---

### 2. Context & Session Control
*   **`/clear`** (Alias: **`/reset`**):
    *   Deletes the entire conversation history for the current channel/thread from the SQLite database.
    *   Recursively deletes the session's workspace staging folder on disk (cleaning up downloaded files/attachments).
    *   Removes active views and button elements.
*   **`/compact`**: Manually forces context compaction on the active history of this channel immediately, replacing old turns with scaffold summary nodes to free up context window space.
*   **`/context`**: Generates and returns a download link to an interactive dark-mode HTML trace file (`lcm_context.html`) detailing the exact LLM thinking trajectory, prompt logs, and tools executed in the current turn.
*   **`/grep {query}`** (Alias: **`/memory-grep`**): Searches active memories and past chat messages for the current channel's role persona matching the query (keyword match or wildcard `*`).
*   **`/search {query}`** (Alias: **`/memory-search`**): Performs semantic (vector) search against active memories and past chat messages for the current channel's role persona.

---

### 3. Monitoring & Debugging
*   **`/status`**: Retrieves and displays detailed session execution statistics:
    *   *Total Session Turns*: Number of turns processed in the current session.
    *   *Context Window Size*: Active tokens loaded in the context window (in K).
    *   *Executed Tool Calls*: Total tools run in the current turn.
    *   *Turn Token Usage*: Input + output tokens consumed.
    *   *Uptime & Speed*: Execution time of the latest turn.
*   **`/debug`**: Toggles debug mode in-place. If enabled, Kesoku outputs raw LLM inference JSONs to logs and returns the session's staging directory absolute path for auditing.

---

### 4. System & Daemon
*   **`/restart`**: Safely restarts the Kesoku background daemon service.
    *   Sends an ephemeral notification to confirm receipt.
    *   Closes active connections, stops background listeners.
    *   Reloads the OS systemd daemon or triggers an in-place `execv` swap.
*   **`/cronjob`** (Alias: **`/cronjob {tag}`**): Lists or manually triggers scheduled automation tasks registered under the config.
