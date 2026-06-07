# Testing & Debugging Reference

This guide details the debugging resources, logging structure, direct database inspection queries, and unit-testing conventions in Kesoku.

---

## 📜 LLM Turn Trajectory Logging (`TurnLogger`)

To audit LLM prompts, tool execution arguments, and reasoning trajectories, Kesoku registers a `TurnLogger` (`src/kesoku/agent/turn_logger.py`):
*   **Log Location**: Saved sequentially under the session's workspace staging directory: `sessions/<session_workspace_name>/llm-turn-{idx}.log.yaml`.
*   **Format**: Logged as a structured YAML document for both human-readability and script parsing.

### Sample YAML Structure:
```yaml
metadata:
  timestamp: 1717765103.541
  session_id: "sess_abc123"
  turn_index: 3
  llm_provider: "gemini"
history:
  - role: "user"
    sender: "my_username"
    type: "text"
    content: "Run test compile"
  - role: "assistant"
    sender: "Kesoku"
    type: "thought"
    content: "I need to run the shell command tool."
tools:
  - name: "run_shell_command"
    description: "Execute terminal commands safely..."
response:
  content: "Shell execution finished."
  thought: "Command executed successfully. Informing user."
  prompt_tokens: 1450
  candidates_tokens: 340
  total_tokens: 1790
```

Use these log files to review the exact inputs passed to model context windows and check if tools were converted correctly.

---

## 💾 Direct Database Inspection (`sqlite3`)

To inspect or debug session histories directly, open the SQLite database from your workspace terminal:

```bash
sqlite3 kesoku.db
```

### Core Database Schema

1.  **`messages`**: Stores flat chat logs (role, type, sender, content, parent_id, timestamp, status).
2.  **`sessions`**: Stores session configuration details (system prompt overrides, workspace folder mappings).
3.  **`channel_sessions`**: Maps chat platform channels/threads to session IDs:
    ```sql
    SELECT * FROM channel_sessions WHERE chatbot_id = 'discord';
    ```
4.  **`channel_roles`**: Maps channels to character personas.
5.  **`cross_session_contexts`**: Stores lock-guarded summaries per persona scope for cross-session knowledge sharing.
6.  **`agent_memories`**: Stores key-value persistent memories for the agent.

---

## 📜 HTML Trace Trajectory Viewer (`LcmHtmlReporter`)

When debugging complex multi-turn trajectories, developers can review an interactive browser trace:
*   **Source File**: `src/kesoku/gateway/chatbot/lcm_reporter.py`
*   **Mechanism**: The reporter reads the session messages database, formats them into a dark-mode styled HTML page, and saves it inside the session's workspace.
*   **Trigger**:
    *   In the CLI: Generate the trace URL path by running `/lcm` or `/context` command commands.
    *   In Discord: Click the `📜` (View Trajectory) button on the message header to stream the generated HTML file directly.

---

## 🧪 Running Unit & Integration Tests

Kesoku uses `pytest` and `pytest-asyncio` for testing.

### Execute the Test Suite
Ensure all development packages are synced, and run tests via `uv`:

```bash
# Run all unit tests
uv run pytest

# Run tests with output logs enabled
uv run pytest -s
```

### Key Areas Tested
*   `tests/agent/test_llm.py`: Validates translation logic for user attachments and multi-tool parallel calls payloads.
*   `tests/gateway/chatbot/`: Verifies command registries, markdown table rendering, and platforms event listeners.
*   **Database Lock Tests**: Checks for self-healing locks and CAS (Compare-And-Swap) updates for cross-session contexts inside SQLite connection pools.
