# Memory, Database Design & Lossless Context Management (LCM)

This document outlines the architecture for Kesoku's memory and context management systems. It details the structured SQLite persistence layers, the dynamic prompt injection lifecycle, and how the Local Context Management (LCM) system optimizes long conversation histories without losing context.

---

## 1. Pitfalls of Previous Designs

### 1.1 Flat Markdown Files (`Progress.md`, `Agent.md`)
1. **Full-Overwrite Hazard**: When updating progress for a single project, LLM-generated full-file rewrites are highly prone to "tunnel vision," leading to accidental truncation or deletion of unrelated projects in the same file.
2. **Key Duplication & Drift**: Lacking database constraints, the LLM often creates duplicate, overlapping keys (e.g., `standard_japanese`, `standard_japan`, `标日学习`) for the same entity.
3. **Weak Rule Enforcement**: System rules (such as *"Always run with `uv run`"*) stored in a flat file rely on the LLM proactively loading and reading them, which lacks the enforcement of a hard constraint.

### 1.2 Memory V1 Full Prompt Injection
Bloating the system prompt with everything (preferences, progress, rules) on every turn quickly hits the context window limit and causes **Attention Distraction**—where the LLM ignores crucial rules because it is overwhelmed by irrelevant historical summaries.

---

## 2. Structured Long-Term Memory (SQLite `agent_memories`)

To ensure transactional safety and clean boundaries, Kesoku uses a structured SQLite persistence layer (`agent_memories` table):

```sql
CREATE TABLE IF NOT EXISTS agent_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,         -- 'progress', 'user_preferences', 'memo'
    key TEXT NOT NULL,              -- unique snake_case identifier (e.g. 'standard_japanese')
    title TEXT NOT NULL,            -- human-readable label
    content TEXT NOT NULL,          -- Markdown or JSON content (max 500 chars)
    updated_at REAL NOT NULL,       -- UNIX timestamp float
    role TEXT NOT NULL DEFAULT 'default', -- active persona scope
    embedding BLOB,                 -- binary vector embedding
    UNIQUE(category, key, role)     -- safe upsert/integrity constraint
);
```

### 2.1 Category Descriptions & Scoping
*   **`progress`**: Used for tracking project work and development milestones. These are **persona-isolated** and bound to the active channel persona.
*   **`user_preferences`**: Stores user details, speech/TTS pronunciation guidelines, personality background, and style preferences. These are **persona-isolated** and bound to the active channel persona.
*   **`memo`**: Custom user-defined notes or memories. These are **persona-isolated**.

### 2.2 Security & Write Lifecycles
*   **User Preferences & Rules**: **Protected (Read-Only for Agent)**. To prevent model hallucinations from fabricating or overwriting preferences, these are only modifiable by the user.
*   **Progress**: **Collaborative (Read-Write for Agent)**. The agent is allowed to write and update these records atomically via tool calls (`INSERT OR REPLACE` mapping), avoiding any overwrite hazards.

---

## 3. Dynamic Prompt Injection Lifecycle

To maximize focus, Kesoku prepends operational memory and preferences to the **latest user message** rather than bloating the global system prompt.

### 3.1 Injection Rules
*   **Bootstrap Turn** (First turn of a session, or idle resumption > 30 minutes):
    *   Prepend **Sync Guidelines** and **User Preferences** to the latest user message.
    *   *Sync Guidelines* inform the LLM that it is active under `role="{active_role}"` and has the `view_chat_history_summary` tool to fetch global timelines if the user refers to external channels.
*   **Regular Turn**:
    *   Prepend **User Preferences** only. Avoids cluttering the prompt with guidelines once the conversation is active.

### 3.2 Injection Template Example
```markdown
[Background Context: Sync Guidelines]
======
# Passive Synchronization Guidelines:
- 💡 You are playing the active persona role: {active_role}.
- 💡 You have access to the `view_chat_history_summary` tool, which retrieves a consolidated chat history summary and chronological timeline of recent events across active threads/channels.
- 💡 If the user's current request below refers to external threads, other chats, or events you cannot locate in this session's history, you MUST call `view_chat_history_summary` to read the global context and synchronize before providing a response.
======

[User Preferences]
- Preferred Programming Language: Python
- Code Style: PEP 8 compliant, explicit type hints
- Preferred Test Framework: pytest with uv run

[Current Request]
{original_user_message}
```

---

## 4. Context Compaction & Session History Management

Long conversational threads cannot be kept in their raw transcripts without hitting context limits or causing attention drift. Kesoku uses a custom hierarchical turn-based compaction manager to compress history while retaining searchable retrieval capabilities.

### 4.1 Hierarchical Turn-Based Compaction (`HistoryCompressor`)
1. Before every LLM inference step, the agent checks whether compaction is needed. Two strategies are selectable via `[agent].compression_strategy`:
    *   **`chunk`** (legacy, default): as soon as the middle turns (excluding the protected head and the last `protect_tail_turns` turns) accumulate `base_node_turns` turns **and** `base_node_min_tokens` tokens (legacy fallback estimator), they are compressed into a Level-0 node. The raw context therefore oscillates between the protected tail and tail + ~25K estimated tokens.
    *   **`watermark`**: the whole prompt (system prompt + tool declarations + scaffold + uncompressed history) is estimated with a calibrated estimator (~0.25 tokens/ASCII char, ~0.9 tokens/non-ASCII char, fitted against Gemini `count_tokens`). Nothing happens until it exceeds `context_high_watermark`; then the oldest unprotected turns are summarized in chunks of ~`watermark_chunk_tokens` (at least `watermark_chunk_min_turns` turns) until the estimate drops below `context_low_watermark`. The tail is protected by both `protect_tail_turns` and `protect_tail_min_tokens`. Compaction is skipped when less than one chunk of tokens is compressible (anti-thrash), and config validation requires `protect_tail_min_tokens < context_low_watermark < context_high_watermark`. Summary length budgets scale with the source size (~8% of source tokens, 1200–3000 chars for Level-0; half of the children's length, up to 4000 chars, when merging).
2. Summaries are generated with the dedicated context-compression model (`[agent].context_llm` / `[gemini|claude].context_model_name`).
3. A summary node is a *continuation* summary with four sections: **Timeline**; **Dead Ends** (approaches tried and failed, reverted or rejected, with the reason, so they are not retried); **Pending State** (`Unfinished` work with its next step, `Constraints` the user set that are still in force — quoted verbatim when short — and `Key files` later work depends on); and **Learnings** (tool & skill execution insights only). When merging summaries, dead ends are deduplicated, pending items that a later summary shows as done are dropped, later rules supersede conflicting earlier ones, and a constraint that is still in force is never dropped. `notes.md` (`<context name="notes">`) remains the agent-maintained source of truth for current rules. Legacy "User Asks" / "Tools & Skills" sections of old nodes are stripped before being shown to the LLM. Level-0 summaries get 1500–3000 chars.
4. In SQLite, the source messages are updated to link to the new Level-0 summary node.
5. If Level-L summary nodes accumulate to $2 \times K$ nodes, the oldest $K$ nodes are consolidated and merged into a Level-L+1 summary node (where $K$ is the configured `context_consolidation_k`).
6. The active history context is assembled dynamically: uncompressed protected head turns, a scaffold message, a scaffold acknowledgment, then every remaining turn that is not compressed. A turn that carries a `summary_node_id` is represented only by the scaffold, wherever it sits (manual compaction may reach into the tail). The scaffold starts with a notice that summaries are historical and that `<context name="notes">` and recent messages win on conflict, followed by the summary forest and a `<recent_user_messages>` block with the last `recent_user_messages_limit` verbatim user messages from compressed turns. Their size is allotted by **max-min water-filling** over a pool of `recent_user_messages_total_chars` (default 12000) with a per-message ceiling of `recent_user_message_ceiling_chars` (default 6000): short messages keep their full text and donate the rest of their share, so a long specification survives intact while an oversized paste is shortened in the middle. The scaffold only changes on compaction, so it stays cache-friendly.
6b. Live state is injected per turn next to `notes`, not in the scaffold: `<context name="running_jobs">` (background jobs of the session from `ActiveJobsRegistry`, with process group, log paths and the exact `kill -TERM -- -<pgid>` stop command, plus recently finished jobs) and, once the session has summary nodes, `<context name="staging_artifacts">` (a deterministic listing of files the agent created in `$STAGING_DIR`, internal logs omitted). Both are computed once per turn so every LLM call of the turn sees byte-identical text (implicit prefix caching).
6c. `/compact [instructions]` forces compaction of every uncompressed turn between the protected head and tail, regardless of the watermark; the optional text is a focus directive for the new Level-0 summaries (in consolidation it only steers prioritisation, never drops still-open items). It refuses while a turn is in progress, and all compaction paths of a session are serialised by a per-session lock.
7. Historical (completed-turn) tool calls of `update_file` (and the removed legacy `update_session_notes`, still present in old history) have long string arguments truncated to 300 characters; tool results keep the existing 500-character cap.

### 4.2 Search & Retrieval Tools
To recall specific details of compacted sections or search across the history of the active role persona, the agent can call the following tools:
*   **`memory_grep(query, start_time, end_time, limit)`**: Search active memories and past chat messages for the current role matching the query (keyword match or wildcard `*`). Supports optional time-range filtering.
*   **`memory_search(query)`**: Perform semantic (vector) search against active memories for the current role.
*   **`view_message(message_id)`**: Retrieve the complete content of a specific historical chat message by its database ID.

### 4.3 Complementary Coexistence
*   **Active Memory System (AMS)** (`agent_memories` table): Stores long-term, structured, semantic knowledge and constraints across categories: `user_preferences`, `progress`, and `memo`.
*   **Hierarchical Compaction System** (`summary_nodes` table): Manages short-term, operational conversational history in a token-efficient, hierarchically consolidated manner.
