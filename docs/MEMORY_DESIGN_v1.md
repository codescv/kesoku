# Kesoku Memory System Design

## 1. Executive Summary
This document outlines the architecture for Kesoku's Unified Memory System. It addresses the "Full-Overwrite Hazard" and "Context Drift" common in standard flat-file LLM memory designs (e.g., direct write-back to Markdown files). By decoupling **Structured Storage (SQLite)** from **Semantic Representation (Dynamic Markdown rendering)** and applying **Framework-level System Prompt Injection**, this memory architecture ensures transactional safety, prevents memory fragmentation, and strictly enforces hard runtime rules.

---

## 2. Problem Statement & Design Objectives

### 2.1 Key Vulnerabilities in Flat-File Memory (`Progress.md`, `Agent.md`)
1. **Full-Overwrite Hazard**: When updating progress for a single project, LLM-generated full-file rewrites are highly prone to "tunnel vision," leading to accidental truncation or deletion of unrelated projects in the same file.
2. **Key Duplication & Drift**: Lacking strict database constraints, the LLM often creates duplicate, overlapping keys (e.g., `standard_japanese`, `Standard_Japanese`, `标日学习`) for the same entity.
3. **Weak Rule Enforcement**: Runtime execution rules (such as *"Always run with `uv run`"*) stored in a flat file rely on the LLM's proactive tool calls to read them. This lacks the physical force of a hard constraint.

### 2.2 Core Objectives
- **Transactional Safety**: Key-Value isolation prevents updates on Item A from corrupting Item B.
- **Strict Hard Rules**: Crucial operational rules must be injected directly into the **System Prompt** by the runtime engine, ensuring they are always active.
- **Token Efficiency**: Highly stable, slow-changing records (Profile, Rules) are dynamically injected, while volatile or large records (Progress) are retrieved on-demand via tools to save context window space.

---

## 3. Architecture Overview

The Kesoku Memory System separates memory into three core logical categories stored in a unified SQLite table, but routed differently during the session lifecycle.

```
                  +-----------------------------------+
                  |          SQLite Storage           |
                  |         `agent_memories`          |
                  +-----------------------------------+
                                    |
          +-------------------------+-------------------------+
          | (Category: 'preference')| (Category: 'rule')      | (Category: 'progress')
          v                         v                         v
+-------------------+     +-------------------+     +-------------------------+
|   User Profile    |     |  Execution Rules  |     |   Activity Progress     |
| (Static Facts)    |     | (Hard Constraints)|     |  (Project Tracking)     |
+-------------------+     +-------------------+     +-------------------------+
          |                         |                         |
          v                         v                         | (On-demand tool call)
+---------------------------------------------+               v
|        Framework-level Injection            |     +-------------------------+
|  (Assembled into LLM System Prompt on Boot) |     |  Tool: `view_progress`  |
+---------------------------------------------+     +-------------------------+
```

---

## 4. Data Storage Layer (SQLite Schema)

All structured memories are stored in the `agent_memories` table in `kesoku.db`.

```sql
CREATE TABLE IF NOT EXISTS agent_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,         -- 'progress', 'preference', 'rule'
    key TEXT NOT NULL,              -- Unified snake_case identifier (e.g. 'standard_japanese')
    title TEXT NOT NULL,            -- Human-readable label (e.g. '《标准日本语》学习进度')
    content TEXT NOT NULL,          -- JSON string or structured payload
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(category, key)           -- Enforces atomic overwrite via INSERT OR REPLACE (UPSERT)
);
```

---

## 5. Memory Governance & Lifecycles

Different categories of memory require distinct access privileges and lifecycles:

### 5.1 `preference` (User Profiles & Backgrounds)
- **Privilege**: **Protected (Read-Only for Agent)**.
- **Description**: Contains human traits (timezone, interests, job description).
- **Enforcement**: If the Agent attempts to update a protected key (e.g., `user_profile`), the tool handler rejects the execution to prevent hallucinations from corrupting user-defined profiles. Changes must be manually initiated by the User.

### 5.2 `rule` (Runtime Engineering Safeguards)
- **Privilege**: **Protected (Read-Only for Agent)**.
- **Description**: High-priority constraints (e.g., using `uv run`, outputting to `$STAGING_DIR`, activating role-play skills).
- **Enforcement**: Injected directly into the topmost segment of the System Prompt on session boot.

### 5.3 `progress` (Learning & Project Progress Tracker)
- **Privilege**: **Collaborative (Read-Write for Agent)**.
- **Description**: Current position in books, games, or development milestones.
- **Enforcement**: Written using atomic `INSERT OR REPLACE` operations to safeguard other entries. Retrieved on-demand through tool execution.

---

## 6. Execution Pipeline & Prompt Injection

### 6.1 Lifecycle Boot Sequence
When a new conversational session starts or resumes:
1. **Fetch Dynamic Context**: The Kesoku framework queries the SQLite database:
   ```sql
   SELECT category, key, title, content FROM agent_memories WHERE category IN ('preference', 'rule');
   ```
2. **Translate to Compact Markdown**: The engine renders raw JSON into a highly compact, non-redundant list format:
   ```markdown
   # 1. User Profile & Preferences
   - Name: 小张 (Japanese TTS Pronunciation: "シャオジャン")
   - Current Role: Customer Solutions Engineer at Google Ads
   - Interests: Guitar, Switch Games, Finance

   # 2. Hard Execution Constraints (CRITICAL)
   - ⚠️ You must use 'uv run' to run Python programs. Never use native 'python', 'pip' or 'pytest'.
   - ⚠️ Strictly adhere to the Staging Directory protocol: write all generated files to $STAGING_DIR.
   ```
3. **Prepend to System Prompt**: This rendered string is merged into the LLM `system_instruction` parameter prior to dispatch.

### 6.2 Preventing Key Duplication (Fuzzy Guardrail)
To prevent the LLM from creating redundant keys (e.g., `standard_japanese` and `standard_japan`), the `update_memory` tool implements **Fuzzy Key Matching**:
1. Clean the input key: `lowercase`, strip whitespace, and replace spaces with underscores.
2. Check existing keys under the category using `difflib.get_close_matches` with a threshold of `0.8`.
3. If an existing key matches, redirect the update to the existing key and output an informative warning log to the agent.

---

## 7. Migration Plan (Legacy Markdown to DB)

To transition from the legacy `memory/*.md` system:
1. Run a setup script to execute the SQLite migration (DDL creation).
2. Read legacy files:
   - Extract sections from `memory/User.md` and seed them as `preference` records.
   - Extract sections from `memory/Progress.md` and seed them as `progress` records.
   - Extract rules from `memory/Agent.md` and seed them as `rule` records.
3. Remove legacy Markdown files to maintain a single source of truth.
