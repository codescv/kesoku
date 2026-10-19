# Kesoku System Design Document (`DESIGN.md`)

Kesoku is a modular autonomous AI coding and chat agent built around a stateless Pub/Sub Gateway broker, per-session asynchronous workers (`SessionWorker`), SQLite persistence, hierarchical context compaction, and a coding-first tool suite.

For detailed user and developer guides, see [`docs/en/dev/architecture.md`](en/dev/architecture.md), [`docs/en/dev/agent-cycle.md`](en/dev/agent-cycle.md), and [`docs/en/user/configuration.md`](en/user/configuration.md).

---

## 1. Core Architecture

- **Stateless Pub/Sub Gateway (`src/kesoku/gateway/gateway.py`)**: Persists all `Message` records (`user`, `assistant`, `tool`, `system`) to SQLite and broadcasts them to filtered async listeners.
- **Agent Dispatcher & `SessionWorker` (`src/kesoku/agent/agent.py`)**: Spawns one `SessionWorker` task per active `session_id`. Tool calls execute atomically, while incoming user messages queued during LLM generation trigger thought interruption and pivot the turn cleanly.
- **Turn Executor (`src/kesoku/agent/turn_executor.py`)**: Orchestrates per-turn context assembly, automatic watermark compaction, implicit/explicit Gemini & Claude prompt caching, concurrent tool execution, post-tool/stop hooks, and turn metrics logging (`llm-turn-{idx}.log.yaml`).

---

## 2. Coding & Compaction Architecture (P0 / P1 / P2)

### 2.1 Phase 1 (P0): Output Truncation, File Tools & Static Skill Index

1. **Middle-Out Head+Tail Truncation (`src/kesoku/utils/text.py`)**:
   - `truncate_middle(text, max_chars, marker)` preserves both the head (command invocation, headers) and tail (tracebacks, test summaries, exit status) of long outputs (`head_ratio=0.4` for shell/history previews, `0.5` balanced default).
   - Used across live `run_shell_command` output, historical tool output previews (`prepare_history_for_llm`), compaction serialization, and chatbot error messages.
2. **Dedicated `read_file` & Enhanced `update_file` (`src/kesoku/agent/tools/file.py`)**:
   - `read_file(file_name, offset=1, limit=500)`: Line-numbered output (`  12: ...`), per-line and total character caps, binary sniff guard, directory listing hint, and fuzzy path suggestions on `FileNotFoundError`.
   - `update_file`: Expands `$STAGING_DIR` and `$AWD` variables, auto-strips accidental `read_file` line-number prefixes from `old_content`, reports nearest-match line numbers and diffs when `old_content` is not found, reports all match line numbers when `replace_all=False` hits multiple occurrences, and returns a compact unified diff (`<= 40` lines) on every successful edit.
3. **Static `# Available Skills` Prompt Index (`src/kesoku/agent/prompt.py`, `src/kesoku/agent/skills.py`)**:
   - `SkillManager.build_skills_index_block()` compiles a sorted one-line-per-skill index (`- **[name]**: description`) into the system prompt at session creation, allowing the model to invoke `use_skill(skill_name)` directly without spending a turn calling `list_skills()`.

### 2.2 Phase 2 (P1): Continuation-Oriented Compaction & State Snapshots

1. **Structured Continuation Summary Schema (`src/kesoku/agent/compressor.py`)**:
   - Level-0 and Level-1+ summaries use a continuation-focused JSON schema rendered into markdown sections:
     - `Summary` (current state & goal)
     - `Key Decisions & Rationale`
     - `Dead Ends (Do Not Repeat)` (approaches/commands/hypotheses that were tried and failed or were rejected)
     - `Pending State` (next step, active background jobs/PIDs/ports, modified files, git branch/commit, open blockers)
     - `External Files` (`path (description)` entries outside `$STAGING_DIR`)
2. **Water-Fill `<recent_user_messages>` Allocation**:
   - `water_fill_allocation` distributes `recent_user_messages_total_chars` (`12,000` default) across up to `recent_user_messages_limit` (`100`) user messages from compacted turns, with a per-message ceiling (`recent_user_message_ceiling_chars = 6,000`), keeping short prompts verbatim and middle-truncating oversized ones.
3. **Deterministic Per-Turn State Snapshots (`src/kesoku/agent/state_snapshot.py`)**:
   - Frozen once per turn in `TurnExecutor._snapshot_cache` to keep the prompt prefix byte-identical across all tool iterations in a turn:
     - `<context name="running_jobs">`: Active and recently finished background jobs from `ActiveJobsRegistry.snapshot(session_id)`.
     - `<context name="staging_artifacts">`: User-created artifacts in `$STAGING_DIR` (injected once the session has summary nodes; excludes internal `llm-turn-*`, `tool_*`, `err_*`, `background_job_*`, `subagent_*`, `notes.md`, `repo_rules.json`, and `context/`).
4. **User-Steered `/compact [instructions]` (`src/kesoku/cli/commands.py`)**:
   - Compacts all eligible turns down to `protect_tail_turns=1` (falling back to `0` when needed) and passes optional user `focus_instructions` into the Level-0 summarization prompt.

### 2.3 Phase 3 (P2): Job Lifecycle, Repo Rules & Hooks, and Sub-Agent Delegation

1. **`manage_job` Tool (`src/kesoku/agent/tools/shell.py`)**:
   - `manage_job(action="list" | "status" | "kill", job_id=..., tail_chars=2000)` provides structured inspection and process-group termination (`os.killpg(pid, SIGTERM)` -> `2.0s` grace -> `SIGKILL`) for background jobs started by `run_shell_command`.
   - Jobs intentionally killed via `manage_job(action="kill")` are recorded in `ActiveJobsRegistry._killed_by_agent` so `monitor_background_job` suppresses redundant `[System Alert]` wake-up messages.
2. **Touch-Driven `AGENTS.md` / `GEMINI.md` Discovery & Hooks (`src/kesoku/agent/repo_rules.py`, `src/kesoku/agent/hooks.py`)**:
   - `RepoRulesManager` walks from touched file/command paths up to `$AWD`, discovering `AGENTS.md` and `GEMINI.md` files (persisted per session in `$STAGING_DIR/repo_rules.json`).
   - Newly discovered rules are appended once to the triggering tool result (`[repo rules discovered: <rel_path>]`) and rendered on subsequent turns as `<context name="repo_rules">`.
   - When `repo_rules_stop_reminder = true`, if a turn modified files inside a discovered rules subtree (outside `$STAGING_DIR`), `TurnExecutor._handle_final_response` injects a one-shot reminder before the turn finishes.
   - Configurable `[[hooks.post_tool_use]]` and `[[hooks.stop]]` commands allow automated lint/test gates.
3. **Sub-Agent Delegation (`src/kesoku/agent/tools/subagent.py`, `src/kesoku/agent/subagent.py`)**:
   - `delegate_subagent(task, context_hint="", tools=None, max_turns=8)` runs an isolated `SubagentRunner` loop with its own ephemeral message list, read-only tool subset (`read_file`, `run_shell_command`, `list_skills`, `use_skill` by default), foreground-only shell execution (`subagent_shell_timeout_seconds = 120`), and a `<= 1,500`-char report cap.
   - Full sub-agent trajectories are written to `$STAGING_DIR/subagent_<id>.log.yaml`.
4. **Role-Isolated Markdown Memory & Progressive Disclosure (`src/kesoku/agent/memory_store.py`, `src/kesoku/agent/tools/memory.py`)**:
   - **Per-Role Isolation (`$MEMORY_DIR`)**: Durable facts, decisions, and references are curated as self-contained `.md` files with YAML frontmatter `description` under `$AWD/roles/<role>/memories/` (exposed to `read_file` and `update_file` via `$MEMORY_DIR`).
   - **Structural Breadcrumb Chunking & Incremental SQLite Index (`memory_files`, `memory_chunks`)**: Each `.md` file is incrementally synchronized by `mtime` + SHA-256 `content_hash` and split by Markdown headings with ancestor breadcrumbs (`# rel_path (description)` + `# H1 > ## H2 ...`).
   - **Hybrid File-Level Ranking**: `search_role_memories` ranks files by `MAX(chunk_cosine_similarity) + lexical_boost` over a multi-turn context query (`render_context_query` on the last 2 user turns).
   - **Progressive Disclosure (`<memory>`) & `memory_search`**: `TurnExecutor` injects only matched file paths and 1-line descriptions (`<memory>...</memory>`), and the `memory_search(query, limit)` tool returns file paths + descriptions so the agent calls `read_file("$MEMORY_DIR/...")` on demand.

---

## 3. Evaluation Benchmark (`benchmarks/kesoku_bench/`)

`kesoku-bench` is a 16-task reproducible benchmark suite (`A1`–`A4` file/diff editing, `B1`–`B4` output truncation & investigation, `C1`–`C4` multi-turn compaction survival, `D1`–`D4` skills, `/compact`, background jobs, and `AGENTS.md` conventions) with deterministic behavioral graders, `MeteredLLM` cost/token accounting, and stage ablation support (`baseline`, `p0`, `p1`, `p2`).
