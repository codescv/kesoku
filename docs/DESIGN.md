# Kesoku System Design Document (`DESIGN.md`)

Kesoku is a modular autonomous AI coding and chat agent built around a stateless Pub/Sub Gateway broker, per-session asynchronous workers (`SessionWorker`), SQLite persistence, hierarchical context compaction, and a coding-first tool suite.

For detailed user and developer guides, see [`docs/en/dev/architecture.md`](en/dev/architecture.md), [`docs/en/dev/agent-cycle.md`](en/dev/agent-cycle.md), and [`docs/en/user/configuration.md`](en/user/configuration.md).

---

## 1. Core Architecture

- **Stateless Pub/Sub Gateway (`src/kesoku/gateway/gateway.py`)**: Persists all `Message` records (`user`, `assistant`, `tool`, `system`) to SQLite and broadcasts them to filtered async listeners.
- **Agent Dispatcher & `SessionWorker` (`src/kesoku/agent/agent.py`)**: Spawns one `SessionWorker` task per active `session_id`. Tool calls execute atomically, while incoming user messages queued during LLM generation trigger thought interruption and pivot the turn cleanly.
- **Turn Executor (`src/kesoku/agent/turn_executor.py`)**: Orchestrates per-turn context assembly, post-turn automatic watermark compaction, implicit/explicit Gemini & Claude prompt caching, optional real-time token streaming (`_StreamPreviewPublisher`), concurrent tool execution, post-tool/stop hooks, and turn metrics logging (`llm-turn-{idx}.log.yaml`).

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

### 2.4 TTFT Optimizations & Per-Channel LLM Streaming

1. **Active-Only SQLite History Loading (`src/kesoku/db/manager.py`, `src/kesoku/agent/history.py`, `src/kesoku/agent/turn_executor.py`)**:
   - `DatabaseManager.get_session_history(..., active_only=True)` adds `AND (summary_node_id IS NULL OR role IN ('user', 'system'))` in SQL so bulky `ASSISTANT` and `TOOL` rows of already-compacted turns are skipped at the storage layer, while `USER`/`SYSTEM` anchors (retaining their `summary_node_id`) preserve exact logical turn indices, `protect_front_turns`, and `<recent_user_messages>` water-fill content.
2. **Immutable Tool-Output Disk Archive Guard (`src/kesoku/agent/history.py`)**:
   - `prepare_history_for_llm` checks `os.path.exists(filepath)` before calling `os.makedirs` and writing `{staging_dir}/tool_results/{safe_id}.txt` (or `tool_errors/`), avoiding redundant synchronous disk writes on every subsequent turn while keeping preview strings byte-identical for implicit prefix caching.
3. **Role Pass-Through & SDK Client Pooling (`src/kesoku/gateway/chatbot/base.py`, `src/kesoku/context.py`)**:
   - `Chatbot.preprocess_message` passes `role=current_role or None` into `build_sys_prompt(...)` so default-role turns skip reopening `DatabaseManager`.
   - `KesokuContext.get_llm` caches `BaseLLM` instances by `(provider, model_name, thinking_level, use_context_compression)` (clearing automatically on config reload) so underlying `genai.Client` and `AnthropicVertex` HTTP/TLS keep-alive pools persist across turns.
4. **Post-Turn Auto-Compaction (`src/kesoku/agent/turn_executor.py`)**:
   - `TurnExecutor.process_turn` assembles existing summary scaffolds immediately (`_assemble_scaffold_history`) before invoking the LLM and defers `_run_auto_compaction` until **after** the final assistant response has been persisted and delivered (invalidating stale explicit context caches post-turn, with a `_compaction_pending` pre-turn fallback safety net if post-turn compaction is skipped due to queued pivot messages or raises an error).
5. **Streaming LLM Inference & Discord Live Previews (`src/kesoku/config.py`, `src/kesoku/agent/llm.py`, `src/kesoku/gateway/gateway.py`, `src/kesoku/gateway/chatbot/discord/adapter.py`)**:
   - **Configuration & Resolution**: `[gemini].stream` and `[claude].stream` (default `false`) plus per-channel `thinking_level` and `stream` overrides on `DiscordChannelOverride` (matched across direct channels and child threads via `parent_channel_id` / `parent_channel_name`, gated on `Gateway.has_stream_handler(chatbot_id)`).
   - **Non-Blocking Stream Pump & UTF-8 SSE Decoding**: `_drain_blocking_stream` runs `GeminiLLM` (`generate_content_stream`) and `ClaudeLLM` (`messages.stream`) iterators in a background thread feeding an `asyncio.Queue`, emitting text-only `(delta, accumulated)` callbacks while reconstructing full `LLMResponse` objects (thoughts, `thought_signature` bytes, tool calls, and token usage). `_ensure_genai_utf8_streaming` patches `google.genai`'s `HttpResponse._iter_response_stream` so `requests.Response` SSE streams over Vertex AI mTLS (`AuthorizedSession`, where `Content-Type: text/event-stream` lacks a `charset` parameter) split lines on raw bytes before decoding UTF-8 instead of defaulting to `ISO-8859-1` / splitting on `0x85` (`U+0085`).
   - **Throttled Preview & In-Place Final Delivery**: `_StreamPreviewPublisher` publishes the first text chunk immediately via `Gateway.publish_stream`, throttles subsequent updates (`1.5s`, single non-blocking in-flight task), and calls `discard()` on tool calls, retries, or interruptions. `DiscordChatbot` renders `<= 1900`-char `▌` previews (auto-closing odd code fences), edits Chunk 0 in-place when the final text arrives without duplication, and skips intermediate `💭` thought send-then-delete REST round-trips on final responses (`metadata["final_response"] = True`).
6. **Generic Deferred Media Protocol (`src/kesoku/gateway/chatbot/deferred_media.py`, `src/kesoku/gateway/chatbot/base.py`, `src/kesoku/gateway/chatbot/discord/adapter.py`, `src/kesoku/gateway/chatbot/wechat/adapter.py`)**:
   - **Capability Plumbing**: Adapters opt in via `Chatbot.supports_deferred_media = True` (enabled on `DiscordChatbot` and `WechatChatbot`; disabled on `CLIChatbot`, `GoogleChatChatbot`, and `CronjobChatbot`). `TurnExecutor` resolves `Gateway.supports_deferred_media(chatbot_id)` onto `ToolContext.supports_deferred_media` (refreshing on `drain_queue_and_pivot`), and `run_shell_command` injects `KESOKU_DEFERRED_MEDIA=1` only when `not foreground_only and supports_deferred_media`.
   - **Sidecar Resolution & Non-Blocking Watchers**: Before fuzzy path matching (`resolve_outbound_path`), `_attempt_send_attachment` inspects `<target>.pending.json` and `<target>.error.json` (including `$STAGING_DIR` recursive basename lookup). Ready files (`os.path.isfile(target)` and `mtime >= created_at - 0.05` when `.pending.json` is present) send synchronously; fast `.error.json` failures surface a sanitized user notice without triggering `DeliveryAbortedError` self-healing; in-flight `[file:]` attachments emit an optional placeholder (`send_deferred_placeholder`), persist `message.metadata["deferred_media"]` via `copy.deepcopy`, and spawn a per-session `asyncio.Task` watcher (`0.5s -> 2.0s` backoff, `job_id` ownership checks, dead-PID grace period, and clamped `[10s, 600s]` timeout).
   - **Inline In-Order Voice Delivery (`_voice_inline_wait_sec = 60.0`, `_voice_inline_anchor_delay_sec = 1.0`)**: Pending `[voice:]` segments wait inline up to `min(_voice_inline_wait_sec, remaining_marker_timeout)`—silently for the first `1.0s` (Phase A, avoiding reaction flicker on near-ready clips) and then attaching a `🎤` reaction anchor on the preceding text chunk via `send_deferred_placeholder` (Phase B; `None` on WeChat)—so synthesized audio posts in exact segment order before trailing text/stickers/`ui` (`🎤` removed via `discard_deferred_placeholder`), `.error.json` and dead-PID/missing-marker failures swap `🎤 -> ⚠️` and post an in-order notice without spawning a watcher, and only hung workers still alive after the full budget fall back to `_start_deferred_tracking` reusing the existing `🎤` handle. Trade-offs: the per-chatbot subscriber loop is serial so an inline wait can head-of-line block other sessions/users sharing that `chatbot_id` for up to `60s` (`_voice_inline_wait_sec` remains an overridable class attribute for multi-user adapters such as WeChat), and `deferred_media` metadata is persisted only after budget expiry (a service restart mid-wait re-renders the message once ready and may leave a stale `🎤` on the pre-restart chunk).
   - **Lifecycle Split & Startup Recovery**: `Chatbot.stop()` cancels only in-memory watcher tasks while preserving `pending` metadata and detached worker processes so `_recover_deferred_media()` (scanning the last 24h on startup after `wait_until_ready()`) restores Discord `PartialMessage` handles and completes delivery across service restarts; explicit session clears (`clear_session_by_channel`, `Gateway.delete_session`) terminate worker PIDs/transient units and mark entries `cancelled`. `analyze_media` also awaits active `.pending.json` targets before reading.
7. **Persistent Discord Button Cards with Per-Option Re-Click (`src/kesoku/gateway/chatbot/discord/persistent_ui.py`)**:
   - Single-select button cards (`not uses_select(spec)`) disable and highlight (`ButtonStyle.success`) only the clicked option `idx`, keeping unpicked buttons clickable (`spec["chosen"]` accumulates indices in click order, `spec["answered"]` joins picked labels with `；`, and the embed stays blurple until closed).
   - Re-clicking an already-picked `idx` returns an ephemeral `✓ Already answered` notice without reposting; clicking a reroll option (`再帮我想` / `换一批` / `换一组`) or picking all regular options sets `spec["closed"] = True` (dropdowns close on first submit; legacy rows without `closed` fall back to `bool(answered)`).

### 2.5 App Channel, Per-Mode Memory Policy & `confirm_only` Proposals (v0.3.0)

1. **`AppChatbot` (`src/kesoku/gateway/chatbot/app/`)**: a fourth gateway subscriber (`chatbot_id = "app"`, one channel
   `app:<session_id>` per session). `EngineLink` (`link.py`) keeps a single outbound WebSocket to a relay
   (`Authorization: Bearer <token from env>`, `engine.hello{boot_id, conn_seq}`, 25 s ping, seamless `relay.rotate`).
   Every client request is a V1 envelope (`protocol.py`) answered by exactly one reply (`<type>.result` unless
   overridden, or `error{code}`), dispatched in `adapter.py` / `commands.py` / `persona.py` / `proposals.py` /
   `memory_policy.py`.
2. **Sequenced outbox (`outbox.py`, `app_outbox`)**: only the adapter allocates per-session `seq` for persisted session
   events (`message.user`, `message.final`, `turn.event`, `media.*`, `state.sync`, `session.updated`,
   `memory.proposal*`); `sync.replay` re-sends them with their original ids, then `sync.done` (or `sync.reset` when the
   cursor predates retention). Streaming previews (`stream.delta` / `stream.clear`, code-point offsets, per-round
   buffers) are never sequenced. `stream_min_interval_sec` is registered per chatbot (`Gateway.get_stream_min_interval`).
3. **Rich fenced blocks (`utils/ui_markup.py`)**: ```` ```bilingual ```` and ```` ```card ```` (free-form `kind`) parse into
   structured segments only when `parse_message_content(..., rich_blocks=True)`; every other platform receives a clean
   text fallback (media tags inside are still delivered).
4. **Generic per-message metadata (`turn_executor.py`)**: `llm_override` (validated against the channel's
   `allowed_models`), `mode_prefs` (injected as `<context name="app-prefs">`), `ignore_context_globs` (render-time
   filter; memory policy still sees every block).
5. **Four-state memory policy (`context_policy.py`)**: `full < real_facts_only < confirm_only < isolated`
   (+ `[agent] memory_confirm_contexts`). `resolve_memory_policy` first asks the registered
   `KesokuContext.memory_policy_override_provider`; a non-empty answer is authoritative (no rank merge). The app
   adapter answers with `app_mode_memory_prefs[current mode]`, the mode being read generically from
   `$STAGING_DIR/game/games.json`.
6. **`confirm_only` proposals (`memory_proposals.py`)**: the consolidator splits a window into an automatic pass and a
   confirmation pass; confirmation-pass ops are filtered against `memory_proposal_blocklist` and staged in
   `memory_proposals` without touching `$MEMORY_DIR`. Only `ProposalStore.confirm` calls `MemoryWriter.apply_ops`;
   `dismiss` blocklists the fact fingerprints; pending proposals expire after 7 days.

---

## 3. Evaluation Benchmark (`benchmarks/kesoku_bench/`)

`kesoku-bench` is a 16-task reproducible benchmark suite (`A1`–`A4` file/diff editing, `B1`–`B4` output truncation & investigation, `C1`–`C4` multi-turn compaction survival, `D1`–`D4` skills, `/compact`, background jobs, and `AGENTS.md` conventions) with deterministic behavioral graders, `MeteredLLM` cost/token accounting, and stage ablation support (`baseline`, `p0`, `p1`, `p2`).
