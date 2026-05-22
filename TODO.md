# Agent Features
- [x] History / context management (compaction etc)
- [x] System Prompt Design
  - Inject current date, time, timezone etc.

# Tools
- [x] Run shell command
- [x] Search tool
- [x] Use skills

# Sub Agent
- [ ] Support sub agent

# Discord
- [x] Basic Messages
- [x] Attachments (Outgoing)
- [x] Attachments (Incoming)
- [x] Multi choice UI

# Bug fix
- [x] cleanup the config not found defensive code.

# Architecture Refactor (from code review 2026-05-22)

## P0 - High Priority
- [ ] **Split `Gateway` into Broker + SessionStore + Facade** (SRP violation)
  - `gateway/broker.py`: pure pub/sub (`post`, `listen`, `Listener`)
  - `gateway/session_store.py`: session CRUD + workspace lifecycle (transactional)
  - `gateway/gateway.py`: thin facade composing both
  - Make `delete_session` transactional (DB + filesystem rollback safety)
- [ ] **Optimize SQLite connection management** in `db.py`
  - Use thread-local cached connections instead of new connection per call
  - Enable `PRAGMA journal_mode=WAL`
  - Set `PRAGMA busy_timeout=5000`
  - Enable `PRAGMA foreign_keys=ON`
  - Consider migration to `aiosqlite` to remove `asyncio.to_thread` wrapping
- [ ] **Optimize `get_session_history` performance** (currently O(N) full-load + Python-side sort)
  - Add `root_id` column to messages, populate on insert
  - Add index `(session_id, root_id, timestamp)`
  - Add `sort_phase INT` column to avoid recomputing
  - Push ORDER BY into SQL, eliminate Python-side sorting for hot path
  - Consider CTE recursive query or materialized recent-N-turns cache

## P1 - Medium Priority
- [ ] **Refactor `agent.py::_process_turn` (200+ line god method) into a state machine**
  - `PrepareStep`: history + resolve_llm + tools
  - `InferStep`: LLM call + metrics
  - `ToolDispatch`: parallel tool execution + interrupt handling
  - `FinalizeStep`: final text + turn_metrics persistence
  - `InterruptPolicy`: centralize `_drain_queue_and_pivot` logic
  - Each step ≤100 lines, independently unit-testable
- [ ] **Tame `DiscordChatbot` (941 lines) — consolidate scattered turn UI state**
  - Replace 5 scattered dicts (`_intermediate_messages`, `_typing_tasks`,
    `_header_views`, `_turn_special_items`, `_turn_special_msg`) with a single
    `DiscordTurnUIState` dataclass
  - Add unified `_release_ui_state(session_id)` cleanup method
  - Audit session deletion paths to prevent state leaks
- [ ] **Extract shared `RichChatbot(Chatbot)` base class**
  - DRY out duplicated logic between `discord.py` and `google_chat.py`:
    custom prompt building, turn-level UI state machine, attachment collection,
    thought-interruption handling, reaction emoji
  - Platform subclasses only override render methods

## Q - Code Quality
- [ ] **Fix function-level imports** (`llm.py::function_to_anthropic_tool`,
      `tools.py::_create_schema_func`) — move to module top per project style guide
- [ ] **Replace `Literal[...]` + parallel string constants with `StrEnum`**
  - Define `Role`, `MessageType`, `Status` as `StrEnum` (Python 3.11+)
  - Eliminates dual source of truth between `constants.py` and Pydantic models
- [ ] **Replace hand-rolled `_ensure_migrations` with proper schema versioning**
  - Either alembic, or minimal `schema_version` table + ordered migration scripts
  - Current ad-hoc ALTER TABLE pattern doesn't scale
- [ ] **DRY up `Message` row mapping in `db.py`**
  - Single `_row_to_message(row)` helper replacing 4× duplicated 20-line blocks
  - Future field additions only touch one place
- [ ] **Make global config singleton testable**
  - Add `autouse` pytest fixture in `conftest.py` to reset `_global_config`
  - Audit tests for cross-test pollution
- [ ] **Move hardcoded tool constants into `ShellConfig`**
  - `MAX_OUTPUT_LENGTH = 10_000`, `TIMEOUT_SECONDS = 1800` in `tools.py`
  - Expose as user-configurable settings
- [ ] **Fix `# noqa: ASYNC240` suppressions**
  - Sync `os.path.exists`/`os.makedirs` calls inside async functions block the
    event loop under load
  - Use `aiofiles.os.*` or wrap in `asyncio.to_thread`
