# Tools & Execution System

This guide outlines how tools are registered, validated, and executed within Kesoku's autonomous agent framework.

---

## 🛠️ Tool Registry & Decorators

Kesoku uses a centralized `ToolRegistry` pattern. Any Python function can be converted into an agent tool by decorating it with `@default_registry.register`:

```python
from kesoku.agent.tools import default_registry, ToolContext

@default_registry.register
async def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two integers.

    Args:
        a: The first integer.
        b: The second integer.
    """
    return a + b
```

### Docstring Parsing & Schema Extraction
When initializing LLM backends (like `GeminiLLM` or `ClaudeLLM`), the client reads the registered tools and parses their signatures:

*   **Descriptions**: Extracted from the function's docstring (using Google-style conventions).
*   **Argument Types**: Extracted from Python type annotations (e.g. `a: int`).
*   **JSON Schema**: Automatically generated and transmitted to the LLM (as `FunctionDeclaration` blocks for Gemini, or `Tool` schemas for Claude).

---

## ⚙️ Execution Lifecycle (`ToolRunner`)

Tool execution is managed by the `ToolRunner` class (`src/kesoku/agent/tool_runner.py`):

```text
┌────────────────────────────────────────────────────────┐
│ 1. Parse ToolRequest Arguments                         │
├────────────────────────────────────────────────────────┤
│ 2. Check Interruption callback (Pre-execution check)   │
├────────────────────────────────────────────────────────┤
│ 3. inspect.signature() validation (Ensure args match)  │
├────────────────────────────────────────────────────────┤
│ 4. Run coroutine or thread (asyncio.to_thread)         │
├────────────────────────────────────────────────────────┤
│ 5. Wrap response in MessageDTO (TOOL / TOOL_RESULT)    │
└────────────────────────────────────────────────────────┘
```

### 1. Interruption Check
Before executing the tool, `ToolRunner` evaluates the `is_interrupted` callback:

```python
if is_interrupted and is_interrupted():
    # Abort execution immediately and return aborted status message
```
This ensures that if a new user prompt arrived in the queue while the agent was in a thinking step preceding this tool, execution is skipped cleanly.

### 2. Parameter Validation
Using `inspect.signature(tool_func)`, the runner checks that all required arguments (parameters without default values) were emitted by the LLM:

*   If a parameter named `context` exists in the signature, the runner automatically injects the current `ToolContext` object (which holds the active `gateway`, `session_id`, and `original_msg_id`).
*   JSON `null` reaches Optional parameters (`str | None`, `Annotated[int | None, ...]`) as `None` (never the string `"None"`); for non-Optional parameters a `null` is treated as absent so the default applies.
*   If any required arguments are missing (usually LLM output truncation), it returns `Missing required argument(s) for tool '<name>': <args>.` plus a hint to split large content into ≤4000-character chunks, rather than attempting execution.

### File tools (`tools/file.py`)
*   **`read_file(file_name, start_line=1, end_line=None, max_lines=500)`**: numbered (`N: `), paged view of a text file (hard caps: 2000 lines, 30000 chars, lines over 2000 chars are cut). Missing files get `Did you mean: repo/...?` suggestions (missing first-level directory prefix or same basename under the AWD); directories list their entries; binary files are refused.
*   **`update_file(file_name, old_content=None, new_content=None)`**: omit `old_content` to create a file (rejected if it exists); otherwise replace an exact block that occurs once (twice: both replaced, with a warning; more: rejected with the matching line numbers and context). `new_content` is required (`""` deletes the block). Failed matches return recovery hints: the whitespace-insensitive match with its exact lines, or the closest lines plus a `read_file` range. `N: ` prefixes accidentally copied from `read_file` are stripped automatically. Success messages report the affected line range.

### Output truncation (middle-out)
*   `run_shell_command` keeps at most `max_output_chars` (default 5000, max 30000): the preview keeps ~45% head and ~55% tail around a placeholder, and the full output is saved to `$STAGING_DIR/cmd_output_*.txt`.
*   In completed turns, `prepare_history_for_llm` shrinks tool results/errors to a 500-char head+tail preview (`history_preview`), after stripping the shell's own truncation banner so the head budget shows real output (a leading `Command failed with exit code N.` line is kept). The full text stays on disk (the shell's file is reused; otherwise a deterministic `hist_*` file per message keeps the prompt prefix cacheable). `HistoryCompressor.estimate_prepared_turn_tokens` uses the same helper.

### 3. Thread Safe Subprocesses
The runner supports both synchronous and asynchronous functions:

*   **Async Functions**: Executed directly via `await tool_func(**kwargs)`.
*   **Sync Functions**: Wrapped and run in a separate threadpool using `asyncio.to_thread(tool_func, **kwargs)` to prevent blocking the event loop.

### 4. Message Packaging
All output values and exceptions are caught, serialized to strings, and packaged into a standard database `Message` object:

*   **Role**: `MessageRole.TOOL` (`"tool"`)
*   **Type**: `MessageType.TOOL_RESULT` (`"tool_result"`)
*   **Status**: `MessageStatus.RESPONDED`
*   **Parent Link**: `parent_id` points to the triggering `tool_call` message ID, maintaining clean turn alignment.

---

## 📸 Multimodal Tools (`analyze_media`)

Kesoku provides built-in support for analyzing local multimedia files (images, audio, video) via LLM multimodal capabilities:

*   **`analyze_media(path, prompt)`**: Resolves a media file path (absolute or relative to AWD/staging), detects its MIME type, and attaches it as a native multimodal block (`ImageBlock`, `DocumentBlock`, or `MediaBlock`) to execute QA against the file.

### `disabled_tools`
`agent.disabled_tools = ["chat_search", "view_message"]` removes the listed tools from the schema sent to the LLM and refuses to execute them (`ToolRegistry.get_tools_list` / `get_tool`). The bootstrap `sync_guidelines` hint only mentions history tools that are not disabled. Together with `auto_context_search = false` this fully turns off past-chat retrieval.
