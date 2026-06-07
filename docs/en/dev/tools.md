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
*   If any arguments are missing due to LLM parsing errors or truncation, it throws a descriptive validation error rather than attempting execution.

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
