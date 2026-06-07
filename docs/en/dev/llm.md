# LLM Backends & Extensibility

Kesoku decouples the core agent reasoning loop from specific LLM provider SDKs by using a standardized template interface. This page details the `BaseLLM` abstract base class and how to extend it.

---

## 🏗️ The Base Interface (`BaseLLM`)

The interface is defined in `src/kesoku/agent/llm.py` and implements the **Template Method Pattern**:

```python
class BaseLLM(ABC):
    
    async def generate(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        history: list[Message] | None = None,
        tools: list[Callable] | None = None,
        cached_content: str | None = None,
    ) -> LLMResponse:
        # 1. Translate flat Messages list to provider-neutral LLMTurns
        turns, resolved_system_prompt = history_to_turns(history, prompt, system_prompt)
        
        # 2. Call abstract methods implemented by subclass
        native_input = self._build_native_input(turns, resolved_system_prompt, tools, cached_content)
        raw_res = await self._call_llm(native_input)
        return self._parse_native_response(raw_res)

    @abstractmethod
    def _build_native_input(self, turns: list[LLMTurn], system_prompt: str | None, tools: list[Callable] | None, cached_content: str | None) -> Any:
        pass

    @abstractmethod
    async def _call_llm(self, native_input: Any) -> Any:
        pass

    @abstractmethod
    def _parse_native_response(self, raw_response: Any) -> LLMResponse:
        pass
```

---

## 📦 Intermediate Representation (IR)

Before sending history to a model, Kesoku converts database `Message` records into provider-neutral turns:

*   **`LLMTurn`**: Represents a single conversational turn. Contains a `role` (`user` or `assistant`) and a list of `LLMBlock` instances.
*   **`LLMBlock`**: Represents atomic parts of a turn:
    *   `TextBlock`: Plain text prompt or response.
    *   `ThoughtBlock`: Internal model reasoning/chain-of-thought (e.g. Gemini `thinking_config`).
    *   `ImageBlock` / `DocumentBlock`: Attachment file bytes and mime type mappings.
    *   `ToolCallBlock`: Request to run a registered tool.
    *   `ToolResultBlock`: Output returned by a tool.

---

## 🛠️ Concrete Implementations

### 1. `GeminiLLM` (Google GenAI SDK)
*   **Authentication**: Supports `api_key` mode (reading keys directly) and `vertex` mode (leveraging Application Default Credentials on GCP).
*   **Thinking Level**: Maps `"thinking_level"` (minimal, low, medium, high) into corresponding token budgets on `types.GenerateContentConfig(thinking_config=...)`.
*   **Context Caching**: If the history size exceeds `context_caching_threshold` (default: 4K tokens), it calls `client.aio.caches.create` to create a TTL-guarded context cache on Vertex AI or Gemini API, reducing prompt costs.

### 2. `ClaudeLLM` (Anthropic Vertex API)
*   **Alternating Turns Alignment**: Anthropic requires strict user/assistant alternation. The translator automatically merges consecutive user text and tool outcomes into a single user turn, and consecutive assistant thoughts/tool calls into a single assistant turn.
*   **Tool Converter**: Parses Python docstrings and type annotations and compiles them into Anthropic-spec tool schemas.

---

## 🚀 Adding a New LLM Backend

To add a new LLM provider (e.g. `OllamaLLM` or `OpenAILLM`):

### Step 1: Subclass `BaseLLM`
Create a new file or add a class in `src/kesoku/agent/llm.py` implementing the three abstract methods:

```python
class OpenAILLM(BaseLLM):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = AsyncOpenAI(api_key=api_key)

    def _build_native_input(self, turns: list[LLMTurn], system_prompt: str | None, tools: list[Callable] | None, cached_content: str | None) -> Any:
        # Map turns to OpenAI Messages format (role, content blocks)
        # Convert Python Callables to OpenAI JSON tool schemas
        ...

    async def _call_llm(self, native_input: Any) -> Any:
        return await self.client.chat.completions.create(**native_input)

    def _parse_native_response(self, raw_response: Any) -> LLMResponse:
        # Standardize OpenAI choices output, tool calls, and token usage into LLMResponse
        ...
```

### Step 2: Register in the Factory
Update the loader factory inside `src/kesoku/agent/llm.py` (or wherever LLM instances are initialized based on `config.toml`) to recognize your new model config options.
