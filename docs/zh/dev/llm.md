# LLM 驱动引擎与后端扩展

Kesoku 采用标准模板接口，将核心 Agent 推理循环与特定大模型提供商的 SDK 进行了解耦。本指南详细介绍了 `BaseLLM` 抽象基类以及如何对其进行扩展。

---

## 🏗️ 核心抽象接口 (`BaseLLM`)

该基类定义于 `src/kesoku/agent/llm.py` 中，采用**模板方法模式（Template Method Pattern）**规范了模型调用的生命周期：

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
        # 1. 将平面化的数据库 Message 列表转换为与平台无关的中间表达 (LLMTurn)
        turns, resolved_system_prompt = history_to_turns(history, prompt, system_prompt)
        
        # 2. 调用由子类具体实现的三个抽象接口方法
        native_input = self._build_native_input(turns, resolved_system_prompt, tools, cached_content)
        raw_res = await self._call_llm(native_input)
        return self._parse_native_response(raw_res)

    @abstractmethod
    def _build_native_input(self, turns: list[LLMTurn], system_prompt: str | None, tools: list[Callable] | None, cached_content: str | None) -> Any:
        """将标准中间表达转译为各模型厂商原生的 Payload 输入结构。"""
        pass

    @abstractmethod
    async def _call_llm(self, native_input: Any) -> Any:
        """异步调用厂商 SDK 客户端发送请求。"""
        pass

    @abstractmethod
    def _parse_native_response(self, raw_response: Any) -> LLMResponse:
        """将厂商原生的响应结构统一解析并返回标准格式的 LLMResponse。"""
        pass
```

---

## 📦 消息的中间表达形式 (IR)

在将对话历史发送给大模型之前，Kesoku 会将平面化的数据库 `Message` 记录提炼转换为与厂商无关的逻辑结构：

*   **`LLMTurn`**：代表一个完整的对话回合。它包含一个 `role`（`user` 或 `assistant`）和一组 `LLMBlock` 代码块。
*   **`LLMBlock`**：代表回合内的原子数据块：
    *   `TextBlock`：普通的用户 prompt 文本或机器人回复。
    *   `ThoughtBlock`：智能体的内部思维推理链（CoT，如 Gemini 开启思维配置时的输出）。
    *   `ImageBlock` / `DocumentBlock`：下载的附件媒体字节流及其 mime_type。
    *   `ToolCallBlock`：大模型生成的执行外部工具的请求。
    *   `ToolResultBlock`：宿主机工具执行完毕后返回的输出。

---

## 🛠️ 具体厂商实现细节

### 1. `GeminiLLM` (Google GenAI SDK)
*   **权限认证**：支持 `api_key` 模式（直接提供密钥）与 `vertex` 模式（在 GCP 环境中使用 Application Default Credentials 免密钥认证）。
*   **思维链控制**：将配置文件中的 `"thinking_level"`（minimal, low, medium, high）映射并设定至 `types.GenerateContentConfig(thinking_config=...)` 的 Token 预算属性中。
*   **上下文缓存**：当对话历史的 tokens 长度超过 `context_caching_threshold` 阈值（默认 4K tokens）时，会自动调用 `client.aio.caches.create` 在 GCP/Gemini 网关上创建 TTL 缓存，极大缩减高频长对话的提示词开销。

### 2. `ClaudeLLM` (Anthropic Vertex API)
*   **消息交替对齐**：Anthropic 接口强硬要求 `user` 和 `assistant` 的 Turn 必须严格交替出现。转译器会自动合并数据库中连续的用户文本和工具执行结果为一个 `user` 回合，以及合并连续的模型思维与工具调用为一个 `assistant` 回合。
*   **工具转换器**：解析 Python 的 Docstrings 注释和参数类型定义，动态编译为符合 Anthropic 规范 the JSON tools 结构。
*   **提示词缓存 (Prompt Caching)**：当估算的输入 tokens 长度超过 `context_caching_threshold` 阈值（默认 1024 tokens）时，会自动在 system prompt、tools 以及最后的 message 上注入 `"cache_control": {"type": "ephemeral"}` 标记，从而利用 Anthropic 的原生 Ephemeral Cache 机制降低后续交互的延迟与成本。

---

## 🚀 编写新的 LLM 后端

如果您需要接入其他模型供应商（例如 `OllamaLLM` 或 `OpenAILLM`）：

### 步骤 1：继承 `BaseLLM`
在 `src/kesoku/agent/llm.py` 中编写您的新实现类，实现三个核心抽象方法：

```python
class OpenAILLM(BaseLLM):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = AsyncOpenAI(api_key=api_key)

    def _build_native_input(self, turns: list[LLMTurn], system_prompt: str | None, tools: list[Callable] | None, cached_content: str | None) -> Any:
        # 将中间表达 LLMTurn 翻译为 OpenAI 的 messages payload
        # 将 Python Callables 转换为 OpenAI JSON tool schemas
        ...

    async def _call_llm(self, native_input: Any) -> Any:
        return await self.client.chat.completions.create(**native_input)

    def _parse_native_response(self, raw_response: Any) -> LLMResponse:
        # 将 OpenAI 的 choices、tool_calls 以及 token 消耗统一组装返回为标准 LLMResponse
        ...
```

### 步骤 2：注册到加载工厂
修改 `src/kesoku/agent/llm.py` 中的实例化逻辑（或者其他根据 `config.toml` 创建 LLM 实例的代码工厂），加入对您新后端的参数映射支持。
