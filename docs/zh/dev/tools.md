# 工具与并发执行系统

本技术指南介绍了 Kesoku 智能体框架中工具（Tools）的注册、参数校验及异步执行流程。

---

## 🛠️ 工具注册与装饰器机制

Kesoku 采用集中式工具注册中心模式。通过使用 `@default_registry.register` 装饰器，任何标准的 Python 函数都可以一键转换为 Agent 的可用工具：

```python
from kesoku.agent.tools import default_registry, ToolContext

@default_registry.register
async def calculate_sum(a: int, b: int) -> int:
    """计算两个整数的和。

    Args:
        a: 第一个整数。
        b: 第二个整数。
    """
    return a + b
```

### 文档注释解析与 Schema 生成
当大模型适配器（如 `GeminiLLM` 或 `ClaudeLLM`）初始化时，客户端会扫描注册中心，解析工具函数签名：

*   **功能描述**：通过解析函数的 Docstring（采用 Google 风格规范）获取。
*   **参数类型**：通过 Python 的函数参数类型注解（例如 `a: int`）获取。
*   **JSON Schema**：系统自动将上述元数据转换为大模型要求的函数声明（Gemini 对应的 `FunctionDeclaration`，或 Claude 对应的 `Tool` 格式）。

---

## ⚙️ 工具执行周期 (`ToolRunner`)

所有的工具调用都统一由 `ToolRunner` 类（`src/kesoku/agent/tool_runner.py`）进行调度执行：

```text
┌────────────────────────────────────────────────────────┐
│ 1. 解析 ToolRequest 参数                               │
├────────────────────────────────────────────────────────┤
│ 2. 检查中断标志位 (is_interrupted 抢占逻辑)            │
├────────────────────────────────────────────────────────┤
│ 3. 参数校验：inspect.signature() 核对必要参数           │
├────────────────────────────────────────────────────────┤
│ 4. 驱动执行：异步直接 await / 同步线程池转义            │
├────────────────────────────────────────────────────────┤
│ 5. 组装响应：转换返回值/异常为标准 Message (TOOL_RESULT)│
└────────────────────────────────────────────────────────┘
```

### 1. 中断检查 (Interruption Check)
在真正调度执行工具代码前，`ToolRunner` 会首先评估 `is_interrupted` 回调状态：

```python
if is_interrupted and is_interrupted():
    # 立即中止工具执行，返回 aborted 错误消息
```
此机制可确保当用户在模型“思考”阶段插话时，后续排队的工具调用可以直接被安全地忽略和丢弃。

### 2. 参数严密校验
利用 Python 内置的 `inspect.signature(tool_func)`，执行器检查模型生成的参数是否满足函数签名：

*   如果工具函数的参数中包含名为 `context` 的参数，执行器在调用时会**自动注入**当前的 `ToolContext` 上下文对象（其中包含活跃的 `gateway` 数据库实例、`session_id` 以及原始消息 ID `original_msg_id`）。
*   JSON `null` 传给 Optional 参数（`str | None`、`Annotated[int | None, ...]`）时保持为 `None`（不会变成字符串 `"None"`）；传给非 Optional 参数时视为未提供，使用默认值。
*   如果有必填参数缺失（通常是模型输出被截断），返回 `Missing required argument(s) for tool '<name>': <args>.`，并提示把大段内容拆成 ≤4000 字符的多次调用。

### 文件工具（`tools/file.py`）
*   **`read_file(file_name, start_line=1, end_line=None, max_lines=500)`**：带 `N: ` 行号的分页读取（上限 2000 行、30000 字符，单行超过 2000 字符会被截断）。文件不存在时给出 `Did you mean: repo/...?` 路径建议；目录返回条目列表；拒绝二进制文件。
*   **`update_file(file_name, old_content=None, new_content=None)`**：省略 `old_content` 即创建文件（已存在则拒绝）；否则替换只出现一次的精确文本块（出现两次会都替换并给出警告，超过两次拒绝并列出行号和上下文）。`new_content` 必填（`""` 表示删除）。匹配失败时给出恢复提示：忽略空白差异的匹配及其原文行，或最相近的行和 `read_file` 行号范围；误复制的 `N: ` 前缀会被自动去除。成功消息包含受影响的行号范围。

### 输出截断（中间截断）
*   `run_shell_command` 最多保留 `max_output_chars`（默认 5000，上限 30000）：预览保留约 45% 开头和 55% 结尾，完整输出保存到 `$STAGING_DIR/cmd_output_*.txt`。
*   已完成回合中，`prepare_history_for_llm` 把工具结果/错误缩成 500 字符的首尾预览（`history_preview`），先去掉 shell 自己的截断说明，使开头预算展示真实输出（保留开头的 `Command failed with exit code N.`）。完整内容保留在磁盘上。`HistoryCompressor.estimate_prepared_turn_tokens` 使用同一个函数。

### 3. 多线程与并发安全
执行器对同步函数和异步协程均提供了支持：

*   **异步协程 (Async Functions)**：直接在事件循环中通过 `await tool_func(**kwargs)` 执行。
*   **同步阻塞函数 (Sync Functions)**：系统会通过 `asyncio.to_thread(tool_func, **kwargs)` 将其丢入独立的线程池中执行，防止阻塞主事件循环。

### 4. 消息响应打包
无论工具是执行成功返回结果，还是执行中抛出异常，`ToolRunner` 都会捕获这些输出，并将其封装为统一的数据库 `Message` 对象：

*   **角色 (Role)**：`MessageRole.TOOL` (`"tool"`)
*   **类型 (Type)**：`MessageType.TOOL_RESULT` (`"tool_result"`)
*   **状态 (Status)**：`MessageStatus.RESPONDED`
*   **父链绑定**：设置 `parent_id` 指向触发它的 `tool_call` 消息 ID，保持完整的会话回合链。

---

## 📸 多模态工具 (`analyze_media`)

Kesoku 内置了利用大模型多模态能力分析本地多媒体文件（图片、音频、视频）的工具：

*   **`analyze_media(path, prompt)`**：解析媒体文件路径（绝对路径或相对于 AWD/Staging 的相对路径），自动检测其 MIME 类型，并将其作为原生多模态数据块（`ImageBlock`、`DocumentBlock` 或 `MediaBlock`）附加到推理请求中，对该文件执行问答分析。

### `disabled_tools`
`agent.disabled_tools = ["chat_search", "view_message"]` 会把列出的工具从发给 LLM 的工具列表中移除，并拒绝执行（`ToolRegistry.get_tools_list` / `get_tool`）。首回合的 `sync_guidelines` 提示也只会提到未被禁用的历史检索工具。配合 `auto_context_search = false` 可以完全关闭历史聊天检索。
