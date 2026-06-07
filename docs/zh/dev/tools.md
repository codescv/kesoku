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
*   如果有任何非默认值的必填参数在模型输出中缺失，执行器会直接抛出校验异常，终止执行，以防止未定义行为。

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
