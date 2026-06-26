# 测试与调试指南

本开发指南详细介绍了 Kesoku 框架中的调试工具、执行轨迹日志规范、直接数据库查询模式以及 Pytest 测试套件使用方法。

---

## 📜 LLM 推理回合轨迹日志 (`TurnLogger`)

为了便于开发者审计发送给大模型的 Prompt 提示词上下文、分析工具调用的执行参数及模型推理逻辑，系统内置了回合日志记录器 `TurnLogger`（位于 `src/kesoku/agent/turn_logger.py`）：

*   **日志位置**：顺序保存在当前会话的 Staging 工作目录中：`sessions/<workspace_name>/llm-turn-{idx}.log.yaml`。
*   **文件格式**：序列化为结构化 YAML 文件，保证了高可读性，同时方便自动化脚本提取分析。

### 示例 YAML 日志结构：
```yaml
metadata:
  timestamp: 1717765103.541
  session_id: "sess_abc123"
  turn_index: 3
  llm_provider: "gemini"
history:
  - role: "user"
    sender: "my_username"
    type: "text"
    content: "Run test compile"
  - role: "assistant"
    sender: "Kesoku"
    type: "thought"
    content: "I need to run the shell command tool."
tools:
  - name: "run_shell_command"
    description: "Execute terminal commands safely..."
response:
  content: "Shell execution finished."
  thought: "Command executed successfully. Informing user."
  prompt_tokens: 1450
  candidates_tokens: 340
  total_tokens: 1790
```

当模型行为异常或工具参数出错时，直接打开对应的 `.log.yaml`，可以最快还原模型在当前步骤所处的真实上下文。

---

## 💾 直接调试 SQLite 数据库 (`sqlite3`)

若需要审计底层的消息流转与关系绑定，可以在终端中直接使用 `sqlite3` 查询本地数据库文件：

```bash
sqlite3 kesoku.db
```

### 核心表结构解析

1.  **`messages`**：核心消息流表。记录了 role、type、sender、content、parent_id（用于多工具并发对齐）、时间戳以及物理递送状态等。
2.  **`sessions`**：记录会话记录（如重构覆盖后的 `system_prompt` 等）。
3.  **`channel_sessions`**：关联表，维护第三方聊天频道/子线程与系统会话 ID 之间的映射：
    ```sql
    SELECT * FROM channel_sessions WHERE chatbot_id = 'discord';
    ```
4.  **`channel_roles`**：维护频道/子线程与绑定人设角色的映射。
5.  **`cross_session_contexts`**：维护角色多会话同步摘要及锁状态。
6.  **`agent_memories`**：存储 Agent 的持久化结构化记忆条目。

---

## 📜 交互式 HTML 推理轨迹查看器 (`LcmHtmlReporter`)

为了更直观地回溯复杂的长对话执行细节，Kesoku 提供了可视化交互网页报告：

*   **核心模块**：`src/kesoku/gateway/chatbot/lcm_reporter.py`。
*   **实现原理**：该模块读取 SQLite 中的消息历史，经过渲染编译，在会话暂存目录下输出一个带暗黑模式、包含工具调用与思维折叠展示的单页面 `.html` 可视化执行轨迹报告。
*   **查看方式**：
    *   在命令行：运行 `/context` 指令，会返回本地 HTML 网页文件的路径。
    *   在 Discord 中：点击消息头部的 `📜` (查看轨迹) 按钮，机器人会直接以附件形式发送 HTML 网页文件给用户，双击即可在浏览器中打开分析。

---

## 🧪 运行单元与集成测试

Kesoku 采用 `pytest` 及 `pytest-asyncio` 来确保核心链路的质量。

### 执行测试用例
确保您的本地开发依赖项已完全同步，在根目录通过 `uv` 运行测试：

```bash
# 运行所有单元与集成测试
uv run pytest

# 开启标准输出日志流模式运行
uv run pytest -s
```

### 主要测试覆盖范围
*   `tests/agent/test_llm.py`：验证 LLM 客户端的格式拼接、文件附件多模态处理，以及并发调用工具的 Payload 翻译逻辑。
*   `tests/gateway/chatbot/`：测试适配器内置指令注册器工作状态、Markdown 表格转换功能以及长轮询事件接收状态。
*   **数据库并发与锁定测试**：专门针对 SQLite 连接池的多线程/多协程 CAS 锁状态更新（针对跨会话总结 claims 锁自愈）进行了压力与健壮性校验。
