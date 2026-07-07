# 记忆系统、数据库设计与无损上下文管理 (LCM)

本技术文档详述了 Kesoku AI Agent 的记忆与上下文管理系统架构设计，涵盖了结构化 SQLite 存储层、动态提示词注入生命周期以及无损上下文管理（LCM）系统如何优化长对话历史。

---

## 1. 早期设计的缺陷与痛点

### 1.1 纯文本 Markdown 文件（如 `Progress.md`、`Agent.md`）
1. **全量覆盖风险 (Full-Overwrite Hazard)**：大模型修改进度文件时，极易因“幻觉”导致整个文件的其他不相干段落被意外截断或丢失。
2. **键值重复与漂移**：缺少数据库的唯一性主键约束，模型经常为同一个概念生成重复或相近的 Key（例如 `standard_japanese`、`standard_japan` 和 `标日学习`）。
3. **弱约束规则**：放置在扁平文件中的运行规则（如“必须使用 `uv run` 运行测试”），只能被动等待模型主动读取，大模型极易忽略。

### 1.2 Memory V1 全量提示词注入
在每个 Turn（会话回合）中，将所有偏好、进度、规则全都塞入系统提示词（System Prompt）中，会迅速占满上下文窗口，并导致**注意力分散（Attention Distraction）**——模型因为被大量无关的进度摘要包围，进而忽略了真正的硬约束规则。

---

## 2. 结构化长期记忆（SQLite `agent_memories`）

为了实现事务级别的安全隔离，Kesoku 引入了结构化的 SQLite 关系存储层（`agent_memories` 表）：

```sql
CREATE TABLE IF NOT EXISTS agent_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,         -- 'progress', 'user_preferences', 'memo'
    key TEXT NOT NULL,              -- 唯一的 snake_case 键名 (例如 'standard_japanese')
    title TEXT NOT NULL,            -- 人类可读标签
    content TEXT NOT NULL,          -- Markdown 或 JSON 内容 (最大 500 字符)
    updated_at REAL NOT NULL,       -- UNIX 时间戳浮点数
    role TEXT NOT NULL DEFAULT 'default', -- 绑定的角色人设范围
    embedding BLOB,                 -- 向量嵌入数据
    UNIQUE(category, key, role)     -- 安全更新 and 排重约束
);
```

### 2.1 类别描述与作用域
*   **`progress`**：用于跟踪项目开发进度。这些是**角色隔离**的，绑定在当前频道活跃的角色人设下。
*   **`user_preferences`**：存储用户个人基本信息、语音/TTS 发音习惯以及回复风格偏好。这些是**角色隔离**的，绑定在当前频道活跃的角色人设下。
*   **`memo`**：自定义用户备忘笔记或零散记忆。这些是**角色隔离**的。

### 2.2 读写权限生命周期
*   **用户偏好与规则**：**只读保护（Read-Only for Agent）**。为了防止大模型幻觉编造并覆写偏好数据，此类记录只能由用户手动发起指令进行修改。
*   **进度与学习记录**：**读写协同（Read-Write for Agent）**。大模型可以通过调用工具对单条记录进行原子的 `INSERT OR REPLACE` 写入，从而完全消除覆盖损坏其他进度的风险。

---

## 3. 动态提示词注入生命周期

为了保证大模型高度聚焦当前任务，Kesoku 将动态记忆和偏好前置注入到**最新的一条用户消息**中，而不是塞入全局 system prompt 中。

### 3.1 注入规则
*   **引导回合 (Bootstrap Turn)**（会话的第 1 回合，或者空闲超过 30 分钟重新唤醒）：
    *   向前置注入**同步引导指南 (Sync Guidelines)** 与**用户偏好配置 (User Preferences)**。
    *   *引导指南会告知模型其当前扮演的角色 `role="{active_role}"`，并提示其拥有 `view_chat_history_summary` 工具。如果用户提到了当前通道历史之外的事物，模型必须首先调用该工具拉取全局时间轴。*
*   **普通回合**：
    *   仅注入**用户偏好配置 (User Preferences)**。避免在对话激活后重复塞入冗长的同步引导。

### 3.2 注入模板示例
```markdown
[Background Context: Sync Guidelines]
======
# Passive Synchronization Guidelines:
- 💡 You are playing the active persona role: {active_role}.
- 💡 You have access to the `view_chat_history_summary` tool, which retrieves a consolidated chat history summary and chronological timeline of recent events across active threads/channels.
- 💡 If the user's current request below refers to external threads, other chats, or events you cannot locate in this session's history, you MUST call `view_chat_history_summary` to read the global context and synchronize before providing a response.
======

[User Preferences]
- Preferred Programming Language: Python
- Code Style: PEP 8 compliant, explicit type hints
- Preferred Test Framework: pytest with uv run

[Current Request]
{original_user_message}
```

---

## 4. 上下文压缩与会话历史管理

长对话历史如果以原始文本一直堆积，会极快耗尽上下文窗口。Kesoku 采用自定义的分层回合历史压缩器，实现历史压缩与无损找回的平衡。

### 4.1 分层回合历史压缩 (`HistoryCompressor`)
1. 在每次发起 LLM 推理前，系统会自动评估当前的历史消息 Token 长度。
2. 当除去受保护的头部回合（protected head turns）和尾部回合（protected tail turns）之后的中间回合累积了足够的未压缩 Token 数与回合数时，会触发自动压缩流程。
3. 压缩管理器把需要压缩的中间回合打包，调用 LLM 生成结构化、高密度的 0 级**摘要节点（Level-0 Summary Node）**。摘要中包含时间轴事件、关键决策与变更、遇到的坑（Pitfalls）以及变更的根目录/文件等字段。
4. 在 SQLite 数据库中更新原消息记录，将其 `summary_node_id` 字段指向新创建 of 0 级摘要节点。
5. 当 L 级摘要节点累积到 $2 \times K$ 个时，最老的 $K$ 个节点会被进一步合并并巩固为 Level-L+1 级的更高阶摘要节点（其中 $K$ 为配置的 `context_consolidation_k`）。
6. 系统会动态拼装活跃历史上下文，拼装顺序为：受保护的头部回合 -> 指向各个摘要节点的骨架指示文本 -> 系统/助手层面的骨架确认应答消息 -> 剩余的未压缩中间回合缓冲区 -> 受保护的尾部回合。

### 4.2 检索与还原工具
如果大模型在推理过程中需要查阅已被压缩的细节或在当前角色下检索历史对话，它可以主动调用以下工具：
*   **`memory_grep(query, start_time, end_time, limit)`**：从当前角色的所有会话及记忆中，通过关键字或通配符 `*` 检索历史消息和主动记忆，并支持时间范围过滤。
*   **`memory_search(query)`**：利用向量嵌入对当前角色的主动记忆进行语义搜索（Semantic Search）。
*   **`view_message(message_id)`**：输入指定的消息 ID，还原并读取数据库中完整的原始对话内容。

### 4.3 记忆系统的互补共存
*   **主动记忆系统 (AMS)**（`agent_memories` 表）：负责保存长期的、高度固化的结构化事实与规则，分为 `user_preferences`、`progress` 和 `memo` 三大类。
*   **分层压缩管理系统**（`summary_nodes` 表）：负责以极低 Token 损耗且分层归并的方式，维持长对话线程的全部历史轨迹。
