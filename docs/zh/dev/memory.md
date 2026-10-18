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
1. 每次 LLM 推理前检查是否需要压缩。通过 `[agent].compression_strategy` 选择两种策略：
    *   **`chunk`**（旧算法，默认）：中间回合（除受保护的头部和最后 `protect_tail_turns` 轮）一旦同时攒够 `base_node_turns` 轮和 `base_node_min_tokens` token（旧的 fallback 估算），立即压成 Level-0 节点。原文上下文在"受保护尾部"和"尾部 + 约 25K 估算 token"之间来回跳。
    *   **`watermark`**（高低水位）：用校准后的估算器（ASCII 约 0.25 token/字，非 ASCII 约 0.9 token/字，按 Gemini `count_tokens` 拟合）估算整个 prompt（system prompt + 工具定义 + scaffold + 未压缩历史）。不超过 `context_high_watermark` 时什么都不做；超过后从最老的未保护回合开始，按约 `watermark_chunk_tokens`（至少 `watermark_chunk_min_turns` 轮）一块压成 L0，直到估算值低于 `context_low_watermark`。尾部同时受 `protect_tail_turns` 和 `protect_tail_min_tokens` 保护。可压缩内容不足一块时跳过（防抖动），配置校验要求 `protect_tail_min_tokens < context_low_watermark < context_high_watermark`。摘要长度按原文规模缩放（L0 约为原文 token 的 8%，1200–3000 字；合并时为子节点总长度的一半，上限 4000 字）。
2. 摘要由专门的压缩模型生成（`[agent].context_llm` / `[gemini|claude].context_model_name`）。
3. 摘要节点只有 **Timeline** 和 **Learnings**（仅工具与技能使用经验）两段，刻意不包含任何用户要求：原来的 "User Asks" 段和 Learnings 里的用户偏好会把要求冻结在摘要里，之后过时并与新指令冲突；"Tools & Skills" 段则大量重复。当前规则只由 agent 维护在 `notes.md`（`<context name="notes">`）中。旧节点里的 "User Asks" / "Tools & Skills" 段在发送给 LLM 前会被剥离。
4. 在 SQLite 数据库中更新原消息记录，将其 `summary_node_id` 字段指向新创建的 0 级摘要节点。
5. 当 L 级摘要节点累积到 $2 \times K$ 个时，最老的 $K$ 个节点会被进一步合并并巩固为 Level-L+1 级的更高阶摘要节点（其中 $K$ 为配置的 `context_consolidation_k`）。
6. 动态拼装活跃上下文：受保护的头部回合 -> scaffold 消息 -> scaffold 确认应答 -> 未压缩的中间回合缓冲区 -> 受保护的尾部回合。scaffold 开头声明"摘要是历史记录，可能过时；与 `<context name="notes">` 或最近消息冲突时以后者为准"，随后是摘要森林，以及 `<recent_user_messages>` 块：保留已压缩回合中最近 `recent_user_messages_limit` 条用户原话（用户消息只占约 1% token，保留原话成本很低）。scaffold 只在压缩时变化，便于缓存。
7. 已完成回合中 `update_file`（以及旧历史中已移除的 `update_session_notes`）工具调用的长字符串参数截断到 300 字；工具结果仍沿用 500 字上限。

### 4.2 检索与还原工具
如果大模型在推理过程中需要查阅已被压缩的细节或在当前角色下检索历史对话，它可以主动调用以下工具：
*   **`memory_grep(query, start_time, end_time, limit)`**：从当前角色的所有会话及记忆中，通过关键字或通配符 `*` 检索历史消息和主动记忆，并支持时间范围过滤。
*   **`memory_search(query)`**：利用向量嵌入对当前角色的主动记忆进行语义搜索（Semantic Search）。
*   **`view_message(message_id)`**：输入指定的消息 ID，还原并读取数据库中完整的原始对话内容。

### 4.3 记忆系统的互补共存
*   **主动记忆系统 (AMS)**（`agent_memories` 表）：负责保存长期的、高度固化的结构化事实与规则，分为 `user_preferences`、`progress` 和 `memo` 三大类。
*   **分层压缩管理系统**（`summary_nodes` 表）：负责以极低 Token 损耗且分层归并的方式，维持长对话线程的全部历史轨迹。
