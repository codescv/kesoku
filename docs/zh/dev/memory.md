# 记忆系统设计与多会话上下文同步

本技术文档详述了 Kesoku AI Agent 的记忆系统架构设计，涵盖了 Memory V1 数据库结构设计、Memory V2 动态拉取优化（Bootstrap & On-Demand Pull）以及跨会话总结异步锁机制。

---

## 1. 记忆架构设计基础 (Memory v1)

早期基于纯 Markdown 扁平文件（如 `Progress.md`、`Agent.md`）的记忆更新方案存在致命的安全漏洞：
*   **全量覆盖风险 (Full-Overwrite Hazard)**：模型在修改文件内某一项进度时，极易由于幻觉导致整个文件的大片内容被丢失或截断。
*   **键值重复与漂移**：缺少数据库主键约束，模型经常会为同一个概念生成重复或相似的 Key（例如 `japan` 和 `japanese`）。
*   **弱约束规则**：放置在扁平文件中的运行规则（如“必须使用 `uv run` 运行测试”），只能等待模型主动阅读，极易被忽略。

为了解决这些问题，Kesoku 建立了**结构化 SQLite 存储层**，并将核心偏好和规则强行注入到模型系统提示词（System Prompt）中最显眼的位置。

### 1.1 数据存储模型 (`agent_memories` 表)
所有的 KV 结构化记忆均存储在 SQLite 数据库中：
```sql
CREATE TABLE IF NOT EXISTS agent_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,         -- 'progress', 'preference', 'rule'
    key TEXT NOT NULL,              -- 唯一的 snake_case 键名 (例如 'standard_japanese')
    title TEXT NOT NULL,            -- 人类可读标签
    content TEXT NOT NULL,          -- 结构化 JSON 或 Markdown 文本内容
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(category, key)           -- 强约束 Upsert 限制
);
```

### 1.2 记忆类别与生命周期分配
不同功能类别的记忆享有不同的读写级别，防止由于大模型误写导致数据混乱：
1.  **用户偏好 (`preference`)**：【只读保护】。记录用户的性格背景、时区等个人偏好信息。智能体对其仅有 Read-Only 权限，更新必须由用户手动通过指令发起，防止大模型编造幻觉覆盖真实偏好。
2.  **执行规范规则 (`rule`)**：【只读保护】。存储最高优先级的系统硬约束规则（如“限制出站文件路径”、“使用 `uv` 管理包”）。在会话初始化时，被强行注入到系统提示词顶部。
3.  **学习与项目进度 (`progress`)**：【读写协同】。记录各种具体项目进度、学习节点。Agent 可以通过工具进行原子性的 `INSERT OR REPLACE` 写入，确保安全不影响其他项目进度。

---

## 2. 局部按需注入与动态拉取 (Memory v2)

在长对话和多通道运行期间，如果将所有历史和进度直接塞入 system prompt 中，会快速占满 context window，并且会极大地**分散大模型的注意力 (Attention Distraction)**。

因此，Memory V2 推出了**自适应拼装与主动拉取**的设计方案：

```text
+-----------------------+                               +---------------------+
|   `agent_memories`    |                               |`cross_session_ctx`  |
| (用户静态偏好与硬规则) |                               | (全局事件轨迹时间轴) |
+-----------------------+                               +---------------------+
        │                                                               │
        │ (Bootstrap 阶段 Push 注入)                                    │ (大模型按需主动 Pull 拉取)
        v                                                               v
+-----------------------+                               +---------------------+
|   第一回合 / 长空闲激活  |                               |  Tool:              |
|  (Sync Guidelines)    |                               |`view_chat_history_` |
|                       |                               |`summary`            |
+-----------------------+                               +---------------------+
        │                                                               |
        +-------------------------------+-------------------------------+
                                        │
                                        v
                     +-------------------------------------+
                     |           LLM 运行上下文            |
                     |       (高度聚焦当前的研发任务)      |
                     +-------------------------------------+
```

### 2.1 会话回合自适应注入规则 (Bootstrap Injection)
系统并不在每个 Turn 中都注入繁琐的记忆同步说明，而是采用了细分策略：
*   **同步引导指南 (Sync Guidelines)**：仅在**引导回合（Bootstrap Turn）**注入。引导回合被定义为：
    1.  全新启动的会话（回合数小于等于 1）。
    2.  长空闲重新唤醒（当前消息的时间戳与上一回合消息的间隔时间，超过了设置的 `1800` 秒超时阈值）。
    *   *引导指南告知模型其拥有读取全局总结的 tool API，如果发现用户提及了当前线程之外的历史（如“刚才在 Discord 里说的那个 bug”），模型必须首先调用拉取工具，主动获取上下文。*
*   **用户偏好配置 (User Preferences)**：如果数据库存在用户偏好设定，则在**每一个回合**中都无条件前置注入给大模型，确保其始终遵循代码规范和个人习惯。

### 2.2 按需主动拉取工具 (`view_chat_history_summary`)
大模型遇到外部上下文依赖时，会调用此工具拉取跨会话进展：
*   **执行逻辑**：
    1.  确定当前的活动角色名称。
    2.  读取跨会话同步数据库表 `cross_session_contexts` 中，属于该角色的已固化总结时间轴。
    3.  查询在最近一次更新时间点之后，在其他所有活跃通道中产生的高价值新消息（过滤掉垃圾 thoughts 和 tool log），并将它们实时拼接输出。
*   这保证了 Agent 在需要时能够像“翻阅书本”一样获取跨越 Discord、微信和命令行终端的全局上下文，而不需要在每次提问时都带上这些繁重的历史包袱。

---

## 3. 跨会话同步与自愈锁机制 (Cross-Session Context Design)

为了使多通道的全局进展能定时在后台更新并存回数据库，Kesoku 引入了基于 CAS（Compare-And-Swap）的异步总结锁控制。

### 3.1 跨会话数据库结构 (`cross_session_contexts`)
```sql
CREATE TABLE IF NOT EXISTS cross_session_contexts (
    role TEXT PRIMARY KEY,          -- 角色名人设 (如 default)
    content TEXT NOT NULL,          -- 固化的 Markdown 时间轴总结
    updated_at REAL NOT NULL,       -- 最后更新时间戳
    status TEXT NOT NULL DEFAULT 'idle' -- 锁状态：'idle' (空闲) 或 'updating' (更新中)
);
```

### 3.2 异步自愈锁管理 (CAS & Self-Healing)
为了在不阻塞用户实时会话（保证低延迟）的前提下，实现后台平滑总结：

1.  **触发阶段**：每一次会话 Turn 执行结束后，执行器统计自上一次 checkpoint 以来产生的新消息 token 数量。若超过 `4000` tokens，或者时间间隔超过 30 分钟，即触发同步流程。
2.  **原子抢占锁 (CAS)**：调用 `claim_cross_session_context_for_update(role)`：
    *   为防由于意外强杀或断电导致锁死在 `updating` 状态（产生死锁），CAS 逻辑会在抢占前**自动检查并重置**所有创建时间超过 300 秒的僵死锁为 `idle`。
    *   执行 UPDATE 事务将状态原子性修改为 `updating`，确保只有一个 SessionWorker 在后台进行大模型总结。
3.  **后台总结**：主会话立即将最终文本发给用户（不影响交互响应），同时在独立的 background `asyncio.Task` 协程中加载当前上下文以及新产生的对话序列，调用大模型生成一段限制在 300 字以内的紧凑时间轴摘要。
4.  **释放与固化**：计算完毕后，调用 `release_cross_session_context_lock(role, new_summary)`，更新摘要正文，写入时间戳并将状态恢复为 `idle`，完成一个记忆生命周期的闭环。
