# 角色扮演实现原理

本技术文档详述了 Kesoku 如何管理、存储以及跨会话通道（Channel/Thread）动态解析角色人设（Roles）的底层实现机制。

---

## 💾 数据库模型设计 (`channel_roles` 表)

所有的角色通道绑定映射均被保存在 SQLite 数据库的 `channel_roles` 表中：

```sql
CREATE TABLE IF NOT EXISTS channel_roles (
    chatbot_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    role TEXT NOT NULL,
    PRIMARY KEY(chatbot_id, channel_id)
);
```

当用户要求切换角色，或者 Agent 在回合中执行了 `play_role` 工具调用时：

1. 系统会通过 upsert 写入关系映射：
    ```sql
    INSERT OR REPLACE INTO channel_roles (chatbot_id, channel_id, role)
    VALUES (?, ?, ?)
    ```
2. 该绑定关系在服务重启后依然持久保留。

---

## 🔄 角色解析继承模型

当进入新的对话回合，或者调度器编译组装系统提示词时，系统会调用以下方法动态计算当前活动角色名称：
`get_channel_role_with_inheritance(chatbot_id, channel_id, session_id)`

为了保证 Discord 子线程等动态场景能够自动继承主频道的设定，角色解析遵循以下严格的继承路径：

```text
┌────────────────────────────────────────────────────────┐
│ 1. 直接查询 (Direct Lookup)                            │
│    查看当前 channel_id (如子线程) 是否绑定了专属角色？    │
└─────────────────────────┬──────────────────────────────┘
                          │ (若无)
                          ▼
┌────────────────────────────────────────────────────────┐
│ 2. 线程继承查询 (Thread Inheritance, 仅 Discord)        │
│    如果是子线程，查询最近一条消息的 parent_channel_id，   │
│    追溯主频道是否配置了绑定角色？                          │
└─────────────────────────┬──────────────────────────────┘
                          │ (若无)
                          ▼
┌────────────────────────────────────────────────────────┐
│ 3. 全局默认角色 (Fallback)                             │
│    返回全局默认角色名: "default"                         │
└────────────────────────────────────────────────────────┘
```

### 1. 直接查询 (Direct Lookup)
直接在 `channel_roles` 数据库表中检索匹配的 `(chatbot_id, channel_id)`。如果命中，则直接应用并返回该角色。

### 2. 线程继承模型 (Discord)
在 Discord 适配器交互场景下：

*   每次新建会话时，机器人通常会在此文本频道下创建一个新的子线程（Thread）。
*   该子线程在 Discord 体系中拥有独立的 ID（反映在数据库中为新 `channel_id`）。
*   为了防止用户每次在线程内都需要重新设置角色，系统会在数据库中查找该会话最近一条用户消息中的 `metadata` 字段，解析出 `parent_channel_id`（即该子线程所属的父级文本频道 ID，如 `#general` 频道的 ID）。
*   若该父频道绑定了角色设定（如 `"tifa"`），则子线程会自动继承并应用 `"tifa"` 角色。

### 3. 全局回退
如果当前频道和父级频道均没有显式绑定任何角色，则系统统一回退采用默认角色 `"default"`。

---

## 🔄 系统提示词动态重构时序

当 Agent 在会话运行期间执行了 `play_role(role="tifa")` 工具：

1.  工具模块更新数据库的通道角色映射关系：
    ```python
    await db.set_channel_role(chatbot_id, channel_id, "tifa")
    ```
2.  工具指令立即向网关触发该活动会话系统提示词的重新拼装编译：
    ```python
    # 动态重构提示词
    new_sys_prompt = build_sys_prompt(session=session)
    ```
3.  编译得到的完整提示词字符串被更新并覆盖写入 SQLite 数据库的 `sessions` 表：
    ```python
    await db.update_session_system_prompt(session_id, new_sys_prompt)
    ```
4.  在下一个推理步骤中，LLM 客户端会从会话记录中重新读取更新后的 `system_prompt` 字段，并代入到推理上下文中。
