# 记忆与会话管理

Kesoku 将所有的聊天记录、绑定设定和 Agent 的记忆内容都存储在本地 SQLite 数据库中（通常为 `kesoku.db`）。本指南将介绍如何通过命令行管理活动的聊天会话，以及如何手动维护 Agent 的长期结构化记忆。

---

## 💬 通过命令行管理聊天会话 (Sessions)

每个对话分支都被隔离在一个唯一的会话（由 `session_id` 标识）中。在守护进程模式下，这些 ID 会与 Discord 的线程 ID、Google Chat 的 Space ID 或微信的聊天下发上下文进行自动映射。

您可以使用 `kesoku chat` 命令组来管理这些会话：

### 1. 列出所有活跃会话
查询数据库中记录的所有会话，并展示它们的创建时间、绑定角色以及已同步的消息条数：

```bash
uv run kesoku chat -c config.toml -l
```

### 2. 打印会话聊天历史
在终端中以美观、带色彩的 Rich 格式打印出指定会话的完整历史对话轨迹：

```bash
uv run kesoku chat -c config.toml --show-history <session_id>
```

### 3. 恢复会话进行聊天
在已有的特定会话中继续对话：

```bash
uv run kesoku chat -c config.toml -r <session_id> "我们刚才提到的数字是几？"
```
或者，无需复制 ID，快速恢复**最近一次活跃**的会话：

```bash
uv run kesoku chat -c config.toml -z "继续之前的任务。"
```

---

## 🧠 管理 Agent 的长期记忆 (`memory`)

Kesoku 内置了一个长期记忆模块，允许 Agent 跨会话沉淀和读取结构化知识（例如用户偏好、里程碑记录、业务配置）。这些记忆按**分类 (Category)** 和**人设角色 (Role)** 进行命名空间隔离。

管理员可以使用 `kesoku memory` 命令组来维护这些记忆：

### 1. 列出记忆条目
列出指定分类下的所有记忆（可选择过滤特定角色）：

```bash
# 列出默认角色 (default) 在 'user_preference' 分类下的所有记忆
uv run kesoku memory list --category user_preference --role default
```

### 2. 查看具体记忆内容
查看单个记忆 Key 的详细内容与描述：

```bash
uv run kesoku memory view --category user_preference --key user_timezone --role default
```

### 3. 添加或更新记忆
手动录入或修改一条记忆记录：

```bash
uv run kesoku memory update --category user_preference --key user_timezone --title "用户时区" --content "Asia/Tokyo" --role default
```

### 4. 删除记忆
删除指定的记忆条目：

```bash
uv run kesoku memory delete --category user_preference --key user_timezone --role default
```

### 5. 备份与迁移 (导出 / 导入)
如果您需要备份所有角色的记忆，或者迁移到另一个工作环境的 SQLite 数据库：

*   **导出为 JSON 文件**：
    ```bash
    uv run kesoku memory export -o memories_backup.json
    ```
*   **从 JSON 文件导入**：
    ```bash
    uv run kesoku memory import -i memories_backup.json
    ```
