# Discord 适配器技术规范

本技术指南详述了 Kesoku 的 Discord 聊天机器人适配器 (`DiscordChatbot`) 的底层实现，涵盖会话线程隔离、动态交互式 UI 按钮、斜杠命令注册以及附件文件解析下载流程。

---

## 📁 目录结构

Discord 适配器代码均存放于 `src/kesoku/gateway/chatbot/discord/` 路径下：

*   **`adapter.py`**：核心 `DiscordChatbot` 适配器类，继承自 `Chatbot` 基类，管理出站事件与入站消息。
*   **`ui.py`**：UI 交互视图组件，包含 `MessageHeaderView`（状态与控制栏按钮）及 `QuestionView`（动态单选问题按钮）。
*   **`persistent_ui.py`**：重启后仍有效的 `DynamicItem` 组件（选择题按钮/下拉菜单、`<details>` 折叠按钮），数据存于 `ui_prompts` 表。
*   **`command.py`**：斜杠（`/`）应用命令的注册与绑定管理（如重启 `/restart`）。
*   **`voice.py`**：语音与音频流通道绑定。

---

## ⚙️ 核心技术架构

### 1. 基于子线程的会话上下文隔离
为防止在多人讨论频道中产生会话消息冲突与串扰：

*   当用户在群组频道中发话且未匹配特定绑定会话时（且 `auto_thread = true`），适配器会自动创建一个 Discord 子线程（Thread）。
*   该 **子线程 ID** 直接在数据库中被映射绑定为 `channel_id` 和唯一的 `session_id`。
*   此后该子线程内的所有后续对话都共享此唯一的会话上下文。

### 1.1 忽略系统消息
`on_message` 只接收用户真正发出的聊天消息：`message.is_system()` 为真，或 `message.type` 不是 `MessageType.default` / `MessageType.reply` 的消息一律丢弃（见 `is_user_chat_message`）。这样线程改名（`channel_name_change`）、置顶（`pins_add`）、创建线程提示（`thread_created`）、成员加入（`recipient_add`）等事件不会触发 Agent 回合。

### 2. 消息长文本切分与附件转换
*   **出站字数限制**：适配器覆盖基类 `get_max_text_length()` 并返回 2000（对应 Discord 的消息字数物理限制）。
*   **出站附件解析**：当出站消息文本匹配 `[file: /绝对/路径/文件]` 时，适配器会捕获该路径，校验文件是否存在，并实例化 `discord.File` 对象，将其作为物理附件随文本消息一同发送。

### 3. 入站用户附件下载
当用户在 Discord 线程中上传图片、PDF 或日志文件时：

1.  适配器捕获附件信息列表。
2.  定位会话对应的 Staging 暂存目录 (`sessions/<workspace_name>`)。
3.  异步下载文件字节流并以安全文件名保存到本地磁盘中。
4.  将本地绝对路径等元数据序列化保存到 `messages` 表的 `metadata` 字段中（Key 名为 `"attachments"`）：
    ```json
    [{"path": "/absolute/path/to/file", "mime_type": "image/png", "filename": "table.png"}]
    ```
5.  在用户消息末尾拼接读取附件的路径索引说明。
6.  LLM 后端在下一次推理时会解析此 metadata，将文件内容作为多模态 block 提交给大模型（如 Gemini 的 `types.Part.from_bytes`）。

---

## 🎨 交互式 UI 设计 (`ui.py`)

系统使用 Discord 原生的 `discord.ui.View` 来构建界面控制组件：

### 1. 会话控制面板 (`MessageHeaderView`)
该面板被挂载在当前会话线程的第一条标志性消息下。提供三个简明功能按钮：

*   **查看执行轨迹 (`📜`)**：异步调用 `LcmHtmlReporter` 生成当前回合推理轨迹的暗黑模式 HTML 网页报告，并作为临时文件发送至线程中。
*   **强行终止推理 (`🛑`)**：调用 `Agent` 强制取消当前会话 Worker 的 `asyncio.Task` 推理任务协程，并将数据库消息状态标记为 `interrupted`，同时自动将 Discord UI 上属于推理中间步骤的思维文本、工具调用卡片清理删除。
*   **清除当前会话 (`♻️`)**：注销 Worker 协程，删除数据库中该会话的全部消息历史，递归删除磁盘暂存目录，并自动将该 Discord 子线程删除。

### 2. 单选卡片 (`QuestionView`)
当解析结果包含选择题（```` ```ui ```` `type: select` 块，或旧语法 `[question: 1+1等于几？ || 2 | 3]`）时，会进行二次解析：

*   渲染题目文本块。
*   生成带有 `"2"` 和 `"3"` 文字按钮的 `QuestionView` 视图。
*   用户扫码/点击任一按钮后，系统立即锁定禁用所有按钮防止重复提交，修改卡片为已确认样式，并将用户的选择结果作为一条新 `ROLE_USER` 消息发布至网关，继续驱动推理循环。

### 3. 折叠段落 (`DetailsView`)
`<details><summary>标题</summary>正文</details>` 段会发送为一条 `▸ **标题**` 消息并附「展开」按钮。点击后原地编辑消息，显示 `▾ **标题**` 和正文（按 Discord 2000 字符上限截断），按钮变为「收起」；再次点击收起。

### 4. 重启后仍有效的选择组件 (`persistent_ui.py`)
选择题和 `<details>` 的展开按钮使用 `discord.ui.DynamicItem` 渲染，`kesoku service restart` 之后点击仍然有效：

*   **custom_id 模板**：`kui:<session_id>:<message_id>:<idx>`（选项按钮）、`kuis:<session_id>:<message_id>`（下拉菜单）、`kud:<session_id>:<message_id>-d<n>`（折叠按钮）。`<message_id>` 是被渲染的助手消息在 kesoku 中的 id。
*   **存储**：发送时把 spec（`prompt`、`options`、`multi`、`max`，回答后加 `answered`/`chosen`；折叠段为 `title`/`body`）写入 `ui_prompts(session_id, message_id, spec_json, created_at)` 表，点击时读取；删除会话时一并删除。
*   **注册**：`DiscordChatbot._setup_hook` 调用 `Client.add_dynamic_items(QuestionButton, QuestionSelect, DetailsToggle)`；`register_chatbot` 记录 client → 适配器映射，回调借此访问网关。
*   **布局**：单选且 ≤5 个选项 → 第 0 行按钮（任一选项超过 15 字时按钮显示 A/B/…，完整文字列在 embed 中）；超过 5 个选项 → 单选下拉；多选 → 第 1 行 `Select(min_values=1, max_values=max)`。因为 `discord.ui.Select(row=...)` 不会设置 `_row`，代码中显式赋值。
*   **回答**：点击后把选项原文（多选为一条 `选择：A；B；C`）作为模拟用户消息发到同一会话，行为与 `QuestionView` 一致（确认消息 + 用户消息 + 输入中提示）；同时记录答案并编辑消息，禁用全部组件（高亮所选项）。之后的点击只会收到仅自己可见的提示。
*   若 custom_id 会超过 Discord 的 100 字符上限，则退回到内存中的 `QuestionView` / `DetailsView`。
