# Discord 适配器技术规范

本技术指南详述了 Kesoku 的 Discord 聊天机器人适配器 (`DiscordChatbot`) 的底层实现，涵盖会话线程隔离、动态交互式 UI 按钮、斜杠命令注册以及附件文件解析下载流程。

---

## 📁 目录结构

Discord 适配器代码均存放于 `src/kesoku/gateway/chatbot/discord/` 路径下：

*   **`adapter.py`**：核心 `DiscordChatbot` 适配器类，继承自 `Chatbot` 基类，管理出站事件与入站消息。
*   **`ui.py`**：UI 交互视图组件，包含 `MessageHeaderView`（状态与控制栏按钮）及 `QuestionView`（动态单选问题按钮）。
*   **`command.py`**：斜杠（`/`）应用命令的注册与绑定管理（如重启 `/restart`）。
*   **`voice.py`**：语音与音频流通道绑定。

---

## ⚙️ 核心技术架构

### 1. 基于子线程的会话上下文隔离
为防止在多人讨论频道中产生会话消息冲突与串扰：
*   当用户在群组频道中发话且未匹配特定绑定会话时（且 `auto_thread = true`），适配器会自动创建一个 Discord 子线程（Thread）。
*   该 **子线程 ID** 直接在数据库中被映射绑定为 `channel_id` 和唯一的 `session_id`。
*   此后该子线程内的所有后续对话都共享此唯一的会话上下文。

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
当适配器遇到选择语法 `[question: 1+1等于几？ || 2 | 3]` 时，会进行二次解析：
*   渲染题目文本块。
*   生成带有 `"2"` 和 `"3"` 文字按钮的 `QuestionView` 视图。
*   用户扫码/点击任一按钮后，系统立即锁定禁用所有按钮防止重复提交，修改卡片为已确认样式，并将用户的选择结果作为一条新 `ROLE_USER` 消息发布至网关，继续驱动推理循环。
