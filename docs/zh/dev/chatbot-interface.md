# 聊天机器人适配器基类与接口开发

本指南详述了聊天机器人适配器如何与 Kesoku 的网关（Gateway）进行对接交互、订阅事件循环的设计原理，以及如何通过继承 `Chatbot` 基类来编写自定义的聊天机器人平台适配器。

---

## 🏗️ 核心基类 (`Chatbot`)

聊天机器人适配器的顶层抽象接口定义为 `Chatbot(ABC)`，位于 `src/kesoku/gateway/chatbot/base.py` 中。它为各种前端平台提供了统一的状态管理、内置命令解析注册器以及文本切片格式化工具。

### 1. 指令注册中心与多平台斜杠指令映射

每个聊天机器人适配器都会自动继承一个标准的指令注册中心 `CommandRegistry`（定义于 `src/kesoku/gateway/chatbot/base.py`），其中注册了平台无关的控制指令。

#### 核心注册指令：
*   **`clear`**（别名：`reset`）：终止当前活动会话的 SessionWorker 协程任务，清理 SQLite 中的历史数据，并递归清理磁盘上的会话暂存文件夹。
*   **`status`**：收集并输出运行指标：会话轮数、Token 消耗、当前上下文 K-tokens 大小以及最新回合的响应速度。
*   **`compact`**：手动强制触发 `OpenLCM` 上下文压缩流，无需等待系统自动达到 Token 阈值。
*   **`role`**：更新 SQLite 中当前通道与人设角色的绑定关系（例如切换为 `coder`），并动态重构系统提示词。
*   **`lcm`**（别名：`context`）：调用 `LcmHtmlReporter` 生成推理轨迹 HTML 报告并返回文件路径。
*   **`debug`**：开启/关闭调试模式，控制是否将 LLM 原始 JSON 输出到终端并暴露 AWD Staging 暂存区路径。
*   **`restart`**：触发服务进程的安全重载。

#### 架构分发与平台指令注册
根据不同聊天平台的特性，这些通用指令的解析和映射方式分为以下两种：

```text
               ┌──────────────────────────────────────────┐
               │    `Chatbot._register_default_commands()`│
               └────────────────────┬─────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
      [Discord Slash 映射]                   [文本前缀匹配解析]
    - setup_discord_commands(chatbot)     - 微信 (WeChat) 或 Console 适配器
    - 读取注册中心中的指令定义列表         - 拦截所有以 "/" 开头的用户输入
    - 创建 discord.app_commands.Command   - 解析出指令名称和后续参数字符串
    - 绑定回调方法至 CommandTree 树中      - 执行：`commands.execute()`
```

*   **Discord 斜杠指令原生注册**：
    *   Discord 适配器采用动态指令构建器 (`src/kesoku/gateway/chatbot/discord/command.py`)。
    *   它循环遍历通用指令注册表，为每个指令构建原生的 `discord.app_commands.Command` 实例。
    *   对于携带参数的指令（例如 `/role {role_name}` 或 `/cronjob {tag}`），使用闭包工厂方法（Closure Factory）为回调函数附加显式类型声明，以便 Discord 客户端界面能渲染出原生的参数输入框。
    *   构建完成后，调用 `chatbot.tree.sync()` 将指令树全量同步至 Discord 服务器。
*   **文本前缀匹配解析**：
    *   对于不支持原生斜杠指令 API 的平台适配器（例如微信或控制台终端），适配器在收到用户输入时进行检查。
    *   若以 `/` 开头，则拦截该消息（不发送给大模型处理），提取出指令名称与参数，直接执行 `self.commands.execute(cmd_name, reply_func, **kwargs)`。

### 2. 订阅者事件循环 (Subscriber Loop)
当适配器启动 `start()` 时，它会开启一个长轮询协程任务，订阅网关分发的外部事件：

```python
async for msg in self.gateway.listen(
    exclude_statuses=[MessageStatus.DELIVERED, MessageStatus.PENDING_AGENT, MessageStatus.PROCESSING],
    exclude_roles=[MessageRole.USER],
    **filters
):
    await self.handle_message(msg)
```
这种设计保持了聊天机器人适配器的**无状态性**。它们只需作为一个解耦的订阅者，等待网关将最终推理结果或中间思维流分发过来并渲染输出即可。

---

## ⚙️ 出站递送模板方法 (`render_outgoing_message`)

子类通常通过在其 `handle_message` 方法中调用 `self.render_outgoing_message(message)` 来利用基类提供的标准化投递流水线：

```text
┌────────────────────────────────────────────────────────┐
│ 1. 过滤中间步骤消息 (Thoughts/Tool Calls)              │
├────────────────────────────────────────────────────────┤
│ 2. 预处理 Markdown 表格 (自动渲染为 PNG 图像附件)       │
├────────────────────────────────────────────────────────┤
│ 3. 分割解析消息内容块 (Text, Files, Q&A 按钮等)        │
├────────────────────────────────────────────────────────┤
│ 4. 路由各个子数据块，进行物理投递                      │
├────────────────────────────────────────────────────────┤
│ 5. 将数据库消息状态更新为 DELIVERED                     │
├────────────────────────────────────────────────────────┤
│ 6. 触发投递完成回调 hook on_message_delivered()        │
└────────────────────────────────────────────────────────┘
```

### 1. Markdown 表格图像化
如果消息文本中包含 Markdown 语法表格，基类模板方法会自动拦截，通过 `render_table_to_image()` 将其渲染为高分辨率的 PNG 图片并存入会话 Staging 目录，同时在内容中将表格语法自动替换为附件标签：`[file: /path/to/table.png]`。

### 2. 内容段落解析
模板方法利用 `parse_message_content()` 将消息剥离为以下段落单元，进行分别处理：

*   `{"type": "text", "content": "..."}`
*   `{"type": "file", "path": "..."}`
*   `{"type": "voice", "path": "..."}`
*   `{"type": "question", "question": "...", "choices": [...]}`

其中，普通文本段落会依据各平台的字数上限约束（通过 `get_max_text_length()` 获取，默认采用 Discord 的 2000 字符限制）进行安全的行切块和代码块标签重补齐，再行分批投递。

---

## 🚀 编写自定义聊天平台适配器

以开发一个飞书或 Slack 适配器 (`SlackChatbot`) 为例：

### 步骤 1：继承 `Chatbot` 基类
编写实现类，并提供底层的具体消息投递 hook：

```python
from kesoku.gateway.chatbot.base import Chatbot
from kesoku.db import Message

class SlackChatbot(Chatbot):
    
    async def handle_message(self, message: Message) -> None:
        # 移交出站逻辑给模板方法处理
        await self.render_outgoing_message(message)

    async def send_text_chunks(self, channel_id: str, chunks: list[str], message: Message) -> None:
        for chunk in chunks:
            await self.slack_client.chat_postMessage(channel=channel_id, text=chunk)

    async def send_file_segment(self, channel_id: str, file_path: str, message: Message) -> None:
        # 调用 Slack Web API 上传本地文件
        await self.slack_client.files_upload_v2(channel=channel_id, file=file_path)

    async def send_question_segment(self, channel_id: str, question: str, choices: list[str], message: Message) -> None:
        # 发送 Slack Interactive Blocks 交互式单选按钮卡片
        ...

    def get_max_text_length(self) -> int:
        return 3000 # 覆盖设置 Slack 平台更长的字数分片限制
```

### 步骤 2：覆盖控制 Hook (可选)
如果需要处理复杂的输入中状态 (Typing spinner) 或卡片更新动作，可以覆盖重写以下内置回调方法：

*   `supports_intermediate_messages()`：如果平台支持在同一张卡片内更新中间步骤或日志，返回 `true`。
*   `pre_ingest_hook()`：在用户发起会话、消息入队前运行（例如立即开启输入状态动画）。
*   `on_message_delivered()`：在消息发送成功、转换为 DELIVERED 状态后运行（例如关闭输入状态动画，或发送会话计时统计）。
