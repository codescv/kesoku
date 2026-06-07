# Google Chat 适配器技术规范

本技术指南详述了 Kesoku 的 Google Chat 聊天机器人适配器 (`GoogleChatChatbot`) 的底层实现，涵盖 GCP Pub/Sub 异步拉取订阅循环、服务账号身份认证以及卡片（Cards V2）动态渲染引擎。

---

## 📁 目录结构

Google Chat 适配器核心代码存放于 `src/kesoku/gateway/chatbot/google_chat/` 路径下：

*   **`adapter.py`**：核心 `GoogleChatChatbot` 类，继承自 `Chatbot` 基类，管理与 GCP 服务的连接以及 Pub/Sub 消息监听器任务。
*   **`cards.py`**：Google Chat Card V2 卡片组件生成器，用于组装中间思考步骤折叠面板与最终性能指标元数据。

---

## ⚙️ 核心技术架构

### 1. 基于 GCP Pub/Sub 的异步消息拉取循环
不同于传统的 HTTP Webhook 机器人必须接收外部 HTTP 请求并暴露公网 IP/域名，Kesoku 采用了免防火墙配置的 **Pub/Sub 异步拉取机制**：
*   适配器在启动时利用 `google.cloud.pubsub_v1.SubscriberClient` 开启一个后台消息接收协程。
*   持续拉取在 `config.toml` 中配置的 Pull 订阅号 (`[google_chat].subscription_id`)。
*   当用户与机器人对话或在群组空间中提及机器人时，Google Chat API 会将事件推送至 GCP 主题，进而由适配器异步拉取并解析为统一的 `InboundMessageDTO`，推送至网关处理。

### 2. 基于线程名称的会话上下文隔离
*   Google Chat 的每条消息都包含唯一的线程标识符 `thread.name`。
*   适配器直接将该 `thread.name` 作为网关的 `channel_id` 和唯一的 `session_id` 进行关联绑定。
*   所有的出站回复消息都显式包含该线程名，并设置消息重定向规则，以确保回复精准落入相同的会话窗口中：
    ```python
    body = {
        "text": "...",
        "thread": {"name": channel_id},
        "messageReplyOption": "REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"
    }
    ```

---

## 🎨 可折叠交互卡片渲染 (`cards.py`)

由于 Google Chat 平台不支持对已发送的纯文本消息进行局部替换或修改，Kesoku 使用了 **Google Chat Cards V2** 来呈现 Agent 思考和工具调用的中间状态。

### 1. 思考步骤折叠卡片 (Foldable Thoughts Card)
在 Agent 进行推理和调用工具期间：
*   适配器会向聊天空间投递一张可折叠的卡片。
*   所有的中间思考逻辑（Thoughts）、工具调用声明（Tool Calls）及系统通知会被打包到一个名为 **"Thoughts & Tools"** 的折叠面板中：
    *   **思维链**：放置于折叠卡片组件中。
    *   **工具调用**：格式化为带有 `<code>` 标签的代码快，展示调用的工具参数。
*   工具返回结果后，适配器通过接口在原消息卡片上进行原地更新编辑，实时刷新状态。

### 2. 最终回复卡片
*   当回合推理结束或由于抢占被强行中断时，适配器更新卡片为最终样式。
*   **Markdown 解析**：最终文本段落启用 `textSyntax="MARKDOWN"` 配置，确保大模型输出的列表、加粗、代码块在飞书/Google Chat 客户端内以正确的格式进行富文本排版。
*   **指标分析**：在卡片底部自动拼入当前回合的性能指标信息（处理回合数、Token 窗口开销、执行耗时等）。
