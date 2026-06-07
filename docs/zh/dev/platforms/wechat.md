# 微信 (WeChat) 适配器技术规范

本技术指南详述了 Kesoku 微信聊天机器人适配器 (`WechatChatbot`) 的底层实现，涵盖腾讯 iLink 平台 REST API 轮询机制、终端 ASCII 二维码扫码配对流程以及即时递送模式的设计。

---

## 📁 目录结构

微信适配器核心代码存放于 `src/kesoku/gateway/chatbot/wechat/` 路径下：

*   **`adapter.py`**：核心 `WechatChatbot` 适配器类，继承自 `Chatbot`，管理消息流分发与终端交互。
*   **`client.py`**：腾讯 iLink 对话接口的 REST API 封装库。
*   **`listener.py`**：异步长轮询接收器，用于从 iLink 服务拉取用户入站事件。
*   **`media.py`**：媒体转换服务，用于对上传的文件/语音进行编码格式重转换以符合微信 API 规范。

---

## ⚙️ 核心技术架构

### 1. 终端二维码配对流程 (QR Login)
命令行执行 `kesoku wechat pair` 时：

*   适配器调用 `client.py` 向腾讯 iLink 服务发起登录事务申请，获取临时的授权 Ticket。
*   将该 Ticket 生成为 ASCII 二维码矩阵，直接在终端标准输出中渲染打印。
*   开启状态长轮询，等待用户在微信手机端扫码并确认授权。
*   授权通过后，配对逻辑将获得的 `account_id`、授权 `token` 等核心凭证写入配置管理单例，并自动序列化持久保存进本地 `config.toml`。

### 2. 长轮询消息监听协程 (`listener.py`)
由于微信普通对话平台不提供反向推送 Webhook，系统采用主动拉取的 **长轮询接收机制**：

*   `WechatListener` 在后台以非阻塞 `asyncio` 方式循环运行。
*   使用配对成功的 Token，持续请求 iLink 接口获取最新的会话包。
*   一旦收到新消息，解包格式化为统一的 `InboundMessageDTO` 消息体，推送至网关调度运行。

---

## 🛠️ 即时递送模式 (Instant Delivery Mode)

因为微信聊天窗口不具备动态输入提示状态、亦不支持在群聊中原地修改已发送的文本或插入复杂的交互卡片：

1.  **禁用中间消息渲染**：
    微信适配器重写覆盖 `supports_intermediate_messages()` 回调，直接返回 `False`。

2.  **后台静默直通**：
    当网关向适配器分发中间思维过程（`MessageType.THOUGHT`）、工具调用声明（`MessageType.TOOL_CALL`）或系统消息时：

    *   基类投递模板检测到该适配器不支持中间消息渲染。
    *   投递动作立即短路，直接将数据库中该消息的状态更改为 `DELIVERED`（已递送），但**不向**微信聊天室中发送任何垃圾消息。
3.  **最终响应物理投递**：
    *   只有当回合推理彻底结束、Agent 输出角色回复文本（`MessageRole.ASSISTANT`）时，适配器才会分段将文本投递给微信终端用户。
    *   这种设计保证了微信群聊界面的整洁性，同时确保了 SQLite 数据库中推理 Trace 日志的完整性。
