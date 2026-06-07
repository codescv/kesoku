# 微信 (WeChat) 机器人接入指南

Kesoku 支持接入基于腾讯微信智能对话（iLink）平台的微信聊天机器人。本指南将介绍如何通过终端扫码配对、查看活动频道并在 `config.toml` 中进行配置。

---

## 🔗 1. 终端扫码配对 (Pairing)

不同于 Discord 或 Google Chat 拥有静态的开发者 Token，微信机器人需要使用动态的终端二维码将您的账号与 Kesoku 进行绑定配对：

1.  在终端中执行配对命令：
    ```bash
    kesoku wechat pair -c config.toml
    ```
2.  终端屏幕上会渲染输出一个大型的 ASCII 字符二维码（QR Code）。
    *   *如果由于终端字体过大导致二维码换行错位，请缩小终端显示字体（使用 `Cmd -` 或 `Ctrl -`）或放大终端窗口。*
3.  打开手机微信，使用 **“扫一扫”** 功能扫描终端上的二维码进行绑定授权。
4.  在手机端确认授权后：
    *   终端将提示 `WeChat chatbot paired and enabled successfully!`。
    *   配对程序会自动获取连接凭证（`account_id`、`token` 和 `base_url`），并自动将其写入并保存至 `config.toml` 文件的 `[wechat]` 配置分区中。

---

## 📝 2. 查看已绑定的活动微信频道

要检查哪些微信群聊或微信好友已同 Kesoku 建立了活动的会话绑定：

```bash
kesoku wechat show-channels -c config.toml
```

此命令将连接微信适配器，在终端上列出所有活动的会话、群名称以及对应的通道 ID（Channel ID）。

---

## ⚙️ 3. 微信配置参数说明 (`[wechat]`)

扫码配对完成后，`config.toml` 会自动更新。以下是完整的配置块格式：

```toml
[wechat]
enabled = true
chatbot_id = "wechat"
account_id = "你的_ACCOUNT_ID"          # 扫码配对后由程序自动填入
token = "你的_AUTH_TOKEN"              # 扫码配对后由程序自动填入
base_url = "https://ilinkai.weixin.qq.com"  # 默认的腾讯 iLink 对接网关地址
sys_prompt_file = "prompts/wechat.md"   # 可选：微信专用的自定义系统提示词文件路径
```

### 参数详细说明：
*   **`enabled`**（布尔值）：设置为 `true`，以在启动 Kesoku 守护进程 (`kesoku start`) 时自动拉取并响应微信聊天消息。
*   **`sys_prompt_file`**（字符串，可选）：指定一个相对于工作区的 Markdown 文件路径。若指定，该文件的内容会被追加注入到该微信会话的系统提示词中，用于给微信端的机器人设定专属规则。
*   **即时递送机制（Instant Delivery Mode）**：由于微信平台本身不支持消息的撤回/修改或持续的输入中状态（Typing indicator），微信适配器采用了**即时递送模式**。Agent 在思考过程中的中间思维链（Thoughts）和工具执行结果（Tool Call/Result）将完全在后台静默处理，微信群聊中只会在推理结束后收到最终的文本回复。
