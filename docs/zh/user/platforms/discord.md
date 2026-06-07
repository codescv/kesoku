# Discord 机器人接入指南

本指南将指导您如何创建 Discord 机器人、获取访问 Token、生成邀请链接、将机器人加入您的服务器并完成 Kesoku 的配置。

---

## 🛠️ 1. 创建 Discord 开发者应用

要将 Kesoku 对接到 Discord，您需要先注册一个机器人应用：

1. 打开 [Discord Developer Portal](https://discord.com/developers/applications)。
2. 点击右上角的 **New Application**，为机器人命名（例如 `Kesoku Agent`）。
3. 选择左侧菜单的 **Bot**。
4. 点击 **Reset Token** 并复制生成的 Token，请妥善保存。
    *   *此 Token 将作为 `config.toml` 中的 `bot_token` 参数，或者系统环境变量 `DISCORD_TOKEN`。*

---

## ⚙️ 2. 启用网关权限 (Gateway Intents)

为了允许 Kesoku 正常监听频道消息和服务器成员列表，您必须手动开启**特权网关权限**：

1. 在 **Bot** 页面中，向下滚动至 **Privileged Gateway Intents** 区域。
2. 勾选并开启以下三个选项：
    *   **Presence Intent**
    *   **Server Members Intent**
    *   **Message Content Intent**（关键：允许机器人读取用户发送的消息文本内容）
3. 点击 **Save Changes** 保存更改。

---

## 🔗 3. 邀请机器人加入您的服务器

生成 OAuth2 链接以将机器人邀请进您的 Discord 频道：

1. 进入左侧菜单的 **OAuth2 > URL Generator** 选项卡。
2. 在 **Scopes** 权限域中，勾选 **bot** 和 **applications.commands**（用于支持斜杠命令）。
3. 在 **Bot Permissions** 权限细节中，勾选以下权限：
    *   *普通权限*：`Read Messages/View Channel`（读取频道信息）。
    *   *文本权限*：`Send Messages`（发送消息）、`Create Public Threads`（创建公开线程）、`Send Messages in Threads`（在线程中发消息）、`Embed Links`（嵌入链接）、`Attach Files`（上传附件）、`Read Message History`（读取消息历史记录）、`Use Slash Commands`（使用斜杠命令）。
4. 复制页面底部自动生成的 URL 链接。
5. 在浏览器中打开该链接，选择要邀请进的服务器，点击 **Authorize** 授权加入。

---

## 📝 4. 配置 `config.toml`

在 `config.toml` 文件的 `[discord]` 配置块中填入对应的参数：

```toml
[discord]
enabled = true
bot_token = "你的_DISCORD_BOT_TOKEN"    # 若为空，则默认读取 DISCORD_TOKEN 环境变量
chatbot_id = "discord"                  # 机器人的唯一标识符
user_allowlist = ["允许的用户名"]       # 可选：限制仅允许特定的用户名或 ID 触发机器人

# 针对特定频道的配置覆盖
[[discord.channels]]
channels = ["1234567890", "general"]   # 匹配频道 ID 或精确匹配频道名称
llm = "claude"                          # 在这些频道中，强制使用 Claude 模型
auto_thread = false                     # 在这些频道中，关闭自动创建子线程的行为
```

### 参数详细说明：
*   **`enabled`**（布尔值）：设置为 `true`，以在启动 Kesoku 守护进程 (`kesoku start`) 时同时监听 Discord 消息事件。
*   **`user_allowlist`**（字符串列表）：如果填入用户名，则机器人仅对此白名单内的用户进行主动应答。其他用户发送消息时，只有在内容中显式 `@mention` 机器人时，机器人才会进行回复。
*   **`auto_thread`**（布尔值，默认：`true`）：如果设置为 `true`，机器人在收到新会话的指令时，会自动在当前频道下创建一个子线程（Thread）进行对话隔离，防止多用户发言混杂。如果设置为 `false`，则对话直接在当前频道内进行。
