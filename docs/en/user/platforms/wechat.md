# WeChat Bot Setup Guide

Kesoku supports integrating WeChat chatbot adapters powered by the Tencent iLink platform. This page guide will walk you through the pairing flow and how to configure the WeChat adapter in `config.toml`.

---

## 🔗 1. Pairing via Terminal QR Code

Unlike Discord or Google Chat which use static developer tokens, the WeChat bot requires you to pair your active account using a dynamic terminal barcode:

1.  Run the pairing command in your terminal:
    ```bash
    kesoku wechat pair -c config.toml
    ```
2.  A large ASCII barcode (QR Code) will be generated and printed inside your terminal.
    *   *If the QR code blocks look warped, zoom out your terminal (`Cmd -` or `Ctrl -`) or enlarge the window.*
3.  Open the WeChat client on your mobile phone, tap **Scan**, and scan the QR code.
4.  Once authorized on your phone:
    *   The terminal will print `WeChat chatbot paired and enabled successfully!`.
    *   The command automatically writes the retrieved credentials (`account_id`, `token`, and `base_url`) into your `config.toml` file under the `[wechat]` section.

---

## 📝 2. Reviewing Paired Channels

To see which WeChat groups or contacts are currently paired and mapped as active conversation channels in Kesoku:

```bash
kesoku wechat show-channels -c config.toml
```

This queries the active WeChat adapter and lists all chats, group names, and their associated channel identifiers.

---

## ⚙️ 3. Configuration Schema (`[wechat]`)

The pairing command automatically updates your config file. Here is the configuration block schema:

```toml
[wechat]
enabled = true
chatbot_id = "wechat"
account_id = "YOUR_ACCOUNT_ID"          # Generated automatically by the pair command
token = "YOUR_AUTH_TOKEN"              # Generated automatically by the pair command
base_url = "https://ilinkai.weixin.qq.com"  # Default Tencent iLink base URL
sys_prompt_file = "prompts/wechat.md"   # Optional: Path to custom system prompt for WeChat
```

### Options Breakdown:
*   **`enabled`** (boolean): Set to `true` to run the WeChat polling daemon inside the Kesoku daemon loop (`kesoku start`).
*   **`sys_prompt_file`** (string, optional): Path to a markdown file relative to the agent working directory. If specified, its content will be appended to the compiled system prompt specifically for the WeChat chatbot sessions.
*   **Instant Delivery Mode**: Since WeChat does not natively support message status updates (e.g. typing indicators or editing sent messages in-place), the WeChat adapter uses *Instant Delivery Mode*. All intermediate thoughts and tool execution messages are completely bypassed and hidden from the chatroom, and only the final assistant text response is sent.
