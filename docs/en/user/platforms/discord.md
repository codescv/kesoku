# Discord Bot Integration Guide

This guide will walk you through setting up a Discord bot, obtaining credentials, inviting the bot to your server, and configuring Kesoku to run it.

---

## 🛠️ 1. Create a Discord Developer Application

To connect Kesoku to Discord, you must register a bot application:

1.  Navigate to the [Discord Developer Portal](https://discord.com/developers/applications).
2.  Click **New Application** (top right) and name your bot (e.g. `Kesoku Agent`).
3.  Go to the **Bot** tab on the left.
4.  Click **Reset Token** and copy the generated token. Store it securely.
    *   *This token will be set as `bot_token` in `config.toml` or as `DISCORD_TOKEN` in your environment.*

---

## ⚙️ 2. Enable Gateway Intents

To allow Kesoku to receive channel messages and members lists, you must enable **Privileged Gateway Intents**:

1.  In the **Bot** tab, scroll down to **Privileged Gateway Intents**.
2.  Enable the following options:
    *   **Presence Intent**
    *   **Server Members Intent**
    *   **Message Content Intent** (Crucial: Allows the bot to read user prompts)
3.  Click **Save Changes**.

---

## 🔗 3. Invite the Bot to Your Server

Generate an invite URL to add the bot to your target server:

1.  Go to the **OAuth2 > URL Generator** tab.
2.  Under **Scopes**, check **bot** and **applications.commands** (enables slash commands).
3.  Under **Bot Permissions**, check the following permissions:
    *   *General Permissions*: `Read Messages/View Channel`.
    *   *Text Permissions*: `Send Messages`, `Create Public Threads`, `Send Messages in Threads`, `Embed Links`, `Attach Files`, `Read Message History`, `Use Slash Commands`.
4.  Copy the generated URL at the bottom of the page.
5.  Paste the URL into your browser, select your server, and click **Authorize**.

---

## 📝 4. Configure `config.toml`

Add your Discord configurations to the `[discord]` section in `config.toml`:

```toml
[discord]
enabled = true
bot_token = "YOUR_DISCORD_BOT_TOKEN"    # Or leave empty to use DISCORD_TOKEN environment variable
chatbot_id = "discord"                  # Unique chatbot identifier
user_allowlist = ["my_username"]       # Optional: allowed Discord usernames or user IDs

# Channel-specific overrides
[[discord.channels]]
channels = ["1234567890", "general"]   # Match channel ID or exact channel name
llm = "claude"                          # Use Claude model in these channels
auto_thread = false                     # Disable automatic thread creation in these channels
```

### Options Breakdown:
*   **`enabled`** (boolean): Set to `true` to run the Discord listener inside the Kesoku daemon loop (`kesoku start`).
*   **`user_allowlist`** (list of strings): If populated, the bot only responds to these specific users. Other users can only trigger responses if they explicitly `@mention` the bot.
*   **`auto_thread`** (boolean, default: `true`): If `true`, the bot automatically isolates every new session conversation by creating a sub-thread inside the channel. If `false`, the conversation runs directly inside the channel.
