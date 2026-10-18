# Discord Adapter Technical Reference

This guide details the technical implementation of Kesoku's Discord chatbot adapter (`DiscordChatbot`), covering thread mapping, dynamic UI buttons, slash commands, and attachment downloads.

---

## 📁 File Structure

The Discord chatbot adapter is implemented inside `src/kesoku/gateway/chatbot/discord/`:

*   **`adapter.py`**: The core `DiscordChatbot` class inheriting from `Chatbot` and managing event hooks.
*   **`ui.py`**: UI components like `MessageHeaderView` (stop/clear/trajectory buttons) and `QuestionView` (multiple-choice buttons).
*   **`persistent_ui.py`**: Restart-persistent `DynamicItem` components for questions (buttons / select menus) and `<details>` toggles, backed by the `ui_prompts` table.
*   **`command.py`**: Slash commands management (`/restart`).
*   **`voice.py`**: Voice channel integrations.

---

## ⚙️ Core Architecture

### 1. Thread-Based Session Separation
To avoid conversation collisions in multi-user environments:

*   When a user posts a message in an override channel (if `auto_thread = true`), the bot creates a Discord thread for the conversation.
*   The **Thread ID** is mapped directly to the database `channel_id` and `session_id`.
*   All subsequent messages within that thread share the same session ID.
*   Uptime and session creation timestamps are synchronized to keep ordering consistent.

### 1.1 Ignoring System Messages
`on_message` only ingests user-authored chat messages: anything for which `message.is_system()` is true or whose `message.type` is not `MessageType.default` / `MessageType.reply` is dropped (see `is_user_chat_message`). This prevents thread renames (`channel_name_change`), pins (`pins_add`), thread creation notices (`thread_created`), member adds (`recipient_add`) and similar events from triggering an agent turn.

### 2. File and Message Splitting
*   **Length Clamping**: The adapter overrides `get_max_text_length()` to return 2000 (Discord's maximum text length constraint).
*   **Attachments parser**: When rendering outbound messages containing `[file: /abs/path/to/file]`, the adapter intercepts the block, checks if the file exists on disk, instantiates `discord.File`, and transmits it alongside the text block as a native attachment.

### 3. Incoming Attachments Processing
When a user uploads files (photos, code documents, PDFs) in Discord:

1.  The adapter catches the attachments list.
2.  It resolves the session staging workspace directory (`sessions/<session_workspace_name>`).
3.  It downloads the attachment bytes and saves them locally with sanitized filenames.
4.  It adds metadata paths under the key `"attachments"`:
    ```json
    [{"path": "/absolute/path/to/file", "mime_type": "image/png", "filename": "table.png"}]
    ```
5.  It appends reference links to the bottom of the user message.
6.  The LLM client reads these references and passes them to the model as multi-modal content blocks.

---

## 🎨 Interactive Discord UI (`ui.py`)

Kesoku uses Discord's `discord.ui.View` to inject interactive widgets:

### 1. `MessageHeaderView`
This persistent view is attached to the very first message header of each session thread. It contains three emoji-only buttons:

*   **View Trajectory (`📜`)**: Triggers an async generation of an HTML trace trajectory file using `LcmHtmlReporter` and posts it to the thread as a temporary file.
*   **Stop Turn (`🛑`)**: Cancels the worker's active `asyncio.Task` loop, flags the database prompt as `interrupted`, and deletes any intermediate thoughts or tool logs from the Discord UI.
*   **Clear Session (`♻️`)**: Stops the worker task, deletes the session SQLite history, recursively removes the session directory from the disk, and deletes the thread from the channel.

### 2. `QuestionView` (Dynamic Choices Buttons)
When the parsed content contains a question segment (a ```` ```ui ```` `type: select` block, or the legacy `[question: What is 1+1? || 2 | 3]`), it splits the content:

*   Renders the text block.
*   Appends a `QuestionView` containing buttons representing `"2"` and `"3"`.
*   Once a button is clicked, it disables all options to prevent duplicate responses, posts a confirmation message, and submits the choice directly to the Gateway Broker as a new `ROLE_USER` prompt.

### 3. `DetailsView` (Collapsible Sections)
A `<details><summary>TITLE</summary>BODY</details>` segment is sent as a message `▸ **TITLE**` with a `展开` button. Clicking it edits the message in place to show `▾ **TITLE**` and BODY (truncated to Discord's 2000-character limit) and relabels the button `收起`; clicking again collapses it.

### 4. Restart-Persistent Question Components (`persistent_ui.py`)
Questions and `<details>` toggles are rendered with `discord.ui.DynamicItem`s, so clicks keep working after `kesoku service restart`:

*   **custom_id templates**: `kui:<session_id>:<message_id>:<idx>` (option button), `kuis:<session_id>:<message_id>` (select menu) and `kud:<session_id>:<message_id>-d<n>` (details toggle). `<message_id>` is the kesoku id of the rendered assistant message.
*   **Storage**: the spec (`prompt`, `options`, `multi`, `max`, later `answered`/`chosen`; or `title`/`body` for details) is saved in the `ui_prompts(session_id, message_id, spec_json, created_at)` table when the message is sent and loaded on click. Rows are deleted with their session.
*   **Registration**: `DiscordChatbot._setup_hook` calls `Client.add_dynamic_items(QuestionButton, QuestionSelect, DetailsToggle)`; the client → adapter mapping (`register_chatbot`) gives callbacks access to the gateway.
*   **Layout**: single-select with ≤5 options → buttons on row 0 (letters A/B/… plus a list in the embed when an option is longer than 15 chars); more than 5 options → single-select dropdown; multiselect → `Select(min_values=1, max_values=max)` on row 1. `_row` is set explicitly because `discord.ui.Select(row=...)` does not set it.
*   **Answering**: a click posts the option text (multiselect: one message `选择：A；B；C`) into the same session exactly like `QuestionView` (confirmation message + simulated user message + typing indicator), stores the answer, and edits the message so every component is disabled (chosen option highlighted). Later clicks get an ephemeral notice.
*   If a custom_id would exceed Discord's 100-character limit, the adapter falls back to the in-memory `QuestionView` / `DetailsView`.
