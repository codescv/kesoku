# Chatbot Interface & Custom Adapters

This guide outlines how chatbot adapters integrate with Kesoku's Broker gateway, how the subscriber event loops run, and how to write a custom chatbot adapter by subclassing `Chatbot`.

---

## 🏛️ The Base Class (`Chatbot`)

The base class is defined as `Chatbot(ABC)` in `src/kesoku/gateway/chatbot/base.py`. It provides shared state, built-in slash command parser registries, and text processing utilities.

### 1. Command Registry
Every chatbot adapter inherits a standard `CommandRegistry` populated with platform-agnostic commands:

*   `/clear` / `/reset`: Terminate the active session, stop tasks, and delete directories.
*   `/status`: Display CPU uptime, turns processed, context window size, and execution speed metrics.
*   `/compact`: Manually force lossless context pruning.
*   `/role <name>`: Switch or view channel roleplay personas.
*   `/debug`: Toggle raw LLM logs and staging directories visibility.

The adapter checks for incoming commands inside `handle_command(text, reply_func, channel_id)`.

### 2. Subscriber Event Loop
When `start()` is called, the adapter runs a continuous listener loop subscribing to outbound events:

```python
async for msg in self.gateway.listen(
    exclude_statuses=[MessageStatus.DELIVERED, MessageStatus.PENDING_AGENT, MessageStatus.PROCESSING],
    exclude_roles=[MessageRole.USER],
    **filters
):
    await self.handle_message(msg)
```
This keeps adapters stateless, operating as simple, decoupled subscribers that wait for the Gateway to route finished assistant responses or intermediate thoughts.

---

## ⚙️ Outbound Delivery Template (`render_outgoing_message`)

Subclasses typically implement `handle_message` by calling `self.render_outgoing_message(message)`. This base template method handles core rendering steps:

```text
┌────────────────────────────────────────────────────────┐
│ 1. Filter Intermediate Messages (Thoughts/Tool Calls)   │
├────────────────────────────────────────────────────────┤
│ 2. Preprocess Markdown Tables (Render to PNG images)   │
├────────────────────────────────────────────────────────┤
│ 3. Parse Message Content Blocks (Text, Files, Q&A)     │
├────────────────────────────────────────────────────────┤
│ 4. Segment Routing & Delivery Chunks                   │
├────────────────────────────────────────────────────────┤
│ 5. Update Status to DELIVERED                          │
├────────────────────────────────────────────────────────┤
│ 6. Trigger on_message_delivered() Hook                 │
└────────────────────────────────────────────────────────┘
```

### 1. Markdown Table Preprocessing
If the message contains markdown tables, the template method intercepts them, renders them to high-resolution PNG images via `render_table_to_image()`, saves them to the session staging directory, and replaces the markdown table in the content with a file attachment syntax `[file: /path/to/table.png]`.

### 2. Content Block Parsing
Using `parse_message_content()`, the template splits the content into segment blocks:

*   `{"type": "text", "content": "..."}`
*   `{"type": "file", "path": "..."}`
*   `{"type": "voice", "path": "..."}`
*   `{"type": "question", "question": "...", "choices": [...]}`

The text segments are automatically formatted (markdown optimization) and chunked to fit the platform's maximum message lengths (retrieved via `get_max_text_length()`, which defaults to 2000 for Discord compatibility).

---

## 🚀 Creating a Custom Chatbot Adapter

To create a new chatbot adapter (e.g. `SlackChatbot`):

### Step 1: Subclass `Chatbot`
Inherit from `Chatbot` and implement the abstract delivery hooks:

```python
from kesoku.gateway.chatbot.base import Chatbot
from kesoku.db import Message

class SlackChatbot(Chatbot):
    
    async def handle_message(self, message: Message) -> None:
        # Pass messages to the template renderer
        await self.render_outgoing_message(message)

    async def send_text_chunks(self, channel_id: str, chunks: list[str], message: Message) -> None:
        for chunk in chunks:
            await self.slack_client.chat_postMessage(channel=channel_id, text=chunk)

    async def send_file_segment(self, channel_id: str, file_path: str, message: Message) -> None:
        # Upload file using Slack Web API
        await self.slack_client.files_upload_v2(channel=channel_id, file=file_path)

    async def send_question_segment(self, channel_id: str, question: str, choices: list[str], message: Message) -> None:
        # Send interactive Slack block buttons representing choices
        ...

    def get_max_text_length(self) -> int:
        return 3000 # Override max character limit for Slack
```

### Step 2: Override Hook Controls (Optional)
If your platform supports typing indicators or card updates, override these hooks:

*   `supports_intermediate_messages()`: Return `true` if the platform supports collapsible cards or inline updates for thoughts/tool logs.
*   `pre_ingest_hook()`: Run setup actions (like starting the Slack typing spinner) when an inbound prompt is received.
*   `on_message_delivered()`: Run cleanup actions (like stopping the typing spinner) when delivery finishes.
