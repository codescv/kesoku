# Google Chat Adapter Technical Reference

This guide details the technical implementation of Kesoku's Google Chat chatbot adapter (`GoogleChatChatbot`), covering Pub/Sub pulling loops, credential helpers, and interactive Cards V2 generation.

---

## 📁 File Structure

The Google Chat chatbot adapter is implemented inside `src/kesoku/gateway/chatbot/google_chat/`:

*   **`adapter.py`**: Core `GoogleChatChatbot` class inheriting from `Chatbot`, managing the Pub/Sub pull listener and GCP API connections.
*   **`cards.py`**: Builder helpers for Google Chat Cards V2 layouts, generating foldable UI components for agent thoughts and tools.

---

## ⚙️ Core Architecture

### 1. Asynchronous Pub/Sub Pull Listener
Unlike typical HTTP webhook-based bots which require incoming web traffic, public DNS records, and SSL certs, Kesoku uses a firewall-friendly **Pub/Sub Pull Subscription**:

*   The adapter starts a background listener thread using `google.cloud.pubsub_v1.SubscriberClient`.
*   It subscribes to the subscription ID configured in `config.toml` (`[google_chat].subscription_id`).
*   When a user interacts with the chatbot (sends a DM, mentions it in a space, or clicks a card button), Google Workspace posts an event to the Pub/Sub topic, which is pulled asynchronously by the adapter.
*   Incoming payloads are parsed, mapped as `InboundMessageDTO` objects, and posted to the Gateway.

### 2. Thread-Based Context Separation
*   Every Google Chat message includes a `thread.name` parameter.
*   The adapter maps `thread.name` directly as the Kesoku `channel_id` and `session_id`.
*   Replies to that session are explicitly sent back to the same thread using:
    ```python
    body = {
        "text": "...",
        "thread": {"name": channel_id},
        "messageReplyOption": "REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"
    }
```

---

## 🎨 Collapsible Card UI (`cards.py`)

Since Google Chat does not support editing previously sent raw text messages in-place, Kesoku uses **Google Chat Cards V2** to render intermediate steps dynamically.

### 1. Foldable Thoughts Card
While the agent is thinking and running tools:

*   The adapter posts an intermediate Card V2 message.
*   Thoughts, tool calls, and system notifications are grouped inside a single collapsible section titled **"Thoughts & Tools"** (built inside `cards.py`):
    *   **Thoughts**: Displayed under a collapsible panel widget.
    *   **Tool Calls**: Formatted with inline code tags (`<code>`) displaying execution arguments.
*   As the agent executes new steps, the adapter edits the active card in-place using Google Chat's message update endpoints.

### 2. Final Message Card
*   Once the turn completes or gets interrupted, the adapter updates the card to its final state.
*   **Markdown Support**: The final assistant text response is rendered inside a paragraph widget with `textSyntax="MARKDOWN"` enabled, rendering lists, bold/italic, and links correctly.
*   **Turn Metrics**: Appends session turns, window size, execution time, and token counts to the final card metadata view.
