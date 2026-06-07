# WeChat Adapter Technical Reference

This guide details the technical implementation of Kesoku's WeChat chatbot adapter (`WechatChatbot`), covering iLink REST API clients, qr login flow, and instant delivery modes.

---

## 📁 File Structure

The WeChat chatbot adapter is implemented inside `src/kesoku/gateway/chatbot/wechat/`:

*   **`adapter.py`**: Core `WechatChatbot` class inheriting from `Chatbot`, managing the polling daemon and chat message routing.
*   **`client.py`**: REST API client wrapper for Tencent's iLink conversational platform.
*   **`listener.py`**: Asynchronous message receiver polling the iLink gateway.
*   **`media.py`**: Media handling utilities (converting images and files for WeChat compatibility).

---

## ⚙️ Core Architecture

### 1. Terminal QR Code Pairing Flow
The pairing flow is handled inside `entrypoint.py` and calls `qr_login()` from `adapter.py`:

*   Initiates an authorization request to the iLink platform to obtain a temporary pairing ticket.
*   Converts the ticket URL into an ASCII QR Code and prints it directly to the terminal.
*   Polls the pairing status endpoint until the WeChat user scans the QR code and clicks authorize on their mobile phone.
*   Once authorized, it saves the retrieved auth tokens and base URLs directly into the Operator configuration singleton, updating the local `config.toml` file.

### 2. Event Polling Daemon (`listener.py`)
Since WeChat does not offer push subscriptions or inbound HTTP webhooks for normal personal accounts, the adapter runs a continuous polling daemon:

*   `WechatListener` runs a non-blocking `asyncio` loop pulling events from Tencent iLink APIs using the active authentication token.
*   It routes incoming text messages and uploads to the Kesoku Gateway Broker.

---

## 🛠️ Instant Delivery Mode

Because WeChat chatrooms do not support message status updates (e.g. typing status, editing previously sent text blocks, or posting foldable UI widgets):

1.  **Disabled Intermediate Messages**:
    The adapter overrides `supports_intermediate_messages()` to return `False`.

2.  **Stateless Bypass**:
    When the gateway posts intermediate thoughts (`MessageType.THOUGHT`), tool calls (`MessageType.TOOL_CALL`), or system notifications (`MessageRole.SYSTEM`):

    *   The base `render_outgoing_message` template detects that intermediate messages are unsupported.
    *   It immediately transitions their database status to `MessageStatus.DELIVERED` without attempting to deliver them to the WeChat room.
3.  **Final Message Delivery**:
    *   Only final assistant text responses are chunked and delivered to the WeChat user or group.
    *   This ensures that the chatroom stays clean, while the database still retains the full reasoning log.
