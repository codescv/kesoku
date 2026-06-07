# Troubleshooting & FAQ

This page outlines solutions to common configuration errors, runtime crashes, and integration issues you might encounter while deploying Kesoku.

---

## 💾 SQLite Database Issues

### 1. Error: `database is locked`
*   **Cause**: SQLite only allows one process to write to the database at a time. If you run `kesoku start` as a service daemon while simultaneously executing multiple manual `kesoku chat` turns, or if a crashed process didn't release its file handles, SQLite might throw lock errors.
*   **Solutions**:
    *   Ensure you do not have multiple background processes pointing to the same `db_path`.
    *   Check for orphaned processes:
        ```bash
        ps aux | grep kesoku
        ```
    *   If using WAL mode (Write-Ahead Logging), SQLite usually handles concurrent reads and writes gracefully. Ensure the database file is placed in a local directory (not on network-mounted NFS shares which do not support locking protocols).

---

## 🤖 Discord Bot Platform Errors

### 1. Bot is online but does not respond to messages
*   **Cause**: Missing Gateway Intents in your Discord Application settings.
*   **Solutions**:
    1.  Navigate to the [Discord Developer Portal](https://discord.com/developers/applications).
    2.  Select your Application and go to the **Bot** tab.
    3.  Scroll down to **Privileged Gateway Intents**.
    4.  Enable **Message Content Intent**, **Server Members Intent**, and **Presence Intent**.
    5.  Save changes.

### 2. Error: `discord.errors.Forbidden: 403 Forbidden`
*   **Cause**: The bot does not have sufficient channel permissions (such as read/write permissions or permission to create threads).
*   **Solutions**:
    *   Ensure the bot has been granted the **Administrator** role, or at least: `Read Messages/View Channel`, `Send Messages`, `Create Public Threads`, `Send Messages in Threads`, `Embed Links`, `Attach Files`, `Read Message History`.

---

## ☁️ Google Chat & GCP Pub/Sub Errors

### 1. Error: `PermissionDenied: 403 User not authorized to perform this action`
*   **Cause**: The GCP credentials used by the bot do not have access to pull messages from the specified Pub/Sub subscription.
*   **Solutions**:
    *   Verify that your Service Account has been granted the **Pub/Sub Subscriber** role (`roles/pubsub.subscriber`) on the subscription defined in your config.
    *   If using Keyless Service Account Impersonation, verify that the active user running the bot has permission to impersonate the target service account (`iam.serviceAccounts.getAccessToken` / **Service Account Token Creator** role).

---

## 💬 WeChat Platform Errors

### 1. WeChat pairing fails or QR Code does not display
*   **Cause**: The terminal window size is too small to render the QR code cleanly, or the pairing connection timed out.
*   **Solutions**:
    *   Enlarge your terminal window and zoom out so the QR code blocks fit cleanly without wrapping.
    *   Scan the barcode using your WeChat client within 2 minutes. If it times out, re-run `kesoku wechat pair`.
