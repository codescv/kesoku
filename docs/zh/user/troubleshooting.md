# 故障排除与常见问题 (FAQ)

本页面列出了在使用和部署 Kesoku 过程中可能遇到的常见配置错误、运行崩溃以及第三方平台对接问题的解决方法。

---

## 💾 SQLite 数据库问题

### 1. 错误：`database is locked`（数据库被锁定）
*   **可能原因**：SQLite 数据库不支持多进程并发写入。如果您在启动后台守护进程 `kesoku start` 的同时，在终端频繁手动执行 `kesoku chat` 对话命令，或者由于进程非正常退出导致文件句柄未被释放，数据库可能会抛出锁定错误。
*   **解决方法**：
    *   确保没有两个以上的后台进程指向同一个 `db_path` 数据库文件。
    *   清理遗留的僵尸进程：
        ```bash
        ps aux | grep kesoku
        ```
    *   确保您的 SQLite 数据库文件存放于宿主机的本地物理磁盘中。切勿存放在网络挂载卷（如 NFS）上，因为网络卷往往对文件锁协议支持不佳。

---

## 🤖 Discord 机器人对接问题

### 1. 机器人在线，但在频道内发送消息无响应
*   **可能原因**：未在 Discord 开发者控制台开启网关权限 (Gateway Intents)。
*   **解决方法**：
    1.  打开 [Discord Developer Portal](https://discord.com/developers/applications)。
    2.  选择您的应用，点击左侧菜单的 **Bot**。
    3.  滑动至 **Privileged Gateway Intents** 区域。
    4.  勾选开启 **Message Content Intent**（消息内容权限）、**Server Members Intent**（成员列表权限）以及 **Presence Intent**（在线状态权限）。
    5.  保存设置并重启机器人。

### 2. 错误：`discord.errors.Forbidden: 403 Forbidden`
*   **可能原因**：机器人账号在当前的 Discord 服务器或特定频道中缺乏足够的权限。
*   **解决方法**：
    *   在服务器中检查机器人的角色，确保其至少拥有以下权限：`View Channel`（查看频道）、`Send Messages`（发送消息）、`Create Public Threads`（创建公开线程）、`Send Messages in Threads`（在线程中发送消息）、`Embed Links`（嵌入链接）、`Attach Files`（上传附件）、`Read Message History`（读取历史记录）。

---

## ☁️ Google Chat 与 GCP Pub/Sub 问题

### 1. 错误：`PermissionDenied: 403 User not authorized to perform this action`
*   **可能原因**：机器人使用的 GCP 凭证无权从指定的 Google Cloud Pub/Sub 订阅中拉取消息。
*   **解决方法**：
    *   前往 Google Cloud Console，确认所使用的服务账号（Service Account）已被赋予了当前 Pub/Sub 订阅的 **Pub/Sub Subscriber**（订阅者，`roles/pubsub.subscriber`）角色权限。
    *   若使用免密钥的服务账号模拟（Impersonation）机制，确保执行命令的用户身份拥有模拟该服务账号的权限（即 **Service Account Token Creator** 角色）。

---

## 💬 微信平台对接问题

### 1. 微信配对失败，或终端上的二维码显示错位
*   **可能原因**：终端窗口尺寸太小，导致 ASCII 二维码换行错位；或者配对扫码超时。
*   **解决方法**：
    *   请调大您的终端窗口尺寸，或缩小终端字体（Zoom out），使二维码能够在屏幕上完整无错位地渲染出来。
    *   配对二维码的有效扫描时间为 2 分钟，请及时扫码。若超时请重新运行 `kesoku wechat pair` 刷新二维码。
