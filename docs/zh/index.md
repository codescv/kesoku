# Kesoku ⚡️

欢迎来到 **Kesoku** (结束 / Kessoku) 官方文档网站。

Kesoku 是一个极简、易读且高度模块化的自主 AI 智能体 (Agent) 框架。它围绕解耦的网关架构和纯 Pub/Sub 代理 (Broker) 模式设计，支持在多个聊天前端（如 Discord、Google Chat、微信）与自主推理循环（具备并发任务、思维中断和工具执行能力）之间进行可靠的异步通信。

---

## 🌟 核心特性

*   **解耦的代理网关**：适配器 (Discord, Google Chat, WeChat) 与后端 Agent 逻辑完全解耦，纯粹通过状态驱动的消息队列进行通信。
*   **异步并发处理**：调度器为每个会话启动专用的任务循环 (`SessionWorker`)，防止多用户会话碰撞与上下文串扰。
*   **防卡死与思维中断**：当在推理中途收到新的用户输入时，支持即时中断当前思维，同时保证正在运行的工具调用（如 Shell 写入）安全执行完毕。
*   **结构化 TOML 配置**：在 `config.toml` 中集中管理模型参数、数据库路径及各平台凭证。
*   **编码与上下文压缩套件**：内置专用 `read_file` 与带行号提示/Diff 的 `update_file` 工具、首尾保留中间截断 (`truncate_middle`)、面向任务续接的分层压缩（含 `Dead Ends` 与 `Pending State` 以及注水分配的 `<recent_user_messages>`）、每轮冻结状态快照（`<running_jobs>`、`<staging_artifacts>`、`<repo_rules>`）、后台任务管理工具 `manage_job`、只读调查子智能体 `delegate_subagent` 以及可配置工具/结束钩子 (`[hooks]`)。
*   **自治技能系统与静态索引 (Skills)**：支持在系统提示词中注入静态 `# Available Skills` 索引并按需通过 `use_skill` 加载完整的 `SKILL.md`。
*   **操作系统守护进程集成**：提供服务安装命令，一键将 Agent 注册为 `systemd` 或 `launchd` 系统服务。

---

## 🗺️ 文档导览

您可以根据需要选择阅读以下指南：

### 👥 用户手册
*   [**安装指南**](user/installation.md)：使用 `uv` 工具链初始化 Python 运行环境。
*   [**配置指南**](user/configuration.md)：详细了解 `config.toml` 文件中的各项配置参数。
*   [**平台配置**](user/platforms/discord.md)：将 Agent 对接至 Discord、Google Chat 或微信。
*   [**系统服务管理**](user/service.md)：将 Kesoku 配置为后台守护进程运行。

### 💻 开发者指南
*   [**架构与设计**](dev/architecture.md)：深入探索代理网关、会话 Worker 以及并发中断模型的设计原理。
*   [**智能体生命周期**](dev/agent-cycle.md)：分步剖析一条消息如何在各个组件间流动与响应。
*   [**自治技能原理**](dev/skills.md)：了解如何通过创建技能目录为 Agent 扩展新的工具与能力。
