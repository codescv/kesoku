# 配置指南

Kesoku 所有的运行参数均在项目根目录的 TOML 文件（通常为 `config.toml`）中进行集中式管理。在命令行启动时，该文件会被加载解析为 Pydantic 模型，作为全局单例共享。

---

## 🚀 初始化配置文件

在任意工作目录下运行以下命令，即可生成一份包含默认值的配置文件：

```bash
uv run kesoku init -c config.toml
```

这将在当前目录生成一个名为 `config.toml` 的配置文件。

---

## ⚙️ 配置文件结构说明

以下是 `config.toml` 中各主要配置项的详细解析：

### 1. `[workspace]`
管理 SQLite 数据库、运行日志和自定义能力的存放路径。

*   **`db_path`**（字符串，默认：`"kesoku.db"`）：SQLite 数据库文件的相对或绝对路径，用于持久化消息。
*   **`skills_dir`**（字符串，默认：`"skills"`）：自定义技能（包含 `SKILL.md` 操作手册的文件夹）的目录。
*   **`sessions_dir`**（字符串，默认：`"sessions"`）：会话工作区暂存目录，用于存放单次会话的原始 Trace 日志、执行文件及下载的媒体附件。

### 2. `[agent]`
配置活动的 Agent 功能。

*   **`llm`**（字符串，默认：`"gemini"`）：当前选用的核心 LLM 驱动引擎。支持 `"gemini"` 或 `"claude"`。

### 3. `[gemini]`
配置 Google GenAI / Gemini 相关的连接参数。

*   **`model_name`**（字符串，默认：`"gemini-2.5-flash"`）：调用的模型名称。
*   **`auth_mode`**（字符串，默认：`"api_key"`）：认证模式。选用 `"api_key"` 代表直接传入 API Key，选用 `"vertex"` 代表通过 Google Cloud Vertex AI (应用默认凭证) 认证。
*   **`api_key`**（字符串）：当 `auth_mode = "api_key"` 时填入的 API Key。若为空，将自动回退尝试读取系统环境变量 `GEMINI_API_KEY`。
*   **`project_id`**（字符串）：使用 Vertex AI 模式时的 Google Cloud 项目 ID。
*   **`location`**（字符串，默认：`"us-central1"`）：使用 Vertex AI 模式时的接口物理分区区域。
*   **`thinking_level`**（字符串，默认：`"high"`）：推理和思考时间预算。支持：`"minimal"`（最少）、`"low"`（低）、`"medium"`（中）、`"high"`（高）。

### 4. `[claude]`
配置部署于 Google Cloud Vertex AI 上的 Anthropic Claude 接入参数。

*   **`model_name`**（字符串，默认：`"claude-3-5-sonnet@20241022"`）：调用的 Vertex Claude 模型代号。
*   **`project_id`**（字符串）：Google Cloud 项目 ID。
*   **`location`**（字符串，默认：`"us-east5"`）：Vertex AI 分区区域。

### 5. `[shell]`
定义 Agent 运行系统命令行工具时的安全策略。

*   **`enabled`**（布尔值，默认：`true`）：是否允许 Agent 在宿主机上执行终端 shell 命令。
*   **`mode`**（字符串，默认：`"blocklist"`）：指令匹配过滤策略。可选 `"blocklist"`（黑名单限制模式）或 `"allowlist"`（白名单模式）。
*   **`allowlist_patterns`**（正则表达式列表，默认包含 echo/pwd/git/uv 等）：允许执行的命令行正则模式。
*   **`blocklist_patterns`**（正则表达式列表，默认包含 rm/sudo/shutdown 等）：禁止执行的危险命令行正则模式。

---

## 💬 聊天平台与适配器配置

对于 Discord、Google Chat 和微信 (WeChat) 的具体参数配置，请参阅[平台配置指南](platforms/discord.md)。
