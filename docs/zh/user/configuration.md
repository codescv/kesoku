# 配置指南

Kesoku 所有的运行参数均在项目根目录的 TOML 文件（通常为 `config.toml`）中进行集中式管理。在命令行启动时，该文件会被加载解析为 Pydantic 模型，作为全局单例共享。

---

## 🚀 初始化配置文件

在任意工作目录下运行以下命令，即可生成一份包含默认值的配置文件：

```bash
kesoku init -c config.toml
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
配置活动的 Agent 功能、上下文分层压缩、仓库规范发现与子智能体委派参数。

*   **`llm`**（字符串，默认：`"gemini"`）：当前选用的核心 LLM 驱动引擎。支持 `"gemini"` 或 `"claude"`。
*   **`protect_front_turns`** / **`protect_tail_turns`**（整数，默认：`1` / `3`）：自动上下文压缩时在会话开头和末尾保留的完整回合数。
*   **`base_node_min_tokens`**（整数，默认：`25000`）：触发自动 Level-0 摘要压缩的缓冲区 Token 阈值。
*   **`recent_user_messages_limit`** / **`recent_user_messages_total_chars`** / **`recent_user_message_ceiling_chars`**（整数，默认：`100` / `12000` / `6000`）：已压缩回合中保留在 `<recent_user_messages>` 内的用户原话数量上限、注水算法总字符预算与单条上限。
*   **`repo_rules_files`**（字符串列表，默认：`["AGENTS.md", "GEMINI.md"]`）：沿访问路径向上至 `$AWD` 自动发现并注入到 `<context name="repo_rules">` 的仓库规范文件名。
*   **`repo_rules_max_files`** / **`repo_rules_max_chars`**（整数，默认：`3` / `4000`）：最多注入的仓库规范文件数与单文件字符上限。
*   **`repo_rules_stop_reminder`**（布尔值，默认：`true`）：当回合内修改了受规范约束的仓库文件时，在结束回合前自动提醒 Agent 检查测试与规范要求。
*   **`subagent_allowed_tools`**（字符串列表，默认：`["read_file", "run_shell_command", "list_skills", "use_skill"]`）：`delegate_subagent` 子智能体允许调用的工具列表。
*   **`subagent_allow_write`**（布尔值，默认：`false`）：是否允许子智能体使用 `update_file`。
*   **`subagent_max_steps`** / **`subagent_max_concurrent`** / **`subagent_max_per_turn`**（整数，默认：`12` / `1` / `3`）：子智能体单次最大推理步数、并发上限与单回合调用上限。
*   **`subagent_shell_timeout_seconds`**（整数，默认：`120`）：子智能体前台 Shell 命令超时秒数（子智能体命令不允许转入后台）。
*   **`subagent_model_name`**（字符串，默认：`""`）：可选的子智能体专用模型名称覆盖。

### 3. `[gemini]`
配置 Google GenAI / Gemini 相关的连接参数。

*   **`model_name`**（字符串，默认：`"gemini-2.5-flash"`）：调用的模型名称。
*   **`auth_mode`**（字符串，默认：`"api_key"`）：认证模式。选用 `"api_key"` 代表直接传入 API Key，选用 `"vertex"` 代表通过 Google Cloud Vertex AI (应用默认凭证) 认证。
*   **`api_key`**（字符串）：当 `auth_mode = "api_key"` 时填入的 API Key。若为空，将自动回退尝试读取系统环境变量 `GEMINI_API_KEY`。
*   **`project_id`**（字符串）：使用 Vertex AI 模式时的 Google Cloud 项目 ID。
*   **`location`**（字符串，默认：`"us-central1"`）：使用 Vertex AI 模式时的接口物理分区区域。
*   **`thinking_level`**（字符串，默认：`"high"`）：推理和思考时间预算。支持：`"minimal"`（最少）、`"low"`（低）、`"medium"`（中）、`"high"`（高）。
*   **`safety_threshold`**（字符串，默认：`"off"`）：应用到所有可配置安全类别的拦截阈值，可选 `"off"`、`"block_none"`、`"block_low_and_above"`、`"block_medium_and_above"`、`"block_only_high"`。覆盖骚扰、仇恨言论、色情、危险内容、公民诚信五类；当 `auth_mode = "vertex"` 时还会额外覆盖仅 Vertex 支持的图像类（仇恨/危险/骚扰/色情）和越狱（jailbreak）类别。服务端默认阈值会偶发性地拦截较长的角色扮演回复（返回 `finish_reason=SAFETY` 且没有任何内容），因此默认关闭过滤；设为 `null` 即可回退到服务端默认值。

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
*   **`forbidden_patterns`**（键值为正则表达式、Value 为错误信息的字典/映射表）：匹配特定正则表达式的命令将被拒绝执行，并向 Agent 返回对应的自定义错误信息。

### 6. `[hooks]`
定义在工具调用后或回合结束前自动执行的检查钩子：

*   **`[[hooks.post_tool_use]]`**：在匹配 `tools` 及可选 `file_pattern` 正则（如 `"\\.py$"`）的工具执行后运行命令（支持 `{file}`、`{awd}`、`{staging_dir}` 占位符），并根据 `append_on`（`"nonzero"` 或 `"always"`）将 `[hook <name> exit <code>]` 附加至工具输出。
*   **`[[hooks.stop]]`**：当回合内调用过 `when_tools_used` 中的工具且 Agent 准备结束回合时运行命令；若退出码非零则提示 Agent 修复并重试（最多 `max_retries` 次）。

---

## 💬 聊天平台与适配器配置

对于 Discord、Google Chat 和微信 (WeChat) 的具体参数配置，请参阅[平台配置指南](platforms/discord.md)。
