# 系统提示词生成原理

为了确保智能体的行为一致性、安全性以及对其工具和技能的认知，Kesoku 会在每个会话回合（Turn）开始时，动态编译组装一份完整的系统提示词（System Prompt）。本指南详细介绍了系统提示词的拼装结构与加载规则。

---

## 🛠️ 组装流程

提示词是由 `src/kesoku/agent/prompt.py` 文件中的 `build_sys_prompt()` 函数进行动态拼接的。拼接时采用双换行符 (`\n\n`) 作为分界，依次拼入以下各个层级的指示内容：

```text
┌────────────────────────────────────────────────────────┐
│ 1. 活跃角色人设：roles/{role}/intro.md                   │
├────────────────────────────────────────────────────────┤
│ 2. Agent 工作根目录指示 (AWD)                           │
├────────────────────────────────────────────────────────┤
│ 3. 会话暂存工作区目录指示 (STAGING_DIR)                   │
├────────────────────────────────────────────────────────┤
│ 4. 内置功能交互规范 (技能、文件发送、按钮等)                │
├────────────────────────────────────────────────────────┤
│ 5. 全局配置提示词 (config.agent.user_prompts)           │
├────────────────────────────────────────────────────────┤
│ 6. 适配器动态注入上下文 (Discord 频道与成员元数据)          │
└────────────────────────────────────────────────────────┘
```

---

## 🔍 各模块详细解析

### 1. 活跃角色人设 (Active Persona)
*   **解析规则**：通过当前会话 ID 在数据库中查询绑定的角色名（未绑定时默认使用 `"default"`）。
*   **加载逻辑**：读取 `roles/<角色名>/intro.md` 文件的文本内容。
*   **目的**：为模型赋予特定的性格、语气风格、专业背景以及对话边界。

### 2. Agent 工作根目录 (AWD)
*   **注入格式**：
    ```markdown
    # Agent Working Directory
    > AWD='{cfg.agent_working_dir}'
    You are working in the agent working directory (AWD)...
    ```
*   **目的**：明确告诉模型当前执行终端命令的工作空间根路径，限制其去任意检索 AWD 目录之外的其他敏感系统文件。

### 3. 会话暂存工作区 (STAGING_DIR)
*   **注入格式**：
    ```markdown
    # Session Staging Directory
    > STAGING_DIR='{sessions_dir}/{session_workspace_name}'
    - This is where you are supposed to save your output files...
    ```
*   **目的**：告诉模型所有在执行过程中产生的中间产物、下载的文件、生成的图片等，都必须保存到当前会话专属的暂存目录下，避免污染全局工作区。

### 3b. 可用技能索引（Available Skills）
*   `build_skills_section()` 在目录信息之后生成确定性的 `# Available Skills` 段落：每个已安装技能一行 `- <name>: <description>`（全局 + 角色技能，同名时角色技能优先），按名称排序，描述压成一行并截断到 200 字符，最多 60 条、6000 字符，其余提示调用 `list_skills()`。
*   目的：模型直接调用 `use_skill(skill_name=...)`，不必先调用 `list_skills()`。
*   当 `use_skill` 在 `agent.disabled_tools` 中、没有技能或列举失败时省略。
*   注意：系统提示词在会话创建（及切换角色/恢复）时持久化，之后新增的技能只出现在新会话中；`list_skills()` 仍是实时列表。

### 4. 内置功能交互规范
这些是硬编码在 `prompt.py` 中的提示语，用于训练智能体理解和操作 Kesoku 的独特机制：

*   **`SKILLS_INSTRUCTIONS`**：指导模型如何使用 `list_skills()` 检索可用技能，以及使用 `use_skill(name)` 加载特定技能说明。
*   **`FILE_SENDING_INSTRUCTIONS`**：指导模型发送文件的特殊语法：`[file: /绝对/文件/路径]`。如果回复文本中包含此格式，聊天平台适配器会自动将其捕获并作为文件附件上传给最终用户。
*   **选择题**（位于 `OUTPUT_FORMATTING_INSTRUCTIONS`）：```` ```ui ```` YAML 代码块，含 `type: select | multiselect`、`prompt`、`options` 和可选的 `max`。每条回复最多一个；点击的选项以原文作为用户消息返回，多选结果为 `选择：A；B`。严禁提供“我自己手动输入 / 其他”等无意义选项（用户随时可直接打字），提供候选选项时应包含“再帮我想 X 个不同的选项”以便点击换一批。旧的 `[question: ...]` 仍会解析，但不再教给模型。
*   **折叠段落**：`<details><summary>标题</summary>正文</details>`（Discord：`▸ 标题` + 「展开」按钮；其他平台：加粗标题 + 正文）。
*   **隐藏文本**：`<spoiler>…</spoiler>` 行内隐藏文本（用户点击才显示）。由各平台适配器的 `format_text` 渲染：Discord 转为原生 `||…||`；不支持隐藏文本的平台（CLI、微信、Google Chat）去掉标签、内容直接可见。实现见 `kesoku.utils.text.render_spoilers`。
*   **`MEMORY_AND_HISTORY_INSTRUCTIONS`**：指导模型如何读写 SQLite 数据库中的结构化记忆条目。
*   **`SESSION_NOTES_INSTRUCTIONS`**：说明每回合注入的 `<context name="...">` 块（见第 7 节）、如何用 `update_file` 维护 `$STAGING_DIR/notes.md`，以及 `$STAGING_DIR/context/` 下的文件由工具维护、禁止用 `update_file` 编辑。
*   **`FILE_TOOL_INSTRUCTIONS`**：用 `read_file(file_name, start_line, end_line)` 代替 `cat`/`head`/`sed -n` 读文件；省略 `old_content` 调用 `update_file(file_name, new_content=...)` 创建文件；编辑时从 `read_file` 输出中原样复制一小段唯一的文本（去掉 `N: ` 行号前缀）；被截断的 shell 输出用 `read_file` 分页查看。
*   **`BACKGROUND_EXECUTION_INSTRUCTIONS`**：指导模型在遇到耗时较长的 background 终端命令时，如何优雅地向用户发送中间提示并立即结束本回合，等待系统后台完成通知。

### 5. 全局配置提示词 (User Prompts)
*   **解析规则**：读取 `config.toml` 中 `[agent].user_prompts` 数组里配置的所有外部提示词文件。
*   **格式包装**：
    ```markdown
    === BEGIN {文件名} ===
    {文件内容}
    === END {文件名} ===
    ```
*   **目的**：供管理员引入全局自定义的规范文档（如代码编写规范、特定的 API 调用手册等）。

### 6. 适配器动态注入上下文 (Adapter Custom Prompts)
*   **解析规则**：各个平台的 Chatbot 适配器在调用网关时，可以通过 `custom_prompt` 参数动态传入上下文。
*   **目的**：例如 Discord 适配器会读取当前服务器频道的成员列表、频道 ID 和用户代号，并将其拼入提示词，让 Agent 能够准确识别是在对谁说话。

---

## 🧩 7. 每回合注入的上下文（不属于系统提示词）

除了被缓存的系统提示词外，`TurnExecutor` 每回合会把以下内容拼到最新一条用户消息前面（这样系统提示词保持不变，利于前缀缓存）。顺序为：

1.  `<instructions>`：当前角色的 `roles/{role}/preferences.md`（每回合）。
2.  `<context name="...">` 块（每回合，见 `kesoku.agent.context_dir`）：
    *   `<context name="notes">`：会话置顶笔记，读取自 `$STAGING_DIR/notes.md`（Agent 可用 `update_file` 编辑）。
    *   `$STAGING_DIR/context/*.md` 中每个文件一个块，按文件名排序。块名为去掉开头数字序号（`20-`、`05_`）和 `.md` 后的文件名，例如 `20-scene.md` → `<context name="scene">`。这些文件由工具/技能写入，Agent 不得编辑。 块名 `notes` 保留给置顶笔记：`context/notes.md`（或 `NN-notes.md`）会被渲染为 `<context name="notes-1">` 并记录 warning。
    *   预算：单文件上限 6000 字符（截断尾部并追加 `[truncated]`）；总量上限为 `agent.context_dir_max_chars`（默认 20000），超出时从排序列表末尾开始丢弃文件并记录 warning。
3.  `<facts>`：角色的 `facts.md`（仅当天第一条消息）。
4.  启动引导 / 自动检索的历史片段，然后是 `<current_message ...>`。

`/context` 命令会把同样的 `<context>` 块显示在最新尾部回合的用户消息前。
