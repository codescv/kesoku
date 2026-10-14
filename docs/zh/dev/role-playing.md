# 角色扮演实现原理

本技术文档详述了 Kesoku 如何管理、存储以及跨会话通道（Channel/Thread）动态解析角色人设（Roles）的底层实现机制。

---

## 💾 数据库模型设计 (`channel_roles` 表)

所有的角色通道绑定映射均被保存在 SQLite 数据库的 `channel_roles` 表中：

```sql
CREATE TABLE IF NOT EXISTS channel_roles (
    chatbot_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    role TEXT NOT NULL,
    PRIMARY KEY(chatbot_id, channel_id)
);
```

当用户通过 `/role` 机器人指令请求切换角色时：

1. 系统会通过 upsert 写入关系映射：
    ```sql
    INSERT OR REPLACE INTO channel_roles (chatbot_id, channel_id, role)
    VALUES (?, ?, ?)
    ```
2. 该绑定关系在服务重启后依然持久保留。

---

## 🔄 角色解析继承模型

当进入新的对话回合，或者调度器编译组装系统提示词时，系统会调用以下方法动态计算当前活动角色名称：
`get_channel_role_with_inheritance(chatbot_id, channel_id, session_id)`

为了保证 Discord 子线程等动态场景能够自动继承主频道的设定，角色解析遵循以下严格的继承路径：

```text
┌────────────────────────────────────────────────────────┐
│ 1. 直接查询 (Direct Lookup)                            │
│    查看当前 channel_id (如子线程) 是否绑定了专属角色？    │
└─────────────────────────┬──────────────────────────────┘
                          │ (若无)
                          ▼
┌────────────────────────────────────────────────────────┐
│ 2. 线程继承查询 (Thread Inheritance, 仅 Discord)        │
│    如果是子线程，查询最近一条消息的 parent_channel_id，   │
│    追溯主频道是否配置了绑定角色？                          │
└─────────────────────────┬──────────────────────────────┘
                          │ (若无)
                          ▼
┌────────────────────────────────────────────────────────┐
│ 3. 全局默认角色 (Fallback)                             │
│    返回全局默认角色名: "default"                         │
└────────────────────────────────────────────────────────┘
```

### 1. 直接查询 (Direct Lookup)
直接在 `channel_roles` 数据库表中检索匹配的 `(chatbot_id, channel_id)`。如果命中，则直接应用并返回该角色。

### 2. 线程继承模型 (Discord)
在 Discord 适配器交互场景下：

*   每次新建会话时，机器人通常会在此文本频道下创建一个新的子线程（Thread）。
*   该子线程在 Discord 体系中拥有独立的 ID（反映在数据库中为新 `channel_id`）。
*   为了防止用户每次在线程内都需要重新设置角色，系统会在数据库中查找该会话最近一条用户消息中的 `metadata` 字段，解析出 `parent_channel_id`（即该子线程所属的父级文本频道 ID，如 `#general` 频道的 ID）。
*   若该父频道绑定了角色设定（如 `"coder"`），则子线程会自动继承并应用 `"coder"` 角色。

### 3. 全局回退
如果当前频道和父级频道均没有显式绑定任何角色，则系统统一回退采用默认角色 `"default"`。

### 会话角色优先
在走上述频道解析链之前，`build_sys_prompt` 与回合执行器都会先检查会话自身的 `sessions.role_name`：只要它不是 `"default"`，就直接使用。切换角色会切换/新建会话，因此会话绑定是权威来源；这样即使线程会话的频道在 `channel_roles` 中没有记录（例如只通过 `channel_sessions` 映射的 Discord 线程），也能加载正确的 `intro.md` / `preferences.md`，而不是默认人设。

---

## 🧩 角色专属技能 (`roles/<role>/skills/`)

除了全局 `workspace.skills_dir`，`SkillManager` 还会发现只属于某个人设的技能：`<roles_dir>/<role>/skills/<技能名>/SKILL.md`。

*   `list_skills` / `use_skill` 会根据会话解析当前角色，并包含该角色的专属技能。
*   与全局技能同名时，角色技能优先。
*   其他角色（以及没有角色的会话）看不到这些技能。
*   角色名和技能名只允许 `[A-Za-z0-9_-]`，并使用与全局技能相同的规范路径边界检查。

因此人设专属能力（脚本及其 SKILL.md）可以完全放在角色目录内，而不会污染全局技能列表。

## 🔁 抗自我模仿措施

*   **历史选项折叠**：`agent.strip_historical_question_options = true`（默认）时，`prepare_history_for_llm` 会把「最后一个已完成回合」之前（`idx < num_turns - 2`）的助手消息中的 `[question: <题干> || <选项>]` 折叠为 `[question: ...]`。最后一个已完成回合保留完整选项，保证数字回复和点选检测正常；折叠后的前缀在回合间字节稳定，不影响隐式前缀缓存。
*   **Auto-RAG 防护**：当用户消息是上一条助手提问的某个选项（精确匹配）或简短数字选择时，跳过自动上下文检索；检索到的片段会折叠 `[question: ...]` 并截断过长的 `Agent` 行（400 字符）。

---

## 🔄 系统提示词动态重构时序

当用户在会话运行期间执行了 `/role {name}` 机器人命令切换角色时：

1.  聊天平台适配器更新 SQLite 中的通道角色绑定关系：
    ```python
    await db.set_channel_role(chatbot_id, channel_id, role_name)
    ```
2.  系统停止旧会话关联的任何活跃的 SessionWorker 以及后台任务：
    ```python
    await agent.stop_session_worker(old_session.id, immediate=True)
    await context.active_jobs.stop_all_for_session(old_session.id)
    ```
3.  系统获取或为新角色创建会话，将其设为活动会话，重新拼装编译系统提示词，并覆盖写入 SQLite 数据库的 `sessions` 表：
    ```python
    new_sys_prompt = build_sys_prompt(session=session)
    await db.update_session_system_prompt(session.id, new_sys_prompt)
    ```
4.  聊天机器人/工作协程会使用新编译的系统提示词来初始化会话。

---

## 🎨 角色创建与初始化机制 (`role-creator` 技能)

为了让用户能够便捷地创建自定义人设，Kesoku 捆绑内置了基于纯 Agent 交互生成流程的 `role-creator` 技能。它不需要特定的后台脚本，而是由 Agent 通过交互收集信息后，利用文件操作工具和技能自带的模板进行动态渲染和生成。

### 1. 角色目录规范结构

一个完整的角色人设目录被严格存放在 `${AWD}/roles/{name}/` 下，其标准布局和设计规范如下：

*   **`intro.md`**：角色 profile 主文件。包含名字、性格设定、口头禅，以及最重要的语音（TTS）和绘图（Image）脚本调用规则（指导 LLM 在需要语音/图片输出时必须调用对应的专有 shell 脚本）。
*   **`images/`**：角色的基准头像或半身照（若提供）。在调用 `ai-image` 绘画技能时作为 `--image` 基础参考图传入，实现图像生成一致性（Image-to-Image）。
*   **`audio/`**：角色的参考声音音频剪辑（通常为 WAV 格式，若提供）。用于在 `qwen-tts` 声音克隆中作为基准参考音频（Reference Audio）。
*   **`scripts/`**：存放自动生成的执行脚本：
    *   `{name}-tts.sh`：文本转语音的专属 Shell 脚本。
    *   `{name}-image.sh`：角色插图渲染的专属 Shell 脚本。

### 2. 纯 Agent 自动构建流程

当启用 `role-creator` 技能后，智能体将遵循以下构建流在 AWD (Agent Working Directory) 中渲染和拼装角色资产：

1.  **收集和规划**：交互式收集用户需求（名称、性格特点、参考音频/图片）。
2.  **创建目录结构**：使用文件工具在 AWD 下初始化 `roles/{name}/`，并创建 `images/`、`audio/` 和 `scripts/` 子文件夹。
3.  **处理参考资产**：
    *   将用户提供的基准头像图片复制并命名到 `roles/{name}/images/` 下。
    *   将用户提供的参考音频复制到 `roles/{name}/audio/` 下。
4.  **生成配置文件 (`intro.md`)**：
    *   根据交互整理出的人设特质动态组装配置内容并写入 `roles/{name}/intro.md`，重点描述语音与视觉输出规范。
5.  **渲染 Shell 脚本**：
    *   **TTS 脚本**：参考模版 `${SKILL_DIR}/template/scripts/asuka-tts.sh` 进行生成。Agent 需要在脚本内容中将名字替换为对应角色的名字，将 `REF_AUDIO` 指向复制后的 WAV 文件，将 `REF_TEXT` 设置为对应的转写文字，保存为 `roles/{name}/scripts/{name}-tts.sh`。
    *   **Image 脚本**：参考模版 `${SKILL_DIR}/template/scripts/asuka-image.sh` 进行生成。替换名字后，将 `REF_IMAGE` 指向复制后的图片，保存为 `roles/{name}/scripts/{name}-image.sh`。
6.  **赋予执行权限**：
    *   所有生成的 `.sh` 脚本在写入后，Agent 必须通过运行命令（例如 `chmod +x`）赋予它们可执行权限（`0o755`），以便在后续推理中，模型可以直接调用该脚本。

