# 角色扮演与设定配置

Kesoku 拥有灵活且动态的角色扮演 (Role-Playing) 系统，允许用户或 Agent 自身在会话中即时切换不同的人物设定（Persona）。

---

## 🎭 运行机制

每个角色设定均由存放于 `roles/` 目录下对应文件夹中的 `intro.md` 描述文件定义：

```text
kesoku/
└── roles/
    ├── default/
    │   └── intro.md         # 默认的智能体系统提示词
    ├── tifa/
    │   └── intro.md         # Tifa 人设提示词
    └── asuka/
        └── intro.md         # Asuka 人设提示词
```

当某个会话绑定了特定角色时：
1. Kesoku 读取对应 `roles/<角色名>/intro.md` 文件的内容。
2. 将其注入到系统提示词编译链的 **Active Persona** 部分。
3. 模型在当前通道的后续对话中将立即采用此人设。

---

## 🛠️ 配置自定义角色人设

### 步骤 1：创建角色目录
在已配置的 `roles_dir`（默认为工作目录下的 `roles/`）中，创建一个以角色名字命名的新文件夹（例如 `helper`）：

```bash
mkdir -p roles/helper
```

### 步骤 2：写入人设提示词 (`intro.md`)
在此文件夹中创建 `intro.md` 文件：

```bash
touch roles/helper/intro.md
```

写入人设规则，例如：
```markdown
你是一个得力的编程助手。你说话简明扼要，且总是编写整洁、注释详尽的 Python 代码。
```

### 步骤 3：初始化与覆盖角色
如果您是第一次初始化工作区，请确保角色目录已生成：
```bash
uv run kesoku init -c config.toml
```
若需要强制覆盖或恢复默认角色设定，可以运行：
```bash
uv run kesoku init -c config.toml --overwrite-roles
```

---

## 🔄 动态角色切换

在 Kesoku 中，角色切换完全是动态的，并由 Agent 自身通过调用工具进行自我维护。

### 通过对话触发
要切换当前频道/线程的角色设定，只需在对话中以日常语言对 Agent 下达指令：

*   *“请切换到 Tifa 人设”*
*   *“扮演 Asuka”*
*   *“换回默认角色”*

### 幕后原理
收到指令后：
1. Agent 识别出您的切换意图，并调用工具 `play_role(role="tifa")`。
2. 该工具验证 `roles/tifa/intro.md` 文件是否存在。
3. 在 SQLite 数据库中将当前 `(chatbot_id, channel_id)` 与角色 `"tifa"` 进行映射绑定。
4. 重新编译当前活动会话的系统提示词，并持久化更新到数据库中。
5. Agent 回复您确认切换，并在后续的推理和发言中立即生效该人设。
