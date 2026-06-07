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
    ├── coder/
    │   └── intro.md         # Coder 人设提示词
    └── helper/
        └── intro.md         # Helper 人设提示词
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

在 Kesoku 中，角色人设的切换是完全动态的，支持通过斜杠指令（Slash Commands）或直接在聊天中触发。

### 1. 使用斜杠指令
在支持的应用平台（如 Discord 或微信）上，您可以通过以下指令即时管理角色：
*   `/role`：查看当前活跃的角色名称，并下列出当前工作区中所有可用的角色列表。
*   `/role {名称}`：为当前频道或线程绑定并切换到指定的角色（例如 `/role coder`）。

> [!TIP]
> **Discord 线程继承**：在 Discord 中，新建的子线程默认会自动继承父级文本频道的角色绑定关系。当然，您随时可以在子线程中运行 `/role {名称}` 来为其单独绑定其他不同的人物设定。

### 2. 通过对话直接触发
您也可以直接在会话中以日常口吻要求智能体进行角色切换：
*   *“请切换到 coder 人设”*
*   *“扮演 helper”*
*   *“换回默认角色”*

在幕后，智能体将自动识别您的切换意图，调用工具 `play_role(role="coder")`，更新数据库绑定并重构当前会话的系统提示词。

---

## 🎨 教程：如何创建生动的角色人设（集成声音克隆与图像生成）

您可以为 Kesoku 智能体量身定制一个高度个性化的角色，包括专属的头像、图像一致性参考样本，以及原汁原味的声音克隆（TTS）输出。

### 角色文件夹结构
在您的 `roles/` 目录下创建一个与角色名字相同的子文件夹（例如 `roles/alice/`）：

```directory
roles/alice/
├── intro.md              # 角色生平设定、性格规则与生成调用指南
├── images/
│   └── character.jpg     #  Alice 的半身照/头像（用作图像生成一致性参考图）
├── audio/
│   └── character.wav     # Alice 的清晰语音音频片段（5-15 秒，用作声音克隆基准）
└── scripts/
    ├── character-tts.sh  # 执行文本转语音声音克隆的脚本
    └── character-image.sh # 执行 Alice 图像渲染的脚本
```

### 步骤 1：撰写人设介绍文件 `intro.md`
在 `intro.md` 中定义 Alice 的性格，并指导大模型在生成语音和图像时，必须通过调用特定的脚本来完成：

```markdown
# 姓名
Alice 🌸

# 角色人设背景
- 你是 Alice，一名充满活力的初级游戏策划。
- 讲话风格偏口语化，喜欢频繁使用各种可爱的表情符号。

# 语音与视觉输出规范
- **语音生成 (TTS)**：只要用户要求你说话、发语音或者用语音回复，你必须通过运行 `${AWD}/roles/alice/scripts/character-tts.sh` 来生成对应的 WAV 音频文件。
- **图像生成**：只要用户要求发送你的照片或自拍，你必须通过运行 `${AWD}/roles/alice/scripts/character-image.sh` 进行生成，并参考`${AWD}/roles/alice/images/character.jpg`。
```

### 步骤 2：添加声音样本与 TTS 脚本
1. 准备一段 10 秒左右清晰、无背景噪音的 Alice 目标声音音频，保存为 `roles/alice/audio/character.wav`。
2. 编写包装 Qwen-TTS（或其它声音克隆模型）的 Shell 脚本 `roles/alice/scripts/character-tts.sh`：

```bash
#!/bin/bash
# 用法: character-tts.sh "要说的文本内容" "/输出音频绝对路径.wav"
TEXT=$1
OUTPUT_PATH=$2
REF_VOICE="${AWD}/roles/alice/audio/character.wav"

# 执行 TTS 声音克隆
uv run python -m qwen_tts --text "$TEXT" --ref-audio "$REF_VOICE" --output "$OUTPUT_PATH"
```

### 步骤 3：添加头像样本与绘图脚本
1. 将 Alice 的基准半身照图片放入 `roles/alice/images/character.jpg`。
2. 编写 `roles/alice/scripts/character-image.sh` 脚本，在调用 `ai-image` 绘画技能时传入此半身照，确保生成的画作中 Alice 的面容与发型具有高度一致性。

### 步骤 4：专属的备忘记忆隔离
当 Alice 角色处于活跃状态时，智能体通过 `update_memory` 写入类别为 `"memo"`（自定义备忘）的记忆时，会自动打上 `role='alice'` 的专属标签，从而与其它角色的记忆完全隔离，防止不同人设之间的互动记忆发生混乱。
