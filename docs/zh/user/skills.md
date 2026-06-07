# 技能与自定义工具

Kesoku 包含一套自治技能系统（Autonomous Skill System），允许 Agent 在会话运行期间动态地发现、读取并学习特定领域的指令和工具调用能力。

---

## 📁 技能目录结构

技能是以独立文件夹的形式存放在已配置的 `skills_dir` 目录（默认为工作目录下的 `skills/`）中：

```text
kesoku/
└── skills/
    ├── ai-image/                 # 图像生成技能
    │   ├── SKILL.md              # 技能定义文件（包含 YAML 元数据的 Markdown）
    │   └── scripts/
    │       └── gen_image.py      # 被 Agent 调用的辅助脚本
    └── web-search/               # 网页搜索技能
        └── SKILL.md
```

---

## ✍️ 开发自定义技能

要为 Agent 增加一项新能力，您只需在 `skills/` 下创建一个新文件夹，并在其中编写 `SKILL.md` 指导文件。

### 步骤 1：定义技能清单（YAML 元数据）
`SKILL.md` 必须以包裹在 `---` 之间的 YAML 元数据块开头，定义技能名称、描述及平台限制。

示例 `skills/my-git-helper/SKILL.md`：

```yaml
---
name: git-helper
description: 运行自动化的 git 分支管理与代码提交脚本。
metadata:
  tags: [git, automation]
  platforms: [linux, darwin]   # 可选：在不支持的操作系统上自动隐藏此技能
---

# Git 助手操作指南

你拥有位于 `scripts/git_commit.py` 的 Git 自动化提交脚本。
要提交更改，请运行以下终端命令：
`uv run scripts/git_commit.py -m "提交信息"`

在操作前，请先使用 `git branch` 确认当前分支。
```

### 操作系统平台兼容性
如果某个技能依赖于特定操作系统的二进制工具，可以在 `metadata.platforms` 中进行声明：

*   `platforms: [linux, darwin]`：该技能仅在 Linux 和 macOS 宿主机上加载，在 Windows 上自动隐藏。
*   省略 `platforms` 属性：默认该技能为跨平台，在所有操作系统上都会被加载。
*   `platforms: []`：全局禁用此技能。

---

## 🛠️ Agent 如何发现和使用技能

Kesoku 为 Agent 提供了两个内置的自治工具来管理技能：

1.  **`list_skills()`**：
    Agent 可以随时调用此工具扫描 `skills/` 目录。系统会识别当前的宿主机操作系统，自动过滤掉不兼容的技能，并向 Agent 返回可用技能的名称与简要介绍列表。

2.  **`use_skill(skill_name)`**：
    当 Agent 判定需要执行与某个技能相关的复杂任务时：

    *   它会调用 `use_skill(skill_name="git-helper")`。
    *   该工具读取 `skills/git-helper/SKILL.md` 的全部说明内容。
    *   **绝对路径重定向**：系统会自动在返回的说明头部注入该技能目录的绝对路径，并指示 Agent 在执行 CLI 脚本时使用该绝对路径（例如 `python /absolute/path/to/skills/git-helper/scripts/git_commit.py`），确保脚本执行稳定。
    *   加载完毕后，技能说明会被作为系统上下文注入会话，Agent 即刻习得并执行新技能。
