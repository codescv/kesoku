---
name: role-creator
description: Interactive role creator to guide users in designing and bootstrapping custom character personas.
metadata:
  tags: [roleplay, persona, character, configuration]
---

捏人指南: 使用本skill来给系统添加角色.

# 1. Kesoku 角色概述

Kesoku 中的每个自定义角色都存放在全局角色目录 `${AWD}/roles/{character_name}/` 下的独立子目录中。
一个完整的角色包含以下内容(参考: `${SKILL_DIR}/template/`)：
* **intro.md**：Markdown 格式的简介，包含性格特征、习惯癖好、指令提示词以及生成脚本指南。
* **preferences.md**：Markdown 格式的用户偏好/个性化指令，在 bootstrap turn 时会作为 <instructions> 注入到提示词中。
* **images/**：（可选）包含一张参考肖像图（例如 `character.jpg` 或 `{name}.png`），用以保证视觉输出的一致性。
* **audio/**：（可选）包含一段清晰的参考语音片段（通常为 WAV PCM 格式），用以保证 TTS 声音克隆输出的一致性。
* **scripts/**：可执行脚本（`{name}-tts.sh` 和 `{name}-image.sh`），它们预配置了参考资产，并调用现有的 AIGC 技能（`qwen-tts`、`ai-image`）。

# 2. 互动指南：分步创建协议

当用户想要创建一个新角色时，你必须扮演互动顾问的角色，引导他们完成以下 4 个结构化步骤：

## 步骤1：角色的核心概念
询问用户该角色的基本信息。你可以为他们提供选项，也可以让他们自由回答。
* **名字 (Name)**：英文/ID（小写、英文半角字母和数字组合、不含空格）。
* **人设/特质 (Persona/Traits)**：简短的关键词或描述（例如“温柔、贴心”、“傲娇、毒舌”）。
* **语言/说话癖好 (Language/Speech Quirks)**：语言风格（例如“说话口吻非正式”、“经常使用表情符号”、“用振假名注释讲授日语”）。

*提示*：可以使用多选语法提供几个预设的角色模板, 例如:
`您想从预设的角色模板开始吗？ [question: 选择一个模板： || 傲娇动漫少女 | 温柔伴侣 | 专业 AI 助手 | 再给我换几个看看]`
使用[question]语法, 引导用户建立人设, 把人设发给用户看, 不断修改直到用户满意.

## 步骤 2： 收集参考资产
解释添加参考文件如何解锁角色的视觉与配音功能：
* **参考图片（用以实现一致的配图）**：请用户上传或提供高质量参考图片的路径。
* **参考音频（用以声音克隆）**：请用户上传或提供一段干净的 WAV 格式人声录音文件路径（时长为 5-15 秒）。如果他们提供了音频，请同时询问文本转写内容（以提高 Qwen-TTS 的准确率）。

## 步骤3：生成角色档案
使用模板文件 `${SKILL_DIR}/template/` 创建新角色的目录结构：
```
{SKILL_DIR}/template/
├── intro.md
├── preferences.md
├── images/
│   └── {name}.png
├── audio/
│   └── {name}.wav
└── scripts/
    ├── {name}-tts.sh
    └── {name}-image.sh
```
按照用户的输入和要求, 仔细编写 `intro.md` 文件和`scripts`. 把内容发给用户看, 确认用户的修改建议, 直到用户觉得ok.


## 步骤4：启用与测试
成功创建后：
告知用户在channel中使用`/role`命令来切换角色. 角色和channel绑定.