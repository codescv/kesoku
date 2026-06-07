# 自治技能系统架构

本开发指南详细介绍了 Kesoku 自治技能管理器的实现细节，包括元数据清单解析、操作系统平台过滤、目录遍历防御以及绝对路径上下文注入机制。

---

## 🏗️ 核心数据模型

所有技能均通过 Pydantic 模式进行强类型解析与验证（定义于 `src/kesoku/agent/skills.py`）：

```python
class SkillMetadata(BaseModel):
    tags: list[str] = Field(default_factory=list)
    platforms: list[str] | None = Field(default=None) # 可选: "linux", "darwin", "windows"
    related_skills: list[str] = Field(default_factory=list)

class SkillManifest(BaseModel):
    name: str = Field(...)
    description: str = Field(...)
    version: str = Field(default="1.0.0")
    required_permissions: list[str] = Field(default_factory=list) # 例如: "run_shell_command"
    metadata: SkillMetadata = Field(default_factory=SkillMetadata)
```

---

## ⚙️ 技能发现与加载生命周期 (`SkillManager`)

技能文件的管理与装载统一由 `SkillManager` 类负责：

```text
┌────────────────────────────────────────────────────────┐
│ 1. 扫描 skills/ 目录下的所有子文件夹                     │
├────────────────────────────────────────────────────────┤
│ 2. 路径边界检查（Commonpath 防御目录越界）               │
├────────────────────────────────────────────────────────┤
│ 3. 正则提取并解析 YAML 元数据与 Markdown 说明            │
├────────────────────────────────────────────────────────┤
│ 4. 宿主机操作系统平台匹配过滤                            │
├────────────────────────────────────────────────────────┤
│ 5. 权限审计（检查 config.toml 相应工具是否开启）         │
├────────────────────────────────────────────────────────┤
│ 6. 自动注入 SKILL_DIR 绝对路径头部                       │
└────────────────────────────────────────────────────────┘
```

### 1. 路径越界防御 (Path Safety)
为了防止恶意输入导致的敏感路径越界攻击（例如模型企图读取 `use_skill(skill_name="../../../../etc")`），`_resolve_skill_dir` 进行了严格的安全检测：
*   **名称过滤**：技能标识符强制使用正则过滤：`^[a-zA-Z0-9_\-]+$`。
*   **前缀匹配**：系统解析真实规范的绝对路径，并确认基础 `skills/` 目录是该绝对路径的唯一公共前缀：
    ```python
    if os.path.commonpath([base, target]) != base or target == base:
        raise KeyError("Invalid skill name: Path traversal boundary violation.")
    ```

### 2. 元数据解析 (Manifest Parsing)
解析器 `parse_skill_markdown()` 用于抓取 Markdown 文件顶部的 YAML 配置块：
*   利用正则检索文件头部包裹在三个短横线（`---`）之间的块。
*   使用 `yaml.safe_load()` 安全解析该块，并映射为 `SkillManifest` 实例。
*   若解析失败，系统会记录日志并跳过该技能文件夹。

### 3. 操作系统平台匹配
在运行期间，系统会将宿主机内核类型标准化为 `"linux"`、`"darwin"` 或 `"windows"`。执行 `list_skills()` 时：
*   若元数据中声明了 `platforms` 列表且当前系统不在其中，该技能将被过滤隐藏。
*   若 `platforms` 为 `None`，代表兼容所有平台。
*   若 `platforms` 为 `[]` 空列表，该技能在所有系统上均被隐藏。

### 4. 权限与功能审计
在加载具体技能前，管理器会核对系统配置，判定宿主机是否具备运行该技能所需的权限：
*   例如，若技能清单中要求了 `"run_shell_command"`，但 `config.toml` 中设置了 `cfg.shell.enabled = false`，系统将直接抛出 `PermissionError` 拒绝加载。

### 5. 绝对路径重定向
当调用 `get_skill(skill_name)` 返回技能内容时：
*   管理器获取该技能子文件夹的绝对路径。
*   在 Markdown 文本最前部追加一段绝对路径环境变量提示语：
    ```markdown
    # Skill: {manifest.name} (v{manifest.version})
    > [!IMPORTANT] You must replace the SKILL_DIR or explicitly set it as an env variable
    > mentioned in the skill instructions with the following value:
    > SKILL_DIR='{abs_skill_dir}'
    ```
*   这能够明确训练并引导 Agent 使用此绝对路径定位技能附属的脚本或资产文件，确保 shell 工具在运行脚本（例如 `python /absolute/path/to/skills/my-skill/scripts/runner.py`）时能够精准找到目标。
