# Skills System Architecture

This guide details the implementation of Kesoku's autonomous skill manager, manifest parsing, platform matching, and absolute path translation mechanisms.

---

## 🏗️ Data Models

Skills are validated and parsed using Pydantic schemas defined in `src/kesoku/agent/skills.py`:

```python
class SkillMetadata(BaseModel):
    tags: list[str] = Field(default_factory=list)
    platforms: list[str] | None = Field(default=None) # Allowed: "linux", "darwin", "windows"
    related_skills: list[str] = Field(default_factory=list)

class SkillManifest(BaseModel):
    name: str = Field(...)
    description: str = Field(...)
    version: str = Field(default="1.0.0")
    required_permissions: list[str] = Field(default_factory=list) # e.g. "run_shell_command"
    metadata: SkillMetadata = Field(default_factory=SkillMetadata)
```

---

## ⚙️ Execution Lifecycle (`SkillManager`)

The discovery and loading of skills is managed by the `SkillManager` class:

```text
┌────────────────────────────────────────────────────────┐
│ 1. Scan skills/ subdirectories                         │
├────────────────────────────────────────────────────────┤
│ 2. Path Traversal Boundary Check (Common path test)    │
├────────────────────────────────────────────────────────┤
│ 3. Parse YAML Frontmatter & Markdown Body              │
├────────────────────────────────────────────────────────┤
│ 4. Platform Filtering (platform.system() comparison)   │
├────────────────────────────────────────────────────────┤
│ 5. Active Config Permission Audit (e.g. Shell Enabled) │
├────────────────────────────────────────────────────────┤
│ 6. Inject SKILL_DIR absolute path header               │
└────────────────────────────────────────────────────────┘
```

### 1. Path Safety & Containment
To prevent directory traversal attacks (e.g., an LLM requesting `use_skill(skill_name="../../../../etc")`), `_resolve_skill_dir` strictly enforces security boundaries:
*   **Regex Sanitization**: The skill name identifier must match the alphanumeric pattern: `^[a-zA-Z0-9_\-]+$`.
*   **Commonpath Containment**: Resolves the real canonical absolute path and checks that the base `skills/` folder is the common prefix of the resolved path:
    ```python
    if os.path.commonpath([base, target]) != base or target == base:
        raise KeyError("Invalid skill name: Path traversal boundary violation.")
    ```

### 2. Manifest Parsing
The parser `parse_skill_markdown()` extracts YAML frontmatter from the markdown files:
*   It searches for a frontmatter block enclosed in triple dashes (`---`) at the top of the file.
*   It parses the block using `yaml.safe_load()` and validates it against the `SkillManifest` model.
*   If the frontmatter is malformed, the folder is skipped.

### 3. Platform Detection & Filtering
At runtime, `_get_current_platform()` normalizes the host OS to `"linux"`, `"darwin"`, or `"windows"`. When listing skills (`list_skills()`):
*   If the manifest specifies `platforms` and the normalized host OS is not in the list, the skill is skipped.
*   If `platforms` is `None` (omitted), it is treated as compatible with all operating systems.
*   If `platforms` is `[]` (empty list), the skill is skipped on all systems.

### 4. Permission Auditing
Before loading a skill, the manager checks that the host machine supports the necessary permissions defined in the manifest:
*   If `required_permissions` contains `"run_shell_command"` but `cfg.shell.enabled = false` in `config.toml`, loading fails with a `PermissionError`.

### 5. Path Context Injection
When `get_skill(skill_name)` is executed:
*   The manager retrieves the absolute path of the skill subdirectory.
*   It prepends a block containing the absolute path of the skill directory to the instructions:
    ```markdown
    # Skill: {manifest.name} (v{manifest.version})
    > [!IMPORTANT] You must replace the SKILL_DIR or explicitly set it as an env variable
    > mentioned in the skill instructions with the following value:
    > SKILL_DIR='{abs_skill_dir}'
    ```
*   This teaches the agent the exact location of any supporting files/scripts in the workspace, ensuring shell commands resolve their targets correctly (e.g. `python /absolute/path/to/skills/my-skill/scripts/runner.py`).
