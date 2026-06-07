# Skills & Custom Tools

Kesoku features an autonomous skill system that allows agents to dynamically discover and adopt specialized domain capabilities and instructions during chat sessions.

---

## 📁 Skill Directory Structure

Skills are self-contained folders placed under the configured `skills_dir` directory (defaults to `skills/` in your workspace):

```text
kesoku/
└── skills/
    ├── ai-image/                 # Image skill
    │   ├── SKILL.md              # Skill definition (markdown with YAML frontmatter)
    │   └── scripts/
    │       └── gen_image.py      # Support script invoked by the agent
    └── web-search/               # Web search skill
        └── SKILL.md
```

---

## ✍️ Creating a Custom Skill

To teach the agent a new capability, create a folder under `skills/` containing a `SKILL.md` file.

### Step 1: Define the manifest (YAML Frontmatter)
The `SKILL.md` must start with a YAML block enclosed in `---` detailing the name, description, and metadata.

Example `skills/my-git-helper/SKILL.md`:
```yaml
---
name: git-helper
description: Run automated git branching and committing scripts.
metadata:
  tags: [git, automation]
  platforms: [linux, darwin]   # Optional: Exclude skill on unsupported systems
---

# Git Helper Instructions

You are equipped with git helper scripts located at `scripts/git_commit.py`.
To commit changes, run:
`uv run scripts/git_commit.py -m "your message"`

Always verify the branch name first using `git branch`.
```

### Platform Compatibility
If a skill requires specific OS configurations, specify the compatible platforms under `metadata.platforms`:
*   `platforms: [linux, darwin]`: The skill is only loaded on Linux and macOS, and hidden on Windows.
*   Omit the `platforms` key to make the skill cross-platform (loaded on all systems).
*   `platforms: []`: Excludes the skill globally.

---

## 🛠️ How the Agent Uses Skills

Kesoku exposes two built-in tools to the agent to manage its skills:

1.  **`list_skills()`**:
    The agent calls this tool to scan the `skills/` directory. It evaluates the current host operating system, filters out incompatible skills, and returns a list of available skill names and descriptions.
2.  **`use_skill(skill_name)`**:
    When the agent needs to perform a task associated with a skill:
    *   It calls `use_skill(skill_name="git-helper")`.
    *   The tool reads the full text of `skills/git-helper/SKILL.md`.
    *   It automatically prepends the absolute path of the skill directory to the instructions. This ensures that any script path referenced in the markdown can be resolved by the agent using its absolute path on disk (e.g. `python /absolute/path/to/skills/git-helper/scripts/git_commit.py`).
    *   The instructions are loaded into the conversation history, and the agent adopts the instructions immediately.
