"""System prompt construction and management utilities for Kesoku AI Agent."""

import os

from kesoku.agent.skills import SkillManager
from kesoku.config import KesokuConfig, get_config
from kesoku.db import DatabaseManager, Session
from kesoku.logger import setup_logger

logger = setup_logger(__name__)

PREAMBLE = """You are Kesoku Agent, a helpful, highly capable autonomous AI assistant."""

OUTPUT_FORMATTING_INSTRUCTIONS = """
# Output Formatting Rules
To attach files or render interactive UI elements,
place these syntax blocks at the most contextually relevant place in your final response:
- **Attach File**: `[file: path/to/file]` (for general documents, images, video, sound effects)
- **Attach Voice Message**: `[voice: path/to/audio]` (exclusively for speech/spoken audio)
  *Rule*: The file must physically exist on disk first.
  For files in `AWD` or `STAGING_DIR`, relative paths are allowed.
- **Choice Question**: a fenced `ui` code block containing YAML:
  ```ui
  type: select        # select (pick one) | multiselect (pick several)
  prompt: <the question>
  options:
    - Option 1
    - Option 2
  max: 2              # multiselect only; defaults to the number of options
  ```
  *Rule*: Concise, button-like option labels. Use to clarify ambiguous requests or offer shortcuts.
  At most one `ui` block per reply. A clicked option comes back as a user message with the option text;
  a multiselect answer comes back as `选择：A；B`.
- **Collapsible Section**: `<details><summary>Title</summary>Body</details>`
  *Rule*: Use for long explanations or optional detail the user can expand. Platforms without
  collapsing show the title in bold followed by the body.
- **Hidden Text**: `<spoiler>text to hide</spoiler>` (inline; the user clicks to reveal)
  *Rule*: Use for answers, translations, or twists the user should opt in to see. Platforms without
  hidden-text support show the content as plain text.
"""


TOOL_CALLING_INSTRUCTIONS = """
# Tool Execution Rules
Whenever possible, return MULTIPLE tool calls in a single turn to accelerate execution via PARALLEL
processing. For example:
- tool_call 1: run_shell_command <command1>
- tool_call 2: run_shell_command <command2>
NOTE: This is DIFFERENT from combining multiple commands into a single bash/shell execution,
which is still SERIAL.
"""




FILE_TOOL_INSTRUCTIONS = """
# File Tools
- Read source files with `read_file(file_name, start_line, end_line)` instead of `cat`/`head`/`sed -n`; page large
  files with `start_line`.
- Create a file with `update_file(file_name, new_content=...)` (omit `old_content`). Edit by passing a short, unique
  block (3-15 lines) copied verbatim from `read_file` output as `old_content`, without the `N: ` line-number prefixes.
- When a shell output is truncated and saved to a file, page through that file with `read_file`.
"""

# Skills index limits (system prompt size control).
MAX_SKILLS_IN_INDEX = 60
MAX_SKILL_DESCRIPTION_CHARS = 200
MAX_SKILLS_SECTION_CHARS = 6000
SKILLS_SECTION_HEADER = (
    "# Available Skills\n"
    "These skills are installed for this role. When a task matches one, call `use_skill(skill_name=\"<name>\")` "
    "FIRST to load its full instructions; do not call `list_skills` just to discover them (it returns this same list)."
)


def build_skills_section(role: str | None, cfg: KesokuConfig) -> str:
    """Render the deterministic `# Available Skills` index for the system prompt.

    Returns "" when `use_skill` is disabled, no skills are installed, or listing fails (never raises). Entries are
    sorted by name, descriptions are collapsed to one line and cut at 200 chars, and the section is capped at
    60 entries and 6000 chars (the rest is summarised with a pointer to `list_skills`).

    Args:
        role: Active role (role-scoped skills override global ones with the same name).
        cfg: Active configuration.

    Returns:
        The section text, or "".
    """
    if "use_skill" in set(cfg.agent.disabled_tools):
        return ""
    try:
        skills = SkillManager().list_skills(role=role)
    except Exception as e:  # noqa: BLE001 - prompt construction must never fail because of skills
        logger.warning(f"Failed to list skills for the system prompt: {e}")
        return ""
    if not skills:
        return ""
    lines = [SKILLS_SECTION_HEADER]
    size = len(SKILLS_SECTION_HEADER)
    ordered = sorted(skills, key=lambda item: str(item.get("name", "")))
    shown = 0
    for item in ordered:
        if shown >= MAX_SKILLS_IN_INDEX:
            break
        desc = " ".join(str(item.get("description") or "").split())
        if len(desc) > MAX_SKILL_DESCRIPTION_CHARS:
            desc = desc[: MAX_SKILL_DESCRIPTION_CHARS - 1].rstrip() + "…"
        line = f"- {item.get('name')}: {desc}" if desc else f"- {item.get('name')}"
        if size + len(line) + 1 > MAX_SKILLS_SECTION_CHARS:
            break
        lines.append(line)
        size += len(line) + 1
        shown += 1
    remaining = len(ordered) - shown
    if remaining > 0:
        lines.append(f"- … and {remaining} more (call list_skills() for the full list)")
    return "\n".join(lines)


BACKGROUND_EXECUTION_INSTRUCTIONS = """
# Background Tasks
If a command transitions to a background job:
1. Immediately stop executing further tools in this turn.
2. Reply informing the user that the task has been moved to the background and they will be notified.
3. End your turn. You will be automatically alerted once execution completes.
"""

SESSION_NOTES_INSTRUCTIONS = """
# Pinned Session Notes and Context Blocks
- Every turn, `<context name="...">` blocks are injected right before `<current_message>`: first
  `<context name="notes">` (your pinned notes), then one block per file in `$STAGING_DIR/context/`.
- Context blocks other than `notes` are maintained by tools; never edit files under `$STAGING_DIR/context/`
  with `update_file` (`notes.md` remains editable as described below).
- `<context name="running_jobs">` and `<context name="staging_artifacts">` are generated by the system; trust them
  over your memory of earlier turns.
- Maintain pinned notes for the current session in `$STAGING_DIR/notes.md` using the `update_file` tool
  (`file_name` = the absolute `$STAGING_DIR/notes.md` path, `old_content` / `new_content`).
- The exact current contents of `$STAGING_DIR/notes.md` are automatically injected into every turn as
  `<context name="notes">`.
- To edit, copy the exact lines to change from `<context name="notes">` as `old_content`;
  do NOT run shell commands (`cat`, `rg`, etc.) to read `notes.md` first.
- Omit `old_content` (equivalently `old_content=None`) only when creating `notes.md` for the first time (when
  `<context name="notes">` is absent).
- Edit only the specific lines that change: append a new rule by replacing an adjacent line with itself plus the
  new line, replace an outdated line with its new version, or delete a line by replacing it with an empty
  string. Never rewrite the whole file, and never drop unrelated existing entries.
- Whenever the user states requirements, rules, formatting/language preferences, or ongoing scene/task states
  that must remain consistent across this session, immediately update the notes so you never forget them.
- Keep session notes concise and up-to-date: merge new rules with existing ones and remove completed one-off tasks.
- When the user changes or revokes a rule, REPLACE or DELETE the old entry in place; never keep the old and new
  versions side by side, and never re-add a rule the user has said is outdated.
- Session notes and the most recent messages are the source of truth for current requirements. Compacted
  history summaries and retrieved historical snippets may contain outdated rules; ignore them when they conflict.
"""


def build_sys_prompt(
    custom_prompt: str | None = None,
    session: Session | None = None,
    role: str | None = None,
) -> str:
    """Build the complete agent system prompt including instructions on file-sending syntax and staging workspace.

    Args:
        custom_prompt: Optional additional context or platform-specific instructions.
        session: Optional chat session object.
        role: Optional pre-resolved character persona binding. If not provided, resolves from the session.

    Returns:
        A complete, modularly constructed system prompt string.
    """
    cfg = get_config()

    # Resolve current role
    if role is None:
        role = "default"
        if session and session.role_name and session.role_name != "default":
            # The session's own persona binding is authoritative (role switches create/swap sessions).
            # This also covers thread sessions that have no direct channel mapping.
            role = session.role_name
        elif session:
            db = DatabaseManager(cfg.workspace.db_path)
            try:
                mapping = db.get_channel_by_session(session.id)
                if mapping:
                    chatbot_id, channel_id = mapping
                    role = db.get_channel_role_with_inheritance(chatbot_id, channel_id, session.id)
            except Exception as e:
                logger.warning(f"Failed to retrieve session channel role in build_sys_prompt: {e}")

    # Read role's intro.md
    intro_content = ""
    if cfg.agent_working_dir:
        intro_path = os.path.join(cfg.agent_working_dir, "roles", role, "intro.md")
        if os.path.exists(intro_path):
            try:
                with open(intro_path, encoding="utf-8") as f:
                    intro_content = f.read().strip()
            except Exception as e:
                logger.warning(f"Failed to read intro.md for role '{role}': {e}")

    working_dir_info = f"""
# Agent Working Directory
> AWD='{cfg.agent_working_dir}'
You are working in the agent working directory (AWD).
This is where you find the files you need by default.
Unless the user explicitly instructs otherwise, do not refer to any file outside this directory.
    """

    session_dir_info = ""
    if session:
        staging_path = os.path.realpath(os.path.join(cfg.workspace.sessions_dir, session.workspace_name))
        session_dir_info = f"""
# Session Staging Directory
> STAGING_DIR='{staging_path}'
- This is your where you are supposed to save your files, unless the user explicitly instructs otherwise.
  Create it if it doesn't exist.
- Save all output files, including generated images, photos, audios, videos, report documents, scripts,
  command output files, cloned repos, downloaded files or other output files) in this session staging directory.
- If you accidentally saved any files outside of this directory, move them to this directory in the end.
    """

    user_prompts_sections = []
    # Process and load user_prompts files
    for p_path in cfg.agent.user_prompts:
        resolved_path = p_path
        if not os.path.isabs(resolved_path) and cfg.agent_working_dir:
            resolved_path = os.path.join(cfg.agent_working_dir, resolved_path)
        resolved_path = os.path.abspath(resolved_path)

        with open(resolved_path, encoding="utf-8") as f:
            content = f.read()
        base_name = os.path.basename(resolved_path)
        user_prompts_sections.append(f"=== BEGIN {base_name} ===\n{content.strip()}\n\n=== END {base_name} ===")

    sections = []
    if intro_content:
        role_info = f"""
# Active Persona (Role Name): ({role})
{intro_content}
"""
        sections.append(role_info.strip())

    sections.append(working_dir_info.strip())
    if session_dir_info:
        sections.append(session_dir_info.strip())
    skills_section = build_skills_section(role, cfg)
    if skills_section:
        sections.append(skills_section)

    sections.extend(
        [
            OUTPUT_FORMATTING_INSTRUCTIONS.strip(),
            TOOL_CALLING_INSTRUCTIONS.strip(),
            FILE_TOOL_INSTRUCTIONS.strip(),
            BACKGROUND_EXECUTION_INSTRUCTIONS.strip(),
            SESSION_NOTES_INSTRUCTIONS.strip(),
        ]
    )

    sections.extend(user_prompts_sections)

    if custom_prompt:
        sections.append(custom_prompt.strip())

    return "\n\n".join(sections)
