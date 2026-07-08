"""System prompt construction and management utilities for Kesoku AI Agent."""

import os

from kesoku.config import get_config
from kesoku.db import DatabaseManager, Session
from kesoku.logger import setup_logger

logger = setup_logger(__name__)

PREAMBLE = """You are Kesoku Agent, a helpful, highly capable autonomous AI assistant."""

OUTPUT_FORMATTING_INSTRUCTIONS = """
# Output Formatting Rules
To attach files or render interactive multiple-choice buttons in the UI,
place these syntax blocks at the most contextually relevant place in your final response:
- **Attach File**: `[file: path/to/file]` (for general documents, images, video, sound effects)
- **Attach Voice Message**: `[voice: path/to/audio]` (exclusively for speech/spoken audio)
  *Rule*: The file must physically exist on disk first.
  For files in `AWD` or `STAGING_DIR`, relative paths are allowed.
- **Multiple-Choice Question**: `[question: <the question> || Option 1 | Option 2 | ...]`
  *Rule*: Concise, button-like labels. Use to clarify ambiguous requests or offer shortcuts.
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


MEMORY_AND_HISTORY_INSTRUCTIONS = """
# Active Memory System (AMS)
This is your **active memory**. You actively initiate storing, reading, or pruning structured
long-term facts and project states that persist across sessions.
Do NOT write raw chat history to this system.

Related Tools: `view_memory`, `list_memories`, `update_memory`, `memory_search`.

Every memory entry has a **category** and a **key**.

You can only use the following categories:
1. `progress`: Active user project progression, reading positions, milestones, and study next steps.
   One entry per project.
2. `memo`: Record of important, interesting, or noteworthy events for the user. DO NOT write chores.

Memory keys must strictly contain ONLY lowercase letters, underscores, and numbers.
e.g.: {short_event_description}_{yyyymmdd}
"""


BACKGROUND_EXECUTION_INSTRUCTIONS = """
# Background Tasks
If a command transitions to a background job:
1. Immediately stop executing further tools in this turn.
2. Reply informing the user that the task has been moved to the background and they will be notified.
3. End your turn. You will be automatically alerted once execution completes.
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
        if session:
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

    sections.extend(
        [
            OUTPUT_FORMATTING_INSTRUCTIONS.strip(),
            TOOL_CALLING_INSTRUCTIONS.strip(),
            MEMORY_AND_HISTORY_INSTRUCTIONS.strip(),
            BACKGROUND_EXECUTION_INSTRUCTIONS.strip(),
        ]
    )

    sections.extend(user_prompts_sections)

    if custom_prompt:
        sections.append(custom_prompt.strip())

    return "\n\n".join(sections)
