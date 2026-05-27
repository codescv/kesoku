"""System prompt construction and management utilities for Kesoku AI Agent."""

import os

from kesoku.config import get_config
from kesoku.db import Session
from kesoku.logger import setup_logger

logger = setup_logger(__name__)

PREAMBLE = """You are Kesoku Agent, a helpful, highly capable autonomous AI assistant."""

SKILLS_INSTRUCTIONS = """
# Skills
You have access to on demand tools (aka skills) to help with various tasks.
- To list available skills, use the "list_skills" tool.
- To know how to use the skill, use the "use_skill" tool.
"""

FILE_SENDING_INSTRUCTIONS = """
# Sending Files and Voice Messages to the User
You have the capability to send files (such as generated images, photos, audios, videos,
report documents, or scripts) and voice messages directly to the user's conversation thread.

To transmit a file, you MUST include the following exact syntax in your final textual response to the user:
[file: /abs/path/to/file]
Example: 'Here is the requested cat picture: [file: /home/user/Downloads/cat.png]'

To transmit a voice message (speech), you MUST include the following exact syntax:
[voice: /abs/path/to/audio]
Example: 'Here is my verbal response: [voice: /home/user/Downloads/reply.ogg]'

Rules for file sending:
1. For speech (voice messages), ALWAYS use the `[voice: /abs/path/to/audio]` block.
2. For all other types of audio (e.g. music, sound effects, environmental recordings)
   and general files, use the `[file: /abs/path/to/file]` block.
3. The file must physically exist on the local disk before you output either syntax.
4. Do not guess or output fictional/placeholder file paths.
5. Always ensure that the path inside `[file: <path>]` or `[voice: <path>]` is a fully resolved absolute path."""


QUESTION_INSTRUCTION = """
# Asking the User Questions with Multiple-Choice Options
When you need to ask the user a question and want to provide them with clear, clickable multiple-choice buttons,
you MUST include the following exact syntax in your final textual response to the user:
[question: <the question> || choice1 | choice2 | ...]

Example: 'Would you like me to generate code in Python? [question: Choose language: || Python | Go]'

Rules for asking questions:
1. Use '||' to separate the question from the first choice option, then separate subsequent choices with '|'.
2. Ensure choices are concise, actionable button labels.
3. Selecting an option automatically posts a new user message containing that exact choice string.

When to ask:
1. To clarify user requests.
2. To predict user's follow up response and provide them as choices as a convenience.
3. To get user's feedback on how to proceed.
"""


def build_sys_prompt(
    custom_prompt: str | None = None,
    session: Session | None = None,
) -> str:
    """Build the complete agent system prompt including instructions on file-sending syntax and staging workspace.

    Args:
        custom_prompt: Optional additional context or platform-specific instructions.
        session: Optional chat session object.

    Returns:
        A complete, modularly constructed system prompt string.
    """
    cfg = get_config()
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

    sections = [
        PREAMBLE.strip(),
        working_dir_info.strip(),
    ]
    if session_dir_info:
        sections.append(session_dir_info.strip())

    sections.extend(
        [
            SKILLS_INSTRUCTIONS.strip(),
            FILE_SENDING_INSTRUCTIONS.strip(),
            QUESTION_INSTRUCTION.strip(),
        ]
    )

    sections.extend(user_prompts_sections)

    if custom_prompt:
        sections.append(custom_prompt.strip())

    return "\n\n".join(sections)
