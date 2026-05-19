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
# Sending Files to the User
You have the capability to send files (such as generated images, photos, audios, videos,
report documents, or scripts) directly to the user's conversation thread.
To transmit a file, you MUST include the following exact syntax in your final textual response to the user:
[file: /abs/path/to/file]
Example: 'Here is the requested cat picture: [file: /home/user/Downloads/cat.png]'

Rules for file sending:
1. The file must physically exist on the local disk before you output the syntax.
2. Do not guess or output fictional/placeholder file paths.
3. Always ensure that the path inside `[file: <path>]` is a fully resolved absolute path."""


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
Unless the user explicitly instructs otherwise, you MUST save all files (such as generated
images, photos, audios, videos, report documents, scripts, or other output files) in this
session staging directory by default. This is your staging directory for your work in this session.
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
        user_prompts_sections.append(f"=== BEGIN {base_name} ===\n{content.strip()}\n=== END {base_name} ===")

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
        ]
    )

    sections.extend(user_prompts_sections)

    if custom_prompt:
        sections.append(custom_prompt.strip())

    return "\n\n".join(sections)
