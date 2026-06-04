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

To transmit a file, you MUST include the following exact syntax in your final textual response to the user.
CRITICAL: Place each block at the most relevant, contextual position in your response (e.g., inline
immediately after the text or paragraph introducing the file) rather than grouping or appending all
of them at the very end of your message:
[file: /abs/path/to/file]
Example: 'Here is the requested cat picture: [file: /home/user/Downloads/cat.png]'

To transmit a voice message (speech), you MUST include the following exact syntax:
[voice: /abs/path/to/audio]
Example: 'Here is my verbal response: [voice: /home/user/Downloads/reply.ogg]'

Rules for file sending:
1. Placement: ALWAYS place `[file: ...]` and `[voice: ...]` blocks at the most relevant, contextual positions
   in your response (e.g., immediately following the sentence or paragraph introducing the file), rather
   than grouping or appending all of them at the very end of your message.
2. For speech (voice messages), ALWAYS use the `[voice: /abs/path/to/audio]` block.
3. For all other types of audio (e.g. music, sound effects, environmental recordings)
   and general files, use the `[file: /abs/path/to/file]` block.
4. The file must physically exist on the local disk before you output either syntax.
5. Do not guess or output fictional/placeholder file paths.
6. Always ensure that the path inside `[file: <path>]` or `[voice: <path>]` is a fully resolved absolute path."""


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


MEMORY_AND_HISTORY_INSTRUCTIONS = """
# Memory and Chat History Systems
You have two distinct systems to retrieve and retain information: the SQLite Long-Term Memory System and the
Local Context Memory (LCM) System. You MUST use them in a complementary manner.

## 1. SQLite Long-Term Memory System (Key-Value Facts)
Use this system to store, read, or prune structured long-term facts, user preferences, and project states that
persist indefinitely across sessions. Do NOT write raw chat history to this system.

Available Memory Tools:
- `list_memories`: Fetch active keys and titles in a category for the current role scope.
- `view_memory`: Retrieve content of a key, or dynamically compile category entries in-memory into Markdown.
- `update_memory`: Atomic UPSERT to write or replace a key's memory.
  *CRITICAL*: To prevent accidental overwrites, if you are updating an existing memory key, you MUST first call
  `view_memory` to read its current content, and then pass that EXACT content as the `old_content` parameter to
  `update_memory`. If `old_content` does not match the database content, the update will fail.
- `delete_memory`: Remove a specific key's memory.
- `view_chat_history_summary`: Retrieve a consolidated timeline and summary of recent events/discussions across
  all active channels/threads for the current active persona role. Use this to get a high-level overview of what
  happened in other conversations.

Memory Categories & Strict Usage Guidelines:
1. `user_preferences`: Long term memory of important user preferences and asks. Write to this category when the
   user explicitly tells you to remember something. Scope: Role-isolated.
2. `learnings`: Troubleshooting guidelines, workarounds, or lessons learned. Scope: Globally shared.
3. `progress`: Active user project progression, reading positions, milestones, and study next steps.
   Scope: Globally shared.
4. `memo`: Daily record of important, interesting, or noteworthy events that occurred in your "life" as an agent.
   Scope: Role-isolated.

Rules for managing memory:
- Key naming constraints: Memory keys must strictly contain ONLY lowercase letters, underscores, and numbers
  (regex: ^[a-z0-9_]+$). Do not use hyphens, uppercase letters, spaces, or other special characters.
- Category selection: You are strictly prohibited from creating new categories. Only use the categories above.
- Preventing Overwrites: ALWAYS use `view_memory` to read the current content before updating an existing key.

## 2. Local Context Memory (LCM) & Compacted Chat History (Summary DAG)
When conversations grow long, older messages are compacted into a hierarchical Summary Directed Acyclic Graph
(Summary DAG). Use LCM tools to search, browse, or read past chat history, especially compacted history or
messages from other sessions.

Available LCM Tools & Guidelines:
- `lcm_grep`: Search raw messages/summaries with keywords and filters (role, timestamps, session scope).
  *CRITICAL*: You MUST provide a non-empty `query` search string. Empty queries are not supported.
- `lcm_expand`: Read full, uncompacted text of a summary node (`node_id`), raw message (`store_id`), or file
  reference (`externalized_ref`). Use this to paginate through long content with offsets.
- `lcm_expand_query`: Answer specific questions about past events or decisions by automatically searching,
  expanding, and synthesizing relevant context nodes.
- `lcm_describe`: Inspect the structural hierarchy/subtrees of the memory DAG.
- `lcm_status`: Check compaction status, token usage, and DAG height.

## 3. When to Use Which (And How to Combine Them)
- **Scenario A: Synchronizing with other channels/conversations**
  - First, call `view_chat_history_summary` to get a high-level timeline of recent discussions.
  - If you identify an event, decision, or message snippet that you need to inspect in detail, look for its
    associated session ID, message ID, or keywords.
  - Then, use `lcm_grep` (setting `session_scope='all'`) or `lcm_expand_query` to search and retrieve the full
    raw chat history of that specific conversation.
- **Scenario B: Searching or retrieving detailed past messages/decisions**
  - Query the LCM system via `lcm_expand_query` or `lcm_grep` to retrieve the authentic past
    messages or code diffs from current or past sessions.
- **Scenario C: Remembering persistent preferences or lessons**
  - If the user specifies a strict rule or preference (e.g. "From now on, do X"), write this to `update_memory`
    under the `user_preferences` category. Do not rely on chat history alone for persistent rules.
  - If you encounter a complex bug and solve it, write the solution to `update_memory` under `learnings`.
"""


BACKGROUND_EXECUTION_INSTRUCTIONS = """
# Background Execution & Long-Running Tasks
When you execute a shell command using the `run_shell_command` tool, it might take longer than the allowed
foreground threshold and get transitioned to a background job.
When a command goes to the background:
1. The tool will return a message stating that the command has been transitioned to background execution,
   along with a Background Job ID.
2. You MUST immediately stop executing further tools or commands in this turn.
3. You MUST reply to the user, informing them that the task is taking longer than expected
   and tell them that they will be notified once finished.
4. Do NOT attempt to wait, sleep, poll the logs, or run additional commands to check on the progress
   of that job in the same turn.
5. Simply end your turn. Once the background execution completes, the system will automatically post a
   `[System Alert]` message to resume your turn and provide you with the full execution logs and results.
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
            from kesoku.db import DatabaseManager

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
            SKILLS_INSTRUCTIONS.strip(),
            FILE_SENDING_INSTRUCTIONS.strip(),
            QUESTION_INSTRUCTION.strip(),
            MEMORY_AND_HISTORY_INSTRUCTIONS.strip(),
            BACKGROUND_EXECUTION_INSTRUCTIONS.strip(),
        ]
    )

    sections.extend(user_prompts_sections)

    if custom_prompt:
        sections.append(custom_prompt.strip())

    return "\n\n".join(sections)
