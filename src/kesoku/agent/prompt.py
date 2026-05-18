"""System prompt construction and management utilities for Kesoku AI Agent."""

DEFAULT_SYSTEM_PROMPT = """You are Kesoku Agent, a helpful, highly capable autonomous AI assistant.
You can use available tools to calculate equations, search information, and answer user questions precisely."""

FILE_SENDING_INSTRUCTIONS = """# Sending Files to the User
You have the capability to send files (such as generated images, photos, audios, videos, report documents, or scripts) directly to the user's conversation thread.
To transmit a file, you MUST include the following exact syntax in your final textual response to the user:
[file: /abs/path/to/file]
Example: 'Here is the requested cat picture: [file: /home/user/Downloads/cat.png]'

Rules for file sending:
1. The file must physically exist on the local disk before you output the syntax.
2. Do not guess or output fictional/placeholder file paths.
3. Always ensure that the path inside `[file: <path>]` is a fully resolved absolute path."""


def build_sys_prompt(custom_prompt: str | None = None) -> str:
    """Build the complete agent system prompt including instructions on file-sending syntax.

    Args:
        custom_prompt: Optional additional context or platform-specific instructions.

    Returns:
        A complete, modularly constructed system prompt string.
    """
    sections = [
        DEFAULT_SYSTEM_PROMPT.strip(),
        FILE_SENDING_INSTRUCTIONS.strip(),
    ]

    if custom_prompt:
        sections.append(custom_prompt.strip())

    return "\n\n".join(sections)
