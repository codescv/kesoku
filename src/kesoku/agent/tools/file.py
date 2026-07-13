"""File management tools for Kesoku AI Agent."""

import os

from kesoku.agent.tools.registry import ToolContext, default_registry
from kesoku.utils.path import PathResolver


@default_registry.register
def update_file(
    file_name: str,
    old_content: str | None,
    new_content: str,
    context: ToolContext | None = None,
) -> str:
    """Partially modify a file.

    If the file does not exist, old_content must be None.
    If the file exists, old_content must match 1 or 2 occurrences in the file.
    Otherwise, an error is returned.

    Args:
        file_name: Path to the file (relative to AWD or absolute).
        old_content: The content to be replaced. Must be None if file doesn't exist.
        new_content: The content to replace with.
        context: Optional tool execution context.

    Returns:
        A success message or error message.
    """
    resolved_path = PathResolver.resolve(file_name)
    file_exists = os.path.exists(resolved_path)

    if not file_exists:
        if old_content is not None:
            return f"Error: File '{file_name}' does not exist, but old_content was provided."

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
        try:
            with open(resolved_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return f"Success: Created file '{file_name}' with new content."
        except Exception as e:
            return f"Error creating file '{file_name}': {e}"

    # File exists
    if old_content is None:
        return f"Error: File '{file_name}' exists, but old_content is None."

    try:
        with open(resolved_path, encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return f"Error reading file '{file_name}': {e}"

    count = content.count(old_content)
    if count == 0:
        return f"Error: old_content not found in file '{file_name}'."
    if count > 2:
        return f"Error: old_content matches {count} places in file '{file_name}' (max allowed is 2)."

    new_file_content = content.replace(old_content, new_content)

    try:
        with open(resolved_path, "w", encoding="utf-8") as f:
            f.write(new_file_content)
        return f"Success: Updated file '{file_name}' ({count} replacement(s) made)."
    except Exception as e:
        return f"Error writing to file '{file_name}': {e}"
