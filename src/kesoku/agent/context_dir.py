"""Per-turn context directory injection.

Every turn, the agent injects the session's pinned notes (`$STAGING_DIR/notes.md`) and every Markdown file in
`$STAGING_DIR/context/` into the latest user message, each wrapped as `<context name="NAME">...</context>`.
Files in `context/` are maintained by external tools; the agent reads them but does not edit them.
"""

import asyncio
import os
import re
from dataclasses import dataclass

from kesoku.logger import setup_logger
from kesoku.utils.async_fs import async_exists, async_isdir, async_read_text_file

logger = setup_logger(__name__)

CONTEXT_DIR_NAME = "context"
NOTES_FILE_NAME = "notes.md"
NOTES_BLOCK_NAME = "notes"
CONTEXT_FILE_MAX_CHARS = 6000
DEFAULT_CONTEXT_TOTAL_MAX_CHARS = 20000
TRUNCATED_MARKER = "[truncated]"

# Structural parsing of filenames: optional leading numeric ordering prefix such as "20-" or "05_".
_NUMERIC_PREFIX_RE = re.compile(r"^\d+[-_]")


@dataclass(frozen=True)
class ContextEntry:
    """A single named context block."""

    name: str
    content: str


def context_block_name(filename: str) -> str:
    """Derive a context block name from a filename (drop a leading numeric prefix and the `.md` suffix).

    Examples: `20-scene.md` -> `scene`, `notes.md` -> `notes`, `99-test.md` -> `test`.
    """
    base = filename[:-3] if filename.lower().endswith(".md") else filename
    stripped = _NUMERIC_PREFIX_RE.sub("", base, count=1)
    name = stripped or base
    # Keep the attribute value well-formed.
    return name.replace('"', "'")


def truncate_context_content(content: str, max_chars: int = CONTEXT_FILE_MAX_CHARS) -> str:
    """Truncate the tail of a context file to `max_chars` and append a truncation marker."""
    if len(content) <= max_chars:
        return content
    return content[:max_chars].rstrip() + f"\n{TRUNCATED_MARKER}"


def _list_context_files(context_dir: str) -> list[str]:
    """Return sorted Markdown filenames inside the context directory."""
    return sorted(
        name
        for name in os.listdir(context_dir)
        if name.lower().endswith(".md") and os.path.isfile(os.path.join(context_dir, name))
    )


async def _read_stripped(path: str) -> str:
    """Read a text file and strip surrounding whitespace; returns an empty string on failure."""
    try:
        return (await async_read_text_file(path)).strip()
    except Exception as e:
        logger.warning(f"Failed to read context file '{path}': {e}")
        return ""


async def collect_context_entries(staging_dir: str) -> list[ContextEntry]:
    """Collect context entries for a session staging directory.

    The notes file comes first (if non-empty), followed by every non-empty `context/*.md` file in filename
    order. Each entry is already capped at `CONTEXT_FILE_MAX_CHARS`.

    Args:
        staging_dir: Absolute path of the session staging directory.

    Returns:
        Ordered list of context entries.
    """
    entries: list[ContextEntry] = []
    if not staging_dir:
        return entries

    notes_path = os.path.join(staging_dir, NOTES_FILE_NAME)
    if await async_exists(notes_path):
        notes = await _read_stripped(notes_path)
        if notes:
            entries.append(ContextEntry(NOTES_BLOCK_NAME, truncate_context_content(notes)))

    context_dir = os.path.join(staging_dir, CONTEXT_DIR_NAME)
    if not await async_isdir(context_dir):
        return entries
    try:
        filenames = await asyncio.to_thread(_list_context_files, context_dir)
    except Exception as e:
        logger.warning(f"Failed to list context directory '{context_dir}': {e}")
        return entries

    for filename in filenames:
        content = await _read_stripped(os.path.join(context_dir, filename))
        if content:
            entries.append(ContextEntry(context_block_name(filename), truncate_context_content(content)))
    return entries


def apply_total_budget(entries: list[ContextEntry], max_total_chars: int) -> list[ContextEntry]:
    """Drop entries from the end of the list until the total content length fits `max_total_chars`.

    Args:
        entries: Ordered context entries (notes first).
        max_total_chars: Maximum total characters across all entry contents.

    Returns:
        The kept prefix of entries.
    """
    kept = list(entries)
    dropped: list[str] = []
    while kept and sum(len(e.content) for e in kept) > max_total_chars:
        dropped.append(kept.pop().name)
    if dropped:
        logger.warning(
            f"Context blocks exceed {max_total_chars} chars; dropped (from the end): {', '.join(reversed(dropped))}"
        )
    return kept


def render_context_blocks(entries: list[ContextEntry]) -> str:
    """Render entries as consecutive `<context name="...">` blocks (empty string if none)."""
    return "".join(f'<context name="{e.name}">\n{e.content}\n</context>\n' for e in entries)


async def build_context_prefix(staging_dir: str, max_total_chars: int = DEFAULT_CONTEXT_TOTAL_MAX_CHARS) -> str:
    """Collect, budget and render all context blocks for a session staging directory."""
    entries = await collect_context_entries(staging_dir)
    return render_context_blocks(apply_total_budget(entries, max_total_chars))
