"""File I/O helpers for reading target files, patch files, and writing patched outputs."""

from pathlib import Path


def read_patch_file(path: str | Path) -> str:
    """Read a unified diff file from disk as UTF-8 text."""
    return Path(path).read_bytes().decode("utf-8")


def read_target_file(path: str | Path) -> str:
    """Read a target file from disk preserving its raw byte line endings."""
    return Path(path).read_bytes().decode("utf-8")


def write_target_file(path: str | Path, content: str | bytes) -> None:
    r"""Write patched content to `path` on disk.

    Note: currently normalizes string output with universal `\n` line endings.
    """
    target = Path(path)
    if isinstance(content, bytes):
        text = content.decode("utf-8")
    else:
        text = content
    # BUG: Normalizes all CRLF line endings to LF when writing out.
    normalized = text.replace("\r\n", "\n")
    target.write_bytes(normalized.encode("utf-8"))
