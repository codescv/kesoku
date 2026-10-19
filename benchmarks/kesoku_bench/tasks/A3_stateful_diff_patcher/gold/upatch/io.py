"""File I/O helpers for reading target files, patch files, and writing patched outputs."""

from pathlib import Path


def read_patch_file(path: str | Path) -> str:
    """Read a unified diff file from disk as UTF-8 text."""
    return Path(path).read_bytes().decode("utf-8")


def read_target_file(path: str | Path) -> str:
    """Read a target file from disk preserving its raw byte line endings."""
    return Path(path).read_bytes().decode("utf-8")


def write_target_file(path: str | Path, content: str | bytes) -> None:
    """Write patched content to `path` on disk preserving exact byte line endings."""
    target = Path(path)
    if isinstance(content, bytes):
        raw_bytes = content
    else:
        raw_bytes = content.encode("utf-8")
    target.write_bytes(raw_bytes)
