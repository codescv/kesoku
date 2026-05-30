"""Asynchronous filesystem utility functions for Kesoku AI Agent framework."""

import asyncio
import os
from pathlib import Path


async def async_exists(path: str | os.PathLike) -> bool:
    """Asynchronously check if a path exists using a thread pool.

    Args:
        path: The filesystem path to check.

    Returns:
        True if the path exists, False otherwise.
    """
    return await asyncio.to_thread(os.path.exists, path)


async def async_isdir(path: str | os.PathLike) -> bool:
    """Asynchronously check if a path is a directory using a thread pool.

    Args:
        path: The filesystem path to check.

    Returns:
        True if the path is a directory, False otherwise.
    """
    return await asyncio.to_thread(os.path.isdir, path)


async def async_realpath(path: str | os.PathLike) -> str:
    """Asynchronously resolve the absolute realpath using a thread pool.

    Args:
        path: The filesystem path to resolve.

    Returns:
        The resolved absolute path.
    """
    return await asyncio.to_thread(os.path.realpath, path)


async def async_makedirs(path: str | os.PathLike) -> None:
    """Asynchronously create recursive directories using a thread pool.

    Args:
        path: The directory path to create.
    """
    await asyncio.to_thread(os.makedirs, path, exist_ok=True)


async def async_read_text_file(path: str | os.PathLike) -> str:
    """Asynchronously read a text file's full content using a thread pool.

    Args:
        path: The text file path to read.

    Returns:
        The full string content of the file.
    """

    def _read() -> str:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()

    return await asyncio.to_thread(_read)


async def async_write_text_file(path: str | os.PathLike, content: str) -> None:
    """Asynchronously write a text string to a file using a thread pool.

    Args:
        path: The target file path to write.
        content: The text string content to write.
    """

    def _write() -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    await asyncio.to_thread(_write)


async def async_read_bytes(path: str | os.PathLike) -> bytes:
    """Asynchronously read a binary file's full bytes using a thread pool.

    Args:
        path: The binary file path to read.

    Returns:
        The full bytes content of the file.
    """
    return await asyncio.to_thread(Path(path).read_bytes)


async def async_write_binary_file(path: str | os.PathLike, data: bytes) -> None:
    """Asynchronously write binary bytes to a file using a thread pool.

    Args:
        path: The target file path to write.
        data: The binary bytes payload to write.
    """

    def _write() -> None:
        with open(path, "wb") as f:
            f.write(data)

    await asyncio.to_thread(_write)


async def async_get_subdirectories(path: str | os.PathLike) -> list[str]:
    """Asynchronously list all subdirectories within a given directory path.

    Args:
        path: The directory path to search.

    Returns:
        A list of subdirectory names.
    """

    def _get() -> list[str]:
        if not os.path.exists(path):
            return []
        return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    return await asyncio.to_thread(_get)
