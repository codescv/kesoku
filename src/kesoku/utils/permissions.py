"""Best-effort file permission helpers for private Kesoku files (DB, config with tokens)."""

import os

PRIVATE_FILE_MODE = 0o600


def secure_file_permissions(path: str | os.PathLike[str], mode: int = PRIVATE_FILE_MODE) -> bool:
    """Restrict a private file to its owner (default ``0600``).

    Best effort: missing files and ``OSError`` (e.g. foreign owner, unsupported filesystem) are ignored.

    Args:
        path: File to restrict.
        mode: Permission bits to apply.

    Returns:
        True when the mode was applied, False otherwise.
    """
    try:
        if not os.path.isfile(path):
            return False
        os.chmod(path, mode)
        return True
    except OSError:
        return False
