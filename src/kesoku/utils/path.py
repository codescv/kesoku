"""Unified path resolution utility for Kesoku AI Agent."""

import os

from kesoku.config import get_config


class PathResolver:
    """Unified helper to resolve relative and absolute paths against Agent Working Directory (AWD)."""

    @classmethod
    def get_awd(cls) -> str:
        """Get the active Agent Working Directory (AWD).

        Returns:
            Absolute path to the AWD.
        """
        cfg = get_config()
        return os.path.abspath(cfg.agent_working_dir or os.getcwd())

    @classmethod
    def resolve(cls, path: str | None) -> str:
        """Resolve a relative or absolute path against the active AWD.

        Args:
            path: Target path to resolve. If None, returns the AWD root.

        Returns:
            Fully resolved absolute path.
        """
        if not path:
            return cls.get_awd()

        # Check if path is absolute
        if os.path.isabs(path):
            return os.path.abspath(path)

        # Resolve relative to AWD
        return os.path.abspath(os.path.join(cls.get_awd(), path))

    @classmethod
    def get_session_staging_dir(cls, workspace_name: str) -> str:
        """Resolve and create the staging directory for a session workspace.

        Args:
            workspace_name: The escaped workspace folder name of the session.

        Returns:
            Absolute path to the staging directory.
        """
        cfg = get_config()
        sessions_base = cls.resolve(cfg.workspace.sessions_dir)
        staging_dir = os.path.join(sessions_base, workspace_name)
        os.makedirs(staging_dir, exist_ok=True)
        return os.path.abspath(staging_dir)
