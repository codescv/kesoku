"""Touch-driven repository rules (AGENTS.md / GEMINI.md) discovery and persistence."""

import json
import os

from kesoku.agent.state_snapshot import render_repo_rules
from kesoku.logger import setup_logger

logger = setup_logger(__name__)


class RepoRulesManager:
    """Manager for repository rules discovery and persistence."""
    def __init__(
        self,
        staging_dir: str,
        agent_working_dir: str,
        max_files: int = 3,
        max_chars: int = 4000,
        rule_filenames: list[str] | None = None
    ):
        """Initialize."""
        self.staging_dir = staging_dir
        self.agent_working_dir = os.path.abspath(agent_working_dir)
        self.max_files = max_files
        self.max_chars = max_chars
        self.rule_filenames = rule_filenames or ["AGENTS.md", "GEMINI.md"]
        self.persistence_path = os.path.join(self.staging_dir, "repo_rules.json")
        self.discovered_files: list[str] = self._load()

    def _load(self) -> list[str]:
        if os.path.exists(self.persistence_path):
            try:
                with open(self.persistence_path, encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("files", [])
            except Exception as e:
                logger.error(f"Failed to load repo_rules.json: {e}")
        return []

    def _save(self) -> None:
        try:
            with open(self.persistence_path, "w", encoding="utf-8") as f:
                json.dump({"files": self.discovered_files}, f)
        except Exception as e:
            logger.error(f"Failed to save repo_rules.json: {e}")

    def discover(self, paths: list[str]) -> list[str]:
        """Walk up from each path to AWD looking for rule files.

        Return a list of NEWLY discovered files in this batch.
        """
        new_discoveries = []
        # paths are already resolved absolute paths (or should be)
        for path in paths:
            abs_path = os.path.abspath(path)
            # Ensure path is within AWD
            try:
                if os.path.commonpath([abs_path, self.agent_working_dir]) != self.agent_working_dir:
                    continue
            except ValueError:
                continue

            current_dir = abs_path if os.path.isdir(abs_path) else os.path.dirname(abs_path)

            while True:
                # Check for rules in current_dir
                for rule_file in self.rule_filenames:
                    potential_path = os.path.join(current_dir, rule_file)
                    if os.path.exists(potential_path) and os.path.isfile(potential_path):
                        if potential_path not in self.discovered_files and potential_path not in new_discoveries:
                            new_discoveries.append(potential_path)

                # Stop if we reached AWD
                if current_dir == self.agent_working_dir:
                    break

                parent_dir = os.path.dirname(current_dir)
                if parent_dir == current_dir:
                    break
                current_dir = parent_dir

        added = []
        if new_discoveries:
            # Append new ones, respecting max_files
            for f in new_discoveries:
                if len(self.discovered_files) < self.max_files:
                    self.discovered_files.append(f)
                    added.append(f)
                else:
                    logger.info(f"Discovered {f} but max_files ({self.max_files}) reached.")

            if added:
                self._save()

        # Return only the ones actually added
        return added

    def render(self) -> str:
        """Render the rules block for the LLM context."""
        return render_repo_rules(self.discovered_files, self.max_chars, self.agent_working_dir)
