"""Skill discovery and loading management for Kesoku AI Agent framework.

Provides autonomous skill loading, YAML frontmatter manifest parsing, OS platform
filtering, and secure absolute path header injection.
"""

import os
import platform
import re
import sys
from typing import Any

import yaml
from pydantic import BaseModel, Field

from kesoku.config import get_config
from kesoku.logger import setup_logger

logger = setup_logger(__name__)


class SkillMetadata(BaseModel):
    """Metadata attributes for a skill."""

    tags: list[str] = Field(default_factory=list, description="Tags categorizing the skill")
    platforms: list[str] | None = Field(
        default=None,
        description="Supported OS (linux, darwin, windows). If None, all platforms. If [], none.",
    )
    related_skills: list[str] = Field(default_factory=list, description="Names of related skills")


class SkillManifest(BaseModel):
    """Structured representation of a skill's manifest parsed from YAML frontmatter."""

    name: str = Field(..., description="Unique identifier name of the skill")
    description: str = Field(..., description="Short description of the skill's capability")
    version: str = Field(default="1.0.0", description="Version of the skill")
    required_permissions: list[str] = Field(
        default_factory=list, description="Required tool or system permissions (e.g., 'run_shell_command')"
    )
    metadata: SkillMetadata = Field(default_factory=SkillMetadata)


def _get_current_platform() -> str:
    """Retrieve the normalized current operating system name (linux, darwin, windows)."""
    system = platform.system().lower()
    if system == "windows" or sys.platform == "win32":
        return "windows"
    if system == "darwin" or sys.platform == "darwin":
        return "darwin"
    return "linux"


def parse_skill_markdown(filepath: str) -> tuple[SkillManifest, str]:
    """Parse a skill markdown file extracting YAML frontmatter and raw body content.

    Args:
        filepath: Absolute path to the markdown file.

    Returns:
        Tuple of (SkillManifest, raw markdown body content).

    Raises:
        ValueError: If frontmatter is missing or malformed.
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        raise ValueError(f"Failed to read skill file {filepath}: {e}")

    # Regex to extract frontmatter enclosed in ---
    match = re.match(r"^\s*---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
    if not match:
        raise ValueError(f"No YAML frontmatter found in {filepath}. Skill files must begin with ---")

    frontmatter_str, body = match.groups()
    try:
        data = yaml.safe_load(frontmatter_str)
        if not isinstance(data, dict):
            raise ValueError("YAML frontmatter must be a dictionary")
    except Exception as e:
        raise ValueError(f"Failed to parse YAML frontmatter in {filepath}: {e}")

    try:
        manifest = SkillManifest.model_validate(data)
    except Exception as e:
        raise ValueError(f"Invalid skill manifest structure in {filepath}: {e}")

    return manifest, body.strip()


class SkillManager:
    """Manages discovering, validating, filtering, and loading specialized agent skills."""

    def __init__(self, skills_dir: str | None = None, roles_dir: str | None = None) -> None:
        """Initialize SkillManager.

        Args:
            skills_dir: Optional override for skills directory path. If None, uses get_config().
            roles_dir: Optional override for the roles directory (role-scoped skills live in
                `<roles_dir>/<role>/skills/`). If None, uses get_config().
        """
        self._skills_dir_override = skills_dir
        self._roles_dir_override = roles_dir

    def _get_skills_dir(self) -> str:
        """Retrieve the real canonical path to the workspace skills directory."""
        if self._skills_dir_override is not None:
            target = self._skills_dir_override
        else:
            target = get_config().workspace.skills_dir
        real = os.path.realpath(target)
        if not os.path.exists(real):
            os.makedirs(real, exist_ok=True)
        return real

    def _get_role_skills_dir(self, role: str | None) -> str | None:
        """Return the canonical `<roles_dir>/<role>/skills` directory, or None if absent/invalid.

        Role names are validated with the same character whitelist as skill names and the result must stay
        inside the roles directory, so a crafted role name can never escape it.
        """
        if not role or not re.match(r"^[a-zA-Z0-9_\-]+$", role):
            return None
        if self._roles_dir_override is not None:
            roles_dir = self._roles_dir_override
        else:
            try:
                roles_dir = get_config().workspace.roles_dir
            except Exception:
                return None
        if not roles_dir:
            return None
        roles_base = os.path.realpath(roles_dir)
        target = os.path.realpath(os.path.join(roles_base, role, "skills"))
        if os.path.commonpath([roles_base, target]) != roles_base or not os.path.isdir(target):
            return None
        return target

    @staticmethod
    def _resolve_in_base(base: str, skill_name: str) -> str | None:
        """Resolve skill_name inside base with a canonical boundary check; None if it does not exist."""
        target = os.path.realpath(os.path.join(base, skill_name))
        if os.path.commonpath([base, target]) != base or target == base:
            raise KeyError(f"Invalid skill name '{skill_name}': Path traversal boundary violation.")
        return target if os.path.isdir(target) else None

    def get_skill_dir(self, skill_name: str, role: str | None = None) -> str:
        """Resolve a skill directory, preferring the role-scoped skill over the global one.

        Args:
            skill_name: Raw skill name identifier.
            role: Optional active role; enables `<roles_dir>/<role>/skills/` lookup.

        Returns:
            Absolute canonical path to the skill's directory.

        Raises:
            KeyError: If skill_name is invalid, missing, or violates path boundaries.
        """
        if not re.match(r"^[a-zA-Z0-9_\-]+$", skill_name):
            raise KeyError(f"Invalid skill name '{skill_name}': Contains prohibited characters.")
        role_base = self._get_role_skills_dir(role)
        if role_base:
            found = self._resolve_in_base(role_base, skill_name)
            if found:
                return found
        return self._resolve_skill_dir(skill_name)

    def _resolve_skill_dir(self, skill_name: str) -> str:
        """Safely resolve and validate a skill directory ensuring no path traversal.

        Args:
            skill_name: Raw skill name identifier.

        Returns:
            Absolute canonical path to the skill's directory.

        Raises:
            KeyError: If skill_name is invalid or directory does not exist or violates path boundaries.
        """
        if not re.match(r"^[a-zA-Z0-9_\-]+$", skill_name):
            raise KeyError(f"Invalid skill name '{skill_name}': Contains prohibited characters.")

        base = self._get_skills_dir()
        target = os.path.realpath(os.path.join(base, skill_name))

        # Canonical boundary check
        if os.path.commonpath([base, target]) != base or target == base:
            raise KeyError(f"Invalid skill name '{skill_name}': Path traversal boundary violation.")

        if not os.path.isdir(target):
            raise KeyError(f"Skill '{skill_name}' does not exist.")

        return target

    def _find_skill_markdown_file(self, skill_dir: str) -> str:
        """Find the primary markdown file defining a skill (preferring SKILL.md)."""
        primary = os.path.join(skill_dir, "SKILL.md")
        if os.path.isfile(primary):
            return primary

        # Fallback: search for any .md file in the directory
        for item in os.listdir(skill_dir):
            if item.lower().endswith(".md") and os.path.isfile(os.path.join(skill_dir, item)):
                return os.path.join(skill_dir, item)

        raise FileNotFoundError(f"No SKILL.md or markdown file found in {skill_dir}")

    def _check_permissions(self, manifest: SkillManifest) -> None:
        """Verify whether required permissions/tools for a skill are enabled in configuration."""
        try:
            cfg = get_config()
        except RuntimeError:
            # Config not loaded in standalone/test mode without init
            return

        for perm in manifest.required_permissions:
            if perm == "run_shell_command" and not cfg.shell.enabled:
                raise PermissionError(f"Skill '{manifest.name}' requires shell execution tool, which is disabled.")

    def is_platform_supported(self, manifest: SkillManifest, host_platform: str | None = None) -> bool:
        """Check whether the skill is supported on the specified host operating system.

        Args:
            manifest: The parsed SkillManifest.
            host_platform: The host OS name. If None, uses _get_current_platform().

        Returns:
            True if supported, False otherwise.
        """
        if manifest.metadata.platforms is None:
            return True
        if not manifest.metadata.platforms:
            return False

        current = host_platform or _get_current_platform()
        allowed = [p.lower().strip() for p in manifest.metadata.platforms]
        return current in allowed

    def list_skills(self, role: str | None = None) -> list[dict[str, Any]]:
        """List all valid skills supported on the current host OS (global plus the role's own skills).

        Args:
            role: Optional active role. Skills in `<roles_dir>/<role>/skills/` are included and take
                precedence over global skills with the same name.

        Returns:
            List of dictionaries containing summary details of each supported skill.
        """
        results_by_name: dict[str, dict[str, Any]] = {}
        # Global first, then role-scoped so role skills override on name collision.
        for item in self._list_skills_in(self._get_skills_dir()):
            results_by_name[item["name"]] = item
        role_dir = self._get_role_skills_dir(role)
        if role_dir:
            for item in self._list_skills_in(role_dir):
                results_by_name[item["name"]] = {**item, "role": role}
        results = sorted(results_by_name.values(), key=lambda x: x["name"])
        return results

    def _list_skills_in(self, skills_dir: str) -> list[dict[str, Any]]:
        """List valid, platform-supported skills directly under skills_dir."""
        results = []
        current_platform = _get_current_platform()

        try:
            entries = os.listdir(skills_dir)
        except Exception as e:
            logger.error(f"Failed to list skills directory {skills_dir}: {e}")
            return []

        for item in entries:
            full_path = os.path.join(skills_dir, item)
            if not os.path.isdir(full_path):
                continue

            try:
                md_file = self._find_skill_markdown_file(full_path)
                manifest, _ = parse_skill_markdown(md_file)
            except Exception as e:
                logger.debug(f"Skipping directory {item}: {e}")
                continue

            if not self.is_platform_supported(manifest, host_platform=current_platform):
                logger.debug(f"Skipping skill {manifest.name}: Not supported on host OS ({current_platform}).")
                continue

            results.append(
                {
                    "name": manifest.name,
                    "description": manifest.description,
                    "version": manifest.version,
                    "tags": manifest.metadata.tags,
                    "platforms": manifest.metadata.platforms,
                }
            )

        return results

    def get_skill(self, skill_name: str, role: str | None = None) -> tuple[SkillManifest, str]:
        """Retrieve the manifest and complete instruction content for a specific skill.

        Prepares a robust header containing the skill directory absolute path
        to ensure correct script and tool invocations during execution.

        Args:
            skill_name: Unique skill name identifier.
            role: Optional active role; a role-scoped skill with the same name takes precedence.

        Returns:
            Tuple of (SkillManifest, formatted markdown instructions).

        Raises:
            KeyError: If the skill is not found or invalid.
            PermissionError: If required tools/permissions are disabled in config.
            ValueError: If the skill is not supported on the current OS.
        """
        skill_dir = self.get_skill_dir(skill_name, role=role)
        try:
            md_file = self._find_skill_markdown_file(skill_dir)
        except FileNotFoundError as e:
            raise KeyError(f"Skill '{skill_name}' is missing its instruction markdown file: {e}")

        try:
            manifest, raw_body = parse_skill_markdown(md_file)
        except Exception as e:
            raise KeyError(f"Failed to parse skill '{skill_name}': {e}")

        current_platform = _get_current_platform()
        if not self.is_platform_supported(manifest, host_platform=current_platform):
            raise ValueError(f"Skill '{skill_name}' is not supported on current OS ({current_platform}).")

        self._check_permissions(manifest)

        # Format absolute path header to solve script location errors
        abs_skill_dir = os.path.realpath(skill_dir)
        header = f"""
# Skill: {manifest.name} (v{manifest.version})
> [!IMPORTANT] You must replace the SKILL_DIR or explicitly set it as an env variable
> mentioned in the skill instructions with the following value:
> SKILL_DIR='{abs_skill_dir}'
>
        """

        return manifest, f"{header}{raw_body}"
