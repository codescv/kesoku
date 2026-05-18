"""Configuration management for Kesoku AI Agent framework.

Defines structured Pydantic settings and TOML persistence.
"""

import importlib.resources
import os
import shutil
import time
import tomllib
from typing import Literal

import tomli_w
from pydantic import BaseModel, Field

from kesoku.logger import setup_logger

logger = setup_logger(__name__)


class WorkspaceConfig(BaseModel):
    """Workspace-level configuration settings."""

    db_path: str = Field(default="kesoku.db", description="Path to SQLite database file")
    skills_dir: str = Field(default="skills", description="Path to skills directory")
    sessions_dir: str = Field(default="sessions", description="Path to session staging directory")


class AgentConfig(BaseModel):
    """Agent-level configuration settings."""

    llm: str = Field(default="gemini", description="LLM provider identifier (e.g., gemini, mock)")


class GeminiConfig(BaseModel):
    """Google GenAI / Gemini LLM configuration settings."""

    model_name: str = Field(default="gemini-3.1-flash", description="Gemini model identifier")
    auth_mode: Literal["api_key", "vertex"] = Field(default="vertex", description="Authentication mode")
    api_key: str | None = Field(default=None, description="API key (if auth_mode='api_key')")
    project_id: str | None = Field(
        default="gtech-ads-localizer-external", description="GCP Project ID (for Vertex AI mode)"
    )
    location: str | None = Field(default="global", description="GCP Region/Location (for Vertex AI mode)")
    thinking_level: Literal["minimal", "low", "medium", "high"] | None = Field(
        default="high", description="Thinking level allocated for reasoning ('minimal', 'low', 'medium', 'high', or None to use model default)"
    )


class DiscordConfig(BaseModel):
    """Discord chatbot adapter settings."""

    enabled: bool = Field(default=False, description="Whether to launch the Discord chatbot in daemon mode")
    bot_token: str | None = Field(default=None, description="Discord bot token")
    chatbot_id: str = Field(default="discord", description="Unique chatbot identifier")
    user_allowlist: list[str] = Field(
        default_factory=list, description="List of allowed Discord user IDs or usernames"
    )


class ShellConfig(BaseModel):
    """Shell command execution tool configuration settings."""

    enabled: bool = Field(default=True, description="Whether the shell command execution tool is enabled")
    use_shell: bool = Field(
        default=True, description="Whether to execute commands via system shell (enabling pipes/redirection)"
    )
    mode: Literal["allowlist", "blocklist"] = Field(
        default="blocklist",
        description="Filtering mode: 'allowlist' strictly permits matching patterns, 'blocklist' prohibits matching patterns",
    )
    allowlist_patterns: list[str] = Field(
        default_factory=lambda: [r"^(echo|ls|pwd|cat|git|uv|grep|find|python|sed|awk)(\s|$)"],
        description="Regex patterns for allowed commands in allowlist mode",
    )
    blocklist_patterns: list[str] = Field(
        default_factory=lambda: [r"(\b|^)(rm|sudo|shutdown|reboot|mkfs|dd|chmod|chown)(\b|\s|$)"],
        description="Regex patterns for prohibited commands in blocklist mode",
    )
    env: dict[str, str] = Field(
        default_factory=dict, description="Custom environment variables injected into subprocesses"
    )


class KesokuConfig(BaseModel):
    """Root Kesoku configuration structure."""

    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    shell: ShellConfig = Field(default_factory=ShellConfig)

    def resolve_paths(self, config_file_path: str) -> None:
        """Resolve workspace relative paths against the directory containing the config file.

        Args:
            config_file_path: Path to the loaded or target config.toml.
        """
        base_dir = os.path.dirname(os.path.abspath(config_file_path))
        if not os.path.isabs(self.workspace.db_path):
            self.workspace.db_path = os.path.join(base_dir, self.workspace.db_path)
        if not os.path.isabs(self.workspace.skills_dir):
            self.workspace.skills_dir = os.path.join(base_dir, self.workspace.skills_dir)
        if not os.path.isabs(self.workspace.sessions_dir):
            self.workspace.sessions_dir = os.path.join(base_dir, self.workspace.sessions_dir)


_global_config: KesokuConfig | None = None


def get_config() -> KesokuConfig:
    """Get the global KesokuConfig instance.

    Raises:
        RuntimeError: If configuration has not been loaded yet.
    """
    global _global_config
    if _global_config is None:
        raise RuntimeError("Configuration has not been loaded. Call load_config() first.")
    return _global_config


def load_config(config_path: str) -> KesokuConfig:
    """Load Kesoku configuration from a TOML file.

    Args:
        config_path: Path to config.toml.

    Returns:
        KesokuConfig instance populated from file.
    """
    global _global_config
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file {config_path} not found. Using defaults.")
        cfg = KesokuConfig()
        cfg.resolve_paths(config_path)
        _global_config = cfg
        return cfg

    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        cfg = KesokuConfig.model_validate(data)
        cfg.resolve_paths(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        _global_config = cfg
        return cfg
    except Exception as e:
        logger.error(f"Error parsing configuration file {config_path}: {e}")
        raise


def save_config(cfg: KesokuConfig, config_path: str) -> None:
    """Save Kesoku configuration to a TOML file.

    Args:
        cfg: KesokuConfig instance.
        config_path: Target path for config.toml.
    """
    base_dir = os.path.dirname(os.path.abspath(config_path))
    if base_dir and not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    # Store relative paths in the TOML file for portability
    data = cfg.model_dump(mode="json", exclude_none=True)
    if os.path.isabs(data["workspace"]["db_path"]):
        data["workspace"]["db_path"] = os.path.basename(data["workspace"]["db_path"])
    if os.path.isabs(data["workspace"]["skills_dir"]):
        data["workspace"]["skills_dir"] = os.path.basename(data["workspace"]["skills_dir"])
    if os.path.isabs(data["workspace"]["sessions_dir"]):
        data["workspace"]["sessions_dir"] = os.path.basename(data["workspace"]["sessions_dir"])

    try:
        with open(config_path, "wb") as f:
            tomli_w.dump(data, f)
        logger.info(f"Configuration saved successfully to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {e}")
        raise


def init_config(config_path: str, overwrite: bool = False) -> None:
    """Copy config.example.toml template to config_path when initializing workspace.

    Args:
        config_path: Target path for config.toml.
        overwrite: Whether to overwrite existing config (creating a backup).
    """
    if os.path.exists(config_path):
        if not overwrite:
            logger.info(
                f"Configuration file already exists at {config_path}. "
                "Skipping default config creation. Use --overwrite-config to overwrite."
            )
            return
        backup_path = f"{config_path}.bak.{int(time.time())}"
        shutil.copy(config_path, backup_path)
        logger.info(f"Created backup of existing configuration at {backup_path}")

    base_dir = os.path.dirname(os.path.abspath(config_path))
    if base_dir and not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    try:
        ref = importlib.resources.files("kesoku.resources") / "config.example.toml"
        template_bytes = ref.read_bytes()
        with open(config_path, "wb") as f:
            f.write(template_bytes)
        logger.info(f"Configuration template copied successfully to {config_path}")
    except Exception as e:
        logger.error(f"Failed to copy configuration template to {config_path}: {e}")
        raise


def init_skills(skills_dir: str, overwrite: bool = False) -> None:
    """Copy default resource skills from kesoku.resources to workspace skills_dir when initializing.

    Args:
        skills_dir: Target path for workspace skills directory.
        overwrite: Whether to overwrite existing skills in target directory.
    """
    os.makedirs(skills_dir, exist_ok=True)
    try:
        ref = importlib.resources.files("kesoku.resources") / "skills"
        with importlib.resources.as_file(ref) as source_dir:
            if source_dir.exists() and source_dir.is_dir():
                for item in source_dir.iterdir():
                    if item.is_dir():
                        dest = os.path.join(skills_dir, item.name)
                        if os.path.exists(dest):
                            if not overwrite:
                                logger.debug(f"Skill {item.name} already exists at {dest}. Skipping.")
                                continue
                            shutil.rmtree(dest)
                        shutil.copytree(item, dest)
                        logger.info(f"Copied resource skill {item.name} to {dest}")
    except Exception as e:
        logger.error(f"Failed to copy resource skills to {skills_dir}: {e}")


