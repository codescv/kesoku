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


class GeminiConfig(BaseModel):
    """Google GenAI / Gemini LLM configuration settings."""

    model_name: str = Field(default="gemini-3.1-flash", description="Gemini model identifier")
    auth_mode: Literal["api_key", "vertex"] = Field(default="vertex", description="Authentication mode")
    api_key: str | None = Field(default=None, description="API key (if auth_mode='api_key')")
    project_id: str | None = Field(
        default="gtech-ads-localizer-external", description="GCP Project ID (for Vertex AI mode)"
    )
    location: str | None = Field(default="global", description="GCP Region/Location (for Vertex AI mode)")


class DiscordConfig(BaseModel):
    """Discord chatbot adapter settings."""

    enabled: bool = Field(default=False, description="Whether to launch the Discord chatbot in daemon mode")
    bot_token: str | None = Field(default=None, description="Discord bot token")
    chatbot_id: str = Field(default="discord_primary", description="Unique chatbot identifier")


class KesokuConfig(BaseModel):
    """Root Kesoku configuration structure."""

    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)

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

    try:
        with open(config_path, "wb") as f:
            tomli_w.dump(data, f)
        logger.info(f"Configuration saved successfully to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {e}")
        raise


def init_config(config_path: str, force: bool = False) -> None:
    """Copy config.example.toml template to config_path when initializing workspace.

    Args:
        config_path: Target path for config.toml.
        force: Whether to overwrite existing config (creating a backup).
    """
    if os.path.exists(config_path):
        if not force:
            logger.info(
                f"Configuration file already exists at {config_path}. "
                "Skipping default config creation. Use --force to overwrite."
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
