"""Configuration management for Kesoku AI Agent framework.

Defines structured Pydantic settings and TOML persistence.
"""

import importlib.resources
import os
import shutil
import sys
import tempfile
import time
import tomllib
from typing import Literal

import tomli_w
from pydantic import BaseModel, Field

from kesoku.logger import setup_logger

logger = setup_logger(__name__)


def _default_db_path() -> str:
    if "pytest" in sys.modules:
        return os.path.join(tempfile.gettempdir(), "kesoku_tests.db")
    return "kesoku.db"


def _default_skills_dir() -> str:
    if "pytest" in sys.modules:
        return os.path.join(tempfile.gettempdir(), "kesoku_tests_skills")
    return "skills"


def _default_roles_dir() -> str:
    if "pytest" in sys.modules:
        return os.path.join(tempfile.gettempdir(), "kesoku_tests_roles")
    return "roles"


def _default_sessions_dir() -> str:
    if "pytest" in sys.modules:
        return os.path.join(tempfile.gettempdir(), "kesoku_tests_sessions")
    return "sessions"


class WorkspaceConfig(BaseModel):
    """Workspace-level configuration settings."""

    db_path: str = Field(
        default_factory=_default_db_path,
        description="Path to SQLite database file",
    )
    skills_dir: str = Field(
        default_factory=_default_skills_dir,
        description="Path to skills directory",
    )
    roles_dir: str = Field(
        default_factory=_default_roles_dir,
        description="Path to roles directory",
    )
    sessions_dir: str = Field(
        default_factory=_default_sessions_dir,
        description="Path to session staging directory",
    )


class AgentConfig(BaseModel):
    """Agent-level configuration settings."""

    llm: str = Field(default="gemini", description="LLM provider identifier (e.g., gemini, mock)")
    user_prompts: list[str] = Field(
        default_factory=list,
        description="List of custom user prompt file paths relative to agent working directory",
    )
    raw_llm_logs: bool = Field(
        default=True,
        description="Whether to write raw LLM inputs and outputs to turn log files in the session directory",
    )
    compact_history_warning_threshold: float = Field(
        default=80.0,
        description="Threshold percentage of context window limit before warning is shown",
    )
    compact_history_threshold: float = Field(
        default=0.8,
        description="Threshold for automatic in-place history compaction (if >1 raw tokens, if <1 window ratio)",
    )
    lcm_llm: str | None = Field(
        default=None,
        description="LLM provider identifier for LCM and background memory (if None, defaults to llm)",
    )



class GeminiConfig(BaseModel):
    """Google GenAI / Gemini LLM configuration settings."""

    model_name: str = Field(default="gemini-3.1-flash", description="Gemini model identifier")
    lcm_model_name: str | None = Field(
        default=None,
        description="Model name override to use for LCM and background memory",
    )
    auth_mode: Literal["api_key", "vertex"] = Field(default="vertex", description="Authentication mode")
    api_key: str | None = Field(default=None, description="API key (if auth_mode='api_key')")
    project_id: str | None = Field(
        default="gtech-ads-localizer-external", description="GCP Project ID (for Vertex AI mode)"
    )
    location: str | None = Field(default="global", description="GCP Region/Location (for Vertex AI mode)")
    thinking_level: Literal["minimal", "low", "medium", "high"] | None = Field(
        default="high",
        description=(
            "Thinking level allocated for reasoning ('minimal', 'low', 'medium', 'high', or None to use model default)"
        ),
    )
    context_caching: bool = Field(
        default=True,
        description="Whether to enable explicit context caching for long sessions",
    )
    context_caching_threshold: int = Field(
        default=4096,
        description="Minimum token threshold of static history prefix before context caching is triggered",
    )
    context_caching_ttl: int = Field(
        default=3600,
        description="Time-to-Live (TTL) in seconds for the explicit context cache",
    )



class ClaudeConfig(BaseModel):
    """Anthropic / Claude LLM configuration settings on Vertex AI."""

    model_name: str = Field(default="claude-3-5-sonnet@20241022", description="Claude model identifier")
    lcm_model_name: str | None = Field(
        default=None,
        description="Model name override to use for LCM and background memory",
    )
    project_id: str | None = Field(default="gtech-ads-localizer-external", description="GCP Project ID (for Vertex AI)")
    location: str | None = Field(default="us-east5", description="GCP Region/Location (for Vertex AI)")


class DiscordChannelOverride(BaseModel):
    """Override settings for specific Discord channels."""

    channels: list[str] = Field(default_factory=list, description="Channel IDs or names to match")
    llm: str | None = Field(default=None, description="Override LLM provider (e.g., 'claude', 'gemini')")
    auto_thread: bool | None = Field(default=None, description="Override auto_thread behavior")


class DiscordConfig(BaseModel):
    """Discord chatbot adapter settings."""

    enabled: bool = Field(default=False, description="Whether to launch the Discord chatbot in daemon mode")
    bot_token: str | None = Field(default=None, description="Discord bot token")
    chatbot_id: str = Field(default="discord", description="Unique chatbot identifier")
    user_allowlist: list[str] = Field(default_factory=list, description="List of allowed Discord user IDs or usernames")
    channels: list[DiscordChannelOverride] = Field(
        default_factory=list,
        description="Channel-specific configuration overrides",
    )


class ShellConfig(BaseModel):
    """Shell command execution tool configuration settings."""

    enabled: bool = Field(default=True, description="Whether the shell command execution tool is enabled")
    use_shell: bool = Field(
        default=True, description="Whether to execute commands via system shell (enabling pipes/redirection)"
    )
    mode: Literal["allowlist", "blocklist"] = Field(
        default="blocklist",
        description=(
            "Filtering mode: 'allowlist' strictly permits matching patterns, 'blocklist' prohibits matching patterns"
        ),
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
    background_threshold_seconds: float = Field(
        default=300.0,
        description="Time limit in seconds before a shell command is transitioned to background execution",
    )


class GoogleChatConfig(BaseModel):
    """Google Chat chatbot adapter settings."""

    enabled: bool = Field(default=False, description="Whether to launch the Google Chat chatbot in daemon mode")
    chatbot_id: str = Field(default="google_chat", description="Unique chatbot identifier")
    project_id: str | None = Field(default=None, description="GCP Project ID for Pub/Sub")
    topic_id: str | None = Field(default=None, description="GCP Pub/Sub Topic ID")
    subscription_id: str | None = Field(default=None, description="GCP Pub/Sub Pull Subscription ID")
    credentials_json: str | None = Field(default=None, description="Path to service account JSON file (optional)")
    impersonate_service_account: str | None = Field(
        default=None, description="Email of target service account to impersonate (optional)"
    )
    user_allowlist: list[str] = Field(default_factory=list, description="Allowed Google Chat users (emails or IDs)")
    reaction_emoji: str | None = Field(default=None, description="Emoji to react with when receiving a user message")


class WechatConfig(BaseModel):
    """WeChat chatbot adapter settings."""

    enabled: bool = Field(default=False, description="Whether to launch the WeChat chatbot in daemon mode")
    chatbot_id: str = Field(default="wechat", description="Unique chatbot identifier")
    account_id: str | None = Field(default=None, description="WeChat/iLink bot account ID")
    token: str | None = Field(default=None, description="WeChat/iLink bot auth token")
    base_url: str = Field(default="https://ilinkai.weixin.qq.com", description="WeChat/iLink API base URL")
    sys_prompt_file: str | None = Field(
        default=None,
        description="Path to custom system prompt file for WeChat relative to agent working directory",
    )


class ChatbotsConfig(BaseModel):
    """Container for multiple chatbot adapter lists."""

    discord: list[DiscordConfig] = Field(default_factory=list)
    google_chat: list[GoogleChatConfig] = Field(default_factory=list)
    wechat: list[WechatConfig] = Field(default_factory=list)


class KesokuConfig(BaseModel):
    """Root Kesoku configuration structure."""

    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    google_chat: GoogleChatConfig = Field(default_factory=GoogleChatConfig)
    wechat: WechatConfig = Field(default_factory=WechatConfig)
    chatbots: ChatbotsConfig = Field(default_factory=ChatbotsConfig)
    shell: ShellConfig = Field(default_factory=ShellConfig)
    env: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Custom environment variables injected into os.environ on startup",
    )
    agent_working_dir: str | None = Field(
        default=None, exclude=True, description="Absolute path of the directory containing the config file"
    )

    @property
    def active_discords(self) -> list[DiscordConfig]:
        """Get all enabled Discord configs."""
        configs = []
        if self.discord.enabled:
            configs.append(self.discord)
        for d in self.chatbots.discord:
            if d.enabled:
                configs.append(d)
        return configs

    @property
    def active_google_chats(self) -> list[GoogleChatConfig]:
        """Get all enabled Google Chat configs."""
        configs = []
        if self.google_chat.enabled:
            configs.append(self.google_chat)
        for g in self.chatbots.google_chat:
            if g.enabled:
                configs.append(g)
        return configs

    @property
    def active_wechats(self) -> list[WechatConfig]:
        """Get all enabled WeChat configs."""
        configs = []
        if self.wechat.enabled:
            configs.append(self.wechat)
        for w in self.chatbots.wechat:
            if w.enabled:
                configs.append(w)
        return configs

    def get_discord_config(self, chatbot_id: str) -> DiscordConfig | None:
        """Find DiscordConfig with matching chatbot_id."""
        if self.discord.chatbot_id == chatbot_id:
            return self.discord
        for d_cfg in self.chatbots.discord:
            if d_cfg.chatbot_id == chatbot_id:
                return d_cfg
        return None

    def get_google_chat_config(self, chatbot_id: str) -> GoogleChatConfig | None:
        """Find GoogleChatConfig with matching chatbot_id."""
        if self.google_chat.chatbot_id == chatbot_id:
            return self.google_chat
        for g_cfg in self.chatbots.google_chat:
            if g_cfg.chatbot_id == chatbot_id:
                return g_cfg
        return None

    def get_wechat_config(self, chatbot_id: str) -> WechatConfig | None:
        """Find WechatConfig with matching chatbot_id."""
        if self.wechat.chatbot_id == chatbot_id:
            return self.wechat
        for w_cfg in self.chatbots.wechat:
            if w_cfg.chatbot_id == chatbot_id:
                return w_cfg
        return None

    def resolve_paths(self, config_file_path: str) -> None:
        """Resolve workspace relative paths against the directory containing the config file.

        Args:
            config_file_path: Path to the loaded or target config.toml.
        """
        base_dir = os.path.dirname(os.path.abspath(config_file_path))
        self.agent_working_dir = base_dir
        if not os.path.isabs(self.workspace.db_path):
            self.workspace.db_path = os.path.join(base_dir, self.workspace.db_path)
        if not os.path.isabs(self.workspace.skills_dir):
            self.workspace.skills_dir = os.path.join(base_dir, self.workspace.skills_dir)
        if not os.path.isabs(self.workspace.roles_dir):
            self.workspace.roles_dir = os.path.join(base_dir, self.workspace.roles_dir)
        if not os.path.isabs(self.workspace.sessions_dir):
            self.workspace.sessions_dir = os.path.join(base_dir, self.workspace.sessions_dir)


_global_config: KesokuConfig | None = None


def get_config() -> KesokuConfig:
    """Get the global KesokuConfig instance.

    If configuration has not been explicitly loaded yet, lazily initializes and returns
    a default KesokuConfig instance to ensure calls never fail.
    """
    global _global_config
    if _global_config is None:
        logger.debug("Global configuration not loaded yet. Initializing defaults.")
        _global_config = KesokuConfig()
    return _global_config


def load_config(config_path: str) -> KesokuConfig:
    """Load Kesoku configuration from a TOML file.

    Args:
        config_path: Path to config.toml.

    Returns:
        KesokuConfig instance populated from file.

    Raises:
        FileNotFoundError: If the configuration file is not found.
    """
    global _global_config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        cfg = KesokuConfig.model_validate(data)
        cfg.resolve_paths(config_path)

        # Inject custom environment variables
        for k, v in cfg.env.items():
            if isinstance(v, bool):
                os.environ[k] = str(v).lower()
            else:
                os.environ[k] = str(v)
            logger.debug(f"Injected env var: {k}={os.environ[k]}")

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
    if os.path.isabs(data["workspace"]["roles_dir"]):
        data["workspace"]["roles_dir"] = os.path.basename(data["workspace"]["roles_dir"])
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

    # Copy cronjob.example.toml to cronjob.toml if not exists
    cron_toml_path = os.path.join(base_dir, "cronjob.toml")
    if not os.path.exists(cron_toml_path):
        try:
            cron_ref = importlib.resources.files("kesoku.resources") / "cronjob.example.toml"
            cron_bytes = cron_ref.read_bytes()
            with open(cron_toml_path, "wb") as f:
                f.write(cron_bytes)
            logger.info(f"Cronjob template copied successfully to {cron_toml_path}")
        except Exception as e:
            logger.error(f"Failed to copy cronjob template to {cron_toml_path}: {e}")


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


def init_roles(roles_dir: str, overwrite: bool = False) -> None:
    """Copy default resource roles from kesoku.resources to workspace roles_dir when initializing.

    Args:
        roles_dir: Target path for workspace roles directory.
        overwrite: Whether to overwrite existing roles in target directory.
    """
    os.makedirs(roles_dir, exist_ok=True)
    try:
        ref = importlib.resources.files("kesoku.resources") / "roles"
        with importlib.resources.as_file(ref) as source_dir:
            if source_dir.exists() and source_dir.is_dir():
                for item in source_dir.iterdir():
                    if item.is_dir():
                        dest = os.path.join(roles_dir, item.name)
                        if os.path.exists(dest):
                            if not overwrite:
                                logger.debug(f"Role {item.name} already exists at {dest}. Skipping.")
                                continue
                            if os.path.isdir(dest):
                                shutil.rmtree(dest)
                            else:
                                os.remove(dest)
                        shutil.copytree(item, dest)
                        logger.info(f"Copied resource role {item.name} to {dest}")
    except Exception as e:
        logger.error(f"Failed to copy resource roles to {roles_dir}: {e}")
