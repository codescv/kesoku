"""Unit tests for Kesoku config module."""

import os
from typing import Any

import pytest

from kesoku.config import KesokuConfig, init_config, load_config


def test_load_config_file_not_found(tmp_path: Any) -> None:
    """Verify load_config raises FileNotFoundError when the configuration file does not exist."""
    config_path = tmp_path / "nonexistent.toml"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_config(str(config_path))

    assert "Configuration file not found" in str(exc_info.value)


def test_load_config_success(tmp_path: Any) -> None:
    """Verify load_config succeeds and returns a KesokuConfig when config.toml exists."""
    config_path = tmp_path / "config.toml"
    init_config(str(config_path))

    cfg = load_config(str(config_path))
    assert isinstance(cfg, KesokuConfig)
    assert cfg.workspace.db_path == os.path.join(tmp_path, "kesoku.db")


def test_config_overrides(tmp_path: Any) -> None:
    """Verify ClaudeConfig and DiscordChannelOverride are correctly parsed from TOML."""
    config_path = tmp_path / "config.toml"
    toml_content = """
[workspace]
db_path = "kesoku.db"

[claude]
model_name = "custom-claude"
project_id = "test-proj"
location = "us-west1"

[[discord.channels]]
channels = ["12345", "announcements"]
llm = "claude"
auto_thread = false
"""
    with open(config_path, "w") as f:
        f.write(toml_content)

    cfg = load_config(str(config_path))
    assert cfg.claude.model_name == "custom-claude"
    assert cfg.claude.project_id == "test-proj"
    assert cfg.claude.location == "us-west1"

    assert len(cfg.discord.channels) == 1
    override = cfg.discord.channels[0]
    assert "12345" in override.channels
    assert "announcements" in override.channels
    assert override.llm == "claude"
    assert override.auto_thread is False


def test_config_google_chat(tmp_path: Any) -> None:
    """Verify GoogleChatConfig parameters are correctly parsed from TOML."""
    config_path = tmp_path / "config.toml"
    toml_content = """
[workspace]
db_path = "kesoku.db"

[google_chat]
enabled = true
chatbot_id = "gchat-custom"
project_id = "my-gcp-project"
topic_id = "my-topic"
subscription_id = "my-sub"
credentials_json = "/keys/sa.json"
impersonate_service_account = "sa@my-gcp-project.iam.gserviceaccount.com"
user_allowlist = ["users/111", "users/222"]
"""
    with open(config_path, "w") as f:
        f.write(toml_content)

    cfg = load_config(str(config_path))
    assert cfg.google_chat.enabled is True
    assert cfg.google_chat.chatbot_id == "gchat-custom"
    assert cfg.google_chat.project_id == "my-gcp-project"
    assert cfg.google_chat.topic_id == "my-topic"
    assert cfg.google_chat.subscription_id == "my-sub"
    assert cfg.google_chat.credentials_json == "/keys/sa.json"
    assert cfg.google_chat.impersonate_service_account == "sa@my-gcp-project.iam.gserviceaccount.com"
    assert "users/111" in cfg.google_chat.user_allowlist
    assert "users/222" in cfg.google_chat.user_allowlist


def test_config_env_injection(tmp_path: Any) -> None:
    """Verify that custom environment variables under [env] are correctly parsed and injected."""
    config_path = tmp_path / "config.toml"
    toml_content = """
[workspace]
db_path = "kesoku.db"

[env]
LCM_CONTEXT_THRESHOLD = 0.75
LCM_FRESH_TAIL_COUNT = 64
LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED = true
LCM_CACHE_FRIENDLY_CONDENSATION_ENABLED = false
LCM_RESERVE_TOKENS_FLOOR = 50000
CUSTOM_DUMMY_ENV = "hello_world"
"""
    with open(config_path, "w") as f:
        f.write(toml_content)

    # Clean up if they were already set
    for k in [
        "LCM_CONTEXT_THRESHOLD",
        "LCM_FRESH_TAIL_COUNT",
        "LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED",
        "LCM_CACHE_FRIENDLY_CONDENSATION_ENABLED",
        "LCM_RESERVE_TOKENS_FLOOR",
        "CUSTOM_DUMMY_ENV",
    ]:
        os.environ.pop(k, None)

    cfg = load_config(str(config_path))

    assert cfg.env["LCM_CONTEXT_THRESHOLD"] == 0.75
    assert cfg.env["LCM_FRESH_TAIL_COUNT"] == 64
    assert cfg.env["LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED"] is True
    assert cfg.env["LCM_CACHE_FRIENDLY_CONDENSATION_ENABLED"] is False
    assert cfg.env["LCM_RESERVE_TOKENS_FLOOR"] == 50000
    assert cfg.env["CUSTOM_DUMMY_ENV"] == "hello_world"

    # Check injection
    assert os.environ["LCM_CONTEXT_THRESHOLD"] == "0.75"
    assert os.environ["LCM_FRESH_TAIL_COUNT"] == "64"
    assert os.environ["LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED"] == "true"
    assert os.environ["LCM_CACHE_FRIENDLY_CONDENSATION_ENABLED"] == "false"
    assert os.environ["LCM_RESERVE_TOKENS_FLOOR"] == "50000"
    assert os.environ["CUSTOM_DUMMY_ENV"] == "hello_world"


def test_config_multiple_chatbots(tmp_path: Any) -> None:
    """Verify that multiple chatbots under [[chatbots.discord]] etc are correctly parsed."""
    config_path = tmp_path / "config.toml"
    toml_content = """
[workspace]
db_path = "kesoku.db"

[[chatbots.discord]]
enabled = true
chatbot_id = "discord-1"
bot_token = "token-1"

[[chatbots.discord]]
enabled = false
chatbot_id = "discord-2"
bot_token = "token-2"

[[chatbots.google_chat]]
enabled = true
chatbot_id = "gchat-1"
project_id = "gchat-proj"

[[chatbots.wechat]]
enabled = true
chatbot_id = "wechat-1"
account_id = "wechat-acc"
"""
    with open(config_path, "w") as f:
        f.write(toml_content)

    cfg = load_config(str(config_path))

    # Verify lists are populated
    assert len(cfg.chatbots.discord) == 2
    assert cfg.chatbots.discord[0].chatbot_id == "discord-1"
    assert cfg.chatbots.discord[0].enabled is True
    assert cfg.chatbots.discord[1].chatbot_id == "discord-2"
    assert cfg.chatbots.discord[1].enabled is False

    assert len(cfg.chatbots.google_chat) == 1
    assert cfg.chatbots.google_chat[0].chatbot_id == "gchat-1"
    assert cfg.chatbots.google_chat[0].project_id == "gchat-proj"

    assert len(cfg.chatbots.wechat) == 1
    assert cfg.chatbots.wechat[0].chatbot_id == "wechat-1"
    assert cfg.chatbots.wechat[0].account_id == "wechat-acc"

    # Verify helper properties (active list resolution)
    active_discords = cfg.active_discords
    assert len(active_discords) == 1
    assert active_discords[0].chatbot_id == "discord-1"

    # Verify lookup helper
    d_cfg = cfg.get_discord_config("discord-2")
    assert d_cfg is not None
    assert d_cfg.bot_token == "token-2"

    g_cfg = cfg.get_google_chat_config("gchat-1")
    assert g_cfg is not None
    assert g_cfg.project_id == "gchat-proj"

    w_cfg = cfg.get_wechat_config("wechat-1")
    assert w_cfg is not None
    assert w_cfg.account_id == "wechat-acc"


def test_lcm_custom_config(tmp_path: Any) -> None:
    """Verify custom LCM settings (lcm_llm and lcm_model_name) are parsed from TOML."""
    config_path = tmp_path / "config.toml"
    toml_content = """
[workspace]
db_path = "kesoku.db"

[agent]
llm = "gemini"
lcm_llm = "claude"

[gemini]
model_name = "gemini-3.5-pro"
lcm_model_name = "gemini-2.0-flash-lite"

[claude]
model_name = "claude-3-5-sonnet"
lcm_model_name = "claude-3-5-haiku"
"""
    with open(config_path, "w") as f:
        f.write(toml_content)

    cfg = load_config(str(config_path))
    assert cfg.agent.llm == "gemini"
    assert cfg.agent.lcm_llm == "claude"
    assert cfg.gemini.model_name == "gemini-3.5-pro"
    assert cfg.gemini.lcm_model_name == "gemini-2.0-flash-lite"
    assert cfg.claude.model_name == "claude-3-5-sonnet"
    assert cfg.claude.lcm_model_name == "claude-3-5-haiku"


