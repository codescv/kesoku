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

