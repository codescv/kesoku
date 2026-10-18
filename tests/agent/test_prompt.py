"""Unit tests for the modular system prompt construction utility."""

import re
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import kesoku.config
from kesoku.agent.prompt import SESSION_NOTES_INSTRUCTIONS, build_sys_prompt
from kesoku.config import KesokuConfig, init_config, load_config
from kesoku.db import Session


def test_build_sys_prompt_with_custom_context() -> None:
    """Verify build_sys_prompt appends custom context instructions correctly."""
    custom_context = "You are inside a specialized testing environment."
    prompt = build_sys_prompt(custom_prompt=custom_context)

    assert "# Output Formatting Rules" in prompt
    assert custom_context in prompt


def test_build_sys_prompt_with_working_directory(tmp_path) -> None:
    """Verify build_sys_prompt includes agent working directory when config is loaded."""
    import kesoku.config
    from kesoku.config import init_config, load_config

    config_path = tmp_path / "config.toml"
    init_config(str(config_path))

    original_config = kesoku.config._global_config
    try:
        # Load a dummy configuration path
        cfg = load_config(str(config_path))

        prompt = build_sys_prompt()

        # Check that Agent Working Directory header and path are included
        assert "# Agent Working Directory" in prompt
        assert "AWD=" in prompt
        assert cfg.agent_working_dir in prompt
    finally:
        kesoku.config._global_config = original_config


def test_build_sys_prompt_with_user_prompts(tmp_path) -> None:
    """Verify build_sys_prompt resolves and injects user_prompts files correctly."""
    import kesoku.config
    from kesoku.config import init_config, load_config

    # Create a couple of dummy prompt files
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    file_a = prompt_dir / "prompt_a.txt"
    file_a.write_text("Instruction A from file.", encoding="utf-8")
    file_b = prompt_dir / "prompt_b.md"
    file_b.write_text("Instruction B from markdown.", encoding="utf-8")

    original_config = kesoku.config._global_config
    try:
        # Load config and manually inject the relative paths for the test prompts
        config_path = tmp_path / "config.toml"
        init_config(str(config_path))
        cfg = load_config(str(config_path))
        cfg.agent.user_prompts = [
            "prompts/prompt_a.txt",
            "prompts/prompt_b.md",
        ]

        prompt = build_sys_prompt()

        # Verify they are present in the specified format
        assert "=== BEGIN prompt_a.txt ===" in prompt
        assert "Instruction A from file." in prompt
        assert "=== END prompt_a.txt ===" in prompt

        assert "=== BEGIN prompt_b.md ===" in prompt
        assert "Instruction B from markdown." in prompt
        assert "=== END prompt_b.md ===" in prompt
    finally:
        kesoku.config._global_config = original_config


def test_build_sys_prompt_with_session(tmp_path) -> None:
    """Verify build_sys_prompt includes session staging directory when session is provided."""
    import kesoku.config
    from kesoku.config import init_config, load_config
    from kesoku.db import Session

    config_path = tmp_path / "config.toml"
    init_config(str(config_path))

    original_config = kesoku.config._global_config
    try:
        cfg = load_config(str(config_path))
        sess = Session(id="sessionid", title="title", created_at=1779264000.0)

        prompt = build_sys_prompt(session=sess)

        # Check that Session Staging Directory and Pinned Session Notes instructions are included
        assert "# Session Staging Directory" in prompt
        assert "STAGING_DIR=" in prompt
        assert sess.workspace_name in prompt
        assert "# Pinned Session Notes" in prompt
        assert "update_file" in prompt
        assert "update_session_notes" not in prompt
    finally:
        kesoku.config._global_config = original_config


# ---------------------------------------------------------------------------
# Role resolution from session.role_name (thread sessions without a channel role)
# ---------------------------------------------------------------------------

ROLE_INTRO_MARKER = "ROLE_A_INTRO_MARKER_7f3a"


@pytest.fixture
def role_awd(tmp_path: Path) -> Iterator[KesokuConfig]:
    """Isolated global config whose agent working dir contains roles/role_a/intro.md."""
    (tmp_path / "roles" / "role_a").mkdir(parents=True)
    (tmp_path / "roles" / "role_a" / "intro.md").write_text(f"# Role A\n{ROLE_INTRO_MARKER}\n", encoding="utf-8")
    original = kesoku.config._global_config
    config_path = tmp_path / "config.toml"
    init_config(str(config_path))
    cfg = load_config(str(config_path))
    cfg.agent_working_dir = str(tmp_path)
    cfg.workspace.roles_dir = str(tmp_path / "roles")
    try:
        yield cfg
    finally:
        kesoku.config._global_config = original


def _fake_db(mapping: tuple[str, str] | None, channel_role: str = "default") -> MagicMock:
    """DatabaseManager stand-in with a controllable channel mapping and channel role."""
    db = MagicMock()
    db.get_channel_by_session.return_value = mapping
    db.get_channel_role_with_inheritance.return_value = channel_role
    return db


def test_prompt_session_role_name_without_channel_mapping(role_awd: KesokuConfig) -> None:
    """A session whose role_name is set but which has no channel mapping loads that role's intro."""
    sess = Session(id="thread01", title="t", role_name="role_a")
    with patch("kesoku.agent.prompt.DatabaseManager", return_value=_fake_db(None)):
        prompt = build_sys_prompt(session=sess)
    assert ROLE_INTRO_MARKER in prompt


def test_prompt_session_role_name_beats_default_channel_role(role_awd: KesokuConfig) -> None:
    """A mapped channel without a role row resolves to 'default'; session.role_name must still win."""
    sess = Session(id="thread02", title="t", role_name="role_a")
    with patch("kesoku.agent.prompt.DatabaseManager", return_value=_fake_db(("cb", "ch"), "default")):
        prompt = build_sys_prompt(session=sess)
    assert ROLE_INTRO_MARKER in prompt


def test_prompt_default_role_name_uses_channel_lookup(role_awd: KesokuConfig) -> None:
    """role_name='default' is treated as unset, so the channel role lookup still applies."""
    sess = Session(id="s1", title="t", role_name="default")
    db = _fake_db(("cb", "ch"), "role_a")
    with patch("kesoku.agent.prompt.DatabaseManager", return_value=db):
        prompt = build_sys_prompt(session=sess)
    assert ROLE_INTRO_MARKER in prompt
    db.get_channel_role_with_inheritance.assert_called()


def test_prompt_no_role_anywhere_has_no_persona(role_awd: KesokuConfig) -> None:
    """Without role_name and without a mapping no role intro leaks into the prompt."""
    for role_name in ("default", None):
        sess = Session(id="s2", title="t", role_name=role_name)
        with patch("kesoku.agent.prompt.DatabaseManager", return_value=_fake_db(None)):
            prompt = build_sys_prompt(session=sess)
        assert ROLE_INTRO_MARKER not in prompt


def test_prompt_explicit_role_argument_respected(role_awd: KesokuConfig) -> None:
    """An explicitly passed role is used as-is."""
    sess = Session(id="s3", title="t", role_name="default")
    with patch("kesoku.agent.prompt.DatabaseManager", return_value=_fake_db(None)):
        prompt = build_sys_prompt(session=sess, role="role_a")
    assert ROLE_INTRO_MARKER in prompt


def test_prompt_channel_lookup_failure_still_uses_role_name(role_awd: KesokuConfig) -> None:
    """A DB error during channel lookup never crashes prompt building."""
    db = MagicMock()
    db.get_channel_by_session.side_effect = RuntimeError("db locked")
    sess = Session(id="s4", title="t", role_name="role_a")
    with patch("kesoku.agent.prompt.DatabaseManager", return_value=db):
        prompt = build_sys_prompt(session=sess)
    assert "# Output Formatting Rules" in prompt
    assert ROLE_INTRO_MARKER in prompt


# ---------------------------------------------------------------------------
# Session notes are maintained with update_file (incremental edits), not a full-overwrite tool
# ---------------------------------------------------------------------------


def test_session_notes_instructions_use_update_file() -> None:
    """Notes are edited via update_file old_content/new_content; the full-overwrite tool is gone."""
    text = SESSION_NOTES_INSTRUCTIONS
    assert "update_file" in text
    assert "old_content" in text and "new_content" in text
    assert "update_session_notes" not in text
    assert "notes.md" in text
    assert '<context name="notes">' in text
    assert "<session_notes>" not in text


def test_session_notes_instructions_first_creation_uses_none() -> None:
    """old_content=None is reserved for creating notes.md when <context name=\"notes\"> is absent."""
    text = SESSION_NOTES_INSTRUCTIONS
    assert re.search(r"old_content\s*=\s*None", text)


def test_session_notes_instructions_forbid_redundant_shell_reads() -> None:
    """The agent copies old_content from <context name="notes"> instead of reading notes.md with shell tools."""
    text = SESSION_NOTES_INSTRUCTIONS
    assert re.search(r"(do not|don't|never)[^.\n]*(rg|cat|shell)", text, re.IGNORECASE), text
    tag = re.escape('<context name="notes">')
    assert re.search(rf"cop(y|ied)[^.\n]*{tag}|{tag}[^.\n]*cop(y|ied)", text, re.IGNORECASE)


def test_session_notes_instructions_partial_edits_only() -> None:
    """Only the outdated lines are replaced/deleted; the whole file is never rewritten."""
    text = SESSION_NOTES_INSTRUCTIONS.lower()
    assert "whole file" in text or "entire file" in text
    assert "replace" in text and "delete" in text


def test_build_sys_prompt_includes_notes_instructions() -> None:
    """A session system prompt carries the update_file-based notes instructions."""
    prompt = build_sys_prompt(session=Session(id="notes01", title="t"))
    assert SESSION_NOTES_INSTRUCTIONS.strip() in prompt
    assert "update_session_notes" not in prompt


def test_context_blocks_instructions_forbid_editing_context_dir() -> None:
    """Context blocks are tool-maintained: files under $STAGING_DIR/context/ must not be edited."""
    text = SESSION_NOTES_INSTRUCTIONS
    assert "$STAGING_DIR/context/" in text
    assert re.search(r"never edit files under `\$STAGING_DIR/context/`", text)
    assert "maintained by tools" in text
