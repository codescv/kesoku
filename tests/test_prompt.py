"""Unit tests for the modular system prompt construction utility."""

from kesoku.agent.prompt import build_sys_prompt, PREAMBLE


def test_build_sys_prompt_default() -> None:
    """Verify build_sys_prompt includes default system prompt and file-sending instructions."""
    prompt = build_sys_prompt()
    
    # Check default system prompt is included
    assert PREAMBLE.strip() in prompt
    
    # Check file instructions header and syntax are included
    assert "# Sending Files to the User" in prompt
    assert "[file: /abs/path/to/file]" in prompt
    assert "Rules for file sending:" in prompt


def test_build_sys_prompt_with_custom_context() -> None:
    """Verify build_sys_prompt appends custom context instructions correctly."""
    custom_context = "You are inside a specialized testing environment."
    prompt = build_sys_prompt(custom_prompt=custom_context)
    
    assert PREAMBLE.strip() in prompt
    assert "# Sending Files to the User" in prompt
    assert custom_context in prompt


def test_build_sys_prompt_with_working_directory() -> None:
    """Verify build_sys_prompt includes agent working directory when config is loaded."""
    import kesoku.config
    from kesoku.config import load_config

    original_config = kesoku.config._global_config
    try:
        # Load a dummy configuration path
        cfg = load_config("test_workspace/config.toml")

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
    from kesoku.config import load_config
    
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
        cfg = load_config(str(tmp_path / "config.toml"))
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


def test_build_sys_prompt_with_session() -> None:
    """Verify build_sys_prompt includes session staging directory when session is provided."""
    import kesoku.config
    from kesoku.config import load_config
    from kesoku.db import Session

    original_config = kesoku.config._global_config
    try:
        cfg = load_config("test_workspace/config.toml")
        sess = Session(id="sessionid", title="title", created_at=1779264000.0)

        prompt = build_sys_prompt(session=sess)

        # Check that Session Staging Directory instruction is included
        assert "# Session Staging Directory" in prompt
        assert "STAGING_DIR=" in prompt
        assert sess.workspace_name in prompt
    finally:
        kesoku.config._global_config = original_config
