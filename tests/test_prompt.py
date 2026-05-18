"""Unit tests for the modular system prompt construction utility."""

from kesoku.agent.prompt import build_sys_prompt, DEFAULT_SYSTEM_PROMPT


def test_build_sys_prompt_default() -> None:
    """Verify build_sys_prompt includes default system prompt and file-sending instructions."""
    prompt = build_sys_prompt()
    
    # Check default system prompt is included
    assert DEFAULT_SYSTEM_PROMPT.strip() in prompt
    
    # Check file instructions header and syntax are included
    assert "# Sending Files to the User" in prompt
    assert "[file: /abs/path/to/file]" in prompt
    assert "Rules for file sending:" in prompt


def test_build_sys_prompt_with_custom_context() -> None:
    """Verify build_sys_prompt appends custom context instructions correctly."""
    custom_context = "You are inside a specialized testing environment."
    prompt = build_sys_prompt(custom_prompt=custom_context)
    
    assert DEFAULT_SYSTEM_PROMPT.strip() in prompt
    assert "# Sending Files to the User" in prompt
    assert custom_context in prompt
