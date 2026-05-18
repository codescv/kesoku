"""Unit tests for Kesoku agent tools and skill registry."""

from unittest.mock import MagicMock
import pytest

from kesoku.agent.tools import ToolContext, WebSearchTool, web_search, run_shell_command


def test_web_search_tool_success() -> None:
    """Test WebSearchTool successfully executes search and formats grounding sources."""
    mock_client = MagicMock()
    mock_res = MagicMock()
    mock_res.text = "The weather in Tokyo is sunny."

    mock_candidate = MagicMock()
    mock_chunk1 = MagicMock()
    mock_chunk1.web.uri = "https://weather.com/tokyo"
    mock_chunk1.web.title = "Tokyo Weather"

    # Duplicate URI to test deduplication logic
    mock_chunk2 = MagicMock()
    mock_chunk2.web.uri = "https://weather.com/tokyo"
    mock_chunk2.web.title = "Tokyo Weather Update"

    mock_candidate.grounding_metadata.grounding_chunks = [mock_chunk1, mock_chunk2]
    mock_res.candidates = [mock_candidate]

    mock_client.models.generate_content.return_value = mock_res

    # Explicitly inject mock client
    tool = WebSearchTool(client=mock_client)
    ctx = ToolContext(session_id="s1", session_workspace="workspace1")
    result = tool.web_search("What is the weather in Tokyo?", ctx)

    assert "The weather in Tokyo is sunny." in result
    assert "Sources:" in result
    assert "- Tokyo Weather: https://weather.com/tokyo" in result
    # Verify that duplicate sources are successfully deduplicated
    assert result.count("https://weather.com/tokyo") == 1


def test_web_search_tool_api_failure() -> None:
    """Test WebSearchTool handles API call exceptions gracefully."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API Quota Exceeded")

    tool = WebSearchTool(client=mock_client)
    result = tool.web_search("Test query", None)

    assert "Web search failed: API Quota Exceeded" in result


def test_run_shell_command_default_cwd(tmp_path) -> None:
    """Test that run_shell_command defaults to executing inside the AWD."""
    import kesoku.config
    from kesoku.config import load_config

    original_config = kesoku.config._global_config
    try:
        config_path = tmp_path / "config.toml"
        cfg = load_config(str(config_path))
        # Set AWD
        cfg.agent_working_dir = str(tmp_path)

        ctx = ToolContext(session_id="test_sess", session_workspace="test_workspace")

        # Execute pwd command
        res = run_shell_command("pwd", context=ctx)

        assert "=== STDOUT ===" in res
        # Output of pwd should match the AWD (tmp_path)
        assert str(tmp_path) in res
    finally:
        kesoku.config._global_config = original_config


def test_run_shell_command_custom_cwd(tmp_path) -> None:
    """Test that run_shell_command executes in custom cwd if supplied."""
    import kesoku.config
    from kesoku.config import load_config

    original_config = kesoku.config._global_config
    try:
        config_path = tmp_path / "config.toml"
        cfg = load_config(str(config_path))
        cfg.agent_working_dir = str(tmp_path)

        # Create a subfolder to run command in
        subfolder = tmp_path / "subfolder"
        subfolder.mkdir()

        ctx = ToolContext(session_id="test_sess", session_workspace="test_workspace")

        # Execute pwd command inside custom cwd (relative)
        res = run_shell_command("pwd", cwd="subfolder", context=ctx)
        assert "=== STDOUT ===" in res
        assert str(subfolder) in res

        # Execute pwd command inside custom cwd (absolute)
        another_folder = tmp_path / "another"
        another_folder.mkdir()
        res_abs = run_shell_command("pwd", cwd=str(another_folder), context=ctx)
        assert str(another_folder) in res_abs
    finally:
        kesoku.config._global_config = original_config
