"""Unit tests for Kesoku agent tools and skill registry."""

from unittest.mock import MagicMock

import pytest

from kesoku.agent.tools import ShellCommandError, ToolContext, WebSearchTool, run_shell_command


@pytest.mark.asyncio
async def test_web_search_tool_success() -> None:
    """Test WebSearchTool successfully executes search and formats grounding sources."""
    from unittest.mock import AsyncMock

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

    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_res)

    # Explicitly inject mock client
    tool = WebSearchTool(client=mock_client)
    ctx = ToolContext(session_id="s1", session_workspace="workspace1")
    result = await tool.web_search("What is the weather in Tokyo?", ctx)

    assert "The weather in Tokyo is sunny." in result
    assert "Sources:" in result
    assert "- Tokyo Weather: https://weather.com/tokyo" in result
    # Verify that duplicate sources are successfully deduplicated
    assert result.count("https://weather.com/tokyo") == 1


@pytest.mark.asyncio
async def test_web_search_tool_api_failure() -> None:
    """Test WebSearchTool handles API call exceptions gracefully."""
    from unittest.mock import AsyncMock

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=Exception("API Quota Exceeded"))

    tool = WebSearchTool(client=mock_client)
    result = await tool.web_search("Test query", None)

    assert "Web search failed: API Quota Exceeded" in result


@pytest.mark.asyncio
async def test_run_shell_command_default_cwd(tmp_path) -> None:
    """Test that run_shell_command defaults to executing inside the AWD."""
    import kesoku.config
    from kesoku.config import init_config, load_config

    original_config = kesoku.config._global_config
    try:
        config_path = tmp_path / "config.toml"
        init_config(str(config_path))
        cfg = load_config(str(config_path))
        # Set AWD
        cfg.agent_working_dir = str(tmp_path)

        ctx = ToolContext(session_id="test_sess", session_workspace="test_workspace")

        # Execute pwd command
        res = await run_shell_command("pwd", context=ctx)

        assert "=== STDOUT ===" in res
        # Output of pwd should match the AWD (tmp_path)
        assert str(tmp_path) in res
    finally:
        kesoku.config._global_config = original_config


@pytest.mark.asyncio
async def test_run_shell_command_custom_cwd(tmp_path) -> None:
    """Test that run_shell_command executes in custom cwd if supplied."""
    import kesoku.config
    from kesoku.config import init_config, load_config

    original_config = kesoku.config._global_config
    try:
        config_path = tmp_path / "config.toml"
        init_config(str(config_path))
        cfg = load_config(str(config_path))
        cfg.agent_working_dir = str(tmp_path)

        # Create a subfolder to run command in
        subfolder = tmp_path / "subfolder"
        subfolder.mkdir()

        ctx = ToolContext(session_id="test_sess", session_workspace="test_workspace")

        # Execute pwd command inside custom cwd (relative)
        res = await run_shell_command("pwd", cwd="subfolder", context=ctx)
        assert "=== STDOUT ===" in res
        assert str(subfolder) in res

        # Execute pwd command inside custom cwd (absolute)
        another_folder = tmp_path / "another"
        another_folder.mkdir()
        res_abs = await run_shell_command("pwd", cwd=str(another_folder), context=ctx)
        assert str(another_folder) in res_abs
    finally:
        kesoku.config._global_config = original_config


@pytest.mark.asyncio
async def test_run_shell_command_env_variables(tmp_path) -> None:
    """Test that run_shell_command injects AWD and STAGING_DIR environment variables."""
    import os

    import kesoku.config
    from kesoku.config import init_config, load_config

    original_config = kesoku.config._global_config
    try:
        config_path = tmp_path / "config.toml"
        init_config(str(config_path))
        cfg = load_config(str(config_path))
        cfg.agent_working_dir = str(tmp_path)

        ctx = ToolContext(session_id="test_sess", session_workspace="test_ws")

        # Execute command to print env variables
        res = await run_shell_command("echo $AWD && echo $STAGING_DIR", context=ctx)

        assert "=== STDOUT ===" in res
        assert str(tmp_path) in res
        expected_staging = os.path.realpath(os.path.join(cfg.workspace.sessions_dir, "test_ws"))  # noqa: ASYNC240
        assert expected_staging in res
    finally:
        kesoku.config._global_config = original_config


@pytest.mark.asyncio
async def test_run_shell_command_failure(tmp_path) -> None:
    """Test that run_shell_command raises ShellCommandError on non-zero exit code."""
    import kesoku.config
    from kesoku.config import init_config, load_config

    original_config = kesoku.config._global_config
    try:
        config_path = tmp_path / "config.toml"
        init_config(str(config_path))
        cfg = load_config(str(config_path))
        cfg.agent_working_dir = str(tmp_path)

        ctx = ToolContext(session_id="test_sess", session_workspace="test_ws")

        with pytest.raises(ShellCommandError) as exc_info:
            await run_shell_command("echo 'error occurred' >&2 && exit 5", context=ctx)

        assert "Command failed with exit code 5" in str(exc_info.value)
        assert "error occurred" in str(exc_info.value)
    finally:
        kesoku.config._global_config = original_config


@pytest.mark.asyncio
async def test_run_shell_command_timeout(tmp_path) -> None:
    """Test that run_shell_command raises ShellCommandError on timeout."""
    from unittest.mock import patch

    import kesoku.config
    from kesoku.config import init_config, load_config

    original_config = kesoku.config._global_config
    try:
        config_path = tmp_path / "config.toml"
        init_config(str(config_path))
        cfg = load_config(str(config_path))
        cfg.agent_working_dir = str(tmp_path)

        ctx = ToolContext(session_id="test_sess", session_workspace="test_ws")

        with patch("kesoku.agent.tools.TIMEOUT_SECONDS", 0.1):
            with pytest.raises(ShellCommandError) as exc_info:
                await run_shell_command("sleep 10", context=ctx)

            assert "Command timed out after" in str(exc_info.value)
    finally:
        kesoku.config._global_config = original_config


@pytest.mark.asyncio
async def test_run_shell_command_start_failure(tmp_path) -> None:
    """Test that run_shell_command raises ShellCommandError on starting non-existent command."""
    import kesoku.config
    from kesoku.config import init_config, load_config

    original_config = kesoku.config._global_config
    try:
        config_path = tmp_path / "config.toml"
        init_config(str(config_path))
        cfg = load_config(str(config_path))
        cfg.agent_working_dir = str(tmp_path)
        # Disable shell mode to allow shlex splitting and triggering FileNotFoundError for non-existent executable
        cfg.shell.use_shell = False

        ctx = ToolContext(session_id="test_sess", session_workspace="test_ws")

        with pytest.raises(ShellCommandError) as exc_info:
            await run_shell_command("non_existent_command_xyz", context=ctx)

        assert "Error executing command" in str(exc_info.value)
    finally:
        kesoku.config._global_config = original_config


@pytest.mark.asyncio
async def test_run_shell_command_failure_truncation(tmp_path) -> None:
    """Test that run_shell_command truncates extremely long failed outputs and saves to a file."""
    import os
    import re

    import kesoku.config
    from kesoku.config import init_config, load_config

    original_config = kesoku.config._global_config
    try:
        config_path = tmp_path / "config.toml"
        init_config(str(config_path))
        cfg = load_config(str(config_path))
        cfg.agent_working_dir = str(tmp_path)

        ctx = ToolContext(session_id="test_sess", session_workspace="test_ws")

        # Generate extremely long output (more than 10000 characters)
        long_cmd = "python3 -c 'print(\"a\" * 12000); exit(1)'"
        with pytest.raises(ShellCommandError) as exc_info:
            await run_shell_command(long_cmd, context=ctx)

        err_msg = str(exc_info.value)
        assert "Command failed with exit code 1" in err_msg
        assert "Output truncated" in err_msg
        assert "Full output saved to session workspace file:" in err_msg

        # Extract output file path from message to verify it was written
        match = re.search(r"Full output saved to session workspace file: `([^`]+)`", err_msg)
        assert match is not None
        filepath = match.group(1)
        assert os.path.exists(filepath)  # noqa: ASYNC240
        with open(filepath, encoding="utf-8") as f:  # noqa: ASYNC230
            content = f.read()
            assert "a" * 12000 in content
    finally:
        kesoku.config._global_config = original_config

