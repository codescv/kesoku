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
        expected_staging = os.path.realpath(os.path.join(cfg.workspace.sessions_dir, "test_ws"))
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
async def test_run_shell_command_background_transition(tmp_path) -> None:
    """Test that run_shell_command gracefully transitions to background execution on timeout."""
    import kesoku.config
    from kesoku.agent.tools import ActiveJobsRegistry, run_shell_command
    from kesoku.config import init_config, load_config

    original_config = kesoku.config._global_config
    try:
        config_path = tmp_path / "config.toml"
        init_config(str(config_path))
        cfg = load_config(str(config_path))
        cfg.agent_working_dir = str(tmp_path)
        # Set low threshold to trigger background transition
        cfg.shell.background_threshold_seconds = 0.1

        active_jobs = ActiveJobsRegistry()
        ctx = ToolContext(
            session_id="test_sess",
            session_workspace="test_ws",
            active_jobs=active_jobs,
        )

        res = await run_shell_command("sleep 5 && echo 'done'", context=ctx)

        assert "transitioned to background execution" in res
        assert "Background Job ID: `job_" in res
        assert "Stdout path:" in res
        assert "Stderr path:" in res

        # Avoid process leak by stopping running jobs
        await active_jobs.stop_all_for_session("test_sess")
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

        # Generate extremely long output (more than 30000 characters)
        long_cmd = "python3 -c 'print(\"a\" * 40000); exit(1)'"
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
        assert os.path.exists(filepath)
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
            assert "a" * 40000 in content
    finally:
        kesoku.config._global_config = original_config


@pytest.mark.asyncio
async def test_run_shell_command_max_output_chars(tmp_path) -> None:
    """Test that run_shell_command truncates output to max_output_chars and appends help message."""
    import kesoku.config
    from kesoku.config import init_config, load_config

    original_config = kesoku.config._global_config
    try:
        config_path = tmp_path / "config.toml"
        init_config(str(config_path))
        cfg = load_config(str(config_path))
        cfg.agent_working_dir = str(tmp_path)

        ctx = ToolContext(session_id="test_sess", session_workspace="test_ws")

        # Test default (max_output_chars = 1000)
        cmd_1200 = "python3 -c 'print(\"a\" * 1200)'"
        res = await run_shell_command(cmd_1200, context=ctx)

        assert "Output truncated" in res
        assert "Preview (first 1000 chars):" in res
        assert "[Output truncated. If you need to view more output, you can set the 'max_output_chars'" in res

        # Test custom max_output_chars = 100
        res_custom = await run_shell_command(cmd_1200, max_output_chars=100, context=ctx)
        assert "Output truncated" in res_custom
        assert "Preview (first 100 chars):" in res_custom

        # Test no truncation if content is shorter than max_output_chars
        cmd_50 = "python3 -c 'print(\"a\" * 50)'"
        res_short = await run_shell_command(cmd_50, max_output_chars=100, context=ctx)
        assert "Output truncated" not in res_short
        assert "a" * 50 in res_short

    finally:
        kesoku.config._global_config = original_config


@pytest.mark.asyncio
async def test_manual_compact_command(tmp_path) -> None:
    """Verify that manual chatbot /compact command triggers compaction eligibility check."""
    from kesoku.config import KesokuConfig, WorkspaceConfig
    from kesoku.context import KesokuContext
    from kesoku.db import DatabaseManager, Message
    from kesoku.gateway.chatbot.base import Chatbot
    from kesoku.gateway.gateway import Gateway

    temp_db = str(tmp_path / "test_tools_cmd.db")
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    session = await gw.create_session(
        "sess_cmd_test",
        title="Command Session",
        chatbot_id="cli",
        channel_id="ch_cmd",
    )
    # Ingest user message to link session to channel
    await gw.post(
        Message(
            session_id="sess_cmd_test",
            chatbot_id="cli",
            channel_id="ch_cmd",
            sender="u1",
            role="user",
            content="Hello!",
            status="processed",
        )
    )

    # Mock a chatbot instance
    class DummyChatbot(Chatbot):
        async def handle_message(self, message: Message) -> None:
            pass

    bot = DummyChatbot("cli", gw)

    # Execute the manual compact command
    reply_msg = None

    async def mock_reply(content: str) -> None:
        nonlocal reply_msg
        reply_msg = content

    await bot.commands.execute("compact", mock_reply, "ch_cmd")

    assert "Context compaction is not needed right now" in reply_msg


@pytest.mark.asyncio
async def test_analyze_media_tool_success(tmp_path) -> None:
    """Test analyze_media tool successfully reads a file and invokes LLM analysis."""
    from unittest.mock import AsyncMock, patch

    from kesoku.agent.llm import LLMResponse
    from kesoku.agent.tools.media import analyze_media

    media_file = tmp_path / "test_image.png"
    media_file.write_bytes(b"\x89PNG\r\n\x1a\n")

    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value=LLMResponse(content="An image of a cute cat."))

    mock_gateway = MagicMock()
    mock_gateway.context.get_llm.return_value = mock_llm

    ctx = ToolContext(
        session_id="sess_media",
        session_workspace="ws_media",
        gateway=mock_gateway,
    )

    with patch("kesoku.agent.tools.media.PathResolver.resolve", return_value=str(media_file)):
        res = await analyze_media(path="test_image.png", prompt="What is this image?", context=ctx)

    assert res == "An image of a cute cat."
    mock_llm.generate.assert_called_once()
    call_args = mock_llm.generate.call_args
    history_passed = call_args.kwargs["history"]
    assert len(history_passed) == 1
    assert history_passed[0].metadata["attachments"][0]["path"] == str(media_file)
    assert history_passed[0].metadata["attachments"][0]["mime_type"] == "image/png"


@pytest.mark.asyncio
async def test_analyze_media_tool_not_found() -> None:
    """Test analyze_media returns clear error when media file does not exist."""
    from kesoku.agent.tools.media import analyze_media

    ctx = ToolContext(session_id="s1", session_workspace="ws1")
    res = await analyze_media(path="non_existent_video.mp4", context=ctx)
    assert "does not exist" in res


@pytest.mark.asyncio
async def test_grep_slash_command(tmp_path) -> None:
    """Verify that chatbot /grep command triggers memory_search tool under the hood."""
    import time

    from kesoku.config import KesokuConfig, WorkspaceConfig
    from kesoku.constants import MessageRole, MessageStatus, MessageType
    from kesoku.context import KesokuContext
    from kesoku.db import DatabaseManager, Message, Session
    from kesoku.gateway.chatbot.base import Chatbot
    from kesoku.gateway.gateway import Gateway

    temp_db = str(tmp_path / "test_slash_grep.db")
    db_mgr = DatabaseManager(temp_db)
    db_mgr.init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    ctx = KesokuContext(config=cfg, db=db_mgr)
    gw = Gateway(context=ctx)

    sess = Session(
        id="sess_slash",
        title="Slash Session",
        created_at=time.time(),
        updated_at=time.time(),
    )
    db_mgr.create_session(sess)
    db_mgr.set_channel_role("cli", "ch_slash", "default")
    db_mgr.set_active_session_for_channel("cli", "ch_slash", "sess_slash")

    # Save a message that matches
    msg = Message(
        id="m1",
        session_id="sess_slash",
        chatbot_id="cli",
        channel_id="ch_slash",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Testing grep keyword match.",
        timestamp=time.time(),
        status=MessageStatus.PROCESSED,
    )
    db_mgr.save_message(msg)

    class DummyChatbot(Chatbot):
        async def handle_message(self, message: Message) -> None:
            pass

    bot = DummyChatbot("cli", gw)
    reply_msg = None

    async def mock_reply(content: str) -> None:
        nonlocal reply_msg
        reply_msg = content

    await bot.execute_command_from_text("/grep keyword", mock_reply, channel_id="ch_slash")

    assert reply_msg is not None
    assert "Search Results for 'keyword'" in reply_msg
    assert "Testing grep keyword match." in reply_msg
