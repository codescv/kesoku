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
async def test_lcm_grep_role_isolation(tmp_path) -> None:
    """Verify that lcm_grep limits results to the current active persona role during cross-session searches."""
    import json

    from kesoku.agent.tools import ToolContext, lcm_grep
    from kesoku.config import KesokuConfig, WorkspaceConfig
    from kesoku.constants import MessageRole, MessageType
    from kesoku.context import KesokuContext
    from kesoku.db import DatabaseManager, Message
    from kesoku.gateway.gateway import Gateway

    # 1. Setup DBs
    temp_db_dir = tmp_path / "dbs"
    temp_db_dir.mkdir()
    kesoku_db_path = str(temp_db_dir / "kesoku.db")
    db_mgr = DatabaseManager(kesoku_db_path)
    db_mgr.init_tables()

    # 2. Setup Context & Gateway
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=kesoku_db_path))
    kesoku_ctx = KesokuContext(config=cfg, db=db_mgr)
    gw = Gateway(context=kesoku_ctx)

    # 3. Bind Roles to Channels in Kesoku DB
    # Create sessions first to satisfy FOREIGN KEY constraints
    import time

    from kesoku.db import DatabaseManager, Session

    sess_asuka_obj = Session(
        id="sess_asuka",
        title="Asuka Session",
        created_at=time.time(),
        updated_at=time.time(),
        system_prompt="System Asuka",
    )
    db_mgr.create_session(sess_asuka_obj)

    sess_tifa_obj = Session(
        id="sess_tifa",
        title="Tifa Session",
        created_at=time.time(),
        updated_at=time.time(),
        system_prompt="System Tifa",
    )
    db_mgr.create_session(sess_tifa_obj)

    # asuka
    db_mgr.set_channel_role(chatbot_id="cli", channel_id="chan_asuka", role="asuka")
    db_mgr.set_active_session_for_channel(chatbot_id="cli", channel_id="chan_asuka", session_id="sess_asuka")
    # tifa
    db_mgr.set_channel_role(chatbot_id="cli", channel_id="chan_tifa", role="tifa")
    db_mgr.set_active_session_for_channel(chatbot_id="cli", channel_id="chan_tifa", session_id="sess_tifa")

    # 4. Insert dummy messages in Kesoku DB to link original_msg_id
    msg_asuka = Message(
        id="msg_asuka_original",
        session_id="sess_asuka",
        chatbot_id="cli",
        channel_id="chan_asuka",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Trigger",
        status="processed"
    )
    db_mgr.save_message(msg_asuka)

    msg_tifa = Message(
        id="msg_tifa_original",
        session_id="sess_tifa",
        chatbot_id="cli",
        channel_id="chan_tifa",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Trigger",
        status="processed"
    )
    db_mgr.save_message(msg_tifa)

    # 5. Ingest messages into OpenLCM (lcm.db)
    # Asuka memory
    kesoku_ctx.lcm_engine.bind_session("sess_asuka")
    kesoku_ctx.lcm_engine._ingest_messages([
        {"role": "user", "content": "This is Asuka's secret password."}
    ])
    # Tifa memory
    kesoku_ctx.lcm_engine.bind_session("sess_tifa")
    kesoku_ctx.lcm_engine._ingest_messages([
        {"role": "user", "content": "This is Tifa's secret password."}
    ])

    # 6. Test search as Asuka
    ctx_asuka = ToolContext(
        session_id="sess_asuka",
        session_workspace="ws_asuka",
        gateway=gw,
        original_msg_id="msg_asuka_original"
    )

    # We must search with session_scope="all"
    res_asuka_str = await lcm_grep(query="secret", session_scope="all", context=ctx_asuka)
    res_asuka = json.loads(res_asuka_str)

    # Assertions for Asuka search
    assert "results" in res_asuka
    # Should only return Asuka's memory
    assert len(res_asuka["results"]) == 1
    assert res_asuka["results"][0]["session_id"] == "sess_asuka"
    assert "Asuka" in res_asuka["results"][0]["snippet"]
    assert "Tifa" not in res_asuka["results"][0]["snippet"]

    # 7. Test search as Tifa
    ctx_tifa = ToolContext(
        session_id="sess_tifa",
        session_workspace="ws_tifa",
        gateway=gw,
        original_msg_id="msg_tifa_original"
    )

    res_tifa_str = await lcm_grep(query="secret", session_scope="all", context=ctx_tifa)
    res_tifa = json.loads(res_tifa_str)

    # Assertions for Tifa search
    assert "results" in res_tifa
    # Should only return Tifa's memory
    assert len(res_tifa["results"]) == 1
    assert res_tifa["results"][0]["session_id"] == "sess_tifa"
    assert "Tifa" in res_tifa["results"][0]["snippet"]
    assert "Asuka" not in res_tifa["results"][0]["snippet"]

    # 8. Test explicit session_id check (Asuka trying to access Tifa's session)
    res_hack_str = await lcm_grep(query="secret", session_scope="session", session_id="sess_tifa", context=ctx_asuka)
    res_hack = json.loads(res_hack_str)
    assert res_hack["total_results"] == 0
    assert "error" in res_hack
    assert "does not belong to role" in res_hack["error"]




