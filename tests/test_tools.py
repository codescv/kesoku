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
        assert os.path.exists(filepath)
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
            assert "a" * 12000 in content
    finally:
        kesoku.config._global_config = original_config


@pytest.mark.asyncio
async def test_compact_history_tool(tmp_path) -> None:
    """Verify that compact_history tool creates a new session and transitions the channel cleanly."""
    from kesoku.agent.tools import compact_history
    from kesoku.config import KesokuConfig, WorkspaceConfig
    from kesoku.context import KesokuContext
    from kesoku.db import DatabaseManager, Message
    from kesoku.gateway.gateway import Gateway

    temp_db = str(tmp_path / "test_tools_compact.db")
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    # 1. Setup a session and post historical messages
    session = await gw.create_session("sess_compact_test", title="Compaction Test Session")
    msg1 = Message(
        session_id="sess_compact_test",
        chatbot_id="cli",
        channel_id="ch_compact",
        sender="u1",
        role="user",
        content="Original prompt",
        status="processed",
    )
    await gw.post(msg1)

    # 2. Set up ToolContext with injected gateway
    ctx = ToolContext(
        session_id="sess_compact_test",
        session_workspace=session.workspace_name,
        original_msg_id=msg1.id,
        gateway=gw,
    )

    # 3. Call the tool
    summary_text = (
        "### 1. Completed & Ongoing Goals\n- **Completed**: All tasks\n\n"
        "### 2. Critical States & Facts\n- **Key Facts**: None\n\n"
        "### 3. User Preferences\n- **Custom Rules**: None"
    )
    res = await compact_history(summary=summary_text, context=ctx)

    # 4. Verify transition flag was recorded in ToolContext
    assert ctx.transitioned_to_session is not None
    new_sess_id = ctx.transitioned_to_session

    # 5. Fetch the newly created session to verify
    new_session = await gw.get_session(new_sess_id)
    assert new_session is not None
    assert new_session.title.startswith("Compacted")

    # 6. Fetch new session history to verify compacted message and notification
    new_history = await gw.get_session_history(new_sess_id, limit=10)
    assert len(new_history) == 2

    # The message is the compacted starting message (USER role)
    compacted_start_msg = new_history[0]
    assert compacted_start_msg.role == "user"
    assert "[Conversation History Summary]" in compacted_start_msg.content
    assert "Original prompt" in compacted_start_msg.content
    assert compacted_start_msg.channel_id == "ch_compact"
    assert compacted_start_msg.status == "pending_agent"

    notify_msg = new_history[1]
    assert notify_msg.role == "assistant"
    assert "🔄 Conversation history has been automatically compacted" in notify_msg.content


@pytest.mark.asyncio
async def test_manual_compact_command(tmp_path) -> None:
    """Verify that manual chatbot /compact command triggers compaction signal correctly."""
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

    assert "Initiating history compaction" in reply_msg

    # Fetch messages to verify that the trigger signal message was posted with USER role
    history = await gw.get_session_history("sess_cmd_test", limit=10)
    trigger_msg = next(
        m for m in history if m.sender == "System" and "manually requested session compaction" in m.content
    )
    assert trigger_msg.role == "user"  # Crucial to wake up dispatcher
    assert trigger_msg.status == "pending_agent"


@pytest.mark.asyncio
async def test_compact_history_tool_completed_turn(tmp_path) -> None:
    """Verify that compact_history tool copies assistant reply and sets status to PROCESSED when turn is completed."""
    from kesoku.agent.tools import compact_history
    from kesoku.config import KesokuConfig, WorkspaceConfig
    from kesoku.context import KesokuContext
    from kesoku.db import DatabaseManager, Message
    from kesoku.gateway.gateway import Gateway

    temp_db = str(tmp_path / "test_tools_completed.db")
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    import kesoku.config

    original_config = kesoku.config._global_config
    try:
        kesoku.config._global_config = cfg

        # 1. Setup a session and post a completed turn
        session = await gw.create_session("sess_completed_test", title="Completed Compaction Session")
        msg1 = Message(
            session_id="sess_completed_test",
            chatbot_id="cli",
            channel_id="ch_completed",
            sender="u1",
            role="user",
            content="Original prompt",
            status="processed",
        )
        await gw.post(msg1)

        reply1 = Message(
            session_id="sess_completed_test",
            chatbot_id="cli",
            channel_id="ch_completed",
            sender="Kesoku",
            role="assistant",
            content="Original reply Y",
            status="responded",
        )
        await gw.post(reply1)

        # 2. Set up ToolContext
        ctx = ToolContext(
            session_id="sess_completed_test",
            session_workspace=session.workspace_name,
            original_msg_id=msg1.id,
            gateway=gw,
        )

        # 3. Call the tool
        summary_text = "### 1. Completed & Ongoing Goals\n- **Completed**: All"
        res = await compact_history(summary=summary_text, context=ctx)

        # 4. Verify transition flag
        assert ctx.transitioned_to_session is not None
        new_sess_id = ctx.transitioned_to_session

        # 5. Fetch new session history
        new_history = await gw.get_session_history(new_sess_id, limit=10)
        # Should have EXACTLY 3 messages: compacted user message,
        # system transition notification, and copied assistant reply!
        assert len(new_history) == 3

        # The first is compacted message with status PROCESSED!
        compacted_msg = new_history[0]
        assert compacted_msg.role == "user"
        assert compacted_msg.status == "processed"  # Extremely important!
        assert "Original prompt" in compacted_msg.content

        # The second is the system transition notification
        notify_msg = new_history[1]
        assert notify_msg.role == "assistant"
        assert "🔄 Conversation history has been automatically compacted" in notify_msg.content

        # The third is the copied assistant reply with status DELIVERED!
        copied_reply = new_history[2]
        assert copied_reply.role == "assistant"
        assert copied_reply.status == "delivered"
        assert copied_reply.content == "Original reply Y"
    finally:
        kesoku.config._global_config = original_config


@pytest.mark.asyncio
async def test_compact_history_with_cronjob(tmp_path) -> None:
    """Verify that compact_history tool correctly processes and retains a cronjob message."""
    from kesoku.agent.tools import compact_history
    from kesoku.config import KesokuConfig, WorkspaceConfig
    from kesoku.context import KesokuContext
    from kesoku.db import DatabaseManager, Message
    from kesoku.gateway.gateway import Gateway

    temp_db = str(tmp_path / "test_tools_cronjob.db")
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    import kesoku.config

    original_config = kesoku.config._global_config
    try:
        kesoku.config._global_config = cfg

        # 1. Setup a session and post a cronjob message
        session = await gw.create_session("sess_cronjob_test", title="Cronjob Compaction Session")
        cron_msg = Message(
            session_id="sess_cronjob_test",
            chatbot_id="cli",
            channel_id="ch_cronjob",
            sender="Cronjob",
            role="user",
            content="Hello from scheduled cron job prompt!",
            status="processed",
            metadata={"is_cronjob": True},
        )
        await gw.post(cron_msg)

        # 2. Set up ToolContext
        ctx = ToolContext(
            session_id="sess_cronjob_test",
            session_workspace=session.workspace_name,
            original_msg_id=cron_msg.id,
            gateway=gw,
        )

        # 3. Call the tool
        summary_text = "### 1. Completed & Ongoing Goals\n- **Completed**: Scheduled tasks done"
        res = await compact_history(summary=summary_text, context=ctx)

        # 4. Verify transition flag
        assert ctx.transitioned_to_session is not None
        new_sess_id = ctx.transitioned_to_session

        # 5. Fetch new session history
        new_history = await gw.get_session_history(new_sess_id, limit=10)
        assert len(new_history) == 2

        # The message is the compacted starting message (USER role) with sender "Cronjob"
        compacted_start_msg = new_history[0]
        assert compacted_start_msg.role == "user"
        assert compacted_start_msg.sender == "Cronjob"
        assert "[Conversation History Summary]" in compacted_start_msg.content
        assert "Hello from scheduled cron job prompt!" in compacted_start_msg.content
        assert compacted_start_msg.metadata.get("is_cronjob") is True

        # The second is the system transition notification
        notify_msg = new_history[1]
        assert notify_msg.role == "assistant"
        assert "🔄 Conversation history has been automatically compacted" in notify_msg.content
    finally:
        kesoku.config._global_config = original_config
