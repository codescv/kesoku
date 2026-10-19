import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kesoku.agent.tools.shell import ActiveJobsRegistry, manage_job, monitor_background_job


@pytest.fixture
def mock_registry():
    return ActiveJobsRegistry()

@pytest.fixture
def tool_context(mock_registry):
    ctx = MagicMock()
    ctx.session_id = "test_sess"
    ctx.active_jobs = mock_registry
    return ctx

@pytest.mark.asyncio
async def test_manage_job_list(tool_context, mock_registry):
    proc = MagicMock()
    proc.pid = 100
    await mock_registry.register(
        job_id="job1", session_id="test_sess", process=proc,
        log_filepath="out.log", command="sleep 10", stderr_filepath="err.log"
    )
    res = await manage_job(action="list", context=tool_context)
    assert "job1" in res
    assert "sleep 10" in res

@pytest.mark.asyncio
async def test_manage_job_status_unknown(tool_context):
    res = await manage_job(action="status", job_id="unknown", context=tool_context)
    assert "unknown or expired" in res

@pytest.mark.asyncio
async def test_manage_job_status_running(tool_context, mock_registry, tmp_path):
    out_path = tmp_path / "out.log"
    out_path.write_text("hello world\n")
    proc = MagicMock()
    proc.pid = 101
    await mock_registry.register(
        job_id="job2", session_id="test_sess", process=proc,
        log_filepath=str(out_path), command="echo hello", stderr_filepath="/dev/null"
    )
    res = await manage_job(action="status", job_id="job2", context=tool_context)
    assert "RUNNING" in res
    assert "hello world" in res

@pytest.mark.asyncio
@patch("kesoku.agent.tools.shell.os.killpg")
async def test_manage_job_kill(mock_killpg, tool_context, mock_registry):
    proc = MagicMock()
    proc.pid = 102
    future = asyncio.Future()
    future.set_result(None)
    proc.wait.return_value = future
    proc.returncode = -15
    await mock_registry.register(
        job_id="job3", session_id="test_sess", process=proc,
        log_filepath="out.log", command="sleep 10", stderr_filepath="err.log"
    )

    res = await manage_job(action="kill", job_id="job3", context=tool_context)
    assert "Killed job job3" in res
    assert mock_killpg.called
    assert mock_registry.was_killed_by_agent("job3", "test_sess")

@pytest.mark.asyncio
@patch("kesoku.agent.tools.shell.os.killpg")
async def test_monitor_background_job_suppression(mock_killpg, tool_context, mock_registry):
    proc = AsyncMock()
    proc.pid = 99999
    proc.returncode = -15
    proc.wait.return_value = -15

    await mock_registry.register(
        job_id="job4", session_id="test_sess", process=proc,
        log_filepath="out.log", command="sleep", stderr_filepath="err.log"
    )

    # Simulate being killed by agent
    await mock_registry.kill_job("job4", "test_sess")

    gateway = AsyncMock()
    tool_context.gateway = gateway

    # signature: job_id, proc, log_filepath_stdout, log_filepath_stderr, context,
    # original_msg_id, stdout_task, stderr_task
    task1 = asyncio.Future()
    task1.set_result(None)
    task2 = asyncio.Future()
    task2.set_result(None)

    await monitor_background_job(
        "job4", proc, "out.log", "err.log", tool_context, "msg_123", task1, task2
    )

    gateway.post.assert_not_called()
