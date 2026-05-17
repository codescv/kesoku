"""Unit tests for Kesoku Agent, LLM mocking, and Tools."""

import asyncio
from typing import Any

import pytest

from unittest.mock import MagicMock, patch

from kesoku.agent.agent import Agent
from kesoku.agent.llm import GeminiLLM, MockLLM, get_llm
from kesoku.agent.tools import ToolContext, ToolRegistry, calculator, run_shell_command
from kesoku.config import KesokuConfig, WorkspaceConfig
from kesoku.db import DatabaseManager, Message
from kesoku.gateway.gateway import Gateway


def test_tool_registry() -> None:
    """Test tool registration and lookup."""
    reg = ToolRegistry()

    @reg.register
    def add_nums(x: int, y: int) -> int:
        return x + y

    assert len(reg.get_tools_list()) == 1
    func = reg.get_tool("add_nums")
    assert func(5, 10) == 15

    with pytest.raises(KeyError):
        reg.get_tool("non_existent")


def test_calculator_tool() -> None:
    """Test mathematical calculator tool."""
    res = calculator("50 * 2 + 15")
    assert "115.0" in res

    # Test error handling on illegal expressions
    err = calculator("import os; os.system('echo hello')")
    assert "Error" in err


@pytest.fixture
def temp_db(tmp_path: Any) -> str:
    return str(tmp_path / "test_agent.db")


@pytest.mark.asyncio
async def test_agent_execution_loop(temp_db: str) -> None:
    """Test agent processing a message using MockLLM and tool calling."""
    DatabaseManager(temp_db).init_tables()
    gw = Gateway(workspace_config=WorkspaceConfig(db_path=temp_db))
    reg = ToolRegistry()
    reg.register(calculator)

    # Ingest a math question
    await gw.create_session("sess1", title="Math Session")
    await gw.post(
        Message(
            session_id="sess1",
            chatbot_id="cli",
            channel_id="ch1",
            sender="u1",
            role="user",
            type="text",
            content="Please calculate 25 + 10",
            status="pending_agent",
        )
    )

    llm = MockLLM()
    agent = Agent(gw, llm, reg)

    # Start agent loop in background
    agent_task = asyncio.create_task(agent.start())

    # Let it process for a moment
    await asyncio.sleep(0.5)
    agent.stop()
    await asyncio.gather(agent_task, return_exceptions=True)

    # Verify message status was marked as processed
    history = await gw.get_session_history("sess1")
    assert len(history) >= 1
    assert any(m.status == "processed" for m in history)


def test_get_llm() -> None:
    """Test get_llm factory function."""
    with patch("kesoku.agent.llm.get_config") as mock_get_config:
        mock_get_config.return_value = KesokuConfig()
        # Test explicit providers
        assert isinstance(get_llm("mock"), MockLLM)
        assert isinstance(get_llm("gemini"), GeminiLLM)

        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            get_llm("invalid")

        # Test reading from config when provider is None
        mock_get_config.return_value.agent.llm = "mock"
        assert isinstance(get_llm(), MockLLM)


def test_run_shell_command() -> None:
    """Test secure shell command execution tool."""
    ctx = ToolContext(session_id="test_sess", session_workspace="test_ws")
    with patch("kesoku.agent.tools.get_config") as mock_get_config:
        cfg = KesokuConfig()
        cfg.shell.enabled = False
        mock_get_config.return_value = cfg
        assert "disabled" in run_shell_command("echo hello", context=ctx)

        cfg.shell.enabled = True
        cfg.shell.mode = "blocklist"
        res = run_shell_command("echo test_hello", context=ctx)
        assert "test_hello" in res

        assert "Execution denied" in run_shell_command("rm -rf /", context=ctx)

        cfg.shell.mode = "allowlist"
        assert "Execution denied" in run_shell_command("unknown_binary_test", context=ctx)
        assert "test_allow" in run_shell_command("echo test_allow", context=ctx)
