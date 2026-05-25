"""Unit tests for TurnExecutor class."""

from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from kesoku.agent.llm import BaseLLM, LLMResponse, ToolCallRequest
from kesoku.agent.turn_executor import TurnExecutor
from kesoku.agent.turn_logger import TurnLogger
from kesoku.config import KesokuConfig, WorkspaceConfig
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager, Message
from kesoku.gateway.gateway import Gateway


@pytest.fixture
def temp_db(tmp_path: Any) -> str:
    """Database fixture for tests."""
    return str(tmp_path / "test_turn_executor.db")


@pytest.mark.asyncio
async def test_turn_executor_successful_turn(temp_db: str) -> None:
    """Verify that TurnExecutor handles a standard text-only conversational turn successfully."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.create_session("sess_1", title="Success Session")

    # Post a pending user message
    user_msg = Message(
        session_id="sess_1",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role="user",
        content="Hello Agent!",
        status="pending_agent",
    )
    await gw.post(user_msg)

    # Setup MockLLM returning standard content
    class SuccessLLM(BaseLLM):
        async def generate(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
        ) -> LLMResponse:
            return LLMResponse(content="Hello User! How can I help?", total_tokens=10)

    llm = SuccessLLM()
    tool_runner = MagicMock()
    # Configure registry mock inside tool_runner to return empty list
    tool_runner.tool_registry.get_tools_list.return_value = []
    turn_logger = MagicMock(spec=TurnLogger)

    context = KesokuContext(llm=llm)
    executor = TurnExecutor("sess_1", gw, tool_runner, turn_logger, context=context)

    # Configure mock worker
    worker = MagicMock()
    type(worker).running = PropertyMock(side_effect=[True, False])
    async def mock_pivot(m: Message) -> Message:
        return m
    worker.drain_queue_and_pivot.side_effect = mock_pivot
    worker.queue_empty.return_value = True

    cfg = KesokuConfig()
    cfg.agent.raw_llm_logs = True

    with patch("kesoku.context.get_config", return_value=cfg):
        await executor.process_turn(
            current_msg=user_msg,
            worker=worker,
            session_staging_dir="/tmp/sess_1",
        )

    # Check final message and status
    history = await gw.get_session_history("sess_1")
    assistant_msgs = [m for m in history if m.role == "assistant"]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0].content == "Hello User! How can I help?"
    assert assistant_msgs[0].status == "pending"
    assert assistant_msgs[0].metadata["turn_metrics"]["turn_tokens"] == 10

    # Verify turn logging was invoked
    turn_logger.log_llm_turn.assert_called_once()


@pytest.mark.asyncio
async def test_turn_executor_nudging(temp_db: str) -> None:
    """Verify that TurnExecutor nudges the LLM once if it returns an empty content response."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.create_session("sess_nudge", title="Nudge Session")

    user_msg = Message(
        session_id="sess_nudge",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role="user",
        content="Help me!",
        status="pending_agent",
    )
    await gw.post(user_msg)

    class EmptyFirstLLM(BaseLLM):
        def __init__(self) -> None:
            self.calls = 0

        async def generate(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
        ) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(content="", thought="Thinking...")
            return LLMResponse(content="Success after nudge!")

    llm = EmptyFirstLLM()
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    turn_logger = MagicMock(spec=TurnLogger)
    context = KesokuContext(llm=llm)
    executor = TurnExecutor("sess_nudge", gw, tool_runner, turn_logger, context=context)

    # Configure mock worker
    worker = MagicMock()
    type(worker).running = PropertyMock(side_effect=[True, True, False])
    async def mock_pivot(m: Message) -> Message:
        return m
    worker.drain_queue_and_pivot.side_effect = mock_pivot
    worker.queue_empty.return_value = True

    cfg = KesokuConfig()
    cfg.agent.raw_llm_logs = False

    with patch("kesoku.context.get_config", return_value=cfg):
        await executor.process_turn(
            current_msg=user_msg,
            worker=worker,
            session_staging_dir="/tmp/sess_nudge",
        )

    # Check messages in session
    history = await gw.get_session_history("sess_nudge")
    system_msgs = [m for m in history if m.role == "system" and "empty content" in m.content]
    assert len(system_msgs) == 1

    assistant_msgs = [m for m in history if m.role == "assistant" and m.type == "text"]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0].content == "Success after nudge!"


@pytest.mark.asyncio
async def test_turn_executor_tool_calls(temp_db: str) -> None:
    """Verify that TurnExecutor schedules and logs tool calls, executes them via ToolRunner, and continues turn."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.create_session("sess_tools", title="Tools Session")

    user_msg = Message(
        session_id="sess_tools",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role="user",
        content="Run calculator",
        status="pending_agent",
    )
    await gw.post(user_msg)

    class ToolLLM(BaseLLM):
        def __init__(self) -> None:
            self.calls = 0

        async def generate(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
        ) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                # Request calculator tool call
                return LLMResponse(
                    content="",
                    thought="Using tool",
                    tool_calls=[
                        ToolCallRequest(
                            name="calculator",
                            arguments={"expression": "5+5"},
                            tool_call_id="tc_calc",
                        )
                    ],
                )
            return LLMResponse(content="The calculated value is 10.")

    llm = ToolLLM()

    # Setup mock ToolRunner returning a result message
    tool_runner = MagicMock()
    async def execute_tool(call: Any, tc_msg: Message, is_interrupted: Any = None) -> Message:
        return Message(
            session_id="sess_tools",
            chatbot_id="cli",
            channel_id="ch1",
            sender="calculator",
            role="tool",
            type="tool_result",
            content="Tool calculator returned:\n```\n10\n```",
            status="responded",
            parent_id=tc_msg.id,
            metadata={"tool_name": "calculator", "tool_result": "10"},
        )
    tool_runner.execute_tool.side_effect = execute_tool
    tool_runner.tool_registry.get_tools_list.return_value = []

    turn_logger = MagicMock(spec=TurnLogger)
    context = KesokuContext(llm=llm)
    executor = TurnExecutor("sess_tools", gw, tool_runner, turn_logger, context=context)

    # Configure mock worker
    worker = MagicMock()
    type(worker).running = PropertyMock(side_effect=[True, True, False])
    async def mock_pivot(m: Message) -> Message:
        return m
    worker.drain_queue_and_pivot.side_effect = mock_pivot
    worker.queue_empty.return_value = True

    cfg = KesokuConfig()
    cfg.agent.raw_llm_logs = False

    with patch("kesoku.context.get_config", return_value=cfg):
        await executor.process_turn(
            current_msg=user_msg,
            worker=worker,
            session_staging_dir="/tmp/sess_tools",
        )

    # Check messages posted
    history = await gw.get_session_history("sess_tools")
    thoughts = [m for m in history if m.type == "thought"]
    assert len(thoughts) == 1
    assert thoughts[0].content == "Using tool"

    tool_calls = [m for m in history if m.type == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0].metadata["tool_name"] == "calculator"

    tool_results = [m for m in history if m.type == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0].metadata["tool_result"] == "10"

    final_reply = [m for m in history if m.role == "assistant" and m.type == "text"]
    assert len(final_reply) == 1
    assert final_reply[0].content == "The calculated value is 10."
