"""Unit tests for ToolRunner class."""

from kesoku.agent.llm import ToolCallRequest
from kesoku.agent.tool_runner import ToolRunner
from kesoku.agent.tools import ToolContext, ToolRegistry
from kesoku.db import Message


def test_tool_runner_success() -> None:
    """Verify that ToolRunner executes a registered tool and returns a successful ToolResult Message."""
    registry = ToolRegistry()

    @registry.register
    def calculate_sum(x: int, y: int) -> int:
        """Sum two integers."""
        return x + y

    context = ToolContext(session_id="sess_1", session_workspace="ws_1")
    runner = ToolRunner(registry, context)

    call = ToolCallRequest(
        name="calculate_sum",
        arguments={"x": 10, "y": 20},
        tool_call_id="tc_1",
    )
    tc_msg = Message(
        id="msg_tc_1",
        session_id="sess_1",
        chatbot_id="cli",
        channel_id="ch1",
        sender="Kesoku",
        role="tool",
        content="Calling calculate_sum",
    )

    import asyncio

    result_msg = asyncio.run(runner.execute_tool(call, tc_msg))

    assert result_msg.role == "tool"
    assert result_msg.type == "tool_result"
    assert result_msg.parent_id == "msg_tc_1"
    assert result_msg.metadata["tool_name"] == "calculate_sum"
    assert result_msg.metadata["tool_result"] == "30"
    assert "returned:\n```\n30\n```" in result_msg.content


def test_tool_runner_missing_argument() -> None:
    """Verify that ToolRunner detects missing required arguments and returns a clear error Message."""
    registry = ToolRegistry()

    @registry.register
    def calculate_product(x: int, y: int) -> int:
        """Multiply two integers."""
        return x * y

    context = ToolContext(session_id="sess_1", session_workspace="ws_1")
    runner = ToolRunner(registry, context)

    # Omit required argument 'y'
    call = ToolCallRequest(
        name="calculate_product",
        arguments={"x": 10},
        tool_call_id="tc_1",
    )
    tc_msg = Message(
        id="msg_tc_1",
        session_id="sess_1",
        chatbot_id="cli",
        channel_id="ch1",
        sender="Kesoku",
        role="tool",
        content="Calling calculate_product",
    )

    import asyncio

    result_msg = asyncio.run(runner.execute_tool(call, tc_msg))

    assert result_msg.role == "tool"
    assert result_msg.type == "tool_result"
    assert "Command too long!" in result_msg.content
    assert "tool_error" in result_msg.metadata


def test_tool_runner_exception() -> None:
    """Verify that ToolRunner catches exceptions thrown by tools and translates them to error Messages."""
    registry = ToolRegistry()

    @registry.register
    def divide_numbers(x: int, y: int) -> float:
        """Divide x by y."""
        return x / y

    context = ToolContext(session_id="sess_1", session_workspace="ws_1")
    runner = ToolRunner(registry, context)

    # Trigger ZeroDivisionError
    call = ToolCallRequest(
        name="divide_numbers",
        arguments={"x": 10, "y": 0},
        tool_call_id="tc_1",
    )
    tc_msg = Message(
        id="msg_tc_1",
        session_id="sess_1",
        chatbot_id="cli",
        channel_id="ch1",
        sender="Kesoku",
        role="tool",
        content="Calling divide_numbers",
    )

    import asyncio

    result_msg = asyncio.run(runner.execute_tool(call, tc_msg))

    assert result_msg.role == "tool"
    assert result_msg.type == "tool_result"
    assert "division by zero" in result_msg.content
    assert "tool_error" in result_msg.metadata
    assert "division by zero" in result_msg.metadata["tool_error"]


def test_tool_runner_context_injection() -> None:
    """Verify that ToolRunner injects ToolContext into parameters if the tool signature expects it."""
    registry = ToolRegistry()

    @registry.register
    def get_workspace_path(context: ToolContext) -> str:
        """Return path from context."""
        return f"path_is_{context.session_workspace}"

    context = ToolContext(session_id="sess_1", session_workspace="special_ws")
    runner = ToolRunner(registry, context)

    call = ToolCallRequest(
        name="get_workspace_path",
        arguments={},
        tool_call_id="tc_1",
    )
    tc_msg = Message(
        id="msg_tc_1",
        session_id="sess_1",
        chatbot_id="cli",
        channel_id="ch1",
        sender="Kesoku",
        role="tool",
        content="Calling get_workspace_path",
    )

    import asyncio

    result_msg = asyncio.run(runner.execute_tool(call, tc_msg))

    assert result_msg.role == "tool"
    assert result_msg.metadata["tool_result"] == "path_is_special_ws"


def test_tool_runner_interrupted() -> None:
    """Verify that ToolRunner aborts execution and returns an interrupted result when is_interrupted returns True."""
    registry = ToolRegistry()

    @registry.register
    def long_running_tool() -> str:
        """Long running tool."""
        return "completed"

    context = ToolContext(session_id="sess_1", session_workspace="ws_1")
    runner = ToolRunner(registry, context)

    call = ToolCallRequest(
        name="long_running_tool",
        arguments={},
        tool_call_id="tc_1",
    )
    tc_msg = Message(
        id="msg_tc_1",
        session_id="sess_1",
        chatbot_id="cli",
        channel_id="ch1",
        sender="Kesoku",
        role="tool",
        content="Calling long_running_tool",
    )

    import asyncio

    # Set interruption callback to always return True
    def is_interrupted() -> bool:
        return True

    result_msg = asyncio.run(runner.execute_tool(call, tc_msg, is_interrupted=is_interrupted))

    assert result_msg.role == "tool"
    assert result_msg.type == "tool_result"
    assert "aborted due to thought interruption" in result_msg.content
    assert result_msg.metadata["tool_error"] == "Tool execution was aborted due to thought interruption."
