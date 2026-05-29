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


@pytest.mark.asyncio
async def test_turn_executor_pivot_resets_nudged(temp_db: str) -> None:
    """Verify that when a pivot happens inside the loop, turn metrics and nudge state are reset."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.create_session("sess_pivot", title="Pivot Session")

    msg1 = Message(
        session_id="sess_pivot", chatbot_id="cli", channel_id="ch1", sender="u1",
        role="user", content="First prompt", status="pending_agent",
    )
    await gw.post(msg1)

    msg2 = Message(
        session_id="sess_pivot", chatbot_id="cli", channel_id="ch1", sender="u1",
        role="user", content="Pivoted prompt", status="pending_agent",
    )
    await gw.post(msg2)

    class NudgeAndPivotLLM(BaseLLM):
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
                # First prompt: return empty to trigger nudge
                return LLMResponse(content="", thought="Empty on first")
            # After pivot (which resets nudge state!):
            # 1. First generation after pivot (call 2): return empty content to trigger nudge on Pivoted prompt too!
            if self.calls == 2:
                return LLMResponse(content="", thought="Empty on second")
            # 2. Second generation after pivot (call 3): return success response
            return LLMResponse(content="Pivoted response success!")

    llm = NudgeAndPivotLLM()
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    turn_logger = MagicMock(spec=TurnLogger)
    context = KesokuContext(llm=llm)
    executor = TurnExecutor("sess_pivot", gw, tool_runner, turn_logger, context=context)

    # Configure mock worker that returns msg1 in first loop, but pivots to msg2 in second loop
    worker = MagicMock()
    type(worker).running = PropertyMock(side_effect=[True, True, True, False])

    loop_count = 0
    async def mock_pivot(m: Message) -> Message:
        nonlocal loop_count
        loop_count += 1
        if loop_count == 2:
            # Pivot to msg2 on second loop iteration!
            return msg2
        return m

    worker.drain_queue_and_pivot.side_effect = mock_pivot
    worker.queue_empty.return_value = True

    with patch("kesoku.context.get_config", return_value=cfg):
        await executor.process_turn(
            current_msg=msg1,
            worker=worker,
            session_staging_dir="/tmp/sess_pivot",
        )

    # Check messages in session
    history = await gw.get_session_history("sess_pivot")

    # We expect:
    # 1. First WeChat prompt
    # 2. Nudge message on the first WeChat prompt (parent is msg1)
    # 3. Pivoted prompt
    # 4. Nudge message on the pivoted prompt (parent is msg2)
    # 5. Success response to pivoted prompt (parent is msg2)

    nudges = [m for m in history if m.role == "system" and "empty content" in m.content]
    assert len(nudges) == 2
    assert nudges[0].parent_id == msg1.id
    assert nudges[1].parent_id == msg2.id

    final_replies = [m for m in history if m.role == "assistant" and m.content == "Pivoted response success!"]
    assert len(final_replies) == 1
    assert final_replies[0].parent_id == msg2.id


@pytest.mark.asyncio
async def test_turn_executor_context_monitor_warning(temp_db: str) -> None:
    """Verify that TurnExecutor injects context monitor warning based on percentage and threshold."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.create_session("sess_warning", title="Warning Session")

    # Post a pending user message
    user_msg = Message(
        session_id="sess_warning",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role="user",
        content="Hello Agent!",
        status="pending_agent",
    )
    await gw.post(user_msg)

    # Mock LLM with a context limit
    class WarningLLM(BaseLLM):
        @property
        def context_window_limit(self) -> int:
            return 1000

        async def generate(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
        ) -> LLMResponse:
            # Capture generated history in metadata to assert warning injection
            self.captured_history = list(history or [])
            return LLMResponse(content="Responded", total_tokens=10)

        def count_tokens(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
        ) -> int:
            # Will trigger 50% when context_window_limit is 1000
            return 500

    llm = WarningLLM()
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    turn_logger = MagicMock(spec=TurnLogger)

    context = KesokuContext(llm=llm)
    executor = TurnExecutor("sess_warning", gw, tool_runner, turn_logger, context=context)

    # Scenario A: Threshold is 80% (percentage is 50%, so no warning should be injected)
    cfg.agent.compact_history_warning_threshold = 80.0

    worker = MagicMock()
    type(worker).running = PropertyMock(side_effect=[True, False])
    async def mock_pivot(m: Message) -> Message:
        return m
    worker.drain_queue_and_pivot.side_effect = mock_pivot
    worker.queue_empty.return_value = True

    with patch("kesoku.context.get_config", return_value=cfg), \
         patch("kesoku.agent.turn_executor.build_clean_history", return_value=[user_msg]):
        await executor.process_turn(
            current_msg=user_msg,
            worker=worker,
            session_staging_dir="/tmp/sess_warning",
        )

    # Assert no warning in the generated history
    assert len(llm.captured_history) == 1
    assert "[Context Monitor:" not in llm.captured_history[0].content

    # Reset status of the message to process it again
    user_msg.status = "pending_agent"
    await gw.update_message_status(user_msg.id, "pending_agent")

    # Scenario B: Threshold is 40% (percentage is 50%, so warning should be injected)
    cfg.agent.compact_history_warning_threshold = 40.0

    worker2 = MagicMock()
    type(worker2).running = PropertyMock(side_effect=[True, False])
    worker2.drain_queue_and_pivot.side_effect = mock_pivot
    worker2.queue_empty.return_value = True

    # Re-create cloned user message without any previous warning
    user_msg_clean = Message(
        session_id="sess_warning",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role="user",
        content="Hello Agent!",
        status="pending_agent",
    )

    executor2 = TurnExecutor("sess_warning", gw, tool_runner, turn_logger, context=context)
    with patch("kesoku.context.get_config", return_value=cfg), \
         patch("kesoku.agent.turn_executor.build_clean_history", return_value=[user_msg_clean]):
        await executor2.process_turn(
            current_msg=user_msg_clean,
            worker=worker2,
            session_staging_dir="/tmp/sess_warning",
        )

    assert (
        "[Context Monitor: Currently using 500 tokens, "
        "which is 50.0% of your 1,000 window limit. "
        "It is highly recommended that you call the 'compact_history' "
        "tool now to reset the context window.]"
    ) in llm.captured_history[0].content


@pytest.mark.asyncio
async def test_turn_executor_user_preferences_injection(temp_db: str) -> None:
    """Verify that TurnExecutor injects user preferences into the latest user message."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.create_session("sess_pref_inject", title="Preferences Session")

    # Add user preferences to database under active role 'tifa'
    await gw.set_channel_role("cli", "ch_pref", "tifa")
    gw.db.upsert_agent_memory(
        category="user_preferences",
        key="rule_one",
        title="No Codeblocks",
        content="Avoid Markdown",
        role="tifa",
    )

    user_msg = Message(
        session_id="sess_pref_inject",
        chatbot_id="cli",
        channel_id="ch_pref",
        sender="u1",
        role="user",
        content="Run task!",
        status="pending_agent",
    )
    await gw.post(user_msg)

    class MockPrefLLM(BaseLLM):
        async def generate(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
        ) -> LLMResponse:
            self.captured_history = list(history or [])
            return LLMResponse(content="Response", total_tokens=5)

    llm = MockPrefLLM()
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    turn_logger = MagicMock(spec=TurnLogger)

    context = KesokuContext(llm=llm)
    executor = TurnExecutor("sess_pref_inject", gw, tool_runner, turn_logger, context=context)

    worker = MagicMock()
    type(worker).running = PropertyMock(side_effect=[True, False])
    async def mock_pivot(m: Message) -> Message:
        return m
    worker.drain_queue_and_pivot.side_effect = mock_pivot
    worker.queue_empty.return_value = True

    with patch("kesoku.context.get_config", return_value=cfg):
        await executor.process_turn(
            current_msg=user_msg,
            worker=worker,
            session_staging_dir="/tmp/sess_pref_inject",
        )

    # Assert that the user preferences were successfully injected in history
    assert len(llm.captured_history) == 1
    assert "Run task!" in llm.captured_history[0].content
    assert "[User Preferences]" in llm.captured_history[0].content
    assert "- No Codeblocks: Avoid Markdown" in llm.captured_history[0].content


@pytest.mark.asyncio
async def test_turn_executor_user_preferences_truncation(temp_db: str) -> None:
    """Verify that TurnExecutor truncates injected user preferences if they exceed 500 characters."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.create_session("sess_pref_trunc", title="Preferences Truncation Session")

    # Add very long user preferences to trigger 500 characters truncation
    await gw.set_channel_role("cli", "ch_pref_trunc", "tifa")
    gw.db.upsert_agent_memory(
        category="user_preferences",
        key="rule_long",
        title="Long Preference",
        content="A" * 600,
        role="tifa",
    )

    user_msg = Message(
        session_id="sess_pref_trunc",
        chatbot_id="cli",
        channel_id="ch_pref_trunc",
        sender="u1",
        role="user",
        content="Go!",
        status="pending_agent",
    )
    await gw.post(user_msg)

    class MockTruncLLM(BaseLLM):
        async def generate(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
        ) -> LLMResponse:
            self.captured_history = list(history or [])
            return LLMResponse(content="Response", total_tokens=5)

    llm = MockTruncLLM()
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    turn_logger = MagicMock(spec=TurnLogger)

    context = KesokuContext(llm=llm)
    executor = TurnExecutor("sess_pref_trunc", gw, tool_runner, turn_logger, context=context)

    worker = MagicMock()
    type(worker).running = PropertyMock(side_effect=[True, False])
    async def mock_pivot(m: Message) -> Message:
        return m
    worker.drain_queue_and_pivot.side_effect = mock_pivot
    worker.queue_empty.return_value = True

    with patch("kesoku.context.get_config", return_value=cfg):
        await executor.process_turn(
            current_msg=user_msg,
            worker=worker,
            session_staging_dir="/tmp/sess_pref_trunc",
        )

    # Assert that the user preferences block length was truncated and capped
    assert len(llm.captured_history) == 1
    content = llm.captured_history[0].content
    assert "[User Preferences]" in content
    from kesoku.agent.turn_executor import MAX_TOTAL_USER_PREFERENCES_LENGTH
    preference_part = content[content.index("\n\n[User Preferences]"):]
    assert len(preference_part) == MAX_TOTAL_USER_PREFERENCES_LENGTH
    assert preference_part.endswith("...")
