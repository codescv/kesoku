"""Unit tests for TurnExecutor class."""

import time
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
            **kwargs: Any,
        ) -> LLMResponse:
            return LLMResponse(content="Hello User! How can I help?", total_tokens=10)

    llm = SuccessLLM()
    tool_runner = MagicMock()
    # Configure registry mock inside tool_runner to return empty list
    tool_runner.tool_registry.get_tools_list.return_value = []
    turn_logger = MagicMock(spec=TurnLogger)

    context = KesokuContext(config=cfg, llm=llm)
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
    history = await gw.db.get_session_history("sess_1")
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
            **kwargs: Any,
        ) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(content="", thought="Thinking...")
            return LLMResponse(content="Success after nudge!")

    llm = EmptyFirstLLM()
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    turn_logger = MagicMock(spec=TurnLogger)
    context = KesokuContext(config=cfg, llm=llm)
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
    history = await gw.db.get_session_history("sess_nudge")
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
            **kwargs: Any,
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
    context = KesokuContext(config=cfg, llm=llm)
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
    history = await gw.db.get_session_history("sess_tools")
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
        session_id="sess_pivot",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role="user",
        content="First prompt",
        status="pending_agent",
    )
    await gw.post(msg1)

    msg2 = Message(
        session_id="sess_pivot",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role="user",
        content="Pivoted prompt",
        status="pending_agent",
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
            **kwargs: Any,
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
    context = KesokuContext(config=cfg, llm=llm)
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
    history = await gw.db.get_session_history("sess_pivot")

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
async def test_turn_executor_user_preferences_injection(temp_db: str, tmp_path: Any) -> None:
    """Verify that TurnExecutor injects role preferences from preferences.md during bootstrap turn."""
    DatabaseManager(temp_db).init_tables()
    roles_dir = tmp_path / "roles"
    tifa_role_dir = roles_dir / "tifa"
    tifa_role_dir.mkdir(parents=True, exist_ok=True)
    (tifa_role_dir / "preferences.md").write_text("Avoid Markdown", encoding="utf-8")

    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db, roles_dir=str(roles_dir)))
    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.create_session("sess_pref_inject", title="Preferences Session")

    # Add active role 'tifa'
    await gw.db.set_channel_role("cli", "ch_pref", "tifa")

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
            **kwargs: Any,
        ) -> LLMResponse:
            self.captured_history = list(history or [])
            return LLMResponse(content="Response", total_tokens=5)

    llm = MockPrefLLM()
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    turn_logger = MagicMock(spec=TurnLogger)

    context = KesokuContext(config=cfg, llm=llm)
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

    # Assert that the user preferences were injected as a <instructions> block inside the wrapped user message
    assert len(llm.captured_history) == 1
    content = llm.captured_history[0].content
    assert "Run task!" in content
    assert '<background_context type="sync_guidelines">' in content
    assert "User Preferences:" not in content
    assert "<instructions>\nAvoid Markdown\n</instructions>" in content


@pytest.mark.asyncio
async def test_turn_executor_dynamic_context_injection_bootstrap_vs_normal(temp_db: str, tmp_path: Any) -> None:
    """Verify dynamic injection rules: Sync Guidelines and Preferences are Bootstrap-only."""
    DatabaseManager(temp_db).init_tables()
    roles_dir = tmp_path / "roles"
    tifa_role_dir = roles_dir / "tifa"
    tifa_role_dir.mkdir(parents=True, exist_ok=True)
    (tifa_role_dir / "preferences.md").write_text("Python", encoding="utf-8")

    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db, roles_dir=str(roles_dir)))
    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.create_session("sess_dynamic", title="Dynamic Injection Session")

    # 1. Set role
    role = "tifa"
    await gw.db.set_channel_role("cli", "ch_dyn", role)

    class CaptureLLM(BaseLLM):
        def __init__(self) -> None:
            self.captured_history = []

        async def generate(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
            **kwargs: Any,
        ) -> LLMResponse:
            self.captured_history = list(history or [])
            return LLMResponse(content="Replied to user", total_tokens=5)

    llm = CaptureLLM()
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    turn_logger = MagicMock(spec=TurnLogger)
    context = KesokuContext(config=cfg, llm=llm)

    # Helper to run turn
    async def run_turn(msg: Message) -> str:
        executor = TurnExecutor("sess_dynamic", gw, tool_runner, turn_logger, context=context)
        worker = MagicMock()
        type(worker).running = PropertyMock(side_effect=[True, False])

        async def mock_pivot(m: Message) -> Message:
            return m

        worker.drain_queue_and_pivot.side_effect = mock_pivot
        worker.queue_empty.return_value = True

        with patch("kesoku.context.get_config", return_value=cfg):
            await executor.process_turn(
                current_msg=msg,
                worker=worker,
                session_staging_dir="/tmp/sess_dynamic",
            )
        assert len(llm.captured_history) > 0
        return llm.captured_history[-1].content

    now = time.time()

    # --- TURN 1: Bootstrap Turn (New Session) ---
    msg1 = Message(
        session_id="sess_dynamic",
        chatbot_id="cli",
        channel_id="ch_dyn",
        sender="u1",
        role="user",
        content="First message",
        status="pending_agent",
        timestamp=now,
    )
    await gw.post(msg1)
    content1 = await run_turn(msg1)

    # MUST contain Consolidated Sync Guidelines and Preferences
    assert '<background_context type="sync_guidelines">' in content1
    assert "view_message(message_id)" in content1
    assert "memory_grep(query)" in content1
    assert "User Preferences:" not in content1
    assert "<instructions>\nPython\n</instructions>" in content1
    assert 'from="u1"' in content1
    assert 'timezone="' in content1
    assert "CRITICAL: The time" not in content1
    assert "First message" in content1

    # --- TURN 2: Normal Turn (Not Bootstrap) ---
    history = await gw.db.get_session_history("sess_dynamic")
    for m in history:
        if m.role == "assistant" and m.status == "pending":
            await gw.db.update_message_status(m.id, "responded")

    msg2 = Message(
        session_id="sess_dynamic",
        chatbot_id="cli",
        channel_id="ch_dyn",
        sender="u1",
        role="user",
        content="Second message",
        status="pending_agent",
        timestamp=now + 10,  # Only 10 seconds later
    )
    await gw.post(msg2)
    content2 = await run_turn(msg2)

    # MUST NOT contain Sync Guidelines or Preferences
    assert '<background_context type="sync_guidelines">' not in content2
    assert "view_message(message_id)" not in content2
    assert "User Preferences:" not in content2
    assert "<instructions>" not in content2
    assert 'from="u1"' in content2
    assert 'timezone="' in content2
    assert "CRITICAL: The time" not in content2
    assert "Second message" in content2

    # --- TURN 3: Bootstrap Turn (Idle resumption) ---
    history = await gw.db.get_session_history("sess_dynamic")
    for m in history:
        if m.role == "assistant" and m.status == "pending":
            await gw.db.update_message_status(m.id, "responded")

    msg3 = Message(
        session_id="sess_dynamic",
        chatbot_id="cli",
        channel_id="ch_dyn",
        sender="u1",
        role="user",
        content="Third message",
        status="pending_agent",
        timestamp=now + 10 + 2000,  # 2000 seconds later (> 30 min idle threshold)
    )
    await gw.post(msg3)
    content3 = await run_turn(msg3)

    # MUST contain Consolidated Sync Guidelines and Preferences again due to idle resumption
    assert '<background_context type="sync_guidelines">' in content3
    assert "view_message(message_id)" in content3
    assert "memory_grep(query)" in content3
    assert "User Preferences:" not in content3
    assert "<instructions>\nPython\n</instructions>" in content3
    assert 'from="u1"' in content3
    assert 'timezone="' in content3
    assert "CRITICAL: The time" not in content3
    assert "Third message" in content3

    # --- TURN 4: Normal Turn (Not Bootstrap, turn_count=4) ---
    history = await gw.db.get_session_history("sess_dynamic")
    for m in history:
        if m.role == "assistant" and m.status == "pending":
            await gw.db.update_message_status(m.id, "responded")

    msg4 = Message(
        session_id="sess_dynamic",
        chatbot_id="cli",
        channel_id="ch_dyn",
        sender="u1",
        role="user",
        content="Fourth message",
        status="pending_agent",
        timestamp=now + 10 + 2000 + 10,
    )
    await gw.post(msg4)
    content4 = await run_turn(msg4)

    # MUST NOT contain Sync Guidelines or Preferences
    assert '<background_context type="sync_guidelines">' not in content4
    assert "<instructions>" not in content4

    # --- TURN 5: Modulo-4 Preferences Injection Turn (turn_count=5) ---
    history = await gw.db.get_session_history("sess_dynamic")
    for m in history:
        if m.role == "assistant" and m.status == "pending":
            await gw.db.update_message_status(m.id, "responded")

    msg5 = Message(
        session_id="sess_dynamic",
        chatbot_id="cli",
        channel_id="ch_dyn",
        sender="u1",
        role="user",
        content="Fifth message",
        status="pending_agent",
        timestamp=now + 10 + 2000 + 20,
    )
    await gw.post(msg5)
    content5 = await run_turn(msg5)

    # MUST contain Preferences (turn_count=5, 5%4==1), but NOT Sync Guidelines (not a bootstrap turn)
    assert '<background_context type="sync_guidelines">' not in content5
    assert "<instructions>\nPython\n</instructions>" in content5



def test_truncate_context_middle() -> None:
    """Verify that truncate_context_middle preserves start/end and truncates middle correctly."""
    from kesoku.utils.text import truncate_context_middle

    # Scenario A: Under limit, should remain completely untouched
    short_text = "Short content timeline."
    assert truncate_context_middle(short_text, max_len=50) == short_text

    # Scenario B: Over limit, should perform middle truncation preserving newline boundaries
    lines = [f"Line {i}: This is some lengthy timeline memory content." for i in range(1, 41)]
    long_text = "\n".join(lines)
    assert len(long_text) > 500

    # Truncate with a small limit (e.g., 400)
    truncated = truncate_context_middle(long_text, max_len=400)
    assert len(truncated) < len(long_text)
    assert "... [Timeline Truncated for Brevity] ..." in truncated
    # Start must be preserved
    assert "Line 1:" in truncated
    # End must be preserved
    assert "Line 40:" in truncated
    # Middle must be truncated
    assert "Line 20:" not in truncated
    # Newline boundaries must be clean
    assert truncated.startswith("Line 1:")
    assert truncated.endswith("Line 40: This is some lengthy timeline memory content.")


@pytest.mark.asyncio
async def test_turn_executor_user_context_injection(temp_db: str) -> None:
    """Verify that TurnExecutor injects User Context based on chatbot platform (Discord/Google Chat)."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.create_session("sess_user_ctx", title="User Context Session")

    # --- Test Discord Injection ---
    discord_msg = Message(
        session_id="sess_user_ctx",
        chatbot_id="discord",
        channel_id="ch_user_ctx",
        sender="Tifa Lockhart",
        role="user",
        content="Hello",
        status="pending_agent",
        metadata={
            "discord_author_id": "123456",
            "sender_name": "Tifa Lockhart (ID: 123456)",
        },
    )

    await gw.post(discord_msg)

    class CaptureLLM(BaseLLM):
        def __init__(self) -> None:
            self.captured_history = []

        async def generate(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
            **kwargs: Any,
        ) -> LLMResponse:
            self.captured_history = list(history or [])
            return LLMResponse(content="Replied", total_tokens=5)

    llm = CaptureLLM()
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    turn_logger = MagicMock(spec=TurnLogger)
    context = KesokuContext(config=cfg, llm=llm)

    executor = TurnExecutor("sess_user_ctx", gw, tool_runner, turn_logger, context=context)
    worker = MagicMock()
    type(worker).running = PropertyMock(side_effect=[True, False])

    async def mock_pivot(m: Message) -> Message:
        return m

    worker.drain_queue_and_pivot.side_effect = mock_pivot
    worker.queue_empty.return_value = True

    with patch("kesoku.context.get_config", return_value=cfg):
        await executor.process_turn(
            current_msg=discord_msg,
            worker=worker,
            session_staging_dir="/tmp/sess_user_ctx",
        )

    assert len(llm.captured_history) == 1
    discord_content = llm.captured_history[0].content
    assert 'from="Tifa Lockhart (ID: 123456)"' in discord_content

    # --- Test Google Chat Injection ---
    # Mark previous turn processed to start a clean turn
    history = await gw.db.get_session_history("sess_user_ctx")
    for m in history:
        if m.status == "pending":
            await gw.db.update_message_status(m.id, "responded")

    gchat_msg = Message(
        session_id="sess_user_ctx",
        chatbot_id="google_chat",
        channel_id="ch_user_ctx",
        sender="Cloud Strife",
        role="user",
        content="Hi",
        status="pending_agent",
        metadata={
            "google_chat_sender_email": "cloud@shinra.com",
            "sender_name": "Cloud Strife (Email: cloud@shinra.com)",
        },
    )

    await gw.post(gchat_msg)

    # Re-init helper with fresh side effects
    type(worker).running = PropertyMock(side_effect=[True, False])

    with patch("kesoku.context.get_config", return_value=cfg):
        await executor.process_turn(
            current_msg=gchat_msg,
            worker=worker,
            session_staging_dir="/tmp/sess_user_ctx",
        )

    assert len(llm.captured_history) > 0
    gchat_content = llm.captured_history[-1].content
    assert 'from="Cloud Strife (Email: cloud@shinra.com)"' in gchat_content


async def test_turn_executor_auto_compaction(temp_db: str) -> None:
    """Verify that TurnExecutor automatically triggers in-place history compaction when threshold is exceeded."""
    from kesoku.constants import MessageRole

    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.create_session("sess_auto_compact", title="Auto Compact Session")

    # Create a history with 3 turns (User -> Assistant -> User -> Assistant -> User)
    # Note: user prompts must be marked PROCESSED so they are completed turns
    msg1 = Message(
        session_id="sess_auto_compact",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role=MessageRole.USER,
        content="Turn 1 user",
        status="processed",
        timestamp=time.time() - 100,
    )
    await gw.post(msg1)
    msg2 = Message(
        session_id="sess_auto_compact",
        chatbot_id="cli",
        channel_id="ch1",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        content="Turn 1 reply",
        status="delivered",
        timestamp=time.time() - 90,
    )
    await gw.post(msg2)

    msg3 = Message(
        session_id="sess_auto_compact",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role=MessageRole.USER,
        content="Turn 2 user",
        status="processed",
        timestamp=time.time() - 80,
    )
    await gw.post(msg3)
    msg4 = Message(
        session_id="sess_auto_compact",
        chatbot_id="cli",
        channel_id="ch1",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        content="Turn 2 reply",
        status="delivered",
        timestamp=time.time() - 70,
    )
    await gw.post(msg4)

    active_msg = Message(
        session_id="sess_auto_compact",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role=MessageRole.USER,
        content="Turn 3 active",
        status="pending_agent",
        timestamp=time.time(),
    )
    await gw.post(active_msg)

    # Mock LLM
    class CompactLLM(BaseLLM):
        context_window_limit: int = 1000

        async def generate(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
            **kwargs: Any,
        ) -> LLMResponse:
            self.captured_history = list(history or [])
            return LLMResponse(content="Final Assistant Turn Response", total_tokens=10)

    llm = CompactLLM()
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    turn_logger = MagicMock(spec=TurnLogger)

    # Mock only the HistoryCompressor.auto_compact_session method
    from kesoku.db import SummaryNode

    async def mock_auto_compact(session_id, history, llm, config):
        node = SummaryNode(
            id="node-1-uuid",
            session_id=session_id,
            level=0,
            summary="Simulated summary node content",
            start_timestamp=time.time() - 100,
            end_timestamp=time.time() - 70,
            token_count=10,
            source_token_count=100,
            parent_id=None,
        )
        await gw.db.insert_summary_node(node)
        await gw.db.update_messages_summary_node([msg1.id, msg2.id], "node-1-uuid")
        return True

    context = KesokuContext(config=cfg, llm=llm)

    executor = TurnExecutor("sess_auto_compact", gw, tool_runner, turn_logger, context=context)

    cfg.agent.protect_front_turns = 1
    cfg.agent.protect_tail_turns = 5

    worker = MagicMock()
    type(worker).running = PropertyMock(side_effect=[True, False])

    async def mock_pivot(m: Message) -> Message:
        return m

    worker.drain_queue_and_pivot.side_effect = mock_pivot
    worker.queue_empty.return_value = True

    with (
        patch("kesoku.context.get_config", return_value=cfg),
        patch(
            "kesoku.agent.compressor.HistoryCompressor.auto_compact_session",
            side_effect=mock_auto_compact,
        ) as mock_compact,
    ):
        await executor.process_turn(
            current_msg=active_msg,
            worker=worker,
            session_staging_dir="/tmp/sess_auto_compact",
        )

    # Assertions: Verify that the HistoryCompressor was called
    mock_compact.assert_called_once()

    # Reconstructed messages verify they match our custom turn-based structure
    assert len(llm.captured_history) == 7
    assert llm.captured_history[0].role == MessageRole.USER
    assert "Turn 1 user" in llm.captured_history[0].content
    assert llm.captured_history[1].role == MessageRole.ASSISTANT
    assert llm.captured_history[1].content == "Turn 1 reply"

    # Scaffold and scaffold ack
    assert llm.captured_history[2].role == MessageRole.USER
    assert "[Note: This conversation uses custom turn-based" in llm.captured_history[2].content
    assert "Simulated summary node content" in llm.captured_history[2].content
    assert llm.captured_history[3].role == MessageRole.ASSISTANT
    assert "Understood" in llm.captured_history[3].content

    # Tail turns
    assert llm.captured_history[4].role == MessageRole.USER
    assert "Turn 2 user" in llm.captured_history[4].content
    assert llm.captured_history[5].role == MessageRole.ASSISTANT
    assert llm.captured_history[5].content == "Turn 2 reply"
    assert llm.captured_history[6].role == MessageRole.USER
    assert "Turn 3 active" in llm.captured_history[6].content

    # Verify database is completely lossless (nothing was physically deleted!)
    db_history = await gw.db.get_session_history("sess_auto_compact", limit=0)
    db_ids = [m.id for m in db_history]
    assert msg1.id in db_ids
    assert msg2.id in db_ids
    assert msg3.id in db_ids
    assert msg4.id in db_ids
    assert active_msg.id in db_ids


@pytest.mark.asyncio
async def test_turn_executor_context_caching_with_compaction(temp_db: str) -> None:
    """Verify context cache deletion and correct generate request when compaction occurs."""
    from kesoku.constants import MessageRole

    DatabaseManager(temp_db).init_tables()

    # Configure Gemini and Context Caching
    from kesoku.config import GeminiConfig

    cfg = KesokuConfig(
        workspace=WorkspaceConfig(db_path=temp_db),
        gemini=GeminiConfig(
            context_caching=True,
            context_caching_threshold=100,  # small threshold to trigger caching
        ),
    )

    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.create_session("sess_cache_compact", title="Cache and Compact")

    # Post some messages to exceed threshold
    msg1 = Message(
        session_id="sess_cache_compact",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role=MessageRole.USER,
        content="User message 1 that is quite long to accumulate tokens for caching " * 10,
        status="pending",
        timestamp=time.time() - 10,
    )
    await gw.post(msg1)

    msg2 = Message(
        session_id="sess_cache_compact",
        chatbot_id="cli",
        channel_id="ch1",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        content="Assistant response 1 that is also quite long " * 10,
        status="pending",
        timestamp=time.time() - 5,
    )
    await gw.post(msg2)

    active_msg = Message(
        session_id="sess_cache_compact",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role=MessageRole.USER,
        content="Active user query",
        status="pending_agent",
        timestamp=time.time(),
    )
    await gw.post(active_msg)

    token_count_to_return = 150

    # Mock GeminiLLM to capture create_cache, delete_cache, and generate calls
    class GeminiLLM(BaseLLM):
        context_window_limit: int = 1000

        def __init__(self) -> None:
            self.created_caches = []
            self.deleted_caches = []
            self.captured_generates = []
            self.token_calls = 0

        def count_tokens(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
        ) -> int:
            if len(self.captured_generates) == 0:
                return 150
            else:
                return 50

        async def create_cache(
            self,
            contents: list[Message],
            system_prompt: str | None,
            tools: list[Any] | None = None,
            display_name: str | None = None,
            ttl_seconds: int = 300,
        ) -> str | None:
            cache_name = f"mock_cache_{len(self.created_caches)}"
            self.created_caches.append(
                {
                    "name": cache_name,
                    "contents": contents,
                    "system_prompt": system_prompt,
                    "tools": tools,
                }
            )
            return cache_name

        async def delete_cache(self, cache_name: str) -> None:
            self.deleted_caches.append(cache_name)

        async def generate(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
            cached_content: str | None = None,
            **kwargs: Any,
        ) -> LLMResponse:
            self.captured_generates.append(
                {
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "history": history,
                    "tools": tools,
                    "cached_content": cached_content,
                }
            )

            # 1st call (during 1st iteration): return tool call
            if len(self.captured_generates) == 1:
                return LLMResponse(
                    content="I need to call a tool",
                    tool_calls=[
                        ToolCallRequest(
                            name="dummy_tool",
                            arguments={"arg1": "val1"},
                            tool_call_id="call_123",
                        )
                    ],
                    total_tokens=10,
                )
            # 2nd call (during 2nd iteration): return final text
            else:
                return LLMResponse(content="Final response after tool execution", total_tokens=10)

    llm = GeminiLLM()
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = ["dummy_tool"]

    # Mock tool execution: returns a Message representing the tool output
    async def mock_execute_tool(call, tc_msg, **kwargs):
        return Message(
            id=f"tool_res_{time.time()}",
            session_id=tc_msg.session_id,
            chatbot_id=tc_msg.chatbot_id,
            channel_id=tc_msg.channel_id,
            sender="System",
            role=MessageRole.TOOL,
            type=MessageType.TEXT,
            content="Tool output content",
            status="responded",
            parent_id=tc_msg.id,
        )

    tool_runner.execute_tool.side_effect = mock_execute_tool
    tool_runner.tool_context = MagicMock()
    tool_runner.tool_context.transitioned_to_session = None

    turn_logger = MagicMock(spec=TurnLogger)

    # Mock only the HistoryCompressor.auto_compact_session method
    from kesoku.db import SummaryNode

    compaction_calls = 0

    async def mock_auto_compact(session_id, history, llm, config):
        nonlocal compaction_calls
        compaction_calls += 1
        if compaction_calls == 1:
            # First call: compact Turn 1 into a summary node
            node = SummaryNode(
                id="node-1-uuid",
                session_id=session_id,
                level=0,
                summary="Compacted user query",
                start_timestamp=time.time() - 100,
                end_timestamp=time.time() - 70,
                token_count=10,
                source_token_count=100,
                parent_id=None,
            )
            await gw.db.insert_summary_node(node)
            await gw.db.update_messages_summary_node([msg1.id, msg2.id], "node-1-uuid")
            return True
        else:
            # Second call: compact the tool messages too!
            node2 = SummaryNode(
                id="node-2-uuid",
                session_id=session_id,
                level=0,
                summary="Compacted tool result",
                start_timestamp=time.time() - 100,
                end_timestamp=time.time(),
                token_count=15,
                source_token_count=150,
                parent_id=None,
            )
            await gw.db.insert_summary_node(node2)
            # Find the new tool messages and mark them compacted
            tool_msgs = await gw.db.get_session_history(session_id, limit=0)
            uncompacted_ids = [m.id for m in tool_msgs if m.summary_node_id is None]
            await gw.db.update_messages_summary_node(uncompacted_ids, "node-2-uuid")
            return True

    context = KesokuContext(config=cfg, llm=llm)

    executor = TurnExecutor("sess_cache_compact", gw, tool_runner, turn_logger, context=context)

    # Set up worker to run exactly two iterations of the loop (1st turn, 2nd turn)
    worker = MagicMock()
    type(worker).running = PropertyMock(side_effect=[True, True, False])

    async def mock_pivot(m: Message) -> Message:
        return m

    worker.drain_queue_and_pivot.side_effect = mock_pivot
    worker.queue_empty.return_value = True

    from kesoku.constants import MessageType

    # Run the turn! It will run 2 iterations because the first iteration returns a tool call
    # which tells process_turn to "continue" the loop. The second iteration returns a final text,
    # which breaks the loop, exiting process_turn.
    with (
        patch("kesoku.context.get_config", return_value=cfg),
        patch(
            "kesoku.agent.compressor.HistoryCompressor.auto_compact_session",
            side_effect=mock_auto_compact,
        ) as mock_compact,
    ):
        await executor.process_turn(
            current_msg=active_msg,
            worker=worker,
            session_staging_dir="/tmp/sess_cache_compact",
        )

    # ASSERTIONS:
    # 1. During the first iteration:
    # - A context cache was successfully created for the prefix of compacted history.
    # - active_cache_name was set to "mock_cache_0" with cached_messages_len=2
    #   (system prompt + scaffold prefix messages).
    assert len(llm.created_caches) == 1
    assert llm.created_caches[0]["name"] == "mock_cache_0"

    # 2. During the second iteration:
    # - Compaction ran again (but returned False).
    # - Obsolete cache "mock_cache_0" was deleted because history length/content changed.
    # - No new cache was created (since token_count returned 50 < 100).
    assert len(llm.deleted_caches) == 1
    assert llm.deleted_caches[0] == "mock_cache_0"
    assert len(llm.created_caches) == 1  # still only 1 cache from turn 1

    # 3. Captured generates checks:
    assert len(llm.captured_generates) == 2

    # - 1st Generate:
    gen_call_1 = llm.captured_generates[0]
    assert gen_call_1["cached_content"] == "mock_cache_0"
    assert "Agent Working Directory" in gen_call_1["system_prompt"]
    assert gen_call_1["tools"] is None
    assert len(gen_call_1["history"]) == 1
    assert gen_call_1["history"][0].role == MessageRole.USER
    assert "Active user query" in gen_call_1["history"][0].content

    # - 2nd Generate:
    gen_call_2 = llm.captured_generates[1]
    assert gen_call_2["cached_content"] is None
    assert "Agent Working Directory" in gen_call_2["system_prompt"]
    assert gen_call_2["tools"] == ["dummy_tool"]
    # history has: Turn 1 (2) + Scaffold (1) + Ack (1) + Turn 3 (active turn with tool call/response) (5)
    assert len(gen_call_2["history"]) == 9
    assert gen_call_2["history"][0].role == MessageRole.USER
    assert "User message 1" in gen_call_2["history"][0].content
    assert gen_call_2["history"][2].role == MessageRole.USER
    assert "[Note: This conversation uses custom turn-based" in gen_call_2["history"][2].content
    assert "Compacted user query" in gen_call_2["history"][2].content
    assert gen_call_2["history"][5].role == MessageRole.ASSISTANT
    assert "I need to call a tool" in gen_call_2["history"][5].content
    assert gen_call_2["history"][7].role == MessageRole.TOOL
    assert "Tool output content" in gen_call_2["history"][7].content


@pytest.mark.asyncio
async def test_turn_executor_error_handling_truncation(temp_db: str, tmp_path: Any) -> None:
    """Verify that TurnExecutor catches exceptions, writes traceback to staging, and truncates chatbot error msg."""
    import os

    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.create_session("sess_err", title="Error Session")

    user_msg = Message(
        id="user_msg_123",
        session_id="sess_err",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role="user",
        content="Trigger Error!",
        status="pending_agent",
    )
    await gw.post(user_msg)

    # Setup MockLLM throwing a very long exception message
    class ThrowingLLM(BaseLLM):
        async def generate(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
            **kwargs: Any,
        ) -> LLMResponse:
            raise RuntimeError("X" * 600)  # Exceeds 500 characters limit

    llm = ThrowingLLM()
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    turn_logger = MagicMock(spec=TurnLogger)

    context = KesokuContext(config=cfg, llm=llm)
    executor = TurnExecutor("sess_err", gw, tool_runner, turn_logger, context=context)

    # Configure mock worker
    worker = MagicMock()
    type(worker).running = PropertyMock(side_effect=[True, False])

    async def mock_pivot(m: Message) -> Message:
        return m

    worker.drain_queue_and_pivot.side_effect = mock_pivot
    worker.queue_empty.return_value = True

    staging_dir = str(tmp_path / "staging_err")
    os.makedirs(staging_dir, exist_ok=True)

    with patch("kesoku.context.get_config", return_value=cfg):
        await executor.process_turn(
            current_msg=user_msg,
            worker=worker,
            session_staging_dir=staging_dir,
        )

    # 1. Check message posted to database is Assistant error message
    history = await gw.db.get_session_history("sess_err")
    assistant_msgs = [m for m in history if m.role == "assistant"]
    assert len(assistant_msgs) == 1
    err_content = assistant_msgs[0].content

    # 2. Check length of error message is <= MAX_CHATBOT_ERROR_MESSAGE_LENGTH
    from kesoku.agent.turn_executor import MAX_CHATBOT_ERROR_MESSAGE_LENGTH

    assert len(err_content) <= MAX_CHATBOT_ERROR_MESSAGE_LENGTH
    # 3. Check error message ends with the hint
    assert "Full error log saved to staging directory: error_user_msg_123.txt" in err_content

    # 4. Check the traceback file exists and contains traceback of RuntimeError
    error_file_path = os.path.join(staging_dir, "error_user_msg_123.txt")
    assert os.path.exists(error_file_path)
    with open(error_file_path, encoding="utf-8") as f:
        file_content = f.read()
    assert "RuntimeError" in file_content
    assert "X" * 600 in file_content


@pytest.mark.asyncio
async def test_turn_executor_cache_expiration_retry(temp_db: str) -> None:
    """Verify that TurnExecutor retries generation without context cache if it expires."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    cfg.gemini.context_caching = True
    cfg.gemini.context_caching_threshold = 10
    cfg.gemini.context_caching_ttl = 1800

    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.create_session("sess_cache_expire", title="Cache Expire Session")

    msg1 = Message(
        session_id="sess_cache_expire",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role="user",
        content="Long prompt prefix that is cached.",
        status="responded",
    )
    await gw.post(msg1)

    msg2 = Message(
        session_id="sess_cache_expire",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role="user",
        content="The current pending message.",
        status="pending_agent",
    )
    await gw.post(msg2)

    class GeminiLLM(BaseLLM):
        def __init__(self) -> None:
            self.created_caches: list[dict[str, Any]] = []
            self.deleted_caches: list[str] = []
            self.captured_generates: list[dict[str, Any]] = []
            self.cache_ttl_passed: int | None = None

        def count_tokens(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
        ) -> int:
            return 100

        async def create_cache(
            self,
            contents: list[Message],
            system_prompt: str | None,
            tools: list[Any] | None = None,
            display_name: str | None = None,
            ttl_seconds: int = 300,
        ) -> str | None:
            self.cache_ttl_passed = ttl_seconds
            cache_name = f"mock_cache_{len(self.created_caches)}"
            self.created_caches.append(
                {
                    "name": cache_name,
                    "contents": contents,
                    "system_prompt": system_prompt,
                    "tools": tools,
                }
            )
            return cache_name

        async def delete_cache(self, cache_name: str) -> None:
            self.deleted_caches.append(cache_name)

        async def generate(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
            cached_content: str | None = None,
            **kwargs: Any,
        ) -> LLMResponse:
            self.captured_generates.append(
                {
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "history": history,
                    "tools": tools,
                    "cached_content": cached_content,
                }
            )

            if len(self.captured_generates) == 1:
                assert cached_content == "mock_cache_0"
                raise RuntimeError("400 INVALID_ARGUMENT. Cache content mock_cache_0 is expired.")
            else:
                assert cached_content is None
                return LLMResponse(content="Recovered from expired cache!", total_tokens=10)

    llm = GeminiLLM()
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    turn_logger = MagicMock(spec=TurnLogger)

    context = KesokuContext(config=cfg, llm=llm)
    executor = TurnExecutor("sess_cache_expire", gw, tool_runner, turn_logger, context=context)

    worker = MagicMock()
    type(worker).running = PropertyMock(side_effect=[True, False])

    async def mock_pivot(m: Message) -> Message:
        return m

    worker.drain_queue_and_pivot.side_effect = mock_pivot
    worker.queue_empty.return_value = True

    with patch("kesoku.context.get_config", return_value=cfg):
        await executor.process_turn(
            current_msg=msg2,
            worker=worker,
            session_staging_dir="/tmp/sess_cache_expire",
        )

    assert llm.cache_ttl_passed == 1800
    assert len(llm.created_caches) == 1
    assert llm.created_caches[0]["name"] == "mock_cache_0"

    assert len(llm.captured_generates) == 2
    assert llm.captured_generates[0]["cached_content"] == "mock_cache_0"
    assert llm.captured_generates[1]["cached_content"] is None

    history = await gw.db.get_session_history("sess_cache_expire")
    assistant_msgs = [m for m in history if m.role == "assistant"]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0].content == "Recovered from expired cache!"
    assert assistant_msgs[0].status == "pending"


@pytest.mark.asyncio
async def test_turn_executor_auto_compaction_buffer_exclusion(temp_db: str) -> None:
    """Verify that compacted middle turns are correctly excluded from buffer in LLM history."""
    from kesoku.constants import MessageRole

    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.create_session("sess_exclude", title="Exclude Session")

    # Create history: Turn 1 (user/assistant) -> Turn 2 (user/assistant) -> Turn 3 (user)
    msg1 = Message(
        session_id="sess_exclude",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role=MessageRole.USER,
        content="Turn 1 user",
        status="processed",
        timestamp=time.time() - 100,
    )
    await gw.post(msg1)
    msg2 = Message(
        session_id="sess_exclude",
        chatbot_id="cli",
        channel_id="ch1",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        content="Turn 1 reply",
        status="delivered",
        timestamp=time.time() - 90,
    )
    await gw.post(msg2)

    msg3 = Message(
        session_id="sess_exclude",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role=MessageRole.USER,
        content="Turn 2 user",
        status="processed",
        timestamp=time.time() - 80,
    )
    await gw.post(msg3)
    msg4 = Message(
        session_id="sess_exclude",
        chatbot_id="cli",
        channel_id="ch1",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        content="Turn 2 reply",
        status="delivered",
        timestamp=time.time() - 70,
    )
    await gw.post(msg4)

    active_msg = Message(
        session_id="sess_exclude",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role=MessageRole.USER,
        content="Turn 3 active",
        status="pending_agent",
        timestamp=time.time(),
    )
    await gw.post(active_msg)

    # Mock LLM to capture history
    class CaptureLLM(BaseLLM):
        async def generate(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
            **kwargs: Any,
        ) -> LLMResponse:
            self.captured_history = list(history or [])
            return LLMResponse(content="Response content", total_tokens=10)

    llm = CaptureLLM()
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    turn_logger = MagicMock(spec=TurnLogger)

    # Mock auto compact to compress Turn 2 (middle turn) only!
    from kesoku.db import SummaryNode

    async def mock_auto_compact(session_id, history, llm, config):
        node = SummaryNode(
            id="node-2-uuid",
            session_id=session_id,
            level=0,
            summary="Turn 2 summary content",
            start_timestamp=time.time() - 80,
            end_timestamp=time.time() - 70,
            token_count=10,
            source_token_count=100,
            parent_id=None,
        )
        await gw.db.insert_summary_node(node)
        await gw.db.update_messages_summary_node([msg3.id, msg4.id], "node-2-uuid")
        # Update the memory message objects in history to match the database update
        for msg in history:
            if msg.id in (msg3.id, msg4.id):
                msg.summary_node_id = "node-2-uuid"
        return True

    context = KesokuContext(config=cfg, llm=llm)
    executor = TurnExecutor("sess_exclude", gw, tool_runner, turn_logger, context=context)

    # Set parameters:
    # protect_front = 1 (protects Turn 1)
    # protect_tail = 1 (protects Turn 3 active)
    # This leaves Turn 2 as middle/buffer candidate!
    cfg.agent.protect_front_turns = 1
    cfg.agent.protect_tail_turns = 1

    worker = MagicMock()
    type(worker).running = PropertyMock(side_effect=[True, False])

    async def mock_pivot(m: Message) -> Message:
        return m

    worker.drain_queue_and_pivot.side_effect = mock_pivot
    worker.queue_empty.return_value = True

    with (
        patch("kesoku.context.get_config", return_value=cfg),
        patch(
            "kesoku.agent.compressor.HistoryCompressor.auto_compact_session",
            side_effect=mock_auto_compact,
        ) as mock_compact,
    ):
        await executor.process_turn(
            current_msg=active_msg,
            worker=worker,
            session_staging_dir="/tmp/sess_exclude",
        )

    mock_compact.assert_called_once()

    # Reconstructed history must have:
    # 1. Turn 1 (messages 1 and 2)
    # 2. Scaffold message (with Turn 2 summary "Turn 2 summary content")
    # 3. Scaffold ack
    # 4. Turn 3 active message
    # Turn 2 raw messages must NOT be in the history!

    captured_contents = [m.content for m in llm.captured_history]

    # Assert Turn 1 is present
    assert any("Turn 1 user" in c for c in captured_contents)
    assert any("Turn 1 reply" in c for c in captured_contents)

    # Assert Scaffold with Turn 2 summary is present
    assert any("Turn 2 summary content" in c for c in captured_contents)

    # Assert Turn 3 is present
    assert any("Turn 3 active" in c for c in captured_contents)

    # Assert Turn 2 raw messages are NOT present!
    assert not any("Turn 2 user" in c for c in captured_contents)
    assert not any("Turn 2 reply" in c for c in captured_contents)


@pytest.mark.asyncio
async def test_history_compressor_in_place_update(temp_db: str) -> None:
    """Verify that real HistoryCompressor.auto_compact_session updates Message objects in-place in memory."""
    from kesoku.agent.compressor import HistoryCompressor
    from kesoku.constants import MessageRole

    DatabaseManager(temp_db).init_tables()

    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    # Setup configuration to trigger compaction for 1 turn (Turn 1)
    cfg.agent.protect_front_turns = 0
    cfg.agent.protect_tail_turns = 1  # protect only the active Turn 2
    cfg.agent.base_node_turns = 1
    cfg.agent.base_node_min_tokens = 0

    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.create_session("sess_real_compact", title="Real Compact Session")

    # Turn 1: user and assistant (should get compacted)
    msg1 = Message(
        session_id="sess_real_compact",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role=MessageRole.USER,
        content="Turn 1 user text that is fairly long to satisfy logic" * 5,
        status="processed",
        timestamp=time.time() - 100,
    )
    await gw.post(msg1)
    msg2 = Message(
        session_id="sess_real_compact",
        chatbot_id="cli",
        channel_id="ch1",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        content="Turn 1 reply text that is also fairly long" * 5,
        status="delivered",
        timestamp=time.time() - 90,
    )
    await gw.post(msg2)

    # Active Turn 2: pending message (should NOT get compacted due to protect_tail_turns=1)
    active_msg = Message(
        session_id="sess_real_compact",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role=MessageRole.USER,
        content="Active query",
        status="pending_agent",
        timestamp=time.time(),
    )
    await gw.post(active_msg)

    # Load history from gateway (which mimics the list passed to TurnExecutor/HistoryCompressor)
    history = await gw.db.get_session_history("sess_real_compact", limit=0)
    assert len(history) == 3

    # Mock LLM used by HistoryCompressor to summarize the turn content
    class CompactorLLM(BaseLLM):
        async def generate(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
            **kwargs: Any,
        ) -> LLMResponse:
            return LLMResponse(content="Summarized turn content successfully.", total_tokens=10)

    llm = CompactorLLM()

    # Verify that before compaction, summary_node_id is None for all messages
    assert msg1.summary_node_id is None
    assert msg2.summary_node_id is None
    assert active_msg.summary_node_id is None

    # Run the real HistoryCompressor auto_compact_session
    compressor = HistoryCompressor(gw.db)
    compacted = await compressor.auto_compact_session(
        session_id="sess_real_compact",
        history=history,
        llm=llm,
        config=cfg,
    )

    assert compacted is True

    # Find the compacted message objects in the history list we passed!
    mem_msg1 = next(m for m in history if m.id == msg1.id)
    mem_msg2 = next(m for m in history if m.id == msg2.id)
    mem_active = next(m for m in history if m.id == active_msg.id)

    # VERIFY: In-memory Message objects must have summary_node_id updated!
    assert mem_msg1.summary_node_id is not None
    assert mem_msg2.summary_node_id is not None

    # The active message should NOT be compacted (no summary_node_id)
    assert mem_active.summary_node_id is None

    # VERIFY: Database matches memory!
    db_msg1 = await gw.db.get_message(msg1.id)
    db_msg2 = await gw.db.get_message(msg2.id)
    assert db_msg1.summary_node_id == mem_msg1.summary_node_id
    assert db_msg2.summary_node_id == mem_msg2.summary_node_id
