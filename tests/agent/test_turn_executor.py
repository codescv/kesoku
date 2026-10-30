"""Unit tests for TurnExecutor class."""

import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from kesoku.agent.llm import BaseLLM, LLMResponse, MockLLM, ToolCallRequest
from kesoku.agent.turn_executor import (
    EMPTY_RESPONSE_RETRIES,
    TurnExecutor,
    _StreamPreviewPublisher,
    extract_question_options,
    is_clicked_option,
    sanitize_rag_snippet_line,
)
from kesoku.agent.turn_logger import TurnLogger
from kesoku.config import DiscordChannelOverride, GeminiConfig, KesokuConfig, WorkspaceConfig
from kesoku.constants import MessageRole, MessageStatus, MessageType
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
    """Verify that TurnExecutor retries empty responses silently and nudges once retries are exhausted."""
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
            if self.calls <= EMPTY_RESPONSE_RETRIES + 1:
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
    type(worker).running = PropertyMock(side_effect=[True] * (EMPTY_RESPONSE_RETRIES + 2) + [False])

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
    # Exactly one nudge, posted only after the silent retries were exhausted
    assert len(system_msgs) == 1
    assert llm.calls == EMPTY_RESPONSE_RETRIES + 2

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
    """Verify that a pivot inside the loop resets turn metrics and the empty-response counter."""
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
            # One empty response for the first prompt, then EMPTY_RESPONSE_RETRIES + 1 empty
            # responses after the pivot. Since the pivot resets the counter, the nudge must be
            # posted on the last of those and never earlier.
            if self.calls <= EMPTY_RESPONSE_RETRIES + 2:
                return LLMResponse(content="", thought=f"Empty on call {self.calls}")
            return LLMResponse(content="Pivoted response success!")

    llm = NudgeAndPivotLLM()
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    turn_logger = MagicMock(spec=TurnLogger)
    context = KesokuContext(config=cfg, llm=llm)
    executor = TurnExecutor("sess_pivot", gw, tool_runner, turn_logger, context=context)

    # Configure mock worker that returns msg1 in first loop, but pivots to msg2 in second loop
    worker = MagicMock()
    type(worker).running = PropertyMock(side_effect=[True] * (EMPTY_RESPONSE_RETRIES + 3) + [False])

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
    # 1. First prompt, whose single empty response is only silently retried
    # 2. Pivoted prompt, whose empty responses exhaust the retries and earn one nudge (parent is msg2)
    # 3. Success response to pivoted prompt (parent is msg2)

    nudges = [m for m in history if m.role == "system" and "empty content" in m.content]
    assert len(nudges) == 1
    assert nudges[0].parent_id == msg2.id

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
    assert "memory_search(query)" in content1
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

    # MUST NOT contain Sync Guidelines, but MUST contain Preferences
    assert '<background_context type="sync_guidelines">' not in content2
    assert "memory_search(query)" not in content2
    assert "User Preferences:" not in content2
    assert "<instructions>\nPython\n</instructions>" in content2
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
    assert "memory_search(query)" in content3
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

    # MUST NOT contain Sync Guidelines, but MUST contain Preferences
    assert '<background_context type="sync_guidelines">' not in content4
    assert "<instructions>\nPython\n</instructions>" in content4

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
    """Verify scaffold assembly from existing summary nodes and post-turn HistoryCompressor execution."""
    from kesoku.constants import MessageRole
    from kesoku.db import SummaryNode

    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.create_session("sess_auto_compact", title="Auto Compact Session")

    # Turn 1 (uncompressed, protected by protect_front_turns=1)
    msg1 = Message(
        session_id="sess_auto_compact",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role=MessageRole.USER,
        content="Turn 1 user",
        status="processed",
        timestamp=time.time() - 120,
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
        timestamp=time.time() - 110,
    )
    await gw.post(msg2)

    # Previously compacted middle turn (compacted at the end of an earlier turn)
    msg_comp_u = Message(
        session_id="sess_auto_compact",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role=MessageRole.USER,
        content="Compacted turn user",
        status="processed",
        timestamp=time.time() - 100,
    )
    await gw.post(msg_comp_u)
    msg_comp_a = Message(
        session_id="sess_auto_compact",
        chatbot_id="cli",
        channel_id="ch1",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        content="Compacted turn reply",
        status="delivered",
        timestamp=time.time() - 95,
    )
    await gw.post(msg_comp_a)

    node = SummaryNode(
        id="node-1-uuid",
        session_id="sess_auto_compact",
        level=0,
        summary="Simulated summary node content",
        start_timestamp=time.time() - 100,
        end_timestamp=time.time() - 95,
        token_count=10,
        source_token_count=100,
        parent_id=None,
    )
    await gw.db.insert_summary_node(node)
    await gw.db.update_messages_summary_node([msg_comp_u.id, msg_comp_a.id], "node-1-uuid")

    # Uncompressed recent Turn 2 + active Turn 3
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

    compact_call_state: dict[str, Any] = {}

    async def mock_auto_compact(
        session_id: str, history: list[Message], llm: Any, config: Any, *args: Any, **kwargs: Any
    ) -> bool:
        db_at_call = await gw.db.get_session_history(session_id, limit=0)
        compact_call_state["final_already_saved"] = any(
            m.role == MessageRole.ASSISTANT and m.content == "Final Assistant Turn Response"
            for m in db_at_call
        )
        compact_call_state["has_scaffold"] = any(
            isinstance(m.metadata, dict) and (m.metadata.get("is_scaffold") or m.metadata.get("is_scaffold_ack"))
            for m in history
        )
        compact_call_state["raw_history_ids"] = [m.id for m in history]
        return False

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

    # Post-turn compaction ran once AFTER final assistant response was persisted, with raw non-scaffold history
    mock_compact.assert_awaited_once()
    assert compact_call_state["final_already_saved"] is True
    assert compact_call_state["has_scaffold"] is False
    # Slice A + Slice B integration: active_only=True kept compressed USER anchor, skipped compressed ASSISTANT row
    assert msg_comp_u.id in compact_call_state["raw_history_ids"]
    assert msg_comp_a.id not in compact_call_state["raw_history_ids"]

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
    assert msg_comp_u.id in db_ids
    assert msg_comp_a.id in db_ids
    assert msg3.id in db_ids
    assert msg4.id in db_ids
    assert active_msg.id in db_ids


@pytest.mark.asyncio
async def test_turn_executor_context_caching_with_compaction(temp_db: str) -> None:
    """Verify explicit context cache is reused across tool iterations, then deleted post-turn when compaction occurs."""
    from kesoku.config import GeminiConfig
    from kesoku.constants import MessageRole, MessageType
    from kesoku.db import SummaryNode

    DatabaseManager(temp_db).init_tables()

    cfg = KesokuConfig(
        workspace=WorkspaceConfig(db_path=temp_db),
        gemini=GeminiConfig(
            context_caching=True,
            context_caching_threshold=100,
        ),
    )

    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.create_session("sess_cache_compact", title="Cache and Compact")

    msg1 = Message(
        session_id="sess_cache_compact",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role=MessageRole.USER,
        content="User message 1 that is quite long to accumulate tokens for caching " * 10,
        status="processed",
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
        status="delivered",
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

    class GeminiLLM(BaseLLM):
        context_window_limit: int = 1000

        def __init__(self) -> None:
            self.created_caches: list[dict[str, Any]] = []
            self.deleted_caches: list[str] = []
            self.captured_generates: list[dict[str, Any]] = []

        def count_tokens(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
        ) -> int:
            return 150 if len(self.created_caches) == 0 else 50

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
            return LLMResponse(content="Final response after tool execution", total_tokens=10)

    llm = GeminiLLM()
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = ["dummy_tool"]

    async def mock_execute_tool(call: Any, tc_msg: Message, **kwargs: Any) -> Message:
        return Message(
            id=f"tool_res_{time.time()}",
            session_id=tc_msg.session_id,
            chatbot_id=tc_msg.chatbot_id,
            channel_id=tc_msg.channel_id,
            sender="System",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_RESULT,
            content="Tool output content",
            status="responded",
            parent_id=tc_msg.id,
        )

    tool_runner.execute_tool.side_effect = mock_execute_tool
    tool_runner.tool_context = MagicMock()
    tool_runner.tool_context.transitioned_to_session = None

    turn_logger = MagicMock(spec=TurnLogger)
    compaction_calls = 0

    async def mock_auto_compact(
        session_id: str, history: list[Message], llm: Any, config: Any, *args: Any, **kwargs: Any
    ) -> bool:
        nonlocal compaction_calls
        compaction_calls += 1
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

    context = KesokuContext(config=cfg, llm=llm)
    executor = TurnExecutor("sess_cache_compact", gw, tool_runner, turn_logger, context=context)

    worker = MagicMock()
    worker.active_cache_name = None
    worker.active_cache_llm = None
    worker.cached_messages_len = 0
    type(worker).running = PropertyMock(side_effect=[True, True, False])

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
            session_staging_dir="/tmp/sess_cache_compact",
        )

    # Cache created once on iteration 1 and reused across iteration 2 (no mid-turn compaction!)
    assert len(llm.created_caches) == 1
    assert llm.created_caches[0]["name"] == "mock_cache_0"
    assert len(llm.captured_generates) == 2
    assert llm.captured_generates[0]["cached_content"] == "mock_cache_0"
    assert llm.captured_generates[0]["tools"] is None
    assert llm.captured_generates[1]["cached_content"] == "mock_cache_0"

    # Post-turn compaction ran once after final response, invalidated cache, and reset worker cache fields
    mock_compact.assert_awaited_once()
    assert compaction_calls == 1
    assert llm.deleted_caches == ["mock_cache_0"]
    assert worker.active_cache_name is None
    assert worker.active_cache_llm is None
    assert worker.cached_messages_len == 0

    # Follow-up turn sees the newly compacted scaffold without obsolete cache
    followup_msg = Message(
        session_id="sess_cache_compact",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role=MessageRole.USER,
        content="Follow-up query",
        status="pending_agent",
        timestamp=time.time() + 1,
    )
    await gw.post(followup_msg)
    executor2 = TurnExecutor("sess_cache_compact", gw, tool_runner, turn_logger, context=context)
    type(worker).running = PropertyMock(side_effect=[True, False])
    with (
        patch("kesoku.context.get_config", return_value=cfg),
        patch("kesoku.agent.compressor.HistoryCompressor.auto_compact_session", return_value=False),
    ):
        await executor2.process_turn(
            current_msg=followup_msg,
            worker=worker,
            session_staging_dir="/tmp/sess_cache_compact",
        )

    gen_call_3 = llm.captured_generates[2]
    assert gen_call_3["cached_content"] is None
    assert gen_call_3["history"][0].role == MessageRole.USER
    assert "[Note: This conversation uses custom turn-based" in gen_call_3["history"][0].content
    assert "Compacted user query" in gen_call_3["history"][0].content
    assert not any(m.id in (msg1.id, msg2.id) for m in gen_call_3["history"])


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

    # Pre-insert Turn 2 summary node (compacted post-turn at the end of Turn 2)
    from kesoku.db import SummaryNode

    node = SummaryNode(
        id="node-2-uuid",
        session_id="sess_exclude",
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

    seen_scaffold_in_compactor = False

    async def mock_auto_compact(
        session_id: str, history: list[Message], llm: Any, config: Any, *args: Any, **kwargs: Any
    ) -> bool:
        nonlocal seen_scaffold_in_compactor
        seen_scaffold_in_compactor = any(
            isinstance(m.metadata, dict) and (m.metadata.get("is_scaffold") or m.metadata.get("is_scaffold_ack"))
            for m in history
        )
        return False

    context = KesokuContext(config=cfg, llm=llm)
    executor = TurnExecutor("sess_exclude", gw, tool_runner, turn_logger, context=context)

    # protect_front = 1 (protects Turn 1), protect_tail = 1 (protects Turn 3 active)
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

    mock_compact.assert_awaited_once()
    assert seen_scaffold_in_compactor is False

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

    # Assert Turn 2 raw messages are NOT present as raw history; the user's own words survive only
    # verbatim inside the scaffold's <recent_user_messages> block.
    assert all("<recent_user_messages>" in c for c in captured_contents if "Turn 2 user" in c)
    assert any("Turn 2 user" in c for c in captured_contents)
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


@pytest.mark.asyncio
async def test_turn_executor_auto_context_search_injection(temp_db: str, tmp_path: Any, monkeypatch: Any) -> None:
    """Verify that auto context search retrieves and injects progressive disclosure <memory> blocks."""
    from kesoku.agent.turn_executor import TurnExecutor
    from kesoku.constants import MessageRole
    from kesoku.db import DatabaseManager

    DatabaseManager(temp_db).init_tables()

    monkeypatch.setattr("kesoku.utils.embedding.get_embedding", lambda text: [1.0] + [0.0] * 383)
    monkeypatch.setattr(
        "kesoku.utils.embedding.get_embeddings",
        lambda texts: [[1.0] + [0.0] * 383 for _ in texts],
    )

    roles_dir = tmp_path / "roles"
    mem_dir = roles_dir / "role_a" / "memories"
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / "psyduck.md").write_text(
        "---\ndescription: 可达鸭头越痛念力越强！\n---\n- 念力攻击说明\n",
        encoding="utf-8",
    )

    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db, roles_dir=str(roles_dir)))
    cfg.agent.auto_context_search = True
    cfg.agent.auto_context_search_semantic_min = 0.5

    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.db.set_channel_role("cli", "ch1", "role_a")

    # Active session
    await gw.create_session("sess_active", title="Active Session", role="role_a")
    await gw.db.set_active_session_for_channel("cli", "ch1", "sess_active")
    active_msg = Message(
        session_id="sess_active",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role=MessageRole.USER,
        content="你刚才怎么变成可达鸭了？",
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

    context = KesokuContext(config=cfg, llm=llm)
    executor = TurnExecutor("sess_active", gw, tool_runner, turn_logger, context=context)

    worker = MagicMock()
    type(worker).running = PropertyMock(side_effect=[True, False])

    async def mock_pivot(m: Message) -> Message:
        return m

    worker.drain_queue_and_pivot.side_effect = mock_pivot
    worker.queue_empty.return_value = True

    with (
        patch("kesoku.context.get_config", return_value=cfg),
        patch("kesoku.utils.path.get_config", return_value=cfg),
    ):
        await executor.process_turn(
            current_msg=active_msg,
            worker=worker,
            session_staging_dir="/tmp/sess_active",
        )

    captured_contents = [m.content for m in llm.captured_history]
    # Verify <memory> progressive disclosure block was injected with $MEMORY_DIR path and description
    assert any("<memory>" in c for c in captured_contents)
    assert any("$MEMORY_DIR/psyduck.md" in c for c in captured_contents)
    assert any("可达鸭" in c for c in captured_contents)


@pytest.mark.asyncio
async def test_turn_executor_daily_facts_injection(temp_db: str, tmp_path: Any) -> None:
    """Verify that legacy facts.md is no longer injected into user messages."""
    from kesoku.agent.turn_executor import TurnExecutor
    from kesoku.constants import MessageRole
    from kesoku.db import DatabaseManager

    DatabaseManager(temp_db).init_tables()

    roles_dir = tmp_path / "roles"
    role_dir = roles_dir / "role_a"
    role_dir.mkdir(parents=True, exist_ok=True)
    (role_dir / "preferences.md").write_text("Role A preferences")
    (role_dir / "facts.md").write_text("Legacy facts content that should not be injected")

    cfg = KesokuConfig(
        workspace=WorkspaceConfig(db_path=temp_db, roles_dir=str(roles_dir)),
    )
    cfg.agent.auto_context_search = False

    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.db.set_channel_role("cli", "ch1", "role_a")
    await gw.create_session("sess_facts", title="Facts Session", role="role_a")
    await gw.db.set_active_session_for_channel("cli", "ch1", "sess_facts")

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

    context = KesokuContext(config=cfg, llm=llm)
    executor = TurnExecutor("sess_facts", gw, tool_runner, turn_logger, context=context)

    (tmp_path / "staging").mkdir(parents=True, exist_ok=True)
    w = MagicMock()
    type(w).running = PropertyMock(side_effect=[True, False])

    async def mock_pivot(m: Message) -> Message:
        return m

    w.drain_queue_and_pivot.side_effect = mock_pivot
    w.queue_empty.return_value = True

    msg_day1_1 = Message(
        session_id="sess_facts",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role=MessageRole.USER,
        content="早上好！",
        status="pending_agent",
        timestamp=1786400000.0,
    )
    await gw.post(msg_day1_1)

    with patch("kesoku.context.get_config", return_value=cfg):
        await executor.process_turn(
            current_msg=msg_day1_1,
            worker=w,
            session_staging_dir=str(tmp_path / "staging"),
        )

    captured_day1_1 = [m.content for m in llm.captured_history]
    assert not any("<facts>" in c for c in captured_day1_1)
    assert not any("Legacy facts content" in c for c in captured_day1_1)


@pytest.mark.asyncio
async def test_turn_executor_session_notes_injection_and_tool(temp_db: str, tmp_path: Any) -> None:
    """update_file edits notes.md incrementally; TurnExecutor injects notes and context/*.md blocks every turn."""
    from kesoku.agent.tools.file import update_file
    from kesoku.agent.turn_executor import TurnExecutor
    from kesoku.constants import MessageRole
    from kesoku.db import DatabaseManager

    DatabaseManager(temp_db).init_tables()

    roles_dir = tmp_path / "roles"
    asuka_dir = roles_dir / "asuka"
    asuka_dir.mkdir(parents=True, exist_ok=True)
    (asuka_dir / "preferences.md").write_text("Asuka preferences")

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = sessions_dir / "sess_notes"
    staging_dir.mkdir(parents=True, exist_ok=True)

    cfg = KesokuConfig(
        workspace=WorkspaceConfig(
            db_path=temp_db,
            roles_dir=str(roles_dir),
            sessions_dir=str(sessions_dir),
        ),
    )
    cfg.agent.auto_context_search = False

    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.db.set_channel_role("cli", "ch1", "asuka")
    await gw.create_session("sess_notes", title="Notes Session")
    await gw.db.set_active_session_for_channel("cli", "ch1", "sess_notes")

    # 1. Create notes.md with update_file (old_content=None only for first creation)
    notes_path = str(staging_dir / "notes.md")
    with patch("kesoku.utils.path.get_config", return_value=cfg):
        res_str = update_file(
            notes_path,
            None,
            "- 本session只用纯日语交流\n- 当前场景：镰仓海边散步",
        )
    assert res_str.startswith("Success"), res_str
    assert (staging_dir / "notes.md").exists()

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
            return LLMResponse(content="了解！", total_tokens=10)

    llm = CaptureLLM()
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    turn_logger = MagicMock(spec=TurnLogger)

    context = KesokuContext(config=cfg, llm=llm)
    executor = TurnExecutor("sess_notes", gw, tool_runner, turn_logger, context=context)

    def create_mock_worker() -> MagicMock:
        w = MagicMock()
        type(w).running = PropertyMock(side_effect=[True, False])

        async def mock_pivot(m: Message) -> Message:
            return m

        w.drain_queue_and_pivot.side_effect = mock_pivot
        w.queue_empty.return_value = True
        return w

    # 2. Turn 1: Verify notes are injected as a notes context block
    t1 = 1786400000.0
    msg_1 = Message(
        session_id="sess_notes",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role=MessageRole.USER,
        content="我们去买冰淇淋吧",
        status="pending_agent",
        timestamp=t1,
    )
    await gw.post(msg_1)

    with patch("kesoku.context.get_config", return_value=cfg):
        await executor.process_turn(
            current_msg=msg_1,
            worker=create_mock_worker(),
            session_staging_dir=str(staging_dir),
        )

    last_user_1 = [m.content for m in llm.captured_history if m.role == MessageRole.USER][-1]
    assert '<context name="notes">\n- 本session只用纯日语交流\n- 当前场景：镰仓海边散步\n</context>' in last_user_1
    assert "<session_notes>" not in last_user_1

    # 3. Turn 2 (same day): edit ONLY the changed line; the unchanged rule must survive
    with patch("kesoku.utils.path.get_config", return_value=cfg):
        res_edit = update_file(notes_path, "- 当前场景：镰仓海边散步", "- 当前场景：江之电车厢内")
    assert res_edit.startswith("Success"), res_edit

    msg_2 = Message(
        session_id="sess_notes",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role=MessageRole.USER,
        content="现在去哪？",
        status="pending_agent",
        timestamp=t1 + 60.0,
    )
    await gw.post(msg_2)

    with patch("kesoku.context.get_config", return_value=cfg):
        await executor.process_turn(
            current_msg=msg_2,
            worker=create_mock_worker(),
            session_staging_dir=str(staging_dir),
        )

    last_user_2 = [m.content for m in llm.captured_history if m.role == MessageRole.USER][-1]
    assert '<context name="notes">\n- 本session只用纯日语交流\n- 当前场景：江之电车厢内\n</context>' in last_user_2

    # 4. Turn 3: tool-maintained context/*.md files are injected after instructions and notes, sorted by filename.
    context_dir = staging_dir / "context"
    context_dir.mkdir()
    (context_dir / "99-test.md").write_text("今天的暗号是🍊")
    (context_dir / "20-scene.md").write_text("location: beach")
    msg_3 = Message(
        session_id="sess_notes",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role=MessageRole.USER,
        content="今天的暗号是什么",
        status="pending_agent",
        timestamp=t1 + 86400.0,
    )
    await gw.post(msg_3)
    with patch("kesoku.context.get_config", return_value=cfg):
        await executor.process_turn(
            current_msg=msg_3,
            worker=create_mock_worker(),
            session_staging_dir=str(staging_dir),
        )
    last_user_3 = [m.content for m in llm.captured_history if m.role == MessageRole.USER][-1]
    positions = [
        last_user_3.index("<instructions>"),
        last_user_3.index('<context name="notes">'),
        last_user_3.index('<context name="scene">\nlocation: beach\n</context>'),
        last_user_3.index('<context name="test">\n今天的暗号是🍊\n</context>'),
        last_user_3.index("<current_message"),
    ]
    assert positions == sorted(positions)

    # 5. Turn 4: agent.context_dir_max_chars is honoured (files at the end of the sorted list are dropped)
    notes_len = len("- 本session只用纯日语交流\n- 当前场景：江之电车厢内")
    cfg.agent.context_dir_max_chars = notes_len + len("location: beach")
    msg_4 = Message(
        session_id="sess_notes",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role=MessageRole.USER,
        content="再说一次",
        status="pending_agent",
        timestamp=t1 + 86460.0,
    )
    await gw.post(msg_4)
    with patch("kesoku.context.get_config", return_value=cfg):
        await executor.process_turn(
            current_msg=msg_4,
            worker=create_mock_worker(),
            session_staging_dir=str(staging_dir),
        )
    last_user_4 = [m.content for m in llm.captured_history if m.role == MessageRole.USER][-1]
    assert '<context name="notes">' in last_user_4
    assert '<context name="scene">' in last_user_4
    assert '<context name="test">' not in last_user_4



@pytest.mark.asyncio
async def test_turn_executor_implicit_caching_mode_skips_explicit_cache(temp_db: str) -> None:
    """With context_caching_mode='implicit', no explicit cache is created and no cached_content is passed."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    cfg.gemini.context_caching = True
    cfg.gemini.context_caching_threshold = 10
    cfg.gemini.context_caching_mode = "implicit"

    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.create_session("sess_implicit", title="Implicit Cache Session")
    for content, status in [("Long prompt prefix.", "responded"), ("Current message.", "pending_agent")]:
        msg = Message(
            session_id="sess_implicit", chatbot_id="cli", channel_id="ch1", sender="u1",
            role="user", content=content, status=status,
        )
        await gw.post(msg)

    class GeminiLLM(BaseLLM):
        def __init__(self) -> None:
            self.created: list[str] = []
            self.cached_contents: list[str | None] = []

        def count_tokens(self, prompt=None, system_prompt=None, history=None, tools=None) -> int:
            return 100

        async def create_cache(self, *args: Any, **kwargs: Any) -> str | None:
            self.created.append("cache")
            return "mock_cache"

        async def delete_cache(self, cache_name: str) -> None:
            pass

        async def generate(self, prompt=None, system_prompt=None, history=None, tools=None,
                           cached_content=None, **kwargs: Any) -> LLMResponse:
            self.cached_contents.append(cached_content)
            return LLMResponse(content="ok", total_tokens=10)

    llm = GeminiLLM()
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    context = KesokuContext(config=cfg, llm=llm)
    executor = TurnExecutor("sess_implicit", gw, tool_runner, MagicMock(spec=TurnLogger), context=context)
    worker = MagicMock()
    worker.active_cache_name = None
    type(worker).running = PropertyMock(side_effect=[True, False])

    async def mock_pivot(m: Message) -> Message:
        return m

    worker.drain_queue_and_pivot.side_effect = mock_pivot
    worker.queue_empty.return_value = True
    current = (await gw.db.get_session_history("sess_implicit"))[-1]

    with patch("kesoku.context.get_config", return_value=cfg):
        await executor.process_turn(current_msg=current, worker=worker, session_staging_dir="/tmp/sess_implicit")

    assert llm.created == []
    assert llm.cached_contents == [None]


# ---------------------------------------------------------------------------
# Clicked-option detection, auto-RAG guard and snippet sanitisation
# ---------------------------------------------------------------------------

LAST_QUESTION_REPLY = (
    "地图摊在桌上。\n"
    "[question: 接下来做什么？ || 秋叶原逛街扫货：挑选情侣穿搭＋顺便淘限定游戏手柄 | 去浅草看花火 | 回家打游戏]"
)
OLD_Q_BLOCK = "[question: 下一步怎么做？ || 像香槟软木塞一样弹射出去 | 冷静地递上一杯热可可]"


def _tmsg(role: MessageRole, content: str, ts: float, session_id: str = "s", msg_id: str | None = None) -> Message:
    """Build a minimal Message for executor tests."""
    kwargs: dict[str, Any] = {"id": msg_id} if msg_id else {}
    return Message(
        session_id=session_id,
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1" if role == MessageRole.USER else "Kesoku",
        role=role,
        type=MessageType.TEXT,
        content=content,
        status=MessageStatus.RESPONDED,
        timestamp=ts,
        **kwargs,
    )


def test_extract_question_options() -> None:
    """Options of the LAST [question:] block are returned stripped; no block gives []."""
    assert extract_question_options(LAST_QUESTION_REPLY) == [
        "秋叶原逛街扫货：挑选情侣穿搭＋顺便淘限定游戏手柄",
        "去浅草看花火",
        "回家打游戏",
    ]
    assert extract_question_options("[question: A? || x | y]\n中间\n[question: B? || p | q | r]") == ["p", "q", "r"]
    assert extract_question_options("没有按钮") == []
    assert extract_question_options("") == []


@pytest.mark.parametrize(
    "user_text", ["秋叶原逛街扫货：挑选情侣穿搭＋顺便淘限定游戏手柄", "  去浅草看花火 ", "2", "选2", "2."]
)
def test_is_clicked_option_positive(user_text: str) -> None:
    """Exact option clicks (whitespace-insensitive) and short numeric picks are clicked options."""
    assert is_clicked_option(user_text, LAST_QUESTION_REPLY)


@pytest.mark.parametrize(
    "user_text", ["可以耐心一点吗，过程都展示出来 不要狂推进度", "去浅草看花火，然后再去吃拉面好不好", "今天去哪？", ""]
)
def test_is_clicked_option_negative(user_text: str) -> None:
    """Free-form text, even overlapping an option, is not a clicked option."""
    assert not is_clicked_option(user_text, LAST_QUESTION_REPLY)


def test_is_clicked_option_without_question_block() -> None:
    """Without a question block in the previous reply a non-numeric message is never a click."""
    assert not is_clicked_option("去浅草看花火", "普通回复，没有按钮")


async def test_rag_snippets_strip_question_blocks() -> None:
    """sanitize_rag_snippet_line never re-injects historical question stems/options."""
    out = sanitize_rag_snippet_line("Agent", f"去神社吧！\n{OLD_Q_BLOCK}")
    assert "去神社吧！" in out
    assert "香槟软木塞" not in out and "下一步怎么做" not in out and "||" not in out


async def test_rag_snippets_truncate_long_agent_lines() -> None:
    """Long Agent lines are truncated; user lines stay readable."""
    agent_out = sanitize_rag_snippet_line("Agent", "像爆米花一样膨胀起来，" * 300)
    user_out = sanitize_rag_snippet_line("User", "今天去哪里玩")
    assert len(agent_out) <= 500
    assert "今天去哪里玩" in user_out


async def _inject_executor(tmp_path: Path, session_role: str | None, channel_role: str | None) -> TurnExecutor:
    """Real DB/gateway executor with roles/role_a/preferences.md and configurable session/channel roles.

    Args:
        tmp_path: pytest temp dir.
        session_role: role_name stored on the session (None keeps the default).
        channel_role: role bound to cli/ch1 (None leaves the channel without a role row).
    """
    db_path = str(tmp_path / "k.db")
    DatabaseManager(db_path).init_tables()
    roles_dir = tmp_path / "roles"
    (roles_dir / "role_a").mkdir(parents=True)
    (roles_dir / "role_a" / "preferences.md").write_text("ROLE_A_PREFS", encoding="utf-8")
    cfg = KesokuConfig(
        workspace=WorkspaceConfig(db_path=db_path, roles_dir=str(roles_dir), sessions_dir=str(tmp_path / "sessions"))
    )
    cfg.agent_working_dir = str(tmp_path)
    cfg.agent.auto_context_search = True
    gw = Gateway(context=KesokuContext(config=cfg))
    if channel_role:
        await gw.db.set_channel_role("cli", "ch1", channel_role)
    await gw.create_session("sessinj", title="inject", role=session_role)
    await gw.db.set_active_session_for_channel("cli", "ch1", "sessinj")
    return TurnExecutor("sessinj", gw, MagicMock(), MagicMock(), context=KesokuContext(config=cfg))


def _inject_history(user_text: str) -> list[Message]:
    """Previous question turn followed by the current user message."""
    now = time.time()
    return [
        _tmsg(MessageRole.USER, "今天做什么？", now - 120, session_id="sessinj", msg_id="h_u1"),
        _tmsg(MessageRole.ASSISTANT, LAST_QUESTION_REPLY, now - 100, session_id="sessinj", msg_id="h_a1"),
        _tmsg(MessageRole.USER, user_text, now, session_id="sessinj", msg_id="h_u2"),
    ]


async def _run_inject(ex: TurnExecutor, tmp_path: Path, user_text: str, rag: AsyncMock) -> Message | None:
    """Run _inject_context_and_trigger_consolidation with a patched auto-RAG retriever."""
    history = _inject_history(user_text)
    with patch.object(ex, "_retrieve_auto_search_context", rag):
        return await ex._inject_context_and_trigger_consolidation(
            history, history[-1], MagicMock(), session_staging_dir=str(tmp_path / "staging")
        )


async def test_inject_skips_auto_rag_for_clicked_option(tmp_path: Path) -> None:
    """Clicking a button from the previous reply does not trigger auto-RAG."""
    ex = await _inject_executor(tmp_path, session_role=None, channel_role="role_a")
    rag = AsyncMock(return_value="<relevant_historical_context>OLD</relevant_historical_context>\n")
    out = await _run_inject(ex, tmp_path, "秋叶原逛街扫货：挑选情侣穿搭＋顺便淘限定游戏手柄", rag)
    rag.assert_not_called()
    assert out is not None and "OLD" not in out.content


async def test_inject_skips_auto_rag_for_numeric_pick(tmp_path: Path) -> None:
    """A short numeric pick ('选2') is treated like a click."""
    ex = await _inject_executor(tmp_path, session_role=None, channel_role="role_a")
    rag = AsyncMock(return_value="<relevant_historical_context>OLD</relevant_historical_context>\n")
    await _run_inject(ex, tmp_path, "选2", rag)
    rag.assert_not_called()


async def test_inject_runs_auto_rag_for_free_text(tmp_path: Path) -> None:
    """Free-form messages still get auto-RAG context."""
    ex = await _inject_executor(tmp_path, session_role=None, channel_role="role_a")
    rag = AsyncMock(return_value="<relevant_historical_context>OLD</relevant_historical_context>\n")
    out = await _run_inject(ex, tmp_path, "还记得上次我们在浅草吃的那家鲷鱼烧吗", rag)
    rag.assert_called_once()
    assert out is not None and "OLD" in out.content


async def test_inject_uses_session_role_name_without_channel_role(tmp_path: Path) -> None:
    """Thread sessions with role_name set but no channel role row still get the role's preferences."""
    ex = await _inject_executor(tmp_path, session_role="role_a", channel_role=None)
    out = await _run_inject(ex, tmp_path, "还记得上次我们在浅草吃的那家鲷鱼烧吗", AsyncMock(return_value=""))
    assert out is not None
    assert "ROLE_A_PREFS" in out.content
    assert out.content.index("ROLE_A_PREFS") < out.content.index("<current_message")


async def test_inject_default_session_role_falls_back_to_channel(tmp_path: Path) -> None:
    """A 'default' session role defers to the channel role."""
    ex = await _inject_executor(tmp_path, session_role=None, channel_role="role_a")
    out = await _run_inject(ex, tmp_path, "还记得上次我们在浅草吃的那家鲷鱼烧吗", AsyncMock(return_value=""))
    assert out is not None and "ROLE_A_PREFS" in out.content


UI_REPLY = "好的。\n```ui\ntype: select\nprompt: 去哪？\noptions:\n  - 海边 [远]\n  - 公园\n```"
UI_MULTI_REPLY = "```ui\ntype: multiselect\nprompt: 带什么？\noptions: [水, 地图, 雨伞]\n```"


def test_extract_question_options_prefers_ui_block() -> None:
    """The first ```ui block provides the interactive options."""
    assert extract_question_options(UI_REPLY) == ["海边 [远]", "公园"]
    assert extract_question_options(f"[question: A? || x | y]\n{UI_REPLY}") == ["海边 [远]", "公园"]


@pytest.mark.parametrize(
    ("user_text", "reply", "expected"),
    [
        ("海边 [远]", UI_REPLY, True),
        ("公园", UI_REPLY, True),
        ("公园然后去海边", UI_REPLY, False),
        ("选择：水；雨伞", UI_MULTI_REPLY, True),
        ("选择：水", UI_MULTI_REPLY, True),
        ("选择：水；火", UI_MULTI_REPLY, False),
        ("选择：水；雨伞", UI_REPLY, False),
    ],
)
def test_is_clicked_option_ui_blocks(user_text: str, reply: str, expected: bool) -> None:
    """Clicks on ```ui select options and multiselect answers are detected."""
    assert is_clicked_option(user_text, reply) is expected


def test_sanitize_rag_snippet_line_collapses_ui_block() -> None:
    """Retrieved snippets lose ```ui options and fences, keeping only the plain prompt."""
    out = sanitize_rag_snippet_line("Agent", UI_REPLY)
    assert out == "好的。\n去哪？"


@pytest.mark.asyncio
async def test_post_turn_compaction_safety_net_on_queue_or_failure(temp_db: str) -> None:
    """Verify `_compaction_pending` triggers pre-turn compaction on the next turn after skip or failure."""
    from kesoku.constants import MessageRole

    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.create_session("sess_fallback", title="Fallback Compaction")

    msg1 = Message(
        session_id="sess_fallback",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role=MessageRole.USER,
        content="Hello 1",
        status="pending_agent",
        timestamp=time.time() - 10,
    )
    await gw.post(msg1)

    events: list[str] = []

    class OrderLLM(BaseLLM):
        async def generate(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
            cached_content: str | None = None,
        ) -> LLMResponse:
            events.append("generate")
            return LLMResponse(content="Answer", total_tokens=5)

    llm = OrderLLM()
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    context = KesokuContext(config=cfg, llm=llm)

    worker = MagicMock()
    type(worker).running = PropertyMock(side_effect=[True, False])

    async def mock_pivot(m: Message) -> Message:
        return m

    worker.drain_queue_and_pivot.side_effect = mock_pivot
    # Non-empty queue at turn end -> skip post-turn compaction and set `_compaction_pending`
    worker.queue_empty.return_value = False

    exec1 = TurnExecutor("sess_fallback", gw, tool_runner, None, context=context)
    with patch(
        "kesoku.agent.compressor.HistoryCompressor.auto_compact_session",
        new_callable=AsyncMock,
    ) as mock_compact_1:
        await exec1.process_turn(msg1, worker, "/tmp/sess_fallback")
        mock_compact_1.assert_not_awaited()
        assert worker._compaction_pending is True

    # Next turn on a fresh TurnExecutor sharing `worker`: runs pre-turn compaction BEFORE `generate`,
    # then post-turn compaction raises -> turn still succeeds and sets `_compaction_pending` again!
    msg2 = Message(
        session_id="sess_fallback",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role=MessageRole.USER,
        content="Hello 2",
        status="pending_agent",
        timestamp=time.time(),
    )
    await gw.post(msg2)
    worker.queue_empty.return_value = True
    type(worker).running = PropertyMock(side_effect=[True, False])

    call_idx = 0

    async def compact_side_effect(*args: Any, **kwargs: Any) -> bool:
        nonlocal call_idx
        call_idx += 1
        if call_idx == 1:
            events.append("pre_compact")
            return False
        events.append("post_compact_fail")
        raise RuntimeError("transient compaction failure")

    exec2 = TurnExecutor("sess_fallback", gw, tool_runner, None, context=context)
    with patch(
        "kesoku.agent.compressor.HistoryCompressor.auto_compact_session",
        side_effect=compact_side_effect,
    ):
        await exec2.process_turn(msg2, worker, "/tmp/sess_fallback")

    assert events == ["generate", "pre_compact", "generate", "post_compact_fail"]
    assert worker._compaction_pending is True


def test_resolve_channel_override_llm_and_stream_precedence(temp_db: str) -> None:
    """Verify `_resolve_channel_override`, `_resolve_llm` minimal kwargs, and `_resolve_stream` precedence."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(
        workspace=WorkspaceConfig(db_path=temp_db),
        gemini=GeminiConfig(stream=False),
    )
    cfg.discord.channels = [
        DiscordChannelOverride(channels=["stream-only"], stream=True),
        DiscordChannelOverride(channels=["think-only"], thinking_level="low"),
        DiscordChannelOverride(
            channels=["lite"],
            llm="gemini",
            model_name="gemini-3.5-flash-lite",
            thinking_level="low",
            stream=False,
        ),
    ]
    ctx = KesokuContext(config=cfg)
    gw = Gateway(context=ctx)
    executor = TurnExecutor("sess_res", gw, MagicMock(), None, context=ctx)

    # Thread inheriting from parent_channel_name="stream-only" even when override.llm is None
    msg_stream_thread = Message(
        session_id="sess_res",
        chatbot_id="discord",
        channel_id="thread-999",
        sender="u1",
        role="user",
        content="hi",
        metadata={"channel_name": "topic", "parent_channel_name": "stream-only"},
    )
    ov = executor._resolve_channel_override(msg_stream_thread)
    assert ov is not None and ov.channels == ["stream-only"]

    # Without registered stream handler on Gateway, `_resolve_stream` returns False even if override.stream=True
    dummy_llm = MagicMock()
    dummy_llm.config = GeminiConfig(stream=True)
    assert executor._resolve_stream(msg_stream_thread, dummy_llm) is False

    # Register Discord stream handler; CLI still has no handler
    gw.register_stream_handler("discord", AsyncMock())
    assert executor._resolve_stream(msg_stream_thread, MagicMock(config=GeminiConfig(stream=False))) is True

    msg_cli = Message(
        session_id="sess_res",
        chatbot_id="cli",
        channel_id="ch",
        sender="u1",
        role="user",
        content="hi",
    )
    assert executor._resolve_stream(msg_cli, dummy_llm) is False

    # Override stream=False beats provider stream=True
    msg_lite = Message(
        session_id="sess_res",
        chatbot_id="discord",
        channel_id="lite",
        sender="u1",
        role="user",
        content="hi",
    )
    assert executor._resolve_stream(msg_lite, dummy_llm) is False

    # MockLLM (no .config) defaults to False when override.stream is None
    msg_think = Message(
        session_id="sess_res",
        chatbot_id="discord",
        channel_id="think-only",
        sender="u1",
        role="user",
        content="hi",
    )
    assert executor._resolve_stream(msg_think, MockLLM()) is False

    # `_resolve_llm` passes ONLY non-None override kwargs
    with patch.object(ctx, "get_llm", return_value=MockLLM()) as mock_get_llm:
        executor._resolve_llm(msg_stream_thread)
        mock_get_llm.assert_called_once_with()

        mock_get_llm.reset_mock()
        executor._resolve_llm(msg_think)
        mock_get_llm.assert_called_once_with(thinking_level="low")

        mock_get_llm.reset_mock()
        executor._resolve_llm(msg_lite)
        mock_get_llm.assert_called_once_with(
            provider="gemini",
            model_name="gemini-3.5-flash-lite",
            thinking_level="low",
        )


@pytest.mark.asyncio
async def test_stream_preview_publisher_throttling_inflight_and_discard(temp_db: str) -> None:
    """Verify `_StreamPreviewPublisher` publishes first chunk immediately, throttles, skips while busy, and discards."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    published: list[str | None] = []
    block_gate = asyncio.Event()
    block_gate.set()

    async def handler(channel_id: str, session_id: str, turn_id: str, text: str | None) -> None:
        await block_gate.wait()
        published.append(text)

    gw.register_stream_handler("discord", handler)

    now = 100.0
    pub = _StreamPreviewPublisher(
        gateway=gw,
        chatbot_id="discord",
        channel_id="ch1",
        session_id="sess1",
        turn_id="turn1",
        min_interval=1.5,
        clock=lambda: now,
    )

    # First chunk publishes immediately
    await pub.on_chunk("Hel", "Hel")
    assert published == ["Hel"]

    # Within 1.5s -> throttled
    now = 100.8
    await pub.on_chunk("lo", "Hello")
    assert published == ["Hel"]

    # Past 1.5s -> publishes; simulate slow Discord 429 backoff on this task
    block_gate.clear()
    now = 102.0
    await pub.on_chunk(" world", "Hello world")
    # In-flight task hasn't finished yet, and `on_chunk` did NOT block!
    assert published == ["Hel"]

    # Even past another 1.5s, while in-flight task is still running, chunks are skipped cleanly
    now = 105.0
    await pub.on_chunk("!", "Hello world!")
    assert published == ["Hel"]

    # Unblock slow edit and discard -> finishes in-flight edit first, then sends None once
    block_gate.set()
    await pub.discard()
    await pub.discard()  # Idempotent
    assert published == ["Hel", "Hello world", None]


@pytest.mark.asyncio
async def test_process_turn_streaming_tool_discard_and_final_thought_metadata(temp_db: str) -> None:
    """Verify streaming turn discards preview on tool calls and tags only final THOUGHT with `final_response=True`."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    cfg.discord.channels = [DiscordChannelOverride(channels=["flash"], stream=True)]
    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.create_session("sess_stream_turn", title="Streaming Turn")

    stream_events: list[str | None] = []

    async def capture_stream(ch: str, sess: str, turn: str, text: str | None) -> None:
        stream_events.append(text)

    gw.register_stream_handler("discord", capture_stream)

    user_msg = Message(
        session_id="sess_stream_turn",
        chatbot_id="discord",
        channel_id="flash",
        sender="u1",
        role=MessageRole.USER,
        content="Run check",
        status="pending_agent",
    )
    await gw.post(user_msg)

    class StreamingTestLLM(BaseLLM):
        def __init__(self) -> None:
            self.call_num = 0

        async def generate(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
            cached_content: str | None = None,
            stream: bool = False,
            on_stream_chunk: Any = None,
        ) -> LLMResponse:
            self.call_num += 1
            assert stream is True
            assert on_stream_chunk is not None
            if self.call_num == 1:
                await on_stream_chunk("Let me check...", "Let me check...")
                return LLMResponse(
                    content="Let me check...",
                    thought="Tool planning thought",
                    tool_calls=[ToolCallRequest(name="ping", arguments={}, tool_call_id="tc_1")],
                    total_tokens=10,
                )
            await on_stream_chunk("All good!", "All good!")
            return LLMResponse(
                content="All good!",
                thought="Final synthesis thought",
                total_tokens=10,
            )

    llm = StreamingTestLLM()
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []

    async def exec_tool(call: Any, tc_msg: Message, **kwargs: Any) -> Message:
        # At tool execution time, the iteration-1 preview must already have been discarded (`None`)
        assert stream_events == ["Let me check...", None]
        return Message(
            session_id=tc_msg.session_id,
            chatbot_id=tc_msg.chatbot_id,
            channel_id=tc_msg.channel_id,
            sender="ping",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_RESULT,
            content="pong",
            status="responded",
            parent_id=tc_msg.id,
        )

    tool_runner.execute_tool.side_effect = exec_tool
    tool_runner.tool_context = MagicMock(transitioned_to_session=None)

    ctx = KesokuContext(config=cfg, llm=llm)
    executor = TurnExecutor("sess_stream_turn", gw, tool_runner, None, context=ctx)
    worker = MagicMock()
    type(worker).running = PropertyMock(side_effect=[True, True, False])

    async def mock_pivot(m: Message) -> Message:
        return m

    worker.drain_queue_and_pivot.side_effect = mock_pivot
    worker.queue_empty.return_value = True

    with patch("kesoku.agent.compressor.HistoryCompressor.auto_compact_session", return_value=False):
        await executor.process_turn(user_msg, worker, "/tmp/sess_stream_turn")

    # Iteration 1 published + discarded (`None`); Iteration 2 published final preview without discarding
    assert stream_events == ["Let me check...", None, "All good!"]

    db_msgs = await gw.db.get_session_history("sess_stream_turn", limit=0)
    thoughts = [m for m in db_msgs if m.role == MessageRole.ASSISTANT and m.type == MessageType.THOUGHT]
    assert len(thoughts) == 2
    assert thoughts[0].content == "Tool planning thought"
    assert thoughts[0].metadata.get("final_response") is None
    assert thoughts[1].content == "Final synthesis thought"
    assert thoughts[1].metadata.get("final_response") is True

