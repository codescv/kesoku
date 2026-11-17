"""Unit tests for TurnExecutor class."""

import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from kesoku.agent.compressor import HistoryCompressor
from kesoku.agent.llm import BaseLLM, GeminiLLM, LLMResponse, MockLLM, ToolCallRequest
from kesoku.agent.tools import ActiveJobsRegistry
from kesoku.agent.tools.registry import ToolContext
from kesoku.agent.turn_executor import (
    EMPTY_RESPONSE_RETRIES,
    TurnExecutor,
    _StreamPreviewPublisher,
    extract_question_options,
    is_clicked_option,
    sanitize_rag_snippet_line,
)
from kesoku.agent.turn_logger import TurnLogger
from kesoku.config import (
    DiscordChannelOverride,
    DiscordConfig,
    GeminiConfig,
    KesokuConfig,
    WorkspaceConfig,
)
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager, Message, SummaryNode
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

    # Turn 0: initial turn runs uncached on R1, then creates `mock_cache_0` post-turn
    msg1 = Message(
        session_id="sess_cache_compact",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role=MessageRole.USER,
        content="User message 1 that is quite long to accumulate tokens for caching " * 10,
        status="pending_agent",
        timestamp=time.time() - 10,
    )
    await gw.post(msg1)

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
            raise AssertionError("GeminiLLM.count_tokens() RPC must never be called for explicit cache management!")

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
                    content="Assistant response 1 that is also quite long " * 10,
                    total_tokens=10,
                )
            if len(self.captured_generates) == 2:
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
        msg_ids = [m.id for m in history if m.role in (MessageRole.USER, MessageRole.ASSISTANT)][:2]
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
        await gw.db.update_messages_summary_node(msg_ids, "node-1-uuid")
        return True

    context = KesokuContext(config=cfg, llm=llm)
    worker = MagicMock()
    worker.active_cache_name = None
    worker.active_cache_llm = None
    worker.cached_messages_len = 0

    async def mock_pivot(m: Message) -> Message:
        return m

    worker.drain_queue_and_pivot.side_effect = mock_pivot
    worker.queue_empty.return_value = True

    # Run Turn 0: R1 runs uncached (zero blocking create_cache before R1!), then post-turn creates `mock_cache_0`
    executor0 = TurnExecutor("sess_cache_compact", gw, tool_runner, turn_logger, context=context)
    type(worker).running = PropertyMock(side_effect=[True, False])
    with (
        patch("kesoku.context.get_config", return_value=cfg),
        patch("kesoku.agent.compressor.HistoryCompressor.auto_compact_session", return_value=False),
    ):
        await executor0.process_turn(
            current_msg=msg1,
            worker=worker,
            session_staging_dir="/tmp/sess_cache_compact",
        )

    assert llm.captured_generates[0]["cached_content"] is None
    assert len(llm.created_caches) == 1
    assert llm.created_caches[0]["name"] == "mock_cache_0"
    assert worker.active_cache_name == "mock_cache_0"

    # Now Turn 1: reuses `mock_cache_0` across tool iterations R1 & R2, then compacts post-turn
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

    # Raise threshold before Turn 1 post-turn so post-compaction short scaffold stays uncached
    cfg.gemini.context_caching_threshold = 5000
    executor = TurnExecutor("sess_cache_compact", gw, tool_runner, turn_logger, context=context)
    type(worker).running = PropertyMock(side_effect=[True, True, False])

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

    # Cache created post-Turn-0 was reused across both rounds of Turn 1
    assert len(llm.created_caches) == 1
    assert len(llm.captured_generates) == 3
    assert llm.captured_generates[1]["cached_content"] == "mock_cache_0"
    assert llm.captured_generates[1]["tools"] is None
    assert llm.captured_generates[2]["cached_content"] == "mock_cache_0"

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

    gen_call_4 = llm.captured_generates[3]
    assert gen_call_4["cached_content"] is None
    assert gen_call_4["history"][0].role == MessageRole.USER
    assert "[Note: This conversation uses custom turn-based" in gen_call_4["history"][0].content
    assert "Compacted user query" in gen_call_4["history"][0].content
    assert not any(m.id == msg1.id for m in gen_call_4["history"])


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
        status="pending_agent",
    )
    await gw.post(msg1)

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
            raise AssertionError("GeminiLLM.count_tokens() RPC must never be called!")

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
                assert cached_content is None
                return LLMResponse(content="First turn reply", total_tokens=10)
            if len(self.captured_generates) == 2:
                assert cached_content == "mock_cache_0"
                raise RuntimeError("400 INVALID_ARGUMENT. Cache content mock_cache_0 is expired.")
            assert cached_content is None
            return LLMResponse(content="Recovered from expired cache!", total_tokens=10)

    llm = GeminiLLM()
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    turn_logger = MagicMock(spec=TurnLogger)

    context = KesokuContext(config=cfg, llm=llm)
    worker = MagicMock()
    worker.active_cache_name = None
    worker.active_cache_llm = None
    worker.cached_messages_len = 0

    async def mock_pivot(m: Message) -> Message:
        return m

    worker.drain_queue_and_pivot.side_effect = mock_pivot
    worker.queue_empty.return_value = True

    # Turn 1 builds `mock_cache_0` post-turn
    executor1 = TurnExecutor("sess_cache_expire", gw, tool_runner, turn_logger, context=context)
    type(worker).running = PropertyMock(side_effect=[True, False])
    with patch("kesoku.context.get_config", return_value=cfg):
        await executor1.process_turn(
            current_msg=msg1,
            worker=worker,
            session_staging_dir="/tmp/sess_cache_expire",
        )

    assert llm.cache_ttl_passed == 1800
    assert len(llm.created_caches) == 1
    assert llm.created_caches[0]["name"] == "mock_cache_0"

    # Turn 2 attempts `mock_cache_0`, hits expired error, retries same round uncached,
    # and recreates `mock_cache_1` post-turn.
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

    executor2 = TurnExecutor("sess_cache_expire", gw, tool_runner, turn_logger, context=context)
    type(worker).running = PropertyMock(side_effect=[True, False])
    with patch("kesoku.context.get_config", return_value=cfg):
        await executor2.process_turn(
            current_msg=msg2,
            worker=worker,
            session_staging_dir="/tmp/sess_cache_expire",
        )

    assert len(llm.captured_generates) == 3
    assert llm.captured_generates[1]["cached_content"] == "mock_cache_0"
    assert llm.captured_generates[2]["cached_content"] is None
    assert len(llm.created_caches) == 2
    assert worker.active_cache_name == "mock_cache_1"

    history = await gw.db.get_session_history("sess_cache_expire")
    assistant_msgs = [m for m in history if m.role == "assistant"]
    assert len(assistant_msgs) == 2
    assert assistant_msgs[-1].content == "Recovered from expired cache!"
    assert assistant_msgs[-1].status == "pending"


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
    from kesoku.agent.memory_store import sync_role_memories
    from kesoku.agent.turn_executor import TurnExecutor
    from kesoku.constants import MessageRole
    from kesoku.db import DatabaseManager

    db = DatabaseManager(temp_db)
    db.init_tables()

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
    sync_role_memories(db, "role_a", str(mem_dir))

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
    # K4: tool-round thought keeps both the reasoning and the visible round text
    assert thoughts[0].content == "Tool planning thought\n\nLet me check..."
    assert thoughts[0].metadata.get("final_response") is None
    assert thoughts[1].content == "Final synthesis thought"
    assert thoughts[1].metadata.get("final_response") is True


async def test_user_clock_header_gating_elapsed_and_cache_prefix_immutability(tmp_path: Path) -> None:
    """Verify user_clock_file header gating, cronjob/compression skipping, and explicit-cache prefix stability."""
    import datetime as _dt
    from zoneinfo import ZoneInfo

    from kesoku.agent.compressor import SCAFFOLD_HEADER
    from kesoku.agent.history import prepare_history_for_llm
    from kesoku.constants import MessageRole, MessageStatus, MessageType

    ex = await _inject_executor(tmp_path, session_role=None, channel_role="role_a")
    ex.context.config.agent.auto_context_search = False
    ts_cur = _dt.datetime(2026, 11, 9, 21, 29, 58, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
    ts_prev = ts_cur - 31 * 60

    # 1. Default (user_clock_file == ""): header is byte-identical to v0.1.5 (no user_local_* or since_prev_user_msg).
    hist_off = [
        _tmsg(MessageRole.USER, "Earlier hello", ts_prev, session_id="sessinj", msg_id="u_off_1"),
        _tmsg(MessageRole.ASSISTANT, "Hi there", ts_prev + 5, session_id="sessinj", msg_id="a_off_1"),
        _tmsg(MessageRole.USER, "Current question", ts_cur, session_id="sessinj", msg_id="u_off_2"),
    ]
    out_off = await ex._inject_context_and_trigger_consolidation(
        hist_off, hist_off[-1], MagicMock(), session_staging_dir=str(tmp_path / "staging")
    )
    assert out_off is not None
    assert "user_local_time=" not in out_off.content
    assert "user_timezone=" not in out_off.content
    assert "user_place=" not in out_off.content
    assert "since_prev_user_msg=" not in out_off.content
    expected_msg_time = _dt.datetime.fromtimestamp(ts_cur).astimezone().strftime("%Y-%m-%d %H:%M:%S (%A) %Z")
    assert f'time="{expected_msg_time}"' in out_off.content

    # 2. Configured user_clock_file, but file does not exist yet -> still byte-identical to v0.1.5.
    ex.context.config.agent.user_clock_file = "state/user_clock.json"
    hist_missing = [
        _tmsg(MessageRole.USER, "Earlier hello", ts_prev, session_id="sessinj", msg_id="u_m_1"),
        _tmsg(MessageRole.ASSISTANT, "Hi there", ts_prev + 5, session_id="sessinj", msg_id="a_m_1"),
        _tmsg(MessageRole.USER, "Current question", ts_cur, session_id="sessinj", msg_id="u_m_2"),
    ]
    out_missing = await ex._inject_context_and_trigger_consolidation(
        hist_missing, hist_missing[-1], MagicMock(), session_staging_dir=str(tmp_path / "staging")
    )
    assert out_missing is not None
    assert out_missing.content == out_off.content

    # 3. Valid state/user_clock.json (Asia/Kolkata +05:30): adds user_local_*, skips cronjob/system senders.
    clock_path = tmp_path / "state" / "user_clock.json"
    clock_path.parent.mkdir(parents=True, exist_ok=True)
    clock_path.write_text(
        json.dumps(
            {
                "tz": "Asia/Kolkata",
                "place": "新德里",
                "home_tz": "Asia/Shanghai",
                "home_place": "家",
                "expires_at": "2026-11-10T06:00:00+08:00",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    cron_msg = Message(
        id="u_cron",
        session_id="sessinj",
        chatbot_id="cli",
        channel_id="ch1",
        sender="cronjob",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Scheduled morning ping",
        status=MessageStatus.RESPONDED,
        timestamp=ts_cur - 60,
        metadata={"sender_name": "cronjob"},
    )
    hist_ist = [
        _tmsg(MessageRole.USER, "Earlier hello", ts_prev, session_id="sessinj", msg_id="u_ist_1"),
        _tmsg(MessageRole.ASSISTANT, "Hi there", ts_prev + 5, session_id="sessinj", msg_id="a_ist_1"),
        cron_msg,
        _tmsg(MessageRole.ASSISTANT, "Cron reply", ts_cur - 55, session_id="sessinj", msg_id="a_cron"),
        _tmsg(MessageRole.USER, "Boarding gate check", ts_cur, session_id="sessinj", msg_id="u_ist_2"),
    ]
    out_ist = await ex._inject_context_and_trigger_consolidation(
        hist_ist, hist_ist[-1], MagicMock(), session_staging_dir=str(tmp_path / "staging")
    )
    assert out_ist is not None
    assert f'time="{expected_msg_time}"' in out_ist.content
    assert 'user_local_time="2026-11-09 18:59:58 (Monday) IST UTC+05:30"' in out_ist.content
    assert 'user_timezone="Asia/Kolkata"' in out_ist.content
    assert 'user_place="新德里"' in out_ist.content
    assert 'since_prev_user_msg="31m"' in out_ist.content

    # 4. When all turns between protected_head and current_msg were compressed into scaffold, omit since_prev_user_msg.
    scaffold_msg = Message(
        id="scaf_1",
        session_id="sessinj",
        chatbot_id="cli",
        channel_id="ch1",
        sender="system",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content=f"{SCAFFOLD_HEADER}\n[Recent Summary]",
        status=MessageStatus.RESPONDED,
        timestamp=ts_cur - 10,
        metadata={"is_scaffold": True},
    )
    ack_msg = _tmsg(
        MessageRole.ASSISTANT, "Understood.", ts_cur - 9, session_id="sessinj", msg_id="ack_1"
    )
    hist_comp = [
        _tmsg(MessageRole.USER, "Day 1 turn 0", ts_cur - 86400, session_id="sessinj", msg_id="u_t0"),
        _tmsg(MessageRole.ASSISTANT, "Day 1 reply", ts_cur - 86390, session_id="sessinj", msg_id="a_t0"),
        scaffold_msg,
        ack_msg,
        _tmsg(MessageRole.USER, "Fresh turn after compression", ts_cur, session_id="sessinj", msg_id="u_after"),
    ]
    out_comp = await ex._inject_context_and_trigger_consolidation(
        hist_comp, hist_comp[-1], MagicMock(), session_staging_dir=str(tmp_path / "staging")
    )
    assert out_comp is not None
    assert 'user_timezone="Asia/Kolkata"' in out_comp.content
    assert "since_prev_user_msg=" not in out_comp.content

    # 5. Explicit-cache prefix (`prepared_all[:-1]`) stays byte-identical when clock file changes between turns.
    post_raw = [
        _tmsg(MessageRole.USER, "Turn 1 user", ts_prev, session_id="sessinj", msg_id="c_u1"),
        _tmsg(MessageRole.ASSISTANT, "Turn 1 assistant", ts_prev + 5, session_id="sessinj", msg_id="c_a1"),
    ]
    sentinel = Message(
        id="__cache_sentinel__",
        session_id="sessinj",
        chatbot_id="cli",
        channel_id="ch1",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="<current_message>\n</current_message>",
        status=MessageStatus.PENDING_AGENT,
    )
    prepared_before = prepare_history_for_llm([*post_raw, sentinel], staging_dir=str(tmp_path / "staging"))[:-1]

    clock_path.write_text(json.dumps({"tz": "Europe/London", "place": "伦敦"}, ensure_ascii=False), encoding="utf-8")
    prepared_after = prepare_history_for_llm([*post_raw, sentinel], staging_dir=str(tmp_path / "staging"))[:-1]
    assert [m.model_dump() for m in prepared_before] == [m.model_dump() for m in prepared_after]


@pytest.mark.asyncio
async def test_turn_executor_skips_post_turn_explicit_cache_overhead_for_unsupported_model_while_preserving_flash(
    tmp_path: Any,
) -> None:
    """Verify `_manage_post_turn_explicit_cache` skips history prep once a model is marked unsupported."""

    from kesoku.agent.llm import (
        _EXPLICIT_CACHE_UNSUPPORTED_LOCK,
        _EXPLICIT_CACHE_UNSUPPORTED_MODELS,
    )
    from kesoku.agent.turn_executor import build_history, discard_session_cache_state, prepare_history_for_llm
    from kesoku.config import ChatbotsConfig

    with _EXPLICIT_CACHE_UNSUPPORTED_LOCK:
        saved_models = set(_EXPLICIT_CACHE_UNSUPPORTED_MODELS)
        _EXPLICIT_CACHE_UNSUPPORTED_MODELS.clear()

    db_path = str(tmp_path / "h2_cache.db")
    staging_dir = str(tmp_path / "sessions" / "sess_h2")
    Path(staging_dir).mkdir(parents=True, exist_ok=True)
    DatabaseManager(db_path).init_tables()

    cfg = KesokuConfig(
        agent_working_dir=str(tmp_path),
        workspace=WorkspaceConfig(db_path=db_path, sessions_dir=str(tmp_path / "sessions")),
        gemini=GeminiConfig(
            model_name="gemini-4-hydrogen",
            context_model_name="gemini-3.8-flash",
            thinking_level="low",
            stream=True,
            context_caching=True,
            context_caching_mode="explicit",
            context_caching_threshold=20,
        ),
        chatbots=ChatbotsConfig(
            discord=[
                DiscordConfig(
                    enabled=True,
                    chatbot_id="discord",
                    bot_token="tok",
                    channels=[
                        DiscordChannelOverride(
                            channels=["flash"],
                            llm="gemini",
                            model_name="gemini-3.8-flash",
                            thinking_level="low",
                            stream=True,
                        ),
                        DiscordChannelOverride(
                            channels=["general"],
                            llm="gemini",
                            thinking_level="low",
                            stream=True,
                        ),
                    ],
                )
            ]
        ),
    )

    mock_client = MagicMock()
    cache_create_models: list[str] = []

    def _fake_create_cache(*, model: str, config: Any) -> Any:
        _ = config
        cache_create_models.append(model)
        if model == "gemini-4-hydrogen":
            raise RuntimeError(
                "400 INVALID_ARGUMENT: Model 'gemini-4-hydrogen' does not support explicit context caching."
            )
        return SimpleNamespace(name=f"cache_for_{model}_{len(cache_create_models)}")

    mock_client.caches.create.side_effect = _fake_create_cache

    try:
        with patch("google.genai.Client", return_value=mock_client):
            ctx = KesokuContext(config=cfg)
            gw = Gateway(context=ctx)
            await gw.create_session("sess_h2", title="Hydrogen + Flash", chatbot_id="discord", channel_id="general")

            h2_llm = ctx.get_llm()
            flash_llm = ctx.get_llm(provider="gemini", model_name="gemini-3.8-flash", thinking_level="low")
            assert isinstance(h2_llm, GeminiLLM)
            assert isinstance(flash_llm, GeminiLLM)
            assert h2_llm.model_name == "gemini-4-hydrogen"
            assert flash_llm.model_name == "gemini-3.8-flash"

            # Stub `generate` on both real `GeminiLLM` instances so no network calls occur during `process_turn`.
            seen_generates: list[tuple[str, str | None]] = []

            async def _stub_h2_gen(**kw: Any) -> LLMResponse:
                seen_generates.append(("gemini-4-hydrogen", kw.get("cached_content")))
                return LLMResponse(content="Hydrogen reply " * 10, prompt_tokens=200, total_tokens=220)

            async def _stub_flash_gen(**kw: Any) -> LLMResponse:
                seen_generates.append(("gemini-3.8-flash", kw.get("cached_content")))
                return LLMResponse(content="Flash reply " * 10, prompt_tokens=200, total_tokens=220)

            h2_llm.generate = _stub_h2_gen  # type: ignore[method-assign]
            flash_llm.generate = _stub_flash_gen  # type: ignore[method-assign]

            # Ensure `#general` override (`ctx.get_llm(provider="gemini", thinking_level="low")`) also shares stub.
            general_llm = ctx.get_llm(provider="gemini", thinking_level="low")
            general_llm.generate = _stub_h2_gen  # type: ignore[method-assign]

            tool_runner = MagicMock()
            tool_runner.tool_registry.get_tools_list.return_value = []
            tool_runner.tool_context = MagicMock(transitioned_to_session=None)

            worker = MagicMock()
            worker.active_cache_name = None
            worker.explicit_caches = {}

            async def _noop_pivot(m: Message) -> Message:
                return m

            worker.drain_queue_and_pivot.side_effect = _noop_pivot
            worker.queue_empty.return_value = True

            # Turn 1 on `#general` (`gemini-4-hydrogen`): hits 400 once, marks unsupported, logs INFO (no WARNING).
            u1 = Message(
                id="h2_u1",
                session_id="sess_h2",
                chatbot_id="discord",
                channel_id="general",
                sender="u",
                role=MessageRole.USER,
                content="Turn 1 on general " * 10,
                status=MessageStatus.PENDING_AGENT,
                metadata={"channel_name": "general"},
            )
            await gw.post(u1)
            ex1 = TurnExecutor("sess_h2", gw, tool_runner, context=ctx)
            type(worker).running = PropertyMock(side_effect=[True, False])
            with patch("kesoku.agent.compressor.HistoryCompressor.auto_compact_session", return_value=False):
                await ex1.process_turn(current_msg=u1, worker=worker, session_staging_dir=staging_dir)

            assert cache_create_models == ["gemini-4-hydrogen"]
            assert "gemini-4-hydrogen" in _EXPLICIT_CACHE_UNSUPPORTED_MODELS
            assert general_llm.supports_explicit_cache() is False
            assert flash_llm.supports_explicit_cache() is True
            assert worker.explicit_caches == {}

            # Direct check on `_manage_post_turn_explicit_cache`: skips BEFORE `build_history` / `prepare_history`.
            with (
                patch("kesoku.agent.turn_executor.build_history", wraps=build_history) as spy_build,
                patch(
                    "kesoku.agent.turn_executor.prepare_history_for_llm", wraps=prepare_history_for_llm
                ) as spy_prepare,
            ):
                await ex1._manage_post_turn_explicit_cache(
                    worker=worker,
                    llm=general_llm,
                    cfg=cfg,
                    current_msg=u1,
                    system_prompt="sys",
                    tools_list=[],
                    session_staging_dir=staging_dir,
                    compacted_post=False,
                    used_explicit_cache_last_round=False,
                    last_context_tokens=200,
                    last_cached_tokens=0,
                    last_generate_est_tokens=200,
                )
            spy_build.assert_not_called()
            spy_prepare.assert_not_called()
            assert cache_create_models == ["gemini-4-hydrogen"]

            # Turn 2 on `#flash` (`gemini-3.8-flash`): creates explicit cache post-turn!
            u2 = Message(
                id="h2_u2",
                session_id="sess_h2",
                chatbot_id="discord",
                channel_id="flash",
                sender="u",
                role=MessageRole.USER,
                content="Turn 2 on flash " * 10,
                status=MessageStatus.PENDING_AGENT,
                metadata={"channel_name": "flash"},
            )
            await gw.post(u2)
            ex2 = TurnExecutor("sess_h2", gw, tool_runner, context=ctx)
            type(worker).running = PropertyMock(side_effect=[True, False])
            with patch("kesoku.agent.compressor.HistoryCompressor.auto_compact_session", return_value=False):
                await ex2.process_turn(current_msg=u2, worker=worker, session_staging_dir=staging_dir)

            assert cache_create_models == ["gemini-4-hydrogen", "gemini-3.8-flash"]
            assert "GeminiLLM:gemini-3.8-flash" in worker.explicit_caches

            # Turn 3 on `#flash` reuses the cache; Turn 4 back on `#general` leaves `#flash`'s cache intact!
            u3 = Message(
                id="h2_u3",
                session_id="sess_h2",
                chatbot_id="discord",
                channel_id="flash",
                sender="u",
                role=MessageRole.USER,
                content="Turn 3 follow-up on flash",
                status=MessageStatus.PENDING_AGENT,
                metadata={"channel_name": "flash"},
            )
            await gw.post(u3)
            ex3 = TurnExecutor("sess_h2", gw, tool_runner, context=ctx)
            type(worker).running = PropertyMock(side_effect=[True, False])
            with patch("kesoku.agent.compressor.HistoryCompressor.auto_compact_session", return_value=False):
                await ex3.process_turn(current_msg=u3, worker=worker, session_staging_dir=staging_dir)

            assert seen_generates[-1] == ("gemini-3.8-flash", "cache_for_gemini-3.8-flash_2")

            u4 = Message(
                id="h2_u4",
                session_id="sess_h2",
                chatbot_id="discord",
                channel_id="general",
                sender="u",
                role=MessageRole.USER,
                content="Turn 4 back on general",
                status=MessageStatus.PENDING_AGENT,
                metadata={"channel_name": "general"},
            )
            await gw.post(u4)
            ex4 = TurnExecutor("sess_h2", gw, tool_runner, context=ctx)
            type(worker).running = PropertyMock(side_effect=[True, False])
            with patch("kesoku.agent.compressor.HistoryCompressor.auto_compact_session", return_value=False):
                await ex4.process_turn(current_msg=u4, worker=worker, session_staging_dir=staging_dir)

            assert seen_generates[-1] == ("gemini-4-hydrogen", None)
            assert cache_create_models == ["gemini-4-hydrogen", "gemini-3.8-flash"]
            assert "GeminiLLM:gemini-3.8-flash" in worker.explicit_caches
            discard_session_cache_state("sess_h2")
    finally:
        with _EXPLICIT_CACHE_UNSUPPORTED_LOCK:
            _EXPLICIT_CACHE_UNSUPPORTED_MODELS.clear()
            _EXPLICIT_CACHE_UNSUPPORTED_MODELS.update(saved_models)


class _CaptureLLM(BaseLLM):
    """Returns scripted responses and records every history it was called with."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.histories: list[list[Message]] = []

    async def generate(self, prompt: str | None = None, system_prompt: str | None = None,
                       history: list[Message] | None = None, tools: list[Any] | None = None,
                       **kwargs: Any) -> LLMResponse:
        self.histories.append([m.model_copy() for m in history or []])
        return self.responses.pop(0) if self.responses else LLMResponse(content="done", total_tokens=1)


def _worker(n_iterations: int) -> MagicMock:
    worker = MagicMock()
    type(worker).running = PropertyMock(side_effect=[True] * n_iterations + [False])

    async def pivot(m: Message) -> Message:
        return m

    worker.drain_queue_and_pivot.side_effect = pivot
    worker.queue_empty.return_value = True
    return worker


async def _session(tmp_path: Path, n_completed: int) -> tuple[Gateway, KesokuConfig, list[Message], Message]:
    db = str(tmp_path / "p1.db")
    DatabaseManager(db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=db, sessions_dir=str(tmp_path / "sessions")))
    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.create_session("sess_p1", title="p1")
    base = time.time() - 1000
    msgs: list[Message] = []
    for i in range(1, n_completed + 1):
        u = Message(session_id="sess_p1", chatbot_id="cli", channel_id="ch", sender="u", role=MessageRole.USER,
                    content=f"Turn {i} user", status="processed", timestamp=base + i * 10)
        a = Message(session_id="sess_p1", chatbot_id="cli", channel_id="ch", sender="Kesoku",
                    role=MessageRole.ASSISTANT, content=f"Turn {i} reply", status="delivered",
                    timestamp=base + i * 10 + 1)
        for m in (u, a):
            await gw.post(m)
            msgs.append(m)
    active = Message(session_id="sess_p1", chatbot_id="cli", channel_id="ch", sender="u", role=MessageRole.USER,
                     content="Active turn", status="pending_agent", timestamp=time.time())
    await gw.post(active)
    return gw, cfg, msgs, active


async def _identity(history: list[Message], **kwargs: Any) -> list[Message]:
    return list(history)


# origin: tests/agent/test_turn_executor_p1.py
@pytest.mark.asyncio
async def test_compressed_turn_in_tail_is_not_repeated_verbatim(tmp_path: Path) -> None:
    """§5: T4 sits inside the protected tail but carries a summary_node_id -> only the scaffold represents it."""
    gw, cfg, msgs, active = await _session(tmp_path, n_completed=4)
    cfg.agent.protect_front_turns = 1
    cfg.agent.protect_tail_turns = 3  # T3, T4, active turn are "tail"
    t4 = [m for m in msgs if m.content.startswith("Turn 4")]
    node = SummaryNode(id="n-t4", session_id="sess_p1", level=0, summary="Turn 4 summary", start_timestamp=0,
                       end_timestamp=1, token_count=3, source_token_count=30)
    await gw.db.insert_summary_node(node)
    await gw.db.update_messages_summary_node([m.id for m in t4], "n-t4")

    llm = _CaptureLLM([LLMResponse(content="ok", total_tokens=1)])
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    executor = TurnExecutor("sess_p1", gw, tool_runner, MagicMock(spec=TurnLogger),
                            context=KesokuContext(config=cfg, llm=llm))
    with patch("kesoku.context.get_config", return_value=cfg), \
            patch("kesoku.agent.compressor.HistoryCompressor.auto_compact_session", return_value=False):
        await executor.process_turn(current_msg=active, worker=_worker(1),
                                    session_staging_dir=str(tmp_path / "sessions" / "sess_p1"))

    contents = [m.content for m in llm.histories[0]]
    assert not any("Turn 4 reply" in c for c in contents)
    assert sum("Turn 4 summary" in c for c in contents) == 1
    order = [next(i for i, c in enumerate(contents) if marker in c)
             for marker in ("Turn 1 user", "Turn 4 summary", "Turn 2 user", "Turn 3 reply", "Active turn")]
    assert order == sorted(order)


# origin: tests/agent/test_turn_executor_p1.py
@pytest.mark.asyncio
async def test_state_snapshot_blocks_frozen_within_a_turn(tmp_path: Path) -> None:
    """running_jobs / staging_artifacts are computed once per turn: identical across LLM calls of the turn."""
    gw, cfg, msgs, active = await _session(tmp_path, n_completed=1)
    staging = tmp_path / "sessions" / "sess_p1"
    staging.mkdir(parents=True)
    calls = {"jobs": 0, "artifacts": 0}

    def fake_jobs(snapshot: Any, *args: Any, **kwargs: Any) -> str:
        calls["jobs"] += 1
        return f"JOBS-SNAPSHOT-{calls['jobs']}"

    def fake_artifacts(staging_dir: str, *args: Any, **kwargs: Any) -> str:
        calls["artifacts"] += 1
        return f"ARTIFACTS-{calls['artifacts']}"

    llm = _CaptureLLM([
        LLMResponse(content="", tool_calls=[ToolCallRequest(name="noop", arguments={})], total_tokens=1),
        LLMResponse(content="final", total_tokens=1),
    ])
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []

    async def exec_tool(call: Any, tc_msg: Message, is_interrupted: Any = None) -> Message:
        return Message(session_id="sess_p1", chatbot_id="cli", channel_id="ch", sender="noop",
                       role=MessageRole.TOOL, type="tool_result", content="ok", status="responded",
                       parent_id=tc_msg.id, metadata={"tool_name": "noop", "tool_result": "ok"})

    tool_runner.execute_tool.side_effect = exec_tool
    tool_runner.tool_context = MagicMock(transitioned_to_session=None)
    context = KesokuContext(config=cfg, llm=llm)
    executor = TurnExecutor("sess_p1", gw, tool_runner, MagicMock(spec=TurnLogger), context=context)
    executor._has_summaries = True  # pretend the session was compacted before
    with patch("kesoku.context.get_config", return_value=cfg), \
            patch("kesoku.agent.compressor.HistoryCompressor.auto_compact_session", return_value=False), \
            patch("kesoku.agent.turn_executor.render_running_jobs", side_effect=fake_jobs), \
            patch("kesoku.agent.turn_executor.render_staging_artifacts", side_effect=fake_artifacts), \
            patch.object(TurnExecutor, "_assemble_scaffold_history", side_effect=_identity):
        await executor.process_turn(current_msg=active, worker=_worker(2), session_staging_dir=str(staging))

    assert len(llm.histories) == 2
    latest = [next(m.content for m in reversed(h) if m.role == MessageRole.USER) for h in llm.histories]
    assert latest[0] == latest[1]
    assert '<context name="running_jobs">' in latest[0] and "JOBS-SNAPSHOT-1" in latest[0]
    assert '<context name="staging_artifacts">' in latest[0] and "ARTIFACTS-1" in latest[0]
    assert calls == {"jobs": 1, "artifacts": 1}


# origin: tests/agent/test_turn_executor_p1.py
@pytest.mark.asyncio
async def test_snapshot_entries_gating_and_failures(tmp_path: Path) -> None:
    """No artifacts block before any compaction; renderer failures never break the turn."""
    gw, cfg, msgs, active = await _session(tmp_path, n_completed=1)
    context = KesokuContext(config=cfg, llm=_CaptureLLM([]))
    executor = TurnExecutor("sess_p1", gw, MagicMock(), None, context=context)
    with patch("kesoku.agent.turn_executor.render_running_jobs", return_value=None), \
            patch("kesoku.agent.turn_executor.render_staging_artifacts", return_value="ART"):
        entries = await executor._get_state_snapshot_entries(str(tmp_path))
    assert entries == []  # no summary nodes yet -> artifacts gated, no jobs -> no block
    executor._snapshot_cache = None
    with patch("kesoku.agent.turn_executor.render_running_jobs", side_effect=RuntimeError("boom")), \
            patch("kesoku.agent.turn_executor.render_staging_artifacts", side_effect=RuntimeError("boom")):
        executor._has_summaries = True
        assert await executor._get_state_snapshot_entries(str(tmp_path)) == []


# origin: tests/agent/test_turn_executor_p1.py
@pytest.mark.asyncio
async def test_running_jobs_block_from_real_registry(tmp_path: Path) -> None:
    """End-to-end with the real ActiveJobsRegistry: a running job is rendered with its pgid stop command."""
    gw, cfg, msgs, active = await _session(tmp_path, n_completed=1)
    registry = ActiveJobsRegistry()
    proc = await asyncio.create_subprocess_shell("sleep 30", preexec_fn=os.setsid)
    try:
        await registry.register("job_test_1", "sess_p1", proc, str(tmp_path / "out.log"),
                                command="sleep 30", stderr_filepath=str(tmp_path / "err.log"))
        context = KesokuContext(config=cfg, llm=_CaptureLLM([]))
        context.active_jobs = registry
        executor = TurnExecutor("sess_p1", gw, MagicMock(), None, context=context)
        entries = await executor._get_state_snapshot_entries(str(tmp_path))
        jobs = next(e for e in entries if e.name == "running_jobs").content
        assert "job_test_1" in jobs and "sleep 30" in jobs
        assert f"kill -TERM -- -{os.getpgid(proc.pid)}" in jobs
    finally:
        await registry.stop_all_for_session("sess_p1")


# origin: tests/test_round3_fixes.py K5
@pytest.mark.asyncio
async def test_k5_preemption_posts_cancelled_tool_results_and_zero_orphans(tmp_path: Path) -> None:
    from kesoku.agent.llm import ToolCallRequest

    db_path = str(tmp_path / "k5.db")
    db = DatabaseManager(db_path)
    db.init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=db_path))
    ctx = KesokuContext(config=cfg, db=db)
    gw = Gateway(context=ctx)
    await gw.create_session("s_k5", title="K5")

    u_msg = Message(
        id="u_k5",
        session_id="s_k5",
        chatbot_id="discord",
        channel_id="c1",
        sender="u",
        role=MessageRole.USER,
        content="run tools",
        status=MessageStatus.PROCESSING,
    )
    await gw.post(u_msg)

    tool_runner = MagicMock()
    tool_runner.tool_context = ToolContext(gateway=gw, session_id="s_k5")

    async def never_runs(*args: Any, **kwargs: Any) -> Message:
        raise AssertionError("execute_tool must not run when preempted")

    tool_runner.execute_tool = never_runs
    executor = TurnExecutor("s_k5", gw, tool_runner, MagicMock(spec=TurnLogger), context=ctx)

    worker = MagicMock()
    worker.queue_empty.return_value = False  # Preempted!

    res = LLMResponse(
        content="",
        tool_calls=[
            ToolCallRequest(name="run_shell_command", arguments={"command": "echo 1"}, tool_call_id="tc_1"),
            ToolCallRequest(name="read_file", arguments={"file_name": "a.txt"}, tool_call_id="tc_2"),
        ],
    )
    should_continue = await executor._execute_tool_calls(res, u_msg, worker, str(tmp_path))
    assert should_continue is False

    # Zero orphans in DB!
    orphans = await gw.db.get_orphaned_tool_calls("s_k5")
    assert orphans == []

    msgs = await gw.db.get_session_messages("s_k5")
    results = [m for m in msgs if m.type == MessageType.TOOL_RESULT]
    assert len(results) == 2
    assert {r.sender for r in results} == {"run_shell_command", "read_file"}
    for r in results:
        assert r.metadata.get("tool_error") == "cancelled"
        assert "was cancelled before it ran because the user sent a new message." in r.content


# ---------------------------------------------------------------------------------------------- reply channel (K3/K4)

_RC_PLAN = "Plan: summarize the request, then answer in two sentences."
_RC_REPLY = "Here is the answer.\nIt has two lines."


class _ScriptedStreamLLM(BaseLLM):
    """LLM double that streams scripted texts chunk by chunk and records system prompts."""

    def __init__(self, responses: list[LLMResponse], chunk_size: int = 7) -> None:
        self.responses = list(responses)
        self.chunk_size = chunk_size
        self.system_prompts: list[str | None] = []
        self.model_name = "scripted"

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
        self.system_prompts.append(system_prompt)
        res = self.responses.pop(0)
        if stream and on_stream_chunk is not None and res.content:
            acc = ""
            for i in range(0, len(res.content), self.chunk_size):
                delta = res.content[i : i + self.chunk_size]
                acc += delta
                await on_stream_chunk(delta, acc)
        return res.model_copy(deep=True)


async def _rc_env(
    temp_db: str,
    session_id: str,
    responses: list[LLMResponse],
    *,
    reply_channel: bool = True,
    hooks: Any = None,
) -> tuple[Gateway, TurnExecutor, _ScriptedStreamLLM, Message, list[str | None], Any]:
    """Build a streaming Discord-channel turn with an optional reply-channel channel override."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    cfg.discord.channels = [DiscordChannelOverride(channels=["rc"], stream=True, reply_channel=reply_channel)]
    if hooks is not None:
        cfg.hooks = hooks
    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.create_session(session_id, title="Reply channel")
    await gw.db.update_session_system_prompt(session_id, "Base system prompt.")

    stream_events: list[str | None] = []

    async def capture_stream(ch: str, sess: str, turn: str, text: str | None) -> None:
        stream_events.append(text)

    gw.register_stream_handler("discord", capture_stream)
    user_msg = Message(
        session_id=session_id,
        chatbot_id="discord",
        channel_id="rc",
        sender="u1",
        role=MessageRole.USER,
        content="Question",
        status="pending_agent",
    )
    await gw.post(user_msg)

    llm = _ScriptedStreamLLM(responses)
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    tool_runner.tool_context = MagicMock(transitioned_to_session=None)

    async def exec_tool(call: Any, tc_msg: Message, **kwargs: Any) -> Message:
        return Message(
            session_id=tc_msg.session_id,
            chatbot_id=tc_msg.chatbot_id,
            channel_id=tc_msg.channel_id,
            sender=call.name,
            role=MessageRole.TOOL,
            type=MessageType.TOOL_RESULT,
            content="ok",
            status="responded",
            parent_id=tc_msg.id,
        )

    tool_runner.execute_tool.side_effect = exec_tool
    ctx = KesokuContext(config=cfg, llm=llm)
    executor = TurnExecutor(session_id, gw, tool_runner, None, context=ctx)
    worker = MagicMock()
    type(worker).running = PropertyMock(return_value=True)

    async def mock_pivot(m: Message) -> Message:
        return m

    worker.drain_queue_and_pivot.side_effect = mock_pivot
    worker.queue_empty.return_value = True
    return gw, executor, llm, user_msg, stream_events, worker


async def _rc_run(executor: TurnExecutor, user_msg: Message, worker: Any, session_id: str) -> None:
    with patch("kesoku.agent.compressor.HistoryCompressor.auto_compact_session", return_value=False):
        await executor.process_turn(user_msg, worker, f"/tmp/{session_id}")


def _rc_rows(msgs: list[Message], mtype: MessageType) -> list[Message]:
    return [m for m in msgs if m.role == MessageRole.ASSISTANT and m.type == mtype]


@pytest.mark.asyncio
async def test_final_response_pre_reply_text_moved_to_thought_row(temp_db: str) -> None:
    """Pre-marker text goes to a reclassified THOUGHT row, never to the stream or the TEXT row."""
    from kesoku.constants import REPLY_MARKER

    sid = "sess_rc_found"
    content = f"{_RC_PLAN}\n{REPLY_MARKER}\n{_RC_REPLY}"
    gw, executor, llm, user_msg, stream_events, worker = await _rc_env(
        temp_db, sid, [LLMResponse(content=content, thought="hidden reasoning", total_tokens=5)]
    )
    await _rc_run(executor, user_msg, worker, sid)

    # The system prompt carries the rule exactly once, after the base prompt.
    assert llm.system_prompts[0] is not None and REPLY_MARKER in llm.system_prompts[0]
    assert llm.system_prompts[0].startswith("Base system prompt.")

    msgs = await gw.db.get_session_history(sid, limit=0)
    thoughts = _rc_rows(msgs, MessageType.THOUGHT)
    pre_rows = [t for t in thoughts if t.metadata.get("reclassified") == "pre_reply"]
    assert len(pre_rows) == 1
    assert pre_rows[0].content == _RC_PLAN
    assert pre_rows[0].metadata["final_response"] is True
    assert any(t.content == "hidden reasoning" for t in thoughts)

    texts = _rc_rows(msgs, MessageType.TEXT)
    assert len(texts) == 1
    assert texts[0].content == _RC_REPLY
    assert texts[0].metadata["reply_channel"] == "found"
    rounds = texts[0].metadata["turn_metrics"]["rounds"]
    assert rounds[-1]["reply_marker"] == "found"
    assert rounds[-1]["pre_reply_chars"] == len(_RC_PLAN)

    published = [e for e in stream_events if e]
    assert published, "the post-marker reply should be previewed"
    assert all("Plan" not in e and REPLY_MARKER not in e for e in published)


@pytest.mark.asyncio
async def test_final_response_without_marker_is_unchanged(temp_db: str) -> None:
    """A final response without the marker is delivered verbatim (fail-open) and tagged `missing`."""
    sid = "sess_rc_missing"
    gw, executor, _llm, user_msg, stream_events, worker = await _rc_env(
        temp_db, sid, [LLMResponse(content="Plain reply without marker.", total_tokens=5)]
    )
    await _rc_run(executor, user_msg, worker, sid)

    msgs = await gw.db.get_session_history(sid, limit=0)
    assert not [t for t in _rc_rows(msgs, MessageType.THOUGHT) if t.metadata.get("reclassified")]
    texts = _rc_rows(msgs, MessageType.TEXT)
    assert texts[0].content == "Plain reply without marker."
    assert texts[0].metadata["reply_channel"] == "missing"
    assert texts[0].metadata["turn_metrics"]["rounds"][-1]["reply_marker"] == "missing"
    assert texts[0].metadata["turn_metrics"]["rounds"][-1]["pre_reply_chars"] == 0
    # No preview is published while the marker never arrives.
    assert [e for e in stream_events if e] == []


@pytest.mark.asyncio
async def test_reply_channel_off_keeps_legacy_behaviour(temp_db: str) -> None:
    """Without the opt-in and without a marker, no rule is injected, text is untouched, telemetry reports `off`."""
    from kesoku.constants import REPLY_MARKER

    sid = "sess_rc_off"
    gw, executor, llm, user_msg, events, worker = await _rc_env(
        temp_db, sid, [LLMResponse(content=_RC_REPLY, total_tokens=5)], reply_channel=False
    )
    await _rc_run(executor, user_msg, worker, sid)

    assert REPLY_MARKER not in (llm.system_prompts[0] or "")
    texts = _rc_rows(await gw.db.get_session_history(sid, limit=0), MessageType.TEXT)
    assert texts[0].content == _RC_REPLY
    assert "reply_channel" not in texts[0].metadata
    assert texts[0].metadata["turn_metrics"]["rounds"][-1]["reply_marker"] == "off"
    assert [e for e in events if e], "previews pass through when the channel is off"


@pytest.mark.asyncio
async def test_off_turn_marker_from_replayed_rows_never_reaches_user(temp_db: str) -> None:
    """Channel off for this turn, contracted rows in history, model imitates the marker: it is still split."""
    from kesoku.constants import REPLY_MARKER

    sid = "sess_rc_off_replay"
    content = f"{_RC_PLAN}\n{REPLY_MARKER}\n{_RC_REPLY}"
    gw, executor, llm, user_msg, events, worker = await _rc_env(
        temp_db, sid, [LLMResponse(content=content, total_tokens=5)], reply_channel=False
    )
    prior_user = Message(
        session_id=sid, chatbot_id="discord", channel_id="rc", sender="u1", role=MessageRole.USER,
        content="Earlier question", status=MessageStatus.PROCESSED, timestamp=user_msg.timestamp - 100,
    )
    prior_reply = Message(
        session_id=sid, chatbot_id="discord", channel_id="rc", sender="Kesoku", role=MessageRole.ASSISTANT,
        type=MessageType.TEXT, content="Earlier contracted reply", status=MessageStatus.DELIVERED,
        parent_id=prior_user.id, timestamp=user_msg.timestamp - 99, metadata={"reply_channel": "found"},
    )
    await gw.post(prior_user)
    await gw.post(prior_reply)
    await _rc_run(executor, user_msg, worker, sid)

    assert REPLY_MARKER not in (llm.system_prompts[0] or "")
    msgs = await gw.db.get_session_history(sid, limit=0)
    final = [t for t in _rc_rows(msgs, MessageType.TEXT) if t.parent_id == user_msg.id]
    assert len(final) == 1
    assert final[0].content == _RC_REPLY
    assert final[0].metadata["reply_channel"] == "found_off"  # never replayed with the marker later
    rnd = final[0].metadata["turn_metrics"]["rounds"][-1]
    assert rnd["reply_marker"] == "found_off" and rnd["pre_reply_chars"] == len(_RC_PLAN)
    pre_rows = [t for t in _rc_rows(msgs, MessageType.THOUGHT) if t.metadata.get("reclassified") == "pre_reply"]
    assert [p.content for p in pre_rows] == [_RC_PLAN]
    assert all(REPLY_MARKER not in e for e in events if e)


def test_explicit_cache_key_isolated_by_reply_channel(temp_db: str) -> None:
    """Caches created with and without the reply-channel rule are never shared (cached_content has no prompt)."""
    from kesoku.constants import REPLY_MARKER

    DatabaseManager(temp_db).init_tables()
    ctx = KesokuContext(config=KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db)))
    executor = TurnExecutor("sess_rc_cache", Gateway(context=ctx), MagicMock(), None, context=ctx)
    llm = MockLLM()
    executor._turn_reply_marker = None
    off_key = executor._turn_cache_key(llm)
    executor._turn_reply_marker = REPLY_MARKER
    on_key = executor._turn_cache_key(llm)
    assert off_key != on_key
    assert on_key.startswith(off_key)


@pytest.mark.asyncio
async def test_stop_hook_draft_uses_same_split(temp_db: str) -> None:
    """The stop-hook draft persists the same pre/reply split as a final response would."""
    from kesoku.agent.hooks import HookOutcome
    from kesoku.config import HooksConfig, StopHook
    from kesoku.constants import REPLY_MARKER

    sid = "sess_rc_hook"
    content = f"{_RC_PLAN}\n{REPLY_MARKER}\n{_RC_REPLY}"
    hooks = HooksConfig(stop=[StopHook(name="check", when_tools_used=["any"], command="true", max_retries=1)])
    gw, executor, _llm, user_msg, _events, worker = await _rc_env(
        temp_db,
        sid,
        [LLMResponse(content=content, total_tokens=5), LLMResponse(content=content, total_tokens=5)],
        hooks=hooks,
    )
    outcomes = [[HookOutcome(name="check", exit_code=1, output="fix it")], []]

    async def fake_stop_hooks(*_a: Any, **_kw: Any) -> list[Any]:
        return outcomes.pop(0)

    with patch("kesoku.agent.turn_executor.run_stop_hooks", side_effect=fake_stop_hooks):
        await _rc_run(executor, user_msg, worker, sid)

    msgs = await gw.db.get_session_history(sid, limit=0)
    texts = _rc_rows(msgs, MessageType.TEXT)
    drafts = [t for t in texts if t.status == MessageStatus.RESPONDED and t.sender == "Kesoku"]
    finals = [t for t in texts if t.status != MessageStatus.RESPONDED and t.sender == "Kesoku"]
    assert len(drafts) == 1 and len(finals) == 1
    for row in (drafts[0], finals[0]):
        assert row.content == _RC_REPLY
        assert row.metadata["reply_channel"] == "found"
    pre_rows = [t for t in _rc_rows(msgs, MessageType.THOUGHT) if t.metadata.get("reclassified") == "pre_reply"]
    assert [p.content for p in pre_rows] == [_RC_PLAN, _RC_PLAN]


@pytest.mark.asyncio
async def test_empty_reply_after_marker_uses_empty_response_retry(temp_db: str) -> None:
    """A marker followed by nothing is an empty reply: the normal empty-response retry kicks in."""
    from kesoku.constants import REPLY_MARKER

    sid = "sess_rc_empty"
    gw, executor, llm, user_msg, _events, worker = await _rc_env(
        temp_db,
        sid,
        [
            LLMResponse(content=f"{_RC_PLAN}\n{REPLY_MARKER}\n", total_tokens=5),
            LLMResponse(content=f"{REPLY_MARKER}\n{_RC_REPLY}", total_tokens=5),
        ],
    )
    await _rc_run(executor, user_msg, worker, sid)

    assert len(llm.system_prompts) == 2  # retried once
    msgs = await gw.db.get_session_history(sid, limit=0)
    texts = _rc_rows(msgs, MessageType.TEXT)
    assert [t.content for t in texts] == [_RC_REPLY]
    assert texts[0].metadata["reply_channel"] == "found"
    assert REPLY_MARKER not in texts[0].content
    assert EMPTY_RESPONSE_RETRIES >= 1


@pytest.mark.asyncio
async def test_tool_round_keeps_thought_and_content(temp_db: str) -> None:
    """K4: a tool-call round with both reasoning and visible text stores both in its THOUGHT row."""
    from kesoku.constants import REPLY_MARKER

    sid = "sess_rc_k4"
    gw, executor, _llm, user_msg, stream_events, worker = await _rc_env(
        temp_db,
        sid,
        [
            LLMResponse(
                content="Checking the data first.",
                thought="Need a lookup.",
                tool_calls=[ToolCallRequest(name="lookup", arguments={}, tool_call_id="tc1")],
                total_tokens=5,
            ),
            LLMResponse(content=f"{REPLY_MARKER}\nDone.", total_tokens=5),
        ],
    )
    await _rc_run(executor, user_msg, worker, sid)

    msgs = await gw.db.get_session_history(sid, limit=0)
    thoughts = _rc_rows(msgs, MessageType.THOUGHT)
    assert thoughts[0].content == "Need a lookup.\n\nChecking the data first."
    texts = _rc_rows(msgs, MessageType.TEXT)
    assert [t.content for t in texts] == ["Done."]
    rounds = texts[0].metadata["turn_metrics"]["rounds"]
    assert rounds[0]["reply_marker"] == "off" and rounds[0]["tool_count"] == 1
    assert rounds[1]["reply_marker"] == "found"
    # Tool-round text is never previewed under the reply channel (no marker in that round).
    assert all(e is None or "Checking" not in e for e in stream_events)


def test_reply_channel_resolution(temp_db: str) -> None:
    """Discord channel override and app-mode stamp enable the channel; absent sources leave it off."""
    from kesoku.config import AppConfig, AppModeConfig
    from kesoku.constants import REPLY_MARKER

    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    cfg.discord.channels = [
        DiscordChannelOverride(channels=["on"], reply_channel=True),
        DiscordChannelOverride(channels=["plain"], stream=True),
    ]
    cfg.chatbots.app = [AppConfig(enabled=True, modes=[AppModeConfig(id="chat"), AppModeConfig(id="guided")])]
    ctx = KesokuContext(config=cfg)
    executor = TurnExecutor("sess_rc_res", Gateway(context=ctx), MagicMock(), None, context=ctx)

    def _msg(chatbot: str, channel: str, meta: dict[str, Any] | None = None) -> Message:
        return Message(
            session_id="sess_rc_res", chatbot_id=chatbot, channel_id=channel, sender="u", content="x",
            metadata=meta or {},
        )

    assert executor._resolve_reply_marker(_msg("discord", "on")) == REPLY_MARKER
    thread = _msg("discord", "thread-1", {"parent_channel_name": "on"})
    assert executor._resolve_reply_marker(thread) == REPLY_MARKER
    assert executor._resolve_reply_marker(_msg("discord", "plain")) is None
    assert executor._resolve_reply_marker(_msg("discord", "other")) is None
    # App path: the adapter's server-side stamp enables it; absence leaves it off.
    assert executor._resolve_reply_marker(_msg("app", "app:s1", {"reply_channel": True})) == REPLY_MARKER
    assert executor._resolve_reply_marker(_msg("app", "app:s1")) is None
    # The stamp is only honoured for app chatbots, and only as a literal True.
    assert executor._resolve_reply_marker(_msg("discord", "other", {"reply_channel": True})) is None
    assert executor._resolve_reply_marker(_msg("app", "app:s1", {"reply_channel": "yes"})) is None


def test_client_llm_override_and_model_set_cannot_toggle_reply_channel(temp_db: str) -> None:
    """`llm_override` never carries the reply channel: extra keys are rejected and valid overrides are ignored."""
    from kesoku.agent.turn_executor import validate_llm_override
    from kesoku.config import AppConfig, AppLlmOverride, AppModeConfig
    from kesoku.constants import REPLY_MARKER

    # The override schema itself has no such field and rejects unknown keys.
    assert "reply_channel" not in AppLlmOverride.model_fields
    assert validate_llm_override({"reply_channel": True}, ["m"]) is None
    assert validate_llm_override({"model_name": "m", "reply_channel": False}, ["m"]) is None

    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    cfg.discord.channels = [DiscordChannelOverride(channels=["on"], reply_channel=True)]
    cfg.chatbots.app = [AppConfig(enabled=True, allowed_models=["m"], modes=[AppModeConfig(id="chat")])]
    ctx = KesokuContext(config=cfg)
    executor = TurnExecutor("sess_rc_ovr", Gateway(context=ctx), MagicMock(), None, context=ctx)

    def _msg(chatbot: str, channel: str, meta: dict[str, Any]) -> Message:
        return Message(session_id="sess_rc_ovr", chatbot_id=chatbot, channel_id=channel, sender="u", content="x",
                       metadata=meta)

    # A model switch cannot turn the channel off...
    on = _msg("discord", "on", {"llm_override": {"model_name": "m", "reply_channel": False}})
    assert executor._resolve_reply_marker(on) == REPLY_MARKER
    stamped = _msg("app", "app:s", {"reply_channel": True, "llm_override": {"model_name": "m"}})
    assert executor._resolve_reply_marker(stamped) == REPLY_MARKER
    # ...and an override (valid or not) cannot turn it on.
    for ovr in ({"model_name": "m"}, {"model_name": "m", "reply_channel": True}, {"reply_channel": True}):
        assert executor._resolve_reply_marker(_msg("app", "app:s", {"llm_override": ovr})) is None


@pytest.mark.asyncio
async def test_reply_marker_not_applied_to_subagent_or_context_model(temp_db: str, tmp_path: Path) -> None:
    """Only the main turn sees the rule: the LLM object is not mutated and sub-agents keep raw text."""
    from kesoku.agent.tools.subagent import delegate_subagent
    from kesoku.constants import REPLY_MARKER

    sid = "sess_rc_scope"
    content = f"{_RC_PLAN}\n{REPLY_MARKER}\n{_RC_REPLY}"
    gw, executor, llm, user_msg, _events, worker = await _rc_env(
        temp_db, sid, [LLMResponse(content=content, total_tokens=5)]
    )
    await _rc_run(executor, user_msg, worker, sid)
    assert REPLY_MARKER in (llm.system_prompts[0] or "")
    # The marker is turn-executor state, not an LLM attribute shared with other callers.
    assert not any("reply" in name.lower() for name in vars(llm))

    # A context-model / compression LLM resolved from the same context gets no rule and no split.
    assert executor.context.get_llm(use_context_compression=True) is llm
    assert llm.responses == []

    # A sub-agent run with a marker-shaped answer keeps its text verbatim and never sees the rule.
    sub_cfg = KesokuConfig()
    sub_cfg.agent_working_dir = str(tmp_path)
    sub_cfg.workspace.sessions_dir = str(tmp_path / "sessions")
    sub_llm = _ScriptedStreamLLM([LLMResponse(content=content, total_tokens=3)])
    sub_kctx = KesokuContext(config=sub_cfg, llm=sub_llm)
    sub_ctx = ToolContext(session_id=sid, session_workspace=sid, kesoku_context=sub_kctx)
    out = await delegate_subagent(task="Investigate.", context=sub_ctx)
    assert REPLY_MARKER not in (sub_llm.system_prompts[0] or "")
    assert _RC_PLAN in out
