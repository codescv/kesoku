"""Unit tests for Kesoku Agent, LLM mocking, and Tools."""

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kesoku.agent.agent import Agent
from kesoku.agent.history import build_history, prepare_history_for_llm
from kesoku.agent.llm import GeminiLLM, MockLLM, ToolCallRequest, get_llm
from kesoku.agent.tools import ToolContext, ToolRegistry, run_shell_command
from kesoku.config import KesokuConfig, WorkspaceConfig
from kesoku.context import KesokuContext
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


@pytest.fixture
def temp_db(tmp_path: Any) -> str:
    return str(tmp_path / "test_agent.db")


@pytest.mark.asyncio
async def test_agent_execution_loop(temp_db: str) -> None:
    """Test agent processing a message using MockLLM and tool calling."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))
    reg = ToolRegistry()

    @reg.register
    def calculator(expression: str, context: Any = None) -> str:
        return "Result of 25 + 10 = 35.0"

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

    from kesoku.agent.llm import LLMResponse

    llm = MockLLM(
        responses=[
            LLMResponse(
                content="Let me calculate that.",
                tool_calls=[ToolCallRequest(name="calculator", arguments={"expression": "25 + 10"})],
            ),
            LLMResponse(content="The calculation result is 35.", tool_calls=[]),
        ]
    )
    context = KesokuContext(config=cfg, llm=llm, tool_registry=reg)
    agent = Agent(gw, context=context)

    # Start agent loop in background
    agent_task = asyncio.create_task(agent.start())

    # Wait for up to 5 seconds or until at least one message is marked as processed
    for _ in range(50):
        await asyncio.sleep(0.1)
        history = await gw.db.get_session_history("sess1")
        if any(m.status == "processed" for m in history):
            break
    agent.stop()
    await asyncio.gather(agent_task, return_exceptions=True)

    # Verify message status was marked as processed
    history = await gw.db.get_session_history("sess1")
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


@pytest.mark.asyncio
async def test_run_shell_command(tmp_path: Any) -> None:
    """Test secure shell command execution tool."""
    ctx = ToolContext(session_id="test_sess", session_workspace="test_ws")
    with patch("kesoku.agent.tools.shell.get_config") as mock_get_config:
        cfg = KesokuConfig()
        cfg.workspace.sessions_dir = str(tmp_path / "sessions")
        cfg.shell.enabled = False
        mock_get_config.return_value = cfg
        assert "disabled" in await run_shell_command("echo hello", context=ctx)

        cfg.shell.enabled = True
        cfg.shell.mode = "blocklist"
        res = await run_shell_command("echo test_hello", context=ctx)
        assert "test_hello" in res

        assert "Execution denied" in await run_shell_command("rm -rf /", context=ctx)

        cfg.shell.mode = "allowlist"
        assert "Execution denied" in await run_shell_command("unknown_binary_test", context=ctx)
        assert "test_allow" in await run_shell_command("echo test_allow", context=ctx)


@pytest.mark.asyncio
async def test_run_shell_command_background_override(tmp_path: Any) -> None:
    """Verify background_threshold_seconds override in run_shell_command transitions to background."""
    from kesoku.agent.tools import ActiveJobsRegistry

    jobs = ActiveJobsRegistry()
    ctx = ToolContext(
        session_id="test_sess_override",
        session_workspace="test_ws_override",
        active_jobs=jobs,
    )
    with patch("kesoku.agent.tools.shell.get_config") as mock_get_config:
        cfg = KesokuConfig()
        cfg.workspace.sessions_dir = str(tmp_path / "sessions")
        cfg.shell.enabled = True
        cfg.shell.mode = "blocklist"
        mock_get_config.return_value = cfg

        # Override threshold to 0.01 seconds, so sleep 1 command will definitely transition to background!
        try:
            res = await run_shell_command("sleep 1", background_threshold_seconds=0.01, context=ctx)
            assert "transitioned to background execution" in res
            assert "Background Job ID" in res
        finally:
            # Clean up the running background subprocess to avoid unraisable asyncio exception warnings
            await jobs.stop_all_for_session("test_sess_override")


def test_workspace_name() -> None:
    """Test Session.workspace_name returns only the session ID."""
    from kesoku.db import Session

    sess1 = Session(id="12345", title="Math Session", created_at=1779264000.0)
    assert sess1.workspace_name == "12345"

    sess2 = Session(id="54321", title="Hello/World*?!", created_at=1779264000.0)
    assert sess2.workspace_name == "54321"


@pytest.mark.asyncio
async def test_agent_parallel_tool_calls(temp_db: str) -> None:
    """Test that agent processes parallel tool calls and batches TC and TR messages."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))
    reg = ToolRegistry()

    @reg.register
    def dummy_search(query: str, context: Any = None) -> str:
        return f"Search result for {query}"

    await gw.create_session("sess_parallel", title="Parallel Session")
    await gw.post(
        Message(
            session_id="sess_parallel",
            chatbot_id="cli",
            channel_id="ch1",
            sender="u1",
            role="user",
            type="text",
            content="Search A and B",
            status="pending_agent",
        )
    )

    mock_tools = [
        ToolCallRequest(name="dummy_search", arguments={"query": "A"}, thought_signature="sigA"),
        ToolCallRequest(name="dummy_search", arguments={"query": "B"}, thought_signature=None),
    ]
    llm = MockLLM(mock_tools=mock_tools)
    context = KesokuContext(config=cfg, llm=llm, tool_registry=reg)
    agent = Agent(gw, context=context)

    agent_task = asyncio.create_task(agent.start())
    await asyncio.sleep(0.5)
    agent.stop()
    await asyncio.gather(agent_task, return_exceptions=True)

    history = await gw.db.get_session_history("sess_parallel")
    tc_msgs = [m for m in history if m.type == "tool_call"]
    tr_msgs = [m for m in history if m.type == "tool_result"]
    assert len(tc_msgs) == 2
    assert len(tr_msgs) == 2

    # Verify order: both TC messages should appear before both TR messages
    tc_indices = [history.index(m) for m in tc_msgs]
    tr_indices = [history.index(m) for m in tr_msgs]
    assert max(tc_indices) < min(tr_indices)


def test_gemini_llm_thinking_level() -> None:
    """Test that GeminiLLM correctly configures thinking_level in GenerateContentConfig."""
    from google.genai import types

    from kesoku.config import GeminiConfig

    cfg = GeminiConfig(thinking_level="low", auth_mode="api_key", api_key="dummy")

    with patch("google.genai.Client") as mock_client_cls:
        mock_client_inst = MagicMock()
        mock_client_cls.return_value = mock_client_inst
        mock_client_inst.models.generate_content.return_value = MagicMock(parts=[])

        llm = GeminiLLM(config=cfg)
        asyncio.run(llm.generate(prompt="Test"))

        mock_client_inst.models.generate_content.assert_called_once()
        _, kwargs = mock_client_inst.models.generate_content.call_args
        assert "config" in kwargs
        gen_cfg = kwargs["config"]
        assert isinstance(gen_cfg, types.GenerateContentConfig)
        assert gen_cfg.thinking_config is not None
        assert gen_cfg.thinking_config.thinking_level == types.ThinkingLevel.LOW
        assert gen_cfg.thinking_config.include_thoughts is True


@pytest.mark.asyncio
async def test_orphaned_tool_call_healing(temp_db: str) -> None:
    """Verify that orphaned tool calls are healed with synthesized interruption messages."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    # Create session
    await gw.create_session("sess_heal", title="Healing Session")

    # Post a user message and an orphaned tool call
    await gw.post(
        Message(
            session_id="sess_heal",
            chatbot_id="cli",
            channel_id="ch1",
            sender="u1",
            role="user",
            content="Do something",
            status="processed",
        )
    )

    tc_msg = Message(
        session_id="sess_heal",
        chatbot_id="cli",
        channel_id="ch1",
        sender="Kesoku",
        role="tool",
        type="tool_call",
        content="Calling tool...",
        status="responded",
        metadata={"tool_name": "some_tool"},
    )
    await gw.post(tc_msg)

    # Call build clean history directly with heal_orphans=True
    history = await build_history(gateway=gw, session_id="sess_heal", heal_orphans=True)

    # Verify a tool result was synthesized and exists in history
    tr_msgs = [m for m in history if m.type == "tool_result"]
    assert len(tr_msgs) == 1
    assert tr_msgs[0].parent_id == tc_msg.id
    assert "interrupted due to service restart" in tr_msgs[0].content
    assert tr_msgs[0].metadata.get("tool_error") == "Tool execution was interrupted due to service restart."


@pytest.mark.asyncio
async def test_orphaned_tool_call_healing_disabled(temp_db: str) -> None:
    """Verify that orphaned tool calls are NOT healed when heal_orphans is False."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    # Create session
    await gw.create_session("sess_no_heal", title="No Healing Session")

    # Post a user message and an orphaned tool call
    await gw.post(
        Message(
            session_id="sess_no_heal",
            chatbot_id="cli",
            channel_id="ch1",
            sender="u1",
            role="user",
            content="Do something",
            status="processed",
        )
    )

    tc_msg = Message(
        session_id="sess_no_heal",
        chatbot_id="cli",
        channel_id="ch1",
        sender="Kesoku",
        role="tool",
        type="tool_call",
        content="Calling tool...",
        status="responded",
        metadata={"tool_name": "some_tool"},
    )
    await gw.post(tc_msg)

    # Call build clean history directly with heal_orphans=False
    history = await build_history(gateway=gw, session_id="sess_no_heal", heal_orphans=False)

    # Verify no tool result was synthesized/added to history
    tr_msgs = [m for m in history if m.type == "tool_result"]
    assert len(tr_msgs) == 0


@pytest.mark.asyncio
async def test_system_prompt_and_pinned_turns_turn_based(temp_db: str) -> None:
    """Verify that system prompt and the first K turns are always pinned under turn-based logic."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    await gw.create_session("sess_pin", title="Pinning Session")

    # system prompt is automatically created as first message in create_session
    # Let's add 6 turns
    for i in range(1, 7):
        await gw.post(
            Message(
                session_id="sess_pin",
                chatbot_id="cli",
                channel_id="ch1",
                sender="u1",
                role="user",
                content=f"User Prompt {i}",
                status="processed",
            )
        )
        await gw.post(
            Message(
                session_id="sess_pin",
                chatbot_id="cli",
                channel_id="ch1",
                sender="Kesoku",
                role="assistant",
                content=f"Response {i}",
                status="responded",
            )
        )

    # Call build clean history directly
    history = await build_history(gateway=gw, session_id="sess_pin")

    # Verify all 6 turns are kept, meaning we have exactly 12 messages
    assert len(history) == 12
    history_contents = [m.content for m in history]
    for i in range(1, 7):
        assert any(f"User Prompt {i}" in content for content in history_contents)
        assert f"Response {i}" in history_contents


@pytest.mark.asyncio
async def test_skill_pinning_and_parallel_safety_turn_based(temp_db: str) -> None:
    """Verify that use_skill calls and their entire parallel turn batch are never truncated under turn-based logic."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    await gw.create_session("sess_skill", title="Skill Session")

    # Turn 1 (Pinned)
    await gw.post(
        Message(
            session_id="sess_skill",
            chatbot_id="cli",
            channel_id="ch1",
            sender="u1",
            role="user",
            content="Turn 1",
            status="processed",
        )
    )
    await gw.post(
        Message(
            session_id="sess_skill",
            chatbot_id="cli",
            channel_id="ch1",
            sender="Kesoku",
            role="assistant",
            content="Resp 1",
            status="responded",
        )
    )

    # Turn 2 (Sliding window candidate, will be older and would be truncated)
    user2 = Message(
        session_id="sess_skill",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role="user",
        content="Turn 2",
        status="processed",
    )
    await gw.post(user2)

    # Parallel tool calls inside Turn 2:
    # 1. Pinned skill call: use_skill('role-playing')
    # 2. Standard command call: run_shell_command('ls')
    tc_skill = Message(
        session_id="sess_skill",
        chatbot_id="cli",
        channel_id="ch1",
        sender="Kesoku",
        role="tool",
        type="tool_call",
        content="Calling use_skill",
        status="responded",
        parent_id=user2.id,
        metadata={"tool_name": "use_skill", "skill_name": "role-playing"},
    )
    tc_cmd = Message(
        session_id="sess_skill",
        chatbot_id="cli",
        channel_id="ch1",
        sender="Kesoku",
        role="tool",
        type="tool_call",
        content="Calling command",
        status="responded",
        parent_id=user2.id,
        metadata={"tool_name": "run_shell_command"},
    )
    await gw.post(tc_skill)
    await gw.post(tc_cmd)

    tr_skill = Message(
        session_id="sess_skill",
        chatbot_id="cli",
        channel_id="ch1",
        sender="role-playing",
        role="tool",
        type="tool_result",
        content="Skill loaded",
        status="responded",
        parent_id=tc_skill.id,
        metadata={"tool_name": "use_skill", "tool_result": "success"},
    )
    tr_cmd = Message(
        session_id="sess_skill",
        chatbot_id="cli",
        channel_id="ch1",
        sender="run_shell_command",
        role="tool",
        type="tool_result",
        content="Command output",
        status="responded",
        parent_id=tc_cmd.id,
        metadata={"tool_name": "run_shell_command", "tool_result": "files"},
    )
    await gw.post(tr_skill)
    await gw.post(tr_cmd)

    await gw.post(
        Message(
            session_id="sess_skill",
            chatbot_id="cli",
            channel_id="ch1",
            sender="Kesoku",
            role="assistant",
            content="Resp 2",
            status="responded",
        )
    )

    # Turn 3 (Latest turn, stays in sliding window suffix)
    await gw.post(
        Message(
            session_id="sess_skill",
            chatbot_id="cli",
            channel_id="ch1",
            sender="u1",
            role="user",
            content="Turn 3",
            status="processed",
        )
    )
    await gw.post(
        Message(
            session_id="sess_skill",
            chatbot_id="cli",
            channel_id="ch1",
            sender="Kesoku",
            role="assistant",
            content="Resp 3",
            status="responded",
        )
    )

    # Retrieve clean history
    history = await build_history(gateway=gw, session_id="sess_skill")

    # Let's assert all messages of Turn 2 are present!
    history_ids = {m.id for m in history}
    assert user2.id in history_ids
    assert tc_skill.id in history_ids
    assert tc_cmd.id in history_ids
    assert tr_skill.id in history_ids
    assert tr_cmd.id in history_ids


@pytest.mark.asyncio
async def test_priority_based_dropping_and_atomic_batches_turn_based(temp_db: str) -> None:
    """Verify that older resolved tool turns and thoughts are dropped under turn-based logic."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    await gw.create_session("sess_drop", title="Dropping Session")

    # Turn 1 (Will be older, should have thoughts and resolved tool calls dropped)
    user1 = Message(
        session_id="sess_drop",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role="user",
        content="Turn 1",
        status="processed",
    )
    await gw.post(user1)

    # Add a thought message (should be dropped)
    thought1 = Message(
        session_id="sess_drop",
        chatbot_id="cli",
        channel_id="ch1",
        sender="Kesoku",
        role="assistant",
        type="thought",
        content="Thinking...",
        status="responded",
    )
    await gw.post(thought1)

    # Add a resolved tool turn (should be dropped)
    tc1 = Message(
        session_id="sess_drop",
        chatbot_id="cli",
        channel_id="ch1",
        sender="Kesoku",
        role="tool",
        type="tool_call",
        content="Calling tool",
        status="responded",
        parent_id=user1.id,
        metadata={"tool_name": "dummy"},
    )
    await gw.post(tc1)

    tr1 = Message(
        session_id="sess_drop",
        chatbot_id="cli",
        channel_id="ch1",
        sender="dummy",
        role="tool",
        type="tool_result",
        content="Result",
        status="responded",
        parent_id=tc1.id,
        metadata={"tool_name": "dummy", "tool_result": "output"},
    )
    await gw.post(tr1)

    resp1 = Message(
        session_id="sess_drop",
        chatbot_id="cli",
        channel_id="ch1",
        sender="Kesoku",
        role="assistant",
        content="Resp 1",
        status="responded",
    )
    await gw.post(resp1)

    # Turn 2 (Will be recent, should keep thoughts and tool calls)
    user2 = Message(
        session_id="sess_drop",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role="user",
        content="Turn 2",
        status="processed",
    )
    await gw.post(user2)

    thought2 = Message(
        session_id="sess_drop",
        chatbot_id="cli",
        channel_id="ch1",
        sender="Kesoku",
        role="assistant",
        type="thought",
        content="Thinking 2...",
        status="responded",
    )
    await gw.post(thought2)

    tc2 = Message(
        session_id="sess_drop",
        chatbot_id="cli",
        channel_id="ch1",
        sender="Kesoku",
        role="tool",
        type="tool_call",
        content="Calling tool 2",
        status="responded",
        parent_id=user2.id,
        metadata={"tool_name": "dummy"},
    )
    await gw.post(tc2)

    tr2 = Message(
        session_id="sess_drop",
        chatbot_id="cli",
        channel_id="ch1",
        sender="dummy",
        role="tool",
        type="tool_result",
        content="Result 2",
        status="responded",
        parent_id=tc2.id,
        metadata={"tool_name": "dummy", "tool_result": "output 2"},
    )
    await gw.post(tr2)

    resp2 = Message(
        session_id="sess_drop",
        chatbot_id="cli",
        channel_id="ch1",
        sender="Kesoku",
        role="assistant",
        content="Resp 2",
        status="responded",
    )
    await gw.post(resp2)

    # Call history building directly
    history = await build_history(gateway=gw, session_id="sess_drop")
    raw_ids = {m.id for m in history}
    assert thought1.id in raw_ids
    assert thought2.id in raw_ids

    llm_history = prepare_history_for_llm(history)
    history_ids = {m.id for m in llm_history}

    # Turn 1 checks:
    # - User 1 and Resp 1 must be kept
    assert user1.id in history_ids
    assert resp1.id in history_ids
    # - Thought 1 must be dropped (completed turn)
    assert thought1.id not in history_ids
    # - tc1 and tr1 must NOT be dropped
    assert tc1.id in history_ids
    assert tr1.id in history_ids

    # Turn 2 checks:
    # - All kept (thought2, tc2, tr2, resp2) since Turn 2 is the latest active turn
    assert thought2.id in history_ids
    assert tc2.id in history_ids
    assert tr2.id in history_ids
    assert resp2.id in history_ids


@pytest.mark.asyncio
async def test_session_worker_dynamic_llm(temp_db: str) -> None:
    """Verify that SessionWorker resolves the correct LLM based on channel overrides."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    await gw.create_session("sess_override", title="Override Session")

    # Ingest user message with Discord chatbot and channel metadata matching our override
    msg = Message(
        session_id="sess_override",
        chatbot_id="discord",
        channel_id="12345",
        sender="u1",
        role="user",
        type="text",
        content="Hello!",
        metadata={"channel_name": "announcements"},
    )
    await gw.post(msg)

    from kesoku.agent.turn_executor import TurnExecutor
    from kesoku.config import DiscordChannelOverride

    mock_cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    mock_cfg.discord.channels = [
        DiscordChannelOverride(
            channels=["announcements"],
            llm="claude",
        )
    ]

    with (
        patch("kesoku.context.get_config", return_value=mock_cfg),
        patch("kesoku.context.KesokuContext.get_llm") as mock_get_llm,
    ):
        mock_claude = MagicMock()
        mock_get_llm.return_value = mock_claude

        context = KesokuContext(config=mock_cfg, llm=MockLLM())
        executor = TurnExecutor(
            session_id="sess_override",
            gateway=gw,
            tool_runner=MagicMock(),
            context=context,
        )

        resolved_llm = executor._resolve_llm(msg)
        assert resolved_llm == mock_claude
        mock_get_llm.assert_called_once_with(provider="claude")


@pytest.mark.asyncio
async def test_agent_empty_response_nudge(temp_db: str) -> None:
    """Verify that the agent nudges the LLM when the first response is empty, and succeeds on the second try."""
    from kesoku.agent.llm import BaseLLM, LLMResponse

    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))
    reg = ToolRegistry()

    await gw.create_session("sess_nudge", title="Nudge Session")
    await gw.post(
        Message(
            session_id="sess_nudge",
            chatbot_id="cli",
            channel_id="ch1",
            sender="u1",
            role="user",
            type="text",
            content="Hello!",
            status="pending_agent",
        )
    )

    class NudgeLLM(BaseLLM):
        def __init__(self) -> None:
            self.generate_calls = 0

        async def generate(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
            **kwargs: Any,
        ) -> LLMResponse:
            self.generate_calls += 1
            if self.generate_calls == 1:
                # First call returns empty content to trigger nudge
                return LLMResponse(content="", thought="I thought about it but forgot to reply.")
            else:
                # Second call returns content after nudge
                return LLMResponse(content="Hello! Here is the reply after nudge.")

    llm = NudgeLLM()
    context = KesokuContext(config=cfg, llm=llm, tool_registry=reg)
    agent = Agent(gw, context=context)

    agent_task = asyncio.create_task(agent.start())
    await asyncio.sleep(0.6)
    agent.stop()
    await asyncio.gather(agent_task, return_exceptions=True)

    history = await gw.db.get_session_history("sess_nudge")

    # We expect:
    # 1. User Prompt ("Hello!")
    # 2. Thought ("I thought about it...")
    # 3. System nudge message ("[System Notification: Your previous response had empty content...]")
    # 4. Final Assistant Response ("Hello! Here is the reply after nudge.")
    assert len(history) >= 4

    nudge_msgs = [m for m in history if m.sender == "System" and "empty content" in m.content]
    assert len(nudge_msgs) == 1
    assert nudge_msgs[0].role == "system"

    final_msgs = [m for m in history if m.role == "assistant" and m.content == "Hello! Here is the reply after nudge."]
    assert len(final_msgs) == 1


@pytest.mark.asyncio
async def test_llm_turn_logging(temp_db: str, tmp_path: Any) -> None:
    """Verify that raw LLM turns are logged to the session staging directory as YAML files."""
    import os

    import yaml

    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))
    reg = ToolRegistry()

    @reg.register
    def dummy_calculator(expression: str, context: Any = None) -> str:
        """Perform basic calculations."""
        return "4.0"

    with patch("kesoku.context.get_config") as mock_get_context_config:
        cfg = KesokuConfig()
        cfg.workspace.db_path = temp_db
        cfg.workspace.sessions_dir = str(tmp_path / "sessions")
        mock_get_context_config.return_value = cfg

        # Create a session and post a user message
        session = await gw.create_session("sess_log", title="Logging Session")
        user_msg = Message(
            session_id="sess_log",
            chatbot_id="cli",
            channel_id="ch1",
            sender="u1",
            role="user",
            type="text",
            content="Do dummy task",
            status="pending_agent",
        )
        await gw.post(user_msg)

        # Mock LLM that returns a tool call
        mock_tools = [
            ToolCallRequest(name="dummy_calculator", arguments={"expression": "dummy"}),
        ]
        llm = MockLLM(mock_tools=mock_tools)
        context = KesokuContext(llm=llm, tool_registry=reg)
        agent = Agent(gw, context=context)

        # Start agent loop to process the turn
        agent_task = asyncio.create_task(agent.start())
        await asyncio.sleep(0.5)
        agent.stop()
        await asyncio.gather(agent_task, return_exceptions=True)

        # Construct the expected session staging directory path
        staging_dir = os.path.join(cfg.workspace.sessions_dir, session.workspace_name)
        assert os.path.exists(staging_dir)

        # Verify that llm-turn-1.log.yaml exists
        log_path = os.path.join(staging_dir, "llm-turn-1.log.yaml")
        assert os.path.exists(log_path)

        # Load and verify the contents of the log file
        with open(log_path, encoding="utf-8") as f:
            log_data = yaml.safe_load(f)

        assert log_data["metadata"]["session_id"] == "sess_log"
        assert log_data["metadata"]["turn_index"] == 1
        assert log_data["metadata"]["llm_provider"] == "MockLLM"

        # Verify history serialization
        history = log_data["history"]
        assert len(history) >= 2
        assert history[0]["role"] == "system"
        assert history[1]["role"] == "user"
        assert "Do dummy task" in history[1]["content"]


        # Verify tools serialization
        tools = log_data["tools"]
        assert len(tools) >= 1
        dummy_tool = next(t for t in tools if t["name"] == "dummy_calculator")
        assert dummy_tool["description"] == "Perform basic calculations."
        assert "expression" in dummy_tool["parameters"]

        # Verify response serialization
        response = log_data["response"]
        assert len(response["tool_calls"]) == 1
        assert response["tool_calls"][0]["name"] == "dummy_calculator"
        assert response["tool_calls"][0]["arguments"] == {"expression": "dummy"}


@pytest.mark.asyncio
async def test_llm_turn_logging_disabled(temp_db: str, tmp_path: Any) -> None:
    """Verify that raw LLM turns are NOT logged to the session staging directory when disabled."""
    import os

    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))
    reg = ToolRegistry()

    @reg.register
    def dummy_calculator(expression: str, context: Any = None) -> str:
        """Perform basic calculations."""
        return "4.0"

    # Configure workspaces directory to temp_path / "sessions"
    with patch("kesoku.context.get_config") as mock_get_context_config:
        cfg = KesokuConfig()
        cfg.workspace.db_path = temp_db
        cfg.workspace.sessions_dir = str(tmp_path / "sessions")
        cfg.agent.raw_llm_logs = False
        mock_get_context_config.return_value = cfg

        # Create a session and post a user message
        session = await gw.create_session("sess_log_disabled", title="No Logging Session")
        user_msg = Message(
            session_id="sess_log_disabled",
            chatbot_id="cli",
            channel_id="ch1",
            sender="u1",
            role="user",
            type="text",
            content="Do dummy task",
            status="pending_agent",
        )
        await gw.post(user_msg)

        # Mock LLM that returns a tool call
        mock_tools = [
            ToolCallRequest(name="dummy_calculator", arguments={"expression": "dummy"}),
        ]
        llm = MockLLM(mock_tools=mock_tools)
        context = KesokuContext(llm=llm, tool_registry=reg)
        agent = Agent(gw, context=context)

        # Start agent loop to process the turn
        agent_task = asyncio.create_task(agent.start())
        await asyncio.sleep(0.5)
        agent.stop()
        await asyncio.gather(agent_task, return_exceptions=True)

        # Construct the expected session staging directory path
        staging_dir = os.path.join(cfg.workspace.sessions_dir, session.workspace_name)

        # The log file should NOT exist
        log_path = os.path.join(staging_dir, "llm-turn-1.log.yaml")
        assert not os.path.exists(log_path)


@pytest.mark.asyncio
async def test_simplified_history_thought_stripping(temp_db: str) -> None:
    """Verify that thoughts are stripped from completed turns but kept in the active turn."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    await gw.create_session("sess_simp", title="Simplified Session")

    # Turn 1 (Historical completed turn with thoughts)
    user1 = Message(
        session_id="sess_simp",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role="user",
        content="Turn 1 prompt",
        status="processed",
    )
    await gw.post(user1)
    thought1 = Message(
        session_id="sess_simp",
        chatbot_id="cli",
        channel_id="ch1",
        sender="Kesoku",
        role="assistant",
        type="thought",
        content="Turn 1 thoughts",
        status="responded",
    )
    await gw.post(thought1)
    resp1 = Message(
        session_id="sess_simp",
        chatbot_id="cli",
        channel_id="ch1",
        sender="Kesoku",
        role="assistant",
        content="Turn 1 response",
        status="responded",
    )
    await gw.post(resp1)

    # Turn 2 (Active turn with thoughts)
    user2 = Message(
        session_id="sess_simp",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role="user",
        content="Turn 2 prompt",
        status="pending_agent",
    )
    await gw.post(user2)
    thought2 = Message(
        session_id="sess_simp",
        chatbot_id="cli",
        channel_id="ch1",
        sender="Kesoku",
        role="assistant",
        type="thought",
        content="Turn 2 thoughts",
        status="responded",
    )
    await gw.post(thought2)

    # Call build clean history directly
    history = await build_history(gateway=gw, session_id="sess_simp")
    raw_ids = {m.id for m in history}
    assert thought1.id in raw_ids
    assert thought2.id in raw_ids

    llm_history = prepare_history_for_llm(history)
    history_ids = {m.id for m in llm_history}

    # Check Turn 1 (Completed)
    assert user1.id in history_ids
    assert resp1.id in history_ids
    assert thought1.id not in history_ids  # Thought must be stripped!

    # Check Turn 2 (Active)
    assert user2.id in history_ids
    assert thought2.id in history_ids  # Active turn thought must be preserved!


@pytest.mark.asyncio
async def test_agent_llm_error_handling(temp_db: str) -> None:
    """Verify that agent turn processing catches LLM exceptions, posts error message, and marks msg as error."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))
    reg = ToolRegistry()

    # Ingest user message
    await gw.create_session("sess_err", title="Error Handling Session")
    user_msg = Message(
        session_id="sess_err",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role="user",
        type="text",
        content="Hello, will fail",
        status="pending_agent",
    )
    await gw.post(user_msg)

    from kesoku.agent.llm import BaseLLM, LLMResponse

    class FailingLLM(BaseLLM):
        async def generate(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
            **kwargs: Any,
        ) -> LLMResponse:
            raise RuntimeError("LLM API Connection Failed")

    llm = FailingLLM()
    context = KesokuContext(config=cfg, llm=llm, tool_registry=reg)
    agent = Agent(gw, context=context)

    agent_task = asyncio.create_task(agent.start())
    await asyncio.sleep(0.5)
    agent.stop()
    await asyncio.gather(agent_task, return_exceptions=True)

    history = await gw.db.get_session_history("sess_err")

    # Verify that initiating user message is updated to status "error"
    db_user_msg = next(m for m in history if m.id == user_msg.id)
    assert db_user_msg.status == "error"

    # Verify that an error response message is posted with Role Assistant
    err_assistant_msg = next(m for m in history if m.role == "assistant" and "⚠️ An error occurred" in m.content)
    assert err_assistant_msg is not None
    assert "LLM API Connection Failed" in err_assistant_msg.content


@pytest.mark.asyncio
async def test_graceful_shutdown_and_orphaned_recovery(temp_db: str) -> None:
    """Verify graceful shutdown in SessionWorker and orphaned processing message recovery."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))
    reg = ToolRegistry()

    # --- Part 1: Test Graceful Shutdown ---
    await gw.create_session("sess_graceful", title="Graceful Session")

    # 1. Post message
    msg1 = Message(
        session_id="sess_graceful",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role="user",
        type="text",
        content="Test graceful",
        status="pending_agent",
    )
    await gw.post(msg1)

    # Mock LLM that takes some time (0.3s) to return a response
    from kesoku.agent.llm import BaseLLM, LLMResponse

    class SlowLLM(BaseLLM):
        async def generate(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
            **kwargs: Any,
        ) -> LLMResponse:
            await asyncio.sleep(0.3)
            return LLMResponse(content="Finished slowly.")

    llm = SlowLLM()
    context = KesokuContext(config=cfg, llm=llm, tool_registry=reg)
    agent = Agent(gw, context=context)

    agent_task = asyncio.create_task(agent.start())

    # Allow worker to start and begin processing
    await asyncio.sleep(0.15)

    # Verify it's processing
    history = await gw.db.get_session_history("sess_graceful")
    db_msg = next(m for m in history if m.id == msg1.id)
    assert db_msg.status == "processing"

    # Now stop the agent and check that it gracefully waits and completes processing
    agent.stop()
    await asyncio.gather(agent_task, return_exceptions=True)

    # The message should be marked as processed, not interrupted or error, because it finished gracefully!
    history = await gw.db.get_session_history("sess_graceful")
    db_msg = next(m for m in history if m.id == msg1.id)
    assert db_msg.status == "processed"

    # --- Part 2: Test Orphaned Processing Messages Recovery ---
    # Manual post a message with status "processing" and old timestamp
    import time

    old_msg = Message(
        session_id="sess_graceful",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role="user",
        type="text",
        content="Stuck message",
        status="processing",
        timestamp=time.time() - 400.0,  # > 300s ago
    )
    await gw.post(old_msg)

    # Recently updated message (should NOT be recovered)
    recent_msg = Message(
        session_id="sess_graceful",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role="user",
        type="text",
        content="Recent processing message",
        status="processing",
        timestamp=time.time() - 50.0,  # < 300s ago
    )
    await gw.post(recent_msg)

    # Directly trigger database recovery
    recovered_count = await gw.db.recover_orphaned_processing_messages(threshold_seconds=300.0)
    assert recovered_count == 1

    # Verify that old_msg was reverted to pending_agent
    history = await gw.db.get_session_history("sess_graceful")
    db_old_msg = next(m for m in history if m.id == old_msg.id)
    assert db_old_msg.status == "pending_agent"

    # Verify that recent_msg remains in processing status
    db_recent_msg = next(m for m in history if m.id == recent_msg.id)
    assert db_recent_msg.status == "processing"


@pytest.mark.asyncio
async def test_history_attachment_stripping(temp_db: str) -> None:
    """Verify that build_clean_history strips attachments from older user messages while keeping the latest."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    await gw.create_session("sess_attach_strip", title="Attachment Stripping Session")

    # Turn 1 (Historical, user message with attachments)
    msg1 = Message(
        session_id="sess_attach_strip",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role="user",
        content="Please look at my first file.",
        status="processed",
        metadata={
            "attachments": [
                {
                    "path": "/tmp/file1.png",
                    "mime_type": "image/png",
                }
            ]
        },
    )
    await gw.post(msg1)
    await gw.post(
        Message(
            session_id="sess_attach_strip",
            chatbot_id="cli",
            channel_id="ch1",
            sender="Kesoku",
            role="assistant",
            content="I see the first file.",
            status="responded",
        )
    )

    # Turn 2 (Latest active turn, user message with attachments)
    msg2 = Message(
        session_id="sess_attach_strip",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role="user",
        content="Please look at my second file.",
        status="processed",
        metadata={
            "attachments": [
                {
                    "path": "/tmp/file2.png",
                    "mime_type": "image/png",
                }
            ]
        },
    )
    await gw.post(msg2)

    # Call build clean history
    history = await build_history(gateway=gw, session_id="sess_attach_strip")
    assert "attachments" in next(m for m in history if m.id == msg1.id).metadata

    llm_history = prepare_history_for_llm(history)

    # Retrieve and inspect the historical message
    hist_msg = next(m for m in llm_history if m.id == msg1.id)
    # Attachments should be stripped and placeholder appended
    assert "attachments" not in hist_msg.metadata
    assert "[Attachments stripped from history: file1.png]" in hist_msg.content

    # Retrieve and inspect the latest message
    latest_msg = next(m for m in llm_history if m.id == msg2.id)
    # Attachments must be kept in full detail for the active turn
    assert "attachments" in latest_msg.metadata
    assert latest_msg.metadata["attachments"][0]["path"] == "/tmp/file2.png"


@pytest.mark.asyncio
async def test_agent_wakeup_by_system_message(temp_db: str) -> None:
    """Verify that the Agent dispatcher wakes up and processes role=SYSTEM PENDING_AGENT messages."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))
    reg = ToolRegistry()

    # 1. Ingest system message with status "pending_agent"
    await gw.create_session("sess_sys_wakeup", title="System Wakeup Session")
    msg = Message(
        session_id="sess_sys_wakeup",
        chatbot_id="system",
        channel_id="system",
        sender="System",
        role="system",
        type="text",
        content="[System Alert] Background Job Finished",
        status="pending_agent",
    )
    await gw.post(msg)

    from kesoku.agent.llm import LLMResponse

    llm = MockLLM(responses=[LLMResponse(content="System alert processed successfully.", tool_calls=[])])
    context = KesokuContext(config=cfg, llm=llm, tool_registry=reg)
    agent = Agent(gw, context=context)

    # Start agent loop
    agent_task = asyncio.create_task(agent.start())

    # Wait for the message to be processed
    await asyncio.sleep(0.5)
    agent.stop()
    await asyncio.gather(agent_task, return_exceptions=True)

    # Verify the system message was successfully claimed and processed
    history = await gw.db.get_session_history("sess_sys_wakeup")
    assert any(m.id == msg.id and m.status == "processed" for m in history)
