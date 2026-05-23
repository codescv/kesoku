"""Unit tests for Kesoku Agent, LLM mocking, and Tools."""

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kesoku.agent.agent import Agent
from kesoku.agent.llm import GeminiLLM, MockLLM, ToolCallRequest, get_llm
from kesoku.agent.tools import ToolContext, ToolRegistry, run_shell_command
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


@pytest.fixture
def temp_db(tmp_path: Any) -> str:
    return str(tmp_path / "test_agent.db")


@pytest.mark.asyncio
async def test_agent_execution_loop(temp_db: str) -> None:
    """Test agent processing a message using MockLLM and tool calling."""
    DatabaseManager(temp_db).init_tables()
    gw = Gateway(workspace_config=WorkspaceConfig(db_path=temp_db))
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


def test_run_shell_command(tmp_path: Any) -> None:
    """Test secure shell command execution tool."""
    ctx = ToolContext(session_id="test_sess", session_workspace="test_ws")
    with patch("kesoku.agent.tools.get_config") as mock_get_config:
        cfg = KesokuConfig()
        cfg.workspace.sessions_dir = str(tmp_path / "sessions")
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


def test_workspace_name() -> None:
    """Test Session.workspace_name escaping, truncation, and prefix."""
    import time

    from kesoku.db import Session

    sess1 = Session(id="12345", title="Math Session", created_at=1779264000.0)
    ts_str = time.strftime("%y%m%d-%H-%M", time.localtime(sess1.created_at))
    assert sess1.workspace_name == f"{ts_str}_Math_Session_12345"

    sess2 = Session(id="54321", title="Hello/World*?!", created_at=1779264000.0)
    assert sess2.workspace_name == f"{ts_str}_Hello_World_54321"

    sess3 = Session(id="67890", title="a" * 30, created_at=1779264000.0)
    assert sess3.workspace_name == f"{ts_str}_{'a' * 20}_67890"

    sess4 = Session(id="99999", title="___", created_at=1779264000.0)
    assert sess4.workspace_name == f"{ts_str}_session_99999"


@pytest.mark.asyncio
async def test_agent_parallel_tool_calls(temp_db: str) -> None:
    """Test that agent processes parallel tool calls and batches TC and TR messages."""
    DatabaseManager(temp_db).init_tables()
    gw = Gateway(workspace_config=WorkspaceConfig(db_path=temp_db))
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
    agent = Agent(gw, llm, reg)

    agent_task = asyncio.create_task(agent.start())
    await asyncio.sleep(0.5)
    agent.stop()
    await asyncio.gather(agent_task, return_exceptions=True)

    history = await gw.get_session_history("sess_parallel")
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
    llm = GeminiLLM(config=cfg)

    with patch("google.genai.Client") as mock_client_cls:
        mock_client_inst = MagicMock()
        mock_client_cls.return_value = mock_client_inst
        mock_client_inst.models.generate_content.return_value = MagicMock(parts=[])

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
    gw = Gateway(workspace_config=WorkspaceConfig(db_path=temp_db))

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

    # Create worker and build clean history
    from kesoku.agent.agent import SessionWorker

    worker = SessionWorker(
        session_id="sess_heal", gateway=gw, llm=MockLLM(), tool_registry=ToolRegistry(), dispatcher=None
    )

    history = await worker._build_clean_history(max_turns=10)

    # Verify a tool result was synthesized and exists in history
    tr_msgs = [m for m in history if m.type == "tool_result"]
    assert len(tr_msgs) == 1
    assert tr_msgs[0].parent_id == tc_msg.id
    assert "interrupted due to service restart" in tr_msgs[0].content
    assert tr_msgs[0].metadata.get("tool_error") == "Tool execution was interrupted due to service restart."


@pytest.mark.asyncio
async def test_system_prompt_and_pinned_turns_turn_based(temp_db: str) -> None:
    """Verify that system prompt and the first K turns are always pinned under turn-based logic."""
    DatabaseManager(temp_db).init_tables()
    gw = Gateway(workspace_config=WorkspaceConfig(db_path=temp_db))

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

    from kesoku.agent.agent import SessionWorker

    worker = SessionWorker(
        session_id="sess_pin", gateway=gw, llm=MockLLM(), tool_registry=ToolRegistry(), dispatcher=None
    )

    # Call build clean history with turn counts that force Turn 3 and Turn 4 to truncate
    history = await worker._build_clean_history(max_turns=4, pin_initial_turns=2, pin_recent_turns=2)

    # 1. System prompt is at index 0
    assert history[0].role == "system"

    # 2. First 2 turns are pinned
    assert history[1].content == "User Prompt 1"
    assert history[2].content == "Response 1"
    assert history[3].content == "User Prompt 2"
    assert history[4].content == "Response 2"

    # 3. Next should start with Turn 5 (User Prompt 5)
    assert history[5].content == "User Prompt 5"


@pytest.mark.asyncio
async def test_skill_pinning_and_parallel_safety_turn_based(temp_db: str) -> None:
    """Verify that use_skill calls and their entire parallel turn batch are never truncated under turn-based logic."""
    DatabaseManager(temp_db).init_tables()
    gw = Gateway(workspace_config=WorkspaceConfig(db_path=temp_db))

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

    from kesoku.agent.agent import SessionWorker

    worker = SessionWorker(
        session_id="sess_skill", gateway=gw, llm=MockLLM(), tool_registry=ToolRegistry(), dispatcher=None
    )

    # Retrieve clean history with limit that forces Turn 2 to truncate
    history = await worker._build_clean_history(max_turns=2, pin_initial_turns=1, pin_recent_turns=1)

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
    gw = Gateway(workspace_config=WorkspaceConfig(db_path=temp_db))

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

    from kesoku.agent.agent import SessionWorker

    worker = SessionWorker(
        session_id="sess_drop", gateway=gw, llm=MockLLM(), tool_registry=ToolRegistry(), dispatcher=None
    )

    # Call history building with max_turns = 10, pin_initial_turns = 0, pin_recent_turns = 1
    history = await worker._build_clean_history(max_turns=10, pin_initial_turns=0, pin_recent_turns=1)
    history_ids = {m.id for m in history}

    # Turn 1 checks:
    # - User 1 and Resp 1 must be kept
    assert user1.id in history_ids
    assert resp1.id in history_ids
    # - Thought 1, tc1, and tr1 must be dropped
    assert thought1.id not in history_ids
    assert tc1.id not in history_ids
    assert tr1.id not in history_ids

    # Turn 2 checks:
    # - All kept (thought2, tc2, tr2, resp2)
    assert thought2.id in history_ids
    assert tc2.id in history_ids
    assert tr2.id in history_ids
    assert resp2.id in history_ids


@pytest.mark.asyncio
async def test_clean_history_config_loading(temp_db: str) -> None:
    """Verify that build_clean_history loads default parameters from global config when arguments are None."""
    DatabaseManager(temp_db).init_tables()
    gw = Gateway(workspace_config=WorkspaceConfig(db_path=temp_db))

    await gw.create_session("sess_cfg", title="Config Session")

    from kesoku.agent.agent import SessionWorker
    from kesoku.config import KesokuConfig

    cfg = KesokuConfig()
    cfg.agent.history.max_turns = 42
    cfg.agent.history.pin_initial_turns = 7
    cfg.agent.history.pin_recent_turns = 13

    with patch("kesoku.agent.agent.get_config", return_value=cfg), \
         patch("kesoku.agent.history.get_config", return_value=cfg):
        worker = SessionWorker(
            session_id="sess_cfg", gateway=gw, llm=MockLLM(), tool_registry=ToolRegistry(), dispatcher=None
        )
        history = await worker._build_clean_history()
        assert len(history) == 1
        assert history[0].role == "system"


@pytest.mark.asyncio
async def test_session_worker_dynamic_llm(temp_db: str) -> None:
    """Verify that SessionWorker resolves the correct LLM based on channel overrides."""
    DatabaseManager(temp_db).init_tables()
    gw = Gateway(workspace_config=WorkspaceConfig(db_path=temp_db))

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

    from kesoku.agent.agent import SessionWorker
    from kesoku.config import DiscordChannelOverride, KesokuConfig

    cfg = KesokuConfig()
    cfg.discord.channels = [
        DiscordChannelOverride(
            channels=["announcements"],
            llm="claude",
        )
    ]

    with patch("kesoku.agent.agent.get_config", return_value=cfg), \
         patch("kesoku.agent.history.get_config", return_value=cfg), \
         patch("kesoku.agent.agent.get_llm") as mock_get_llm:
        mock_claude = MagicMock()
        mock_get_llm.return_value = mock_claude

        worker = SessionWorker(
            session_id="sess_override",
            gateway=gw,
            llm=MockLLM(),
            tool_registry=ToolRegistry(),
            dispatcher=None,
        )

        resolved_llm = worker._resolve_llm(msg)
        assert resolved_llm == mock_claude
        mock_get_llm.assert_called_once_with("claude")


@pytest.mark.asyncio
async def test_agent_empty_response_nudge(temp_db: str) -> None:
    """Verify that the agent nudges the LLM when the first response is empty, and succeeds on the second try."""
    from kesoku.agent.llm import BaseLLM, LLMResponse

    DatabaseManager(temp_db).init_tables()
    gw = Gateway(workspace_config=WorkspaceConfig(db_path=temp_db))
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
        ) -> LLMResponse:
            self.generate_calls += 1
            if self.generate_calls == 1:
                # First call returns empty content to trigger nudge
                return LLMResponse(content="", thought="I thought about it but forgot to reply.")
            else:
                # Second call returns content after nudge
                return LLMResponse(content="Hello! Here is the reply after nudge.")

    llm = NudgeLLM()
    agent = Agent(gw, llm, reg)

    agent_task = asyncio.create_task(agent.start())
    await asyncio.sleep(0.6)
    agent.stop()
    await asyncio.gather(agent_task, return_exceptions=True)

    history = await gw.get_session_history("sess_nudge")

    # We expect:
    # 1. System prompt (from create_session)
    # 2. User Prompt ("Hello!")
    # 3. Thought ("I thought about it...")
    # 4. System nudge message ("[System Notification: Your previous response had empty content...]")
    # 5. Final Assistant Response ("Hello! Here is the reply after nudge.")
    assert len(history) >= 5

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
    gw = Gateway(workspace_config=WorkspaceConfig(db_path=temp_db))
    reg = ToolRegistry()

    @reg.register
    def dummy_calculator(expression: str, context: Any = None) -> str:
        """Perform basic calculations."""
        return "4.0"

    # Configure workspaces directory to temp_path / "sessions"
    with patch("kesoku.agent.agent.get_config") as mock_get_config, \
         patch("kesoku.agent.history.get_config") as mock_get_history_config:
        cfg = KesokuConfig()
        cfg.workspace.db_path = temp_db
        cfg.workspace.sessions_dir = str(tmp_path / "sessions")
        mock_get_config.return_value = cfg
        mock_get_history_config.return_value = cfg

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
        agent = Agent(gw, llm, reg)

        # Start agent loop to process the turn
        agent_task = asyncio.create_task(agent.start())
        await asyncio.sleep(0.5)
        agent.stop()
        await asyncio.gather(agent_task, return_exceptions=True)

        # Construct the expected session staging directory path
        staging_dir = os.path.join(cfg.workspace.sessions_dir, session.workspace_name)
        assert os.path.exists(staging_dir)  # noqa: ASYNC240

        # Verify that llm-turn-1.log.yaml exists
        log_path = os.path.join(staging_dir, "llm-turn-1.log.yaml")
        assert os.path.exists(log_path)  # noqa: ASYNC240

        # Load and verify the contents of the log file
        with open(log_path, encoding="utf-8") as f:  # noqa: ASYNC230
            log_data = yaml.safe_load(f)

        assert log_data["metadata"]["session_id"] == "sess_log"
        assert log_data["metadata"]["turn_index"] == 1
        assert log_data["metadata"]["llm_provider"] == "MockLLM"

        # Verify history serialization
        history = log_data["history"]
        assert len(history) >= 2
        assert history[0]["role"] == "system"
        assert history[1]["role"] == "user"
        assert history[1]["content"] == "Do dummy task"

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
    gw = Gateway(workspace_config=WorkspaceConfig(db_path=temp_db))
    reg = ToolRegistry()

    @reg.register
    def dummy_calculator(expression: str, context: Any = None) -> str:
        """Perform basic calculations."""
        return "4.0"

    # Configure workspaces directory to temp_path / "sessions"
    with patch("kesoku.agent.agent.get_config") as mock_get_config, \
         patch("kesoku.agent.history.get_config") as mock_get_history_config:
        cfg = KesokuConfig()
        cfg.workspace.db_path = temp_db
        cfg.workspace.sessions_dir = str(tmp_path / "sessions")
        cfg.agent.raw_llm_logs = False
        mock_get_config.return_value = cfg
        mock_get_history_config.return_value = cfg

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
        agent = Agent(gw, llm, reg)

        # Start agent loop to process the turn
        agent_task = asyncio.create_task(agent.start())
        await asyncio.sleep(0.5)
        agent.stop()
        await asyncio.gather(agent_task, return_exceptions=True)

        # Construct the expected session staging directory path
        staging_dir = os.path.join(cfg.workspace.sessions_dir, session.workspace_name)

        # The log file should NOT exist
        log_path = os.path.join(staging_dir, "llm-turn-1.log.yaml")
        assert not os.path.exists(log_path)  # noqa: ASYNC240


@pytest.mark.asyncio
async def test_context_optimization_tool_serialization(temp_db: str, tmp_path: Any) -> None:
    """Verify that tool results are serialized and truncated under the context optimization settings."""
    import os
    DatabaseManager(temp_db).init_tables()
    gw = Gateway(workspace_config=WorkspaceConfig(db_path=temp_db))

    with patch("kesoku.agent.agent.get_config") as mock_get_config, \
         patch("kesoku.agent.history.get_config") as mock_get_history_config:
        cfg = KesokuConfig()
        cfg.workspace.db_path = temp_db
        cfg.workspace.sessions_dir = str(tmp_path / "sessions")
        cfg.agent.history.active_turn_keep_tool_results_for_k_recent_calls = 1
        cfg.agent.history.serialize_historical_tool_results = True
        cfg.agent.history.serialize_tool_results_threshold = 200
        mock_get_config.return_value = cfg
        mock_get_history_config.return_value = cfg

        session = await gw.create_session("sess_opt", title="Optimization Session")

        # --- Turn 1 (Historical Turn) ---
        # 1. User message
        user1 = Message(
            session_id="sess_opt", chatbot_id="cli", channel_id="ch1", sender="u1",
            role="user", content="Turn 1 message", status="processed"
        )
        await gw.post(user1)
        # 2. Tool call (not use_skill)
        tc1 = Message(
            session_id="sess_opt", chatbot_id="cli", channel_id="ch1", sender="Kesoku",
            role="tool", type="tool_call", content="Calling dummy tool", status="responded",
            parent_id=user1.id, metadata={"tool_name": "dummy_tool"}
        )
        await gw.post(tc1)
        # 3. Tool result (long content > 200: should be serialized because it's historical and not use_skill)
        long_content_1 = "Dummy tool result content " * 10
        tr1 = Message(
            session_id="sess_opt", chatbot_id="cli", channel_id="ch1", sender="dummy_tool",
            role="tool", type="tool_result", content=long_content_1, status="responded",
            parent_id=tc1.id, metadata={"tool_name": "dummy_tool", "tool_result": long_content_1}
        )
        await gw.post(tr1)
        # 4. Pinned skill tool call and result (should NOT be serialized because it's use_skill)
        tc_skill = Message(
            session_id="sess_opt", chatbot_id="cli", channel_id="ch1", sender="Kesoku",
            role="tool", type="tool_call", content="Calling use_skill", status="responded",
            parent_id=user1.id, metadata={"tool_name": "use_skill"}
        )
        await gw.post(tc_skill)
        tr_skill = Message(
            session_id="sess_opt", chatbot_id="cli", channel_id="ch1", sender="use_skill",
            role="tool", type="tool_result", content="Skill loaded content", status="responded",
            parent_id=tc_skill.id, metadata={"tool_name": "use_skill", "tool_result": "Skill loaded content"}
        )
        await gw.post(tr_skill)
        # 5. Short tool result (should NOT be serialized because length <= 200)
        tc_short = Message(
            session_id="sess_opt", chatbot_id="cli", channel_id="ch1", sender="Kesoku",
            role="tool", type="tool_call", content="Calling short tool", status="responded",
            parent_id=user1.id, metadata={"tool_name": "short_tool"}
        )
        await gw.post(tc_short)
        tr_short = Message(
            session_id="sess_opt", chatbot_id="cli", channel_id="ch1", sender="short_tool",
            role="tool", type="tool_result", content="Short tool result content", status="responded",
            parent_id=tc_short.id, metadata={"tool_name": "short_tool", "tool_result": "Short tool result content"}
        )
        await gw.post(tr_short)
        # 6. Assistant response
        resp1 = Message(
            session_id="sess_opt", chatbot_id="cli", channel_id="ch1", sender="Kesoku",
            role="assistant", content="Response 1", status="responded"
        )
        await gw.post(resp1)

        # --- Turn 2 (Active Turn) ---
        # 1. User message
        user2 = Message(
            session_id="sess_opt", chatbot_id="cli", channel_id="ch1", sender="u1",
            role="user", content="Turn 2 message", status="pending_agent"
        )
        await gw.post(user2)
        # 2. LLM Call Batch 1 (Older active turn batch, long content > 200: should be serialized)
        tc2_1 = Message(
            session_id="sess_opt", chatbot_id="cli", channel_id="ch1", sender="Kesoku",
            role="tool", type="tool_call", content="Calling tool 2.1", status="responded",
            parent_id=user2.id, metadata={"tool_name": "tool2_1"}
        )
        await gw.post(tc2_1)
        long_content_2_1 = "Tool 2.1 result content " * 15
        tr2_1 = Message(
            session_id="sess_opt", chatbot_id="cli", channel_id="ch1", sender="tool2_1",
            role="tool", type="tool_result", content=long_content_2_1, status="responded",
            parent_id=tc2_1.id, metadata={"tool_name": "tool2_1", "tool_result": long_content_2_1}
        )
        await gw.post(tr2_1)

        # Simulate delay of second LLM call to ensure separate timestamps for batches
        await asyncio.sleep(0.6)

        # 3. LLM Call Batch 2 (Latest active turn batch, keep_k=1, so this should be kept in full detail)
        tc2_2 = Message(
            session_id="sess_opt", chatbot_id="cli", channel_id="ch1", sender="Kesoku",
            role="tool", type="tool_call", content="Calling tool 2.2", status="responded",
            parent_id=user2.id, metadata={"tool_name": "tool2_2"}
        )
        await gw.post(tc2_2)
        tr2_2 = Message(
            session_id="sess_opt", chatbot_id="cli", channel_id="ch1", sender="tool2_2",
            role="tool", type="tool_result", content="Tool 2.2 result content", status="responded",
            parent_id=tc2_2.id, metadata={"tool_name": "tool2_2", "tool_result": "Tool 2.2 result content"}
        )
        await gw.post(tr2_2)

        # Instantiate worker and build clean history
        from kesoku.agent.agent import SessionWorker
        worker = SessionWorker(
            session_id="sess_opt", gateway=gw, llm=MockLLM(), tool_registry=ToolRegistry(), dispatcher=None
        )

        history = await worker._build_clean_history(max_turns=10, pin_initial_turns=2, pin_recent_turns=2)

        # 1. Check that historical long tr1 was serialized
        tr1_msg = next(m for m in history if m.id == tr1.id)
        assert "tool output in" in tr1_msg.content
        assert "tool output in" in tr1_msg.metadata["tool_result"]
        file_path1 = tr1_msg.content.split("tool output in ")[1]
        assert os.path.exists(file_path1)  # noqa: ASYNC240
        with open(file_path1, encoding="utf-8") as f:  # noqa: ASYNC230
            assert f.read() == long_content_1

        # 2. Check that historical tr_skill was NOT serialized (preserved use_skill)
        tr_skill_msg = next(m for m in history if m.id == tr_skill.id)
        assert tr_skill_msg.content == "Skill loaded content"
        assert tr_skill_msg.metadata["tool_result"] == "Skill loaded content"

        # 3. Check that historical short tool result tr_short was NOT serialized
        tr_short_msg = next(m for m in history if m.id == tr_short.id)
        assert tr_short_msg.content == "Short tool result content"
        assert tr_short_msg.metadata["tool_result"] == "Short tool result content"

        # 4. Check that active turn long tr2_1 (older batch) was serialized
        tr2_1_msg = next(m for m in history if m.id == tr2_1.id)
        assert "tool output in" in tr2_1_msg.content
        file_path2_1 = tr2_1_msg.content.split("tool output in ")[1]
        assert os.path.exists(file_path2_1)  # noqa: ASYNC240
        with open(file_path2_1, encoding="utf-8") as f:  # noqa: ASYNC230
            assert f.read() == long_content_2_1

        # 5. Check that active turn tr2_2 (latest batch, K=1) was NOT serialized (kept in full detail)
        tr2_2_msg = next(m for m in history if m.id == tr2_2.id)
        assert tr2_2_msg.content == "Tool 2.2 result content"
        assert tr2_2_msg.metadata["tool_result"] == "Tool 2.2 result content"



