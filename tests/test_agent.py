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

    with patch("kesoku.agent.agent.get_config", return_value=cfg):
        worker = SessionWorker(
            session_id="sess_cfg", gateway=gw, llm=MockLLM(), tool_registry=ToolRegistry(), dispatcher=None
        )
        history = await worker._build_clean_history()
        assert len(history) == 1
        assert history[0].role == "system"
