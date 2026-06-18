"""Unit tests for the Chatbot base class and its utilities."""

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kesoku.constants import (
    MessageRole,
    MessageType,
)
from kesoku.context import KesokuContext
from kesoku.db import Message, Session
from kesoku.gateway.chatbot.base import Chatbot, _format_uptime
from kesoku.gateway.gateway import Gateway


def test_format_uptime() -> None:
    """Test formatting various timedeltas with _format_uptime."""
    assert _format_uptime(datetime.timedelta(seconds=45)) == "45s"
    assert _format_uptime(datetime.timedelta(minutes=2, seconds=15)) == "2m 15s"
    assert _format_uptime(datetime.timedelta(hours=5, minutes=0, seconds=8)) == "5h 0m 8s"
    assert _format_uptime(datetime.timedelta(days=1, hours=12, minutes=30, seconds=5)) == "1d 12h 30m 5s"


class MockChatbot(Chatbot):
    """A concrete chatbot adapter for testing base class functionality."""

    async def handle_message(self, message: Message) -> None:
        """Dummy handler for MockChatbot."""
        pass


@pytest.mark.asyncio
async def test_get_session_status_by_channel() -> None:
    """Test that get_session_status_by_channel returns correct metrics and uptime info."""
    mock_gateway = MagicMock(spec=Gateway)
    mock_db = AsyncMock()
    mock_gateway.db = mock_db
    mock_session = Session(id="session123", title="Test Session")
    mock_db.get_session_by_channel = AsyncMock(return_value=mock_session)

    # Mock history with one user message and one assistant message containing turn metrics
    history = [
        Message(
            session_id="session123",
            chatbot_id="mock_bot",
            channel_id="channel123",
            sender="User",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="Hello",
            timestamp=1000.0,
        ),
        Message(
            session_id="session123",
            chatbot_id="mock_bot",
            channel_id="channel123",
            sender="Agent",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content="Hi",
            timestamp=1002.0,
            metadata={
                "turn_metrics": {
                    "context_tokens": 1200,
                    "cached_tokens": 1000,
                    "turn_tool_calls": 3,
                    "turn_tokens": 150,
                    "turn_time": 2.5,
                }
            },
        ),
    ]
    mock_db.get_session_history = AsyncMock(return_value=history)
    mock_db.get_session_turns_count = AsyncMock(return_value=1)

    chatbot = MockChatbot(chatbot_id="mock_bot", gateway=mock_gateway)

    # Freeze time/started time to verify output matches
    fixed_now = datetime.datetime(2026, 5, 26, 12, 0, 0)
    fixed_start = datetime.datetime(2026, 5, 26, 11, 0, 0)

    with (
        patch("datetime.datetime") as mock_datetime,
        patch("kesoku.gateway.chatbot.base.SYSTEM_START_TIME", fixed_start),
    ):
        mock_datetime.now.return_value = fixed_now
        status_str = await chatbot.get_session_status_by_channel("channel123")

    assert "【Current Stats】" in status_str
    assert "⏰ Uptime: 1h 0m 0s (started: 2026-05-26 11:00:00)" in status_str
    assert "⚡ Session: 1 turns (ID: session123)" in status_str
    assert "📖 Context: 1K tokens (Cached: 1K)" in status_str
    assert "  - Tool Calls: 3" in status_str
    assert "  - Tokens: 0K" in status_str
    assert "  - Time: 2.5s" in status_str


def test_format_text_headers_shifting() -> None:
    """Test shifting and clamping of markdown headers via MockChatbot delegation."""
    mock_gateway = MagicMock(spec=Gateway)
    chatbot = MockChatbot(chatbot_id="mock_bot", gateway=mock_gateway)

    # Simple smoke test to verify delegation works
    input_text = "## Header A\n### Header B\n"
    expected = "# Header A\n\n## Header B\n"
    assert chatbot.format_text(input_text) == expected


@pytest.mark.asyncio
async def test_resolve_outbound_path(tmp_path) -> None:
    """Test resolve_outbound_path under fuzzy matching for misspelled absolute paths."""
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()

    # Create test files in staging
    file_in_staging = staging_dir / "cat.png"
    file_in_staging.write_text("staging cat")

    # Create a nested file to test path scoring
    nested_dir = staging_dir / "output"
    nested_dir.mkdir()
    nested_file = nested_dir / "result.png"
    nested_file.write_text("nested result")

    flat_file = staging_dir / "result.png"
    flat_file.write_text("flat result")

    mock_gateway = MagicMock(spec=Gateway)
    chatbot = MockChatbot(chatbot_id="mock_bot", gateway=mock_gateway)

    from kesoku.config import AgentConfig, KesokuConfig, WorkspaceConfig
    mock_cfg = KesokuConfig(
        workspace=WorkspaceConfig(sessions_dir=str(tmp_path / "sessions"), db_path=":memory:"),
        agent=AgentConfig(user_prompts=[]),
        agent_working_dir=str(tmp_path / "awd"),
    )

    with patch("kesoku.gateway.chatbot.base.get_config", return_value=mock_cfg):
        # Mock session retrieval and staging dir resolution
        mock_session = Session(id="session123", title="Test Session", workspace_name="test_ws")
        mock_gateway.db = AsyncMock()
        mock_gateway.db.get_session = AsyncMock(return_value=mock_session)
        chatbot.get_session_staging_dir = MagicMock(return_value=str(staging_dir))

        # 1. Existing absolute path
        abs_path = str(file_in_staging.resolve())
        resolved = await chatbot.resolve_outbound_path(abs_path, "session123")
        assert resolved == abs_path

        # 2. Fuzzy match misspelled filename (catt.png -> cat.png) with absolute path typo
        staging_dir_abs = str(staging_dir.resolve())
        misspelled_abs = staging_dir_abs.replace("staging", "stating") + "/catt.png"
        resolved = await chatbot.resolve_outbound_path(misspelled_abs, "session123")
        assert resolved == str(file_in_staging.resolve())

        # 3. Fuzzy match path with nested files (comprehensive check):
        # R: "stating/output/results.png"
        # C1 (nested): "staging/output/result.png" -> score will be higher
        # C2 (flat): "staging/result.png"
        misspelled_nested = str(nested_file.resolve())
        misspelled_nested = misspelled_nested.replace("staging", "stating").replace("result.png", "results.png")
        resolved = await chatbot.resolve_outbound_path(misspelled_nested, "session123")
        assert resolved == str(nested_file.resolve())

        misspelled_flat = str(flat_file.resolve())
        misspelled_flat = misspelled_flat.replace("staging", "stating").replace("result.png", "results.png")
        resolved = await chatbot.resolve_outbound_path(misspelled_flat, "session123")
        assert resolved == str(flat_file.resolve())

        # 4. Path not found / low confidence (score < PATH_RESOLUTION_CONFIDENCE_THRESHOLD), returns original
        low_confidence_path = staging_dir_abs.replace("staging", "stating") + "/completely_different.png"
        resolved = await chatbot.resolve_outbound_path(low_confidence_path, "session123")
        assert resolved == low_confidence_path

        # 5. Exact match of a relative path under staging_dir
        resolved = await chatbot.resolve_outbound_path("cat.png", "session123")
        assert resolved == str(file_in_staging.resolve())

        # 6. Fuzzy match of a relative path under staging_dir (catt.png -> cat.png)
        resolved = await chatbot.resolve_outbound_path("catt.png", "session123")
        assert resolved == str(file_in_staging.resolve())

        # 7. Exact match of a nested relative path under staging_dir
        resolved = await chatbot.resolve_outbound_path("output/result.png", "session123")
        assert resolved == str(nested_file.resolve())

        # 8. Fuzzy match of a nested relative path under staging_dir (outtput/results.png -> output/result.png)
        resolved = await chatbot.resolve_outbound_path("outtput/results.png", "session123")
        assert resolved == str(nested_file.resolve())


@pytest.mark.asyncio
async def test_resolve_or_create_session_updates_prompt(tmp_path) -> None:
    """Verify that _resolve_or_create_session updates the system prompt when custom_prompt changes."""
    from kesoku.config import AgentConfig, KesokuConfig, WorkspaceConfig
    from kesoku.db import DatabaseManager
    from kesoku.gateway.chatbot.base import InboundMessageDTO

    temp_db = str(tmp_path / "test_prompt_update.db")
    db_mgr = DatabaseManager(temp_db)
    db_mgr.init_tables()

    # Create a dummy user prompt file
    user_prompt_file = tmp_path / "User.md"
    user_prompt_file.write_text("Initial User Instructions")

    mock_cfg = KesokuConfig(
        workspace=WorkspaceConfig(sessions_dir=str(tmp_path / "sessions"), db_path=temp_db),
        agent=AgentConfig(user_prompts=[str(user_prompt_file)]),
        agent_working_dir=str(tmp_path / "awd"),
    )

    with patch("kesoku.gateway.chatbot.base.get_config", return_value=mock_cfg), \
         patch("kesoku.agent.prompt.get_config", return_value=mock_cfg):

        mock_gateway = Gateway(context=KesokuContext(config=mock_cfg, db=db_mgr))
        chatbot = MockChatbot(chatbot_id="mock_bot", gateway=mock_gateway)

        # 1. Create session first time
        dto1 = InboundMessageDTO(
            sender_id="u1",
            channel_id="c1",
            text="Hello",
            message_id="m1",
            timestamp=1000.0,
            custom_prompt="Custom Instructions V1",
        )
        session, created = await chatbot._resolve_or_create_session(dto1)
        assert created is True
        assert "Initial User Instructions" in session.system_prompt
        assert "Custom Instructions V1" in session.system_prompt

        # 2. Modify User.md on disk
        user_prompt_file.write_text("Updated User Instructions")

        # 3. Resolve session again with same custom prompt (simulating new message in existing session)
        dto2 = InboundMessageDTO(
            sender_id="u1",
            channel_id="c1",
            text="World",
            message_id="m2",
            timestamp=1002.0,
            custom_prompt="Custom Instructions V1",
        )
        session2, created2 = await chatbot._resolve_or_create_session(dto2)
        assert created2 is False

        # Fetch session from DB to verify it was updated
        db_session = await mock_gateway.db.get_session(session.id)
        assert db_session is not None
        assert "Updated User Instructions" in db_session.system_prompt
        assert "Custom Instructions V1" in db_session.system_prompt
        assert "Initial User Instructions" not in db_session.system_prompt

        # 4. Modify User.md again
        user_prompt_file.write_text("Final User Instructions")

        # 5. Resolve session again with NO custom prompt
        dto3 = InboundMessageDTO(
            sender_id="u1",
            channel_id="c1",
            text="Again",
            message_id="m3",
            timestamp=1004.0,
            custom_prompt=None,
        )
        session3, created3 = await chatbot._resolve_or_create_session(dto3)
        assert created3 is False

        db_session3 = await mock_gateway.db.get_session(session.id)
        assert db_session3 is not None
        assert "Final User Instructions" in db_session3.system_prompt
        assert "Updated User Instructions" not in db_session3.system_prompt
        assert "Custom Instructions V1" not in db_session3.system_prompt


@pytest.mark.asyncio
async def test_role_switching_session_isolation(tmp_path) -> None:
    """Test that switching roles in a channel isolates the session contexts and grep searches."""
    from kesoku.agent.tools import memory_grep
    from kesoku.agent.tools.registry import ToolContext
    from kesoku.config import AgentConfig, KesokuConfig, WorkspaceConfig
    from kesoku.constants import MessageRole, MessageStatus, MessageType
    from kesoku.context import KesokuContext
    from kesoku.db import DatabaseManager, Message
    from kesoku.gateway.chatbot.base import InboundMessageDTO

    temp_db = str(tmp_path / "test_role_switch.db")
    db_mgr = DatabaseManager(temp_db)
    db_mgr.init_tables()

    # Create dummy roles dir
    roles_dir = tmp_path / "roles"
    roles_dir.mkdir()
    (roles_dir / "default").mkdir()
    (roles_dir / "asuka").mkdir()

    # Create intro.md for each
    (roles_dir / "default" / "intro.md").write_text("Default role description")
    (roles_dir / "asuka" / "intro.md").write_text("Asuka role description")

    mock_cfg = KesokuConfig(
        workspace=WorkspaceConfig(
            sessions_dir=str(tmp_path / "sessions"),
            db_path=temp_db,
            roles_dir=str(roles_dir)
        ),
        agent=AgentConfig(user_prompts=[]),
        agent_working_dir=str(tmp_path),
    )

    with patch("kesoku.gateway.chatbot.base.get_config", return_value=mock_cfg), \
         patch("kesoku.agent.prompt.get_config", return_value=mock_cfg):

        mock_gateway = Gateway(context=KesokuContext(config=mock_cfg, db=db_mgr))
        chatbot = MockChatbot(chatbot_id="mock_bot", gateway=mock_gateway)

        # 1. Start on channel c1 with default role, resolve a session
        dto1 = InboundMessageDTO(
            sender_id="user1",
            channel_id="c1",
            text="Hello default",
            timestamp=1000.0,
        )
        session_default, created = await chatbot._resolve_or_create_session(dto1)
        assert created is True
        assert session_default.role_name == "default"

        # Save a message to default session
        msg1 = Message(
            id="m1",
            session_id=session_default.id,
            chatbot_id="mock_bot",
            channel_id="c1",
            sender="user1",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="Hello default message content",
            timestamp=1000.0,
            status=MessageStatus.PROCESSED,
        )
        await mock_gateway.post(msg1)

        # 2. Switch role to 'asuka'
        status_msg = await chatbot.update_role_by_channel("c1", "asuka")
        assert "changed to **`asuka`**" in status_msg

        # Resolve session under the new channel role 'asuka'
        dto2 = InboundMessageDTO(
            sender_id="user1",
            channel_id="c1",
            text="Hello asuka",
            timestamp=1005.0,
        )
        session_asuka, created2 = await chatbot._resolve_or_create_session(dto2)
        # Because update_role_by_channel already created and bound the session, created2 should be False
        assert created2 is False
        assert session_asuka.id != session_default.id
        assert session_asuka.role_name == "asuka"

        # Save a message to asuka session
        msg2 = Message(
            id="m2",
            session_id=session_asuka.id,
            chatbot_id="mock_bot",
            channel_id="c1",
            sender="user1",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="Hello asuka message content",
            timestamp=1005.0,
            status=MessageStatus.PROCESSED,
        )
        await mock_gateway.post(msg2)

        # 3. Simulate role switch directly in DB (Scenario B: e.g. third party tool updates channel role)
        # We manually change c1 role to 'default' in DB, but do NOT update channel_sessions
        # (so active session remains session_asuka)
        await mock_gateway.db.set_channel_role("mock_bot", "c1", "default")

        # Resolve session under 'default' channel role.
        # Since active session is session_asuka (role: 'asuka'), it should detect mismatch
        # and switch to last default session (session_default)!
        dto3 = InboundMessageDTO(
            sender_id="user1",
            channel_id="c1",
            text="Back to default",
            timestamp=1010.0,
        )
        session_back, created3 = await chatbot._resolve_or_create_session(dto3)
        # Since it successfully switched back to existing session_default, created3 should be False
        assert created3 is False
        assert session_back.id == session_default.id
        assert session_back.role_name == "default"

        # 4. Perform memory_grep under 'default' role
        ctx_default = ToolContext(
            session_id=session_default.id,
            session_workspace="test_ws",
            gateway=mock_gateway,
        )
        res_default = await memory_grep(query="*", context=ctx_default)
        assert "Hello default message content" in res_default
        assert "Hello asuka message content" not in res_default  # Isolate check!

        # 5. Perform memory_grep under 'asuka' role
        ctx_asuka = ToolContext(
            session_id=session_asuka.id,
            session_workspace="test_ws",
            gateway=mock_gateway,
        )
        res_asuka = await memory_grep(query="*", context=ctx_asuka)
        assert "Hello asuka message content" in res_asuka
        assert "Hello default message content" not in res_asuka  # Isolate check!


