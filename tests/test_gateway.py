"""Unit tests for Kesoku Gateway and Chatbot routing."""

import asyncio
import sqlite3
from typing import Any

import pytest

from kesoku.config import KesokuConfig, WorkspaceConfig
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager, Message
from kesoku.gateway.chatbot.base import Chatbot
from kesoku.gateway.gateway import Gateway


class DummyChatbot(Chatbot):
    """Dummy chatbot adapter for testing."""

    def __init__(self, chatbot_id: str, gateway: Gateway) -> None:
        super().__init__(chatbot_id, gateway)
        self.sent_messages: list[tuple[str, str]] = []

    async def handle_message(self, message: Message) -> None:
        self.sent_messages.append((message.channel_id, message.content))
        await self.gateway.update_message_status(message.id, MessageStatus.DELIVERED)


@pytest.fixture
def temp_db(tmp_path: Any) -> str:
    db_file = tmp_path / "test_kesoku.db"
    return str(db_file)


@pytest.mark.asyncio
async def test_gateway_init_db(temp_db: str) -> None:
    """Verify SQLite database schema is initialized correctly."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
    table = cursor.fetchone()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
    sessions_table = cursor.fetchone()
    conn.close()
    assert table is not None
    assert table[0] == "messages"
    assert sessions_table is not None
    assert sessions_table[0] == "sessions"


@pytest.mark.asyncio
async def test_gateway_routing(temp_db: str) -> None:
    """Test routing outgoing response back to registered chatbot."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))
    bot = DummyChatbot("dummy_bot", gw)
    bot_task = asyncio.create_task(bot.start())

    msg = Message(
        session_id="sess99",
        chatbot_id="dummy_bot",
        channel_id="chan99",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content="Response message",
        status=MessageStatus.PENDING,
    )
    await gw.post(msg)
    await asyncio.sleep(0.05)
    bot_task.cancel()
    await asyncio.gather(bot_task, return_exceptions=True)

    assert len(bot.sent_messages) == 1
    assert bot.sent_messages[0] == ("chan99", "Response message")


@pytest.mark.asyncio
async def test_gateway_history(temp_db: str) -> None:
    """Test retrieving historical messages for a session."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.post(
        Message(
            session_id="sess1",
            chatbot_id="bot1",
            channel_id="ch1",
            sender="u1",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="Msg 1",
            status=MessageStatus.RESPONDED,
        )
    )
    await gw.post(
        Message(
            session_id="sess1",
            chatbot_id="bot1",
            channel_id="ch1",
            sender="u2",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="Msg 2",
            status=MessageStatus.RESPONDED,
        )
    )
    await gw.post(
        Message(
            session_id="sess2",
            chatbot_id="bot1",
            channel_id="ch1",
            sender="u1",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="Other Session",
            status=MessageStatus.RESPONDED,
        )
    )

    history = await gw.get_session_history("sess1")
    assert len(history) == 2
    assert history[0].content == "Msg 1"
    assert history[1].content == "Msg 2"


@pytest.mark.asyncio
async def test_gateway_history_phased_sorting_thought_messages(temp_db: str) -> None:
    """Verify pre-tool and post-tool thoughts are sorted correctly in a turn."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    # Create messages in a single conversational turn
    # USER message at t=1000.0
    await gw.post(
        Message(
            id="msg_user",
            session_id="sess_thought",
            chatbot_id="bot1",
            channel_id="ch1",
            sender="u1",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="Prompt",
            timestamp=1000.0,
            status=MessageStatus.RESPONDED,
        )
    )

    # Pre-tool THOUGHT at t=1001.0
    await gw.post(
        Message(
            id="msg_thought_1",
            session_id="sess_thought",
            chatbot_id="bot1",
            channel_id="ch1",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.THOUGHT,
            content="I should call a tool",
            timestamp=1001.0,
            status=MessageStatus.RESPONDED,
            parent_id="msg_user",
        )
    )

    # TOOL_CALL at t=1002.0
    await gw.post(
        Message(
            id="msg_tool_call",
            session_id="sess_thought",
            chatbot_id="bot1",
            channel_id="ch1",
            sender="Kesoku",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_CALL,
            content="Call tool",
            timestamp=1002.0,
            status=MessageStatus.RESPONDED,
            parent_id="msg_user",
        )
    )

    # TOOL_RESULT at t=1003.0
    await gw.post(
        Message(
            id="msg_tool_result",
            session_id="sess_thought",
            chatbot_id="bot1",
            channel_id="ch1",
            sender="calculator",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_RESULT,
            content="Result is 42",
            timestamp=1003.0,
            status=MessageStatus.RESPONDED,
            parent_id="msg_tool_call",
        )
    )

    # Post-tool THOUGHT at t=1004.0
    await gw.post(
        Message(
            id="msg_thought_2",
            session_id="sess_thought",
            chatbot_id="bot1",
            channel_id="ch1",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.THOUGHT,
            content="Tool returned 42, now answering",
            timestamp=1004.0,
            status=MessageStatus.RESPONDED,
            parent_id="msg_user",
        )
    )

    # Final ASSISTANT response at t=1005.0
    await gw.post(
        Message(
            id="msg_assistant",
            session_id="sess_thought",
            chatbot_id="bot1",
            channel_id="ch1",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content="The answer is 42",
            timestamp=1005.0,
            status=MessageStatus.RESPONDED,
            parent_id="msg_user",
        )
    )

    history = await gw.get_session_history("sess_thought", limit=10)
    assert len(history) == 6

    # The sorted order should be:
    # 0: USER message
    # 1: Pre-tool THOUGHT
    # 2: TOOL_CALL
    # 3: TOOL_RESULT
    # 4: Post-tool THOUGHT
    # 5: Final ASSISTANT response
    assert history[0].id == "msg_user"
    assert history[1].id == "msg_thought_1"
    assert history[2].id == "msg_tool_call"
    assert history[3].id == "msg_tool_result"
    assert history[4].id == "msg_thought_2"
    assert history[5].id == "msg_assistant"


@pytest.mark.asyncio
async def test_gateway_history_phased_sorting_multi_iteration(temp_db: str) -> None:
    """Verify phased sorting preserves multi-iteration conversational waves correctly."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    # 1. USER prompt
    await gw.post(
        Message(
            id="msg_user",
            session_id="sess_multi",
            chatbot_id="bot1",
            channel_id="ch1",
            sender="u1",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="Prompt",
            timestamp=1000.0,
            status=MessageStatus.RESPONDED,
        )
    )

    # 2. Iteration 1 Tool Call 1 & 2
    await gw.post(
        Message(
            id="tc_1",
            session_id="sess_multi",
            chatbot_id="bot1",
            channel_id="ch1",
            sender="Kesoku",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_CALL,
            content="Call 1",
            timestamp=1001.0,
            status=MessageStatus.RESPONDED,
            parent_id="msg_user",
        )
    )
    await gw.post(
        Message(
            id="tc_2",
            session_id="sess_multi",
            chatbot_id="bot1",
            channel_id="ch1",
            sender="Kesoku",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_CALL,
            content="Call 2",
            timestamp=1001.1,
            status=MessageStatus.RESPONDED,
            parent_id="msg_user",
        )
    )

    # 3. Iteration 1 Tool Result 1 & 2
    await gw.post(
        Message(
            id="tr_1",
            session_id="sess_multi",
            chatbot_id="bot1",
            channel_id="ch1",
            sender="calculator",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_RESULT,
            content="Result 1",
            timestamp=1002.0,
            status=MessageStatus.RESPONDED,
            parent_id="tc_1",
        )
    )
    await gw.post(
        Message(
            id="tr_2",
            session_id="sess_multi",
            chatbot_id="bot1",
            channel_id="ch1",
            sender="calculator",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_RESULT,
            content="Result 2",
            timestamp=1002.1,
            status=MessageStatus.RESPONDED,
            parent_id="tc_2",
        )
    )

    # 4. Iteration 2 Thought 1 & Tool Call 3
    await gw.post(
        Message(
            id="thought_1",
            session_id="sess_multi",
            chatbot_id="bot1",
            channel_id="ch1",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.THOUGHT,
            content="I need to call another tool",
            timestamp=1003.0,
            status=MessageStatus.RESPONDED,
            parent_id="msg_user",
        )
    )
    await gw.post(
        Message(
            id="tc_3",
            session_id="sess_multi",
            chatbot_id="bot1",
            channel_id="ch1",
            sender="Kesoku",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_CALL,
            content="Call 3",
            timestamp=1004.0,
            status=MessageStatus.RESPONDED,
            parent_id="msg_user",
        )
    )

    # 5. Iteration 2 Tool Result 3
    await gw.post(
        Message(
            id="tr_3",
            session_id="sess_multi",
            chatbot_id="bot1",
            channel_id="ch1",
            sender="calculator",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_RESULT,
            content="Result 3",
            timestamp=1005.0,
            status=MessageStatus.RESPONDED,
            parent_id="tc_3",
        )
    )

    # 6. Iteration 3 Thought 2 & Final response
    await gw.post(
        Message(
            id="thought_2",
            session_id="sess_multi",
            chatbot_id="bot1",
            channel_id="ch1",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.THOUGHT,
            content="Almost done",
            timestamp=1006.0,
            status=MessageStatus.RESPONDED,
            parent_id="msg_user",
        )
    )
    await gw.post(
        Message(
            id="assistant_final",
            session_id="sess_multi",
            chatbot_id="bot1",
            channel_id="ch1",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content="Final Answer",
            timestamp=1007.0,
            status=MessageStatus.RESPONDED,
            parent_id="msg_user",
        )
    )

    history = await gw.get_session_history("sess_multi", limit=20)
    assert len(history) == 10

    expected_order = [
        "msg_user",
        "tc_1",
        "tc_2",
        "tr_1",
        "tr_2",
        "thought_1",
        "tc_3",
        "tr_3",
        "thought_2",
        "assistant_final",
    ]
    assert [m.id for m in history] == expected_order


@pytest.mark.asyncio
async def test_gateway_sessions(temp_db: str) -> None:
    """Test session creation, retrieval, update, and listing."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    # Initial state: no sessions
    assert await gw.list_sessions() == []
    assert await gw.get_latest_session() is None

    # Create session 1
    await gw.create_session("s1", "Session One")
    s1 = await gw.get_session("s1")
    assert s1 is not None
    assert s1.id == "s1"
    assert s1.title == "Session One"

    # Create session 2
    await gw.create_session("s2", "Session Two")
    latest = await gw.get_latest_session()
    assert latest is not None
    assert latest.id == "s2"

    # Update session 1
    await gw.update_session_updated_at("s1")
    sessions = await gw.list_sessions()
    assert len(sessions) == 2
    assert sessions[0].id == "s1"


@pytest.mark.asyncio
async def test_gateway_get_session_by_channel(temp_db: str) -> None:
    """Test retrieving session by chatbot and channel identifier."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    session = await gw.create_session(
        "sess_chan_1",
        "Chan Session",
        chatbot_id="discord_bot",
        channel_id="thread_777",
    )
    await gw.post(
        Message(
            session_id="sess_chan_1",
            chatbot_id="discord_bot",
            channel_id="thread_777",
            sender="User",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="Hello",
            status=MessageStatus.RESPONDED,
        )
    )

    fetched = await gw.get_session_by_channel("discord_bot", "thread_777")
    assert fetched is not None
    assert fetched.id == "sess_chan_1"

    not_found = await gw.get_session_by_channel("discord_bot", "thread_999")
    assert not_found is None


@pytest.mark.asyncio
async def test_gateway_create_session_created_at(temp_db: str) -> None:
    """Test creating a session with explicit created_at ensures correct system prompt ordering."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    hist_timestamp = 1500000000.0
    session = await gw.create_session(title="Historical", created_at=hist_timestamp)

    # Post a user message with the same timestamp
    await gw.post(
        Message(
            session_id=session.id,
            chatbot_id="discord_bot",
            channel_id="123",
            sender="User",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="First msg",
            timestamp=hist_timestamp,
            status=MessageStatus.RESPONDED,
        )
    )

    history = await gw.get_session_history(session.id)
    assert len(history) == 1
    assert history[0].role == MessageRole.USER

    # Verify system prompt exists directly on session model
    fetched_session = await gw.get_session(session.id)
    assert fetched_session is not None
    assert fetched_session.system_prompt != ""


@pytest.mark.asyncio
async def test_chatbot_ignore_completed_messages(temp_db: str) -> None:
    """Test that Chatbot.start() ignores messages already marked as MessageStatus.DELIVERED."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    # Post a message that is already completed
    await gw.post(
        Message(
            session_id="sess_1",
            chatbot_id="dummy_bot",
            channel_id="chan_1",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content="Completed message",
            status=MessageStatus.DELIVERED,
        )
    )

    bot = DummyChatbot("dummy_bot", gw)
    bot_task = asyncio.create_task(bot.start())
    await asyncio.sleep(0.05)
    bot_task.cancel()
    await asyncio.gather(bot_task, return_exceptions=True)

    # The bot should NOT have handled the completed message
    assert len(bot.sent_messages) == 0


@pytest.mark.asyncio
async def test_gateway_delete_session(temp_db: str, tmp_path: Any) -> None:
    """Test that deleting a session deletes database records and the disk workspace recursively."""
    DatabaseManager(temp_db).init_tables()

    # Setup paths
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    cfg = KesokuConfig(
        workspace=WorkspaceConfig(
            db_path=temp_db,
            sessions_dir=str(sessions_dir),
        )
    )
    gw = Gateway(context=KesokuContext(config=cfg))

    # Create session
    session = await gw.create_session("del_sess_1", "Delete Session Test")

    # Create workspace folder
    workspace_folder = sessions_dir / session.workspace_name
    workspace_folder.mkdir()

    # Create a dummy file in the workspace
    dummy_file = workspace_folder / "draft.txt"
    dummy_file.write_text("some content", encoding="utf-8")

    assert workspace_folder.exists()
    assert dummy_file.exists()

    # Verify session exists in DB
    assert await gw.get_session("del_sess_1") is not None

    # Post a message
    await gw.post(
        Message(
            session_id="del_sess_1",
            chatbot_id="discord_bot",
            channel_id="chan1",
            sender="User",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="Message to be deleted",
            status=MessageStatus.RESPONDED,
        )
    )

    # Verify messages exist
    history = await gw.get_session_history("del_sess_1")
    assert len(history) > 0

    # Delete session
    await gw.delete_session("del_sess_1")

    # Verify DB records deleted
    assert await gw.get_session("del_sess_1") is None
    history_after = await gw.get_session_history("del_sess_1")
    assert len(history_after) == 0

    # Verify workspace deleted from disk
    assert not workspace_folder.exists()


@pytest.mark.asyncio
async def test_gateway_queue_backpressure_full(temp_db: str) -> None:
    """Test that when a listener queue is full, messages are dropped without blocking."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    from kesoku.gateway.gateway import Listener
    # Create a listener with maxsize = 2
    listener = Listener(lambda msg: True, maxsize=2)
    gw._listeners.add(listener)

    # Post 3 messages
    msg1 = Message(
        session_id="sess_bp",
        chatbot_id="bot",
        channel_id="chan",
        sender="User",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Msg 1",
        status=MessageStatus.PENDING_AGENT,
    )
    msg2 = Message(
        session_id="sess_bp",
        chatbot_id="bot",
        channel_id="chan",
        sender="User",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Msg 2",
        status=MessageStatus.PENDING_AGENT,
    )
    msg3 = Message(
        session_id="sess_bp",
        chatbot_id="bot",
        channel_id="chan",
        sender="User",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Msg 3",
        status=MessageStatus.PENDING_AGENT,
    )

    # This should not block and should successfully finish posting
    await gw.post(msg1)
    await gw.post(msg2)
    await gw.post(msg3)

    # The queue should be full with exactly 2 messages
    assert listener.queue.full()
    assert listener.queue.qsize() == 2

    # The messages in queue should be msg1 and msg2 (msg3 dropped)
    m1 = listener.queue.get_nowait()
    m2 = listener.queue.get_nowait()
    assert m1.content == "Msg 1"
    assert m2.content == "Msg 2"


@pytest.mark.asyncio
async def test_gateway_listener_mutation_during_post(temp_db: str) -> None:
    """Test that mutating listeners set during post() doesn't cause issues."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    from kesoku.gateway.gateway import Listener

    # Create a listener that removes itself from gw._listeners when receiving a message
    listener = None
    def filter_func(msg: Message) -> bool:
        nonlocal listener
        # Remove itself during post processing
        gw._listeners.discard(listener)
        return True

    listener = Listener(filter_func)
    gw._listeners.add(listener)

    msg = Message(
        session_id="sess_mut",
        chatbot_id="bot",
        channel_id="chan",
        sender="User",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Trigger",
        status=MessageStatus.PENDING_AGENT,
    )

    # This should not crash and should safely process
    await gw.post(msg)
    assert len(gw._listeners) == 0


@pytest.mark.asyncio
async def test_gateway_claim_message(temp_db: str) -> None:
    """Test that claim_message atomically updates the status and prevents concurrent claims."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    msg = Message(
        session_id="sess_claim",
        chatbot_id="bot",
        channel_id="chan",
        sender="User",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Hello",
        status=MessageStatus.PENDING_AGENT,
    )
    await gw.post(msg)

    # First claim should succeed
    success = await gw.claim_message(msg.id, "processing", [MessageStatus.PENDING_AGENT])
    assert success is True

    # The message status in DB should now be "processing"
    history = await gw.get_session_history("sess_claim")
    assert len(history) == 1
    assert history[0].status == "processing"

    # Second claim with original expected status should fail
    success2 = await gw.claim_message(msg.id, "processing", [MessageStatus.PENDING_AGENT])
    assert success2 is False


@pytest.mark.asyncio
async def test_gateway_update_system_prompt(temp_db: str) -> None:
    """Verify that update_session_system_prompt successfully updates only the main system prompt in the session."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    # Create a new session which saves the initial system prompt
    session = await gw.create_session("sess_update_sys", "Test Session", system_prompt="Initial prompt")

    # Add a system nudge message with different chatbot_id to ensure it is NOT updated
    nudge = Message(
        session_id="sess_update_sys",
        chatbot_id="discord_bot",
        channel_id="chan_1",
        sender="System",
        role=MessageRole.SYSTEM,
        type=MessageType.TEXT,
        content="System nudge",
        status=MessageStatus.RESPONDED,
    )
    await gw.post(nudge)

    # Verify initial state
    history = await gw.get_session_history("sess_update_sys", limit=10)
    assert len(history) == 1
    assert history[0].role == MessageRole.SYSTEM and history[0].chatbot_id == "discord_bot"
    assert history[0].content == "System nudge"

    fetched_session = await gw.get_session("sess_update_sys")
    assert fetched_session is not None
    assert fetched_session.system_prompt == "Initial prompt"

    # Update system prompt
    await gw.update_session_system_prompt("sess_update_sys", "Updated prompt")

    # Verify updated state
    history_after = await gw.get_session_history("sess_update_sys", limit=10)
    assert len(history_after) == 1
    assert history_after[0].role == MessageRole.SYSTEM and history_after[0].chatbot_id == "discord_bot"
    assert history_after[0].content == "System nudge"  # Left untouched!

    fetched_session_after = await gw.get_session("sess_update_sys")
    assert fetched_session_after is not None
    assert fetched_session_after.system_prompt == "Updated prompt"  # Updated successfully!



