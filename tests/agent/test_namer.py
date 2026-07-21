"""Unit tests for SessionNamer."""


import pytest

from kesoku.agent.llm import LLMResponse, MockLLM
from kesoku.agent.namer import SessionNamer
from kesoku.config import KesokuConfig, WorkspaceConfig
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager, Message
from kesoku.gateway.gateway import Gateway


@pytest.fixture
def temp_db(tmp_path: pytest.TempPathFactory) -> str:
    """Provide a temporary database path."""
    return str(tmp_path / "test_namer.db")


@pytest.mark.asyncio
async def test_session_namer_success(temp_db: str) -> None:
    """Test successful auto-naming."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    ctx = KesokuContext(config=cfg)
    gw = Gateway(context=ctx)

    await gw.create_session("sess1", title="New Session")

    # Seed first turn
    msg_user = Message(
        session_id="sess1",
        chatbot_id="discord",
        channel_id="ch1",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Hello, who are you?",
        status=MessageStatus.PROCESSED,
    )
    msg_assistant = Message(
        session_id="sess1",
        chatbot_id="discord",
        channel_id="ch1",
        sender="assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content="I am Kesoku AI.",
        status=MessageStatus.DELIVERED,
    )
    await gw.db.save_message(msg_user)
    await gw.db.save_message(msg_assistant)

    llm = MockLLM(responses=[LLMResponse(content="Kesoku Introduction", tool_calls=[])])

    namer = SessionNamer(db=gw.db, gateway=gw, llm=llm)

    success = await namer.auto_rename_session("sess1")
    assert success is True

    # Verify session title updated in DB
    session = await gw.db.get_session("sess1")
    assert session.title == "Kesoku Introduction"

    # Verify rename message posted
    messages = await gw.db.get_messages_by_filters(filters={"session_id": "sess1"})
    rename_msgs = [
        m
        for m in messages
        if m.role == MessageRole.SYSTEM and m.type == MessageType.SESSION_RENAME
    ]
    assert len(rename_msgs) == 1
    assert rename_msgs[0].content == "Kesoku Introduction"
    assert rename_msgs[0].chatbot_id == "discord"
    assert rename_msgs[0].channel_id == "ch1"


@pytest.mark.asyncio
async def test_session_namer_wrong_turns_count(temp_db: str) -> None:
    """Test that naming is skipped if turns count is not 1 or 2."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    ctx = KesokuContext(config=cfg)
    gw = Gateway(context=ctx)

    await gw.create_session("sess1", title="New Session")

    namer = SessionNamer(db=gw.db, gateway=gw, llm=MockLLM())

    # 0 turns
    success = await namer.auto_rename_session("sess1")
    assert success is False

    # Seed 3 turns (3 user messages)
    for i in range(3):
        msg = Message(
            session_id="sess1",
            chatbot_id="discord",
            channel_id="ch1",
            sender="user",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content=f"msg {i}",
            status=MessageStatus.PROCESSED,
        )
        await gw.db.save_message(msg)

    success = await namer.auto_rename_session("sess1")
    assert success is False


@pytest.mark.asyncio
async def test_session_namer_empty_response(temp_db: str) -> None:
    """Test that naming fails if LLM returns empty response."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    ctx = KesokuContext(config=cfg)
    gw = Gateway(context=ctx)

    await gw.create_session("sess1", title="New Session")

    msg_user = Message(
        session_id="sess1",
        chatbot_id="discord",
        channel_id="ch1",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Hello",
        status=MessageStatus.PROCESSED,
    )
    msg_assistant = Message(
        session_id="sess1",
        chatbot_id="discord",
        channel_id="ch1",
        sender="assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content="Hi",
        status=MessageStatus.DELIVERED,
    )
    await gw.db.save_message(msg_user)
    await gw.db.save_message(msg_assistant)

    llm = MockLLM(responses=[LLMResponse(content="", tool_calls=[])])

    namer = SessionNamer(db=gw.db, gateway=gw, llm=llm)

    success = await namer.auto_rename_session("sess1")
    assert success is False

    # Title should not be updated
    session = await gw.db.get_session("sess1")
    assert session.title == "New Session"
