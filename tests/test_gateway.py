"""Unit tests for Kesoku Gateway and Chatbot routing."""

import asyncio
import sqlite3
from typing import Any

import pytest

from kesoku.config import WorkspaceConfig
from kesoku.constants import (
    ROLE_ASSISTANT,
    ROLE_USER,
    STATUS_PENDING,
    STATUS_RESPONDED,
    TYPE_TEXT,
)
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
        await self.gateway.update_message_status(message.id, STATUS_COMPLETED)


@pytest.fixture
def temp_db(tmp_path: Any) -> str:
    db_file = tmp_path / "test_kesoku.db"
    return str(db_file)


@pytest.mark.asyncio
async def test_gateway_init_db(temp_db: str) -> None:
    """Verify SQLite database schema is initialized correctly."""
    DatabaseManager(temp_db).init_tables()
    gw = Gateway(workspace_config=WorkspaceConfig(db_path=temp_db))

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
    gw = Gateway(workspace_config=WorkspaceConfig(db_path=temp_db))
    bot = DummyChatbot("dummy_bot", gw)
    bot_task = asyncio.create_task(bot.start())

    msg = Message(
        session_id="sess99",
        chatbot_id="dummy_bot",
        channel_id="chan99",
        sender="Kesoku",
        role=ROLE_ASSISTANT,
        type=TYPE_TEXT,
        content="Response message",
        status=STATUS_PENDING,
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
    gw = Gateway(workspace_config=WorkspaceConfig(db_path=temp_db))
    await gw.post(
        Message(
            session_id="sess1",
            chatbot_id="bot1",
            channel_id="ch1",
            sender="u1",
            role=ROLE_USER,
            type=TYPE_TEXT,
            content="Msg 1",
            status=STATUS_RESPONDED,
        )
    )
    await gw.post(
        Message(
            session_id="sess1",
            chatbot_id="bot1",
            channel_id="ch1",
            sender="u2",
            role=ROLE_USER,
            type=TYPE_TEXT,
            content="Msg 2",
            status=STATUS_RESPONDED,
        )
    )
    await gw.post(
        Message(
            session_id="sess2",
            chatbot_id="bot1",
            channel_id="ch1",
            sender="u1",
            role=ROLE_USER,
            type=TYPE_TEXT,
            content="Other Session",
            status=STATUS_RESPONDED,
        )
    )

    history = await gw.get_session_history("sess1")
    assert len(history) == 2
    assert history[0].content == "Msg 1"
    assert history[1].content == "Msg 2"


@pytest.mark.asyncio
async def test_gateway_sessions(temp_db: str) -> None:
    """Test session creation, retrieval, update, and listing."""
    DatabaseManager(temp_db).init_tables()
    gw = Gateway(workspace_config=WorkspaceConfig(db_path=temp_db))

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
