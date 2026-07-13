"""Unit tests for the Chat History Search and Message tools."""

import time

import pytest

from kesoku.agent.tools import (
    ToolContext,
    chat_search,
    view_message,
)
from kesoku.config import KesokuConfig, WorkspaceConfig
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager, Message, Session


@pytest.mark.asyncio
async def test_view_message_tool(tmp_path) -> None:
    """Test that view_message retrieves full message details successfully."""
    from kesoku.gateway.gateway import Gateway

    temp_db = str(tmp_path / "test_view_msg.db")
    db = DatabaseManager(temp_db)
    db.init_tables()

    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    session1 = Session(id="sess_abc", title="Sess ABC", created_at=time.time(), updated_at=time.time())
    await gw.db.create_session(session1)

    msg = Message(
        id="unique_msg_999",
        session_id="sess_abc",
        chatbot_id="discord",
        channel_id="chan_1",
        sender="Asuka",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content="This is a secret long content.",
        status=MessageStatus.RESPONDED,
    )
    await gw.post(msg)

    ctx = ToolContext(
        session_id="sess_abc",
        session_workspace="test_ws",
        original_msg_id="unique_msg_999",
        chatbot_id="discord",
        channel_id="chan_1",
        gateway=gw,
    )

    # Test retrieval
    res = await view_message("unique_msg_999", context=ctx)
    assert "Message Details" in res
    assert "unique_msg_999" in res
    assert "Asuka (assistant)" in res
    assert "This is a secret long content." in res

    # Test non-existent retrieval
    res_fail = await view_message("non_existent_id", context=ctx)
    assert "not found" in res_fail


@pytest.mark.asyncio
async def test_chat_search_hybrid_and_boosting(tmp_path, monkeypatch) -> None:
    """Test chat_search hybrid exact matching and score boosting + text truncation."""
    from kesoku.gateway.gateway import Gateway

    temp_db = str(tmp_path / "test_search_hybrid.db")
    db = DatabaseManager(temp_db)
    db.init_tables()

    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    await gw.db.set_channel_role("discord", "chan_1", "coder")
    session1 = Session(id="sess_1", title="Sess 1", created_at=time.time(), updated_at=time.time())
    await gw.db.create_session(session1)
    await gw.db.set_active_session_for_channel("discord", "chan_1", "sess_1")

    embeddings_map = {
        "Game Console Joystick": [1.0, 0.0],
        "How controller is built": [0.5, 0.5],
    }

    def pad_vector(v):
        return v + [0.0] * (384 - len(v))

    padded_map = {k: pad_vector(v) for k, v in embeddings_map.items()}

    def mock_get_embedding(text: str) -> list[float]:
        if text == "controller":
            return pad_vector([0.2, 0.8])
        for key_text, vec in padded_map.items():
            if text == key_text or key_text in text:
                return vec
        return pad_vector([0.0, 0.0])

    monkeypatch.setattr("kesoku.utils.embedding.get_embedding", mock_get_embedding)
    monkeypatch.setattr("kesoku.utils.embedding.get_embeddings", lambda texts: [mock_get_embedding(t) for t in texts])

    msg_sem = Message(
        id="m_sem",
        session_id="sess_1",
        chatbot_id="discord",
        channel_id="chan_1",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Game Console Joystick",
        timestamp=1.0,
        status=MessageStatus.PROCESSED,
    )
    msg_exact = Message(
        id="m_exact",
        session_id="sess_1",
        chatbot_id="discord",
        channel_id="chan_1",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Hello controller world",
        timestamp=2.0,
        status=MessageStatus.PROCESSED,
    )
    msg_both = Message(
        id="m_both",
        session_id="sess_1",
        chatbot_id="discord",
        channel_id="chan_1",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="How controller is built",
        timestamp=3.0,
        status=MessageStatus.PROCESSED,
    )
    long_content = "controller " + "a" * 600
    msg_long = Message(
        id="m_long",
        session_id="sess_1",
        chatbot_id="discord",
        channel_id="chan_1",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content=long_content,
        timestamp=4.0,
        status=MessageStatus.PROCESSED,
    )

    db.save_message(msg_sem)
    db.save_message(msg_exact)
    db.save_message(msg_both)
    db.save_message(msg_long)

    ctx = ToolContext(
        session_id="sess_1",
        session_workspace="test_ws",
        gateway=gw,
    )

    res = await chat_search(query="controller", limit=10, context=ctx)

    assert "m_both" in res
    assert "m_exact" in res
    assert "m_sem" not in res
    assert "m_long" in res

    pos_both = res.index("m_both")
    pos_exact = res.index("m_exact")

    assert pos_both < pos_exact

    assert "Truncated for Brevity" in res
    assert len(res) < 1500


@pytest.mark.asyncio
async def test_chat_search_wildcard_and_time_filters(tmp_path, monkeypatch) -> None:
    """Test chat_search supporting wildcard queries and time range filtering."""
    from kesoku.gateway.gateway import Gateway

    temp_db = str(tmp_path / "test_search_wildcard.db")
    db = DatabaseManager(temp_db)
    db.init_tables()

    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    await gw.db.set_channel_role("discord", "chan_1", "coder")
    session1 = Session(id="sess_1", title="Sess 1", created_at=time.time(), updated_at=time.time())
    await gw.db.create_session(session1)
    await gw.db.set_active_session_for_channel("discord", "chan_1", "sess_1")

    with db.connection_provider.connection() as conn:
        with conn:
            sql = (
                "INSERT INTO messages ("
                "  id, session_id, chatbot_id, channel_id, sender, role, type, "
                "  content, metadata, timestamp, status, embedding"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
            conn.execute(sql, (
                "m1", "sess_1", "discord", "chan_1", "user", "user", "text",
                "first message", "{}", 1000.0, "processed", None
            ))
            conn.execute(sql, (
                "m2", "sess_1", "discord", "chan_1", "user", "user", "text",
                "second message", "{}", 2000.0, "processed", None
            ))
            conn.execute(sql, (
                "m3", "sess_1", "discord", "chan_1", "user", "user", "text",
                "third message", "{}", 3000.0, "processed", None
            ))

    ctx = ToolContext(
        session_id="sess_1",
        session_workspace="test_ws",
        gateway=gw,
    )

    res_wildcard = await chat_search(query="*", limit=10, context=ctx)
    assert "Search Results for '*'" in res_wildcard
    assert "m3" in res_wildcard
    assert "m2" in res_wildcard
    assert "m1" in res_wildcard

    pos_m3 = res_wildcard.index("m3")
    pos_m2 = res_wildcard.index("m2")
    pos_m1 = res_wildcard.index("m1")
    assert pos_m3 < pos_m2 < pos_m1
    assert "score:" not in res_wildcard

    res_time = await chat_search(
        query="*",
        start_time="1500",
        end_time="2500",
        limit=10,
        context=ctx
    )
    assert "m2" in res_time
    assert "m1" not in res_time
    assert "m3" not in res_time


@pytest.mark.asyncio
async def test_chat_search_chunks_limit(tmp_path, monkeypatch) -> None:
    """Test that chat_search limits matching chunks from the same message to at most 3 in the final ranking."""
    from kesoku.gateway.gateway import Gateway

    temp_db = str(tmp_path / "test_search_chunks_limit.db")
    db = DatabaseManager(temp_db)
    db.init_tables()

    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    await gw.db.set_channel_role("discord", "chan_1", "coder")
    session1 = Session(id="sess_1", title="Sess 1", created_at=time.time(), updated_at=time.time())
    await gw.db.create_session(session1)
    await gw.db.set_active_session_for_channel("discord", "chan_1", "sess_1")

    # Mock embeddings to be neutral so exact matching (literal query match) determines the ranking
    def mock_get_embedding(text: str) -> list[float]:
        return [0.0] * 384
    monkeypatch.setattr("kesoku.utils.embedding.get_embedding", mock_get_embedding)
    monkeypatch.setattr("kesoku.utils.embedding.get_embeddings", lambda texts: [mock_get_embedding(t) for t in texts])

    # A message containing 5 distinct sentences matching the keyword "target"
    # With threshold=15, they will be chunked into 5 separate chunks.
    msg = Message(
        id="m_multi",
        session_id="sess_1",
        chatbot_id="discord",
        channel_id="chan_1",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content=(
            "target sentence one which is quite long. "
            "target sentence two which is also very long. "
            "target sentence three which is also long. "
            "target sentence four which is long. "
            "target sentence five which is long."
        ),
        timestamp=time.time(),
        status=MessageStatus.PROCESSED,
    )
    # This calls db.save_message internally which chunks it
    await gw.post(msg)

    ctx = ToolContext(
        session_id="sess_1",
        session_workspace="test_ws",
        gateway=gw,
    )

    # Search for "target"
    res = await chat_search(query="target", limit=10, context=ctx)

    # Verify that the message ID is shown but only 3 times
    assert res.count("(ID: `m_multi`)") == 3
