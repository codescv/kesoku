"""Unit tests for DatabaseManager and models in kesoku.db."""

import os
import time

import pytest

from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import DatabaseManager, Message, Session


@pytest.fixture
def temp_db_path(tmp_path):
    """Provides a temporary path for testing the SQLite database."""
    db_file = tmp_path / "test_kesoku.db"
    return str(db_file)


@pytest.fixture
def db_manager(temp_db_path):
    """Initializes a DatabaseManager with clean tables."""
    manager = DatabaseManager(temp_db_path)
    manager.init_tables()
    return manager


def test_db_initialization_and_verification(temp_db_path):
    """Tests verifying and initializing database schema."""
    manager = DatabaseManager(temp_db_path)

    # Verification should fail before init
    with pytest.raises(RuntimeError, match="does not exist or is empty"):
        manager.verify_db()

    # Init tables
    manager.init_tables()
    assert os.path.exists(temp_db_path)
    assert os.path.getsize(temp_db_path) > 0

    # Verification should pass after init
    manager.verify_db()


def test_session_crud(db_manager):
    """Tests basic session creation, retrieval, listing, and deletion."""
    session1 = Session(
        id="session_abc",
        title="First Chat Session",
        created_at=time.time(),
        updated_at=time.time(),
        system_prompt="Be a helpful assistant.",
    )
    session2 = Session(
        id="session_xyz",
        title="Second Chat Session",
        created_at=time.time() - 10,
        updated_at=time.time() - 10,
        system_prompt="Be a creative coder.",
    )

    # Create
    db_manager.create_session(session1)
    db_manager.create_session(session2)

    # Retrieve
    retrieved = db_manager.get_session("session_abc")
    assert retrieved is not None
    assert retrieved.id == "session_abc"
    assert retrieved.title == "First Chat Session"
    assert retrieved.system_prompt == "Be a helpful assistant."

    # List
    sessions = db_manager.list_sessions()
    assert len(sessions) == 2
    assert sessions[0].id == "session_abc"  # Most recently updated first

    # Update updated_at
    new_ts = time.time() + 100
    db_manager.update_session_updated_at("session_xyz", new_ts)
    latest = db_manager.get_latest_session()
    assert latest.id == "session_xyz"

    # Delete
    db_manager.delete_session("session_abc")
    assert db_manager.get_session("session_abc") is None
    assert len(db_manager.list_sessions()) == 1


def test_channel_session_mappings(db_manager):
    """Tests binding sessions to specific channels and retrieving them."""
    session = Session(
        id="session_mapping",
        title="Channel Session Mapping",
        created_at=time.time(),
        updated_at=time.time(),
    )
    db_manager.create_session(session)

    # Set mapping
    db_manager.set_active_session_for_channel(chatbot_id="discord", channel_id="123456", session_id="session_mapping")

    # Retrieve session by channel
    bound_session = db_manager.get_session_by_channel(chatbot_id="discord", channel_id="123456")
    assert bound_session is not None
    assert bound_session.id == "session_mapping"

    # Retrieve channel by session
    mapping = db_manager.get_channel_by_session("session_mapping")
    assert mapping == ("discord", "123456")


def test_message_crud(db_manager):
    """Tests saving, status claims, filters, and turn counts for messages."""
    session = Session(id="sess_msg_test", title="Msg Test", created_at=time.time(), updated_at=time.time())
    db_manager.create_session(session)

    msg1 = Message(
        id="msg_1",
        session_id="sess_msg_test",
        chatbot_id="cli",
        channel_id="terminal",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Hello AI",
        timestamp=time.time(),
        status=MessageStatus.PENDING,
    )
    msg2 = Message(
        id="msg_2",
        session_id="sess_msg_test",
        chatbot_id="cli",
        channel_id="terminal",
        sender="assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content="Hello Human",
        timestamp=time.time() + 1,
        status=MessageStatus.PROCESSING,
    )
    msg_thought = Message(
        id="msg_thought",
        session_id="sess_msg_test",
        chatbot_id="cli",
        channel_id="terminal",
        sender="assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.THOUGHT,
        content="Thinking process...",
        timestamp=time.time() + 2,
        status=MessageStatus.PROCESSING,
    )

    # Save
    db_manager.save_message(msg1)
    db_manager.save_message(msg2)
    db_manager.save_message(msg_thought)

    # Verify embeddings are populated correctly
    saved_msg1 = db_manager.get_messages_by_filters(filters={"id": "msg_1"})[0]
    assert saved_msg1.embedding is not None

    saved_msg2 = db_manager.get_messages_by_filters(filters={"id": "msg_2"})[0]
    assert saved_msg2.embedding is not None

    saved_thought = db_manager.get_messages_by_filters(filters={"id": "msg_thought"})[0]
    assert saved_thought.embedding is None

    # Turn count (counting only User messages)
    assert db_manager.get_session_turns_count("sess_msg_test") == 1

    # Message filters
    pending_messages = db_manager.get_messages_by_filters(
        filters={"session_id": "sess_msg_test", "status": MessageStatus.PENDING}
    )
    assert len(pending_messages) == 1
    assert pending_messages[0].id == "msg_1"

    # Update status
    db_manager.update_message_status("msg_1", MessageStatus.DELIVERED)
    updated_msg = db_manager.get_messages_by_filters(filters={"id": "msg_1"})[0]
    assert updated_msg.status == MessageStatus.DELIVERED

    # Claim message atomically
    claimed = db_manager.claim_message(
        message_id="msg_2",
        new_status=MessageStatus.DELIVERED,
        expected_statuses=[MessageStatus.PROCESSING],
    )
    assert claimed is True
    claimed_msg = db_manager.get_messages_by_filters(filters={"id": "msg_2"})[0]
    assert claimed_msg.status == MessageStatus.DELIVERED

    # Claim message with wrong status fails
    failed_claim = db_manager.claim_message(
        message_id="msg_2",
        new_status=MessageStatus.PENDING,
        expected_statuses=[MessageStatus.PROCESSING],
    )
    assert failed_claim is False


def test_agent_memory_crud(db_manager):
    """Tests agent memory upsertion, retrieval, listing, and deletion."""
    db_manager.upsert_agent_memory(
        category="memo",
        key="python_pref",
        title="Python Preference",
        content="User prefers type-annotated Python code.",
        role="default",
    )
    db_manager.upsert_agent_memory(
        category="memo",
        key="rust_pref",
        title="Rust Preference",
        content="User likes cargo clean.",
        role="coder",
    )

    # Get Specific
    mem = db_manager.get_agent_memory(category="memo", key="python_pref", role="default")
    assert mem is not None
    assert mem["title"] == "Python Preference"
    assert mem["content"] == "User prefers type-annotated Python code."

    # List filtered by category & role (should NOT include default)
    mems = db_manager.get_agent_memories(category="memo", role="coder")
    assert len(mems) == 1  # Should fetch only 'coder' memories

    # Delete
    db_manager.delete_agent_memory(category="memo", key="python_pref", role="default")
    assert db_manager.get_agent_memory(category="memo", key="python_pref", role="default") is None


def test_cross_session_context_and_locking(db_manager):
    """Tests cross-session memory updates and lock mechanics with deadlock self-healing."""
    db_manager.upsert_cross_session_context(role="default", content="Initial default context summary")

    # Get context
    ctx = db_manager.get_cross_session_context("default")
    assert ctx is not None
    assert ctx.content == "Initial default context summary"
    assert ctx.status == "idle"

    # Atomically claim update lock
    locked = db_manager.claim_cross_session_context_for_update("default")
    assert locked is True

    # Second lock attempt should fail
    locked_again = db_manager.claim_cross_session_context_for_update("default")
    assert locked_again is False

    # Release lock and update content
    db_manager.release_cross_session_context_lock(role="default", content="Updated context summary")
    updated_ctx = db_manager.get_cross_session_context("default")
    assert updated_ctx.content == "Updated context summary"
    assert updated_ctx.status == "idle"


def test_cross_session_context_stale_lock_self_healing(db_manager):
    """Tests that stale locks older than 5 minutes are self-healed and re-claimable."""
    # Manually inject a stale lock directly into SQLite table
    with db_manager.connection_provider.connection() as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO cross_session_contexts (role, content, updated_at, status)
                VALUES ('coder_role', 'Coder content', ?, 'updating')
                """,
                (time.time() - 301,),  # 301 seconds ago (more than 5 minutes)
            )

    # Claim lock should heal the stale lock and return True
    locked = db_manager.claim_cross_session_context_for_update("coder_role")
    assert locked is True


def test_cronjob_thread_aware_stats_and_timestamp(db_manager):
    """Tests that get_cronjob_sent_stats_today and get_last_message_timestamp
    correctly match parent channels from thread metadata.
    """
    session = Session(
        id="sess_thread_test",
        title="Thread Test",
        created_at=time.time(),
        updated_at=time.time(),
    )
    db_manager.create_session(session)

    now = time.time()

    # 1. Save a cronjob message inside a thread (thread_1) whose parent is parent_chan_1
    msg_cron_thread = Message(
        id="msg_cron_t1",
        session_id="sess_thread_test",
        chatbot_id="discord",
        channel_id="thread_1",
        sender="Cronjob",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Scheduled review question",
        timestamp=now - 60,
        metadata={"parent_channel_id": "parent_chan_1", "is_cronjob": True},
        status=MessageStatus.PENDING,
    )
    db_manager.save_message(msg_cron_thread)

    # 2. Save a regular user message inside another thread (thread_2) whose parent is parent_chan_1
    msg_user_thread = Message(
        id="msg_user_t2",
        session_id="sess_thread_test",
        chatbot_id="discord",
        channel_id="thread_2",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="User reply",
        timestamp=now - 30,
        metadata={"parent_channel_id": "parent_chan_1"},
        status=MessageStatus.PENDING,
    )
    db_manager.save_message(msg_user_thread)

    # 3. Query cronjob stats on the parent channel. It should find the cron message sent in thread_1.
    count, last_ts = db_manager.get_cronjob_sent_stats_today(chatbot_id="discord", channel_id="parent_chan_1")
    assert count == 1
    assert last_ts is not None
    assert abs(last_ts - (now - 60)) < 1e-3

    # 4. Query last message timestamp on the parent channel. It should find the user reply in thread_2.
    last_msg_ts = db_manager.get_last_message_timestamp(chatbot_id="discord", channel_id="parent_chan_1")
    assert last_msg_ts is not None
    assert abs(last_msg_ts - (now - 30)) < 1e-3


def test_search_role_data(db_manager):
    """Tests search_role_memories and search_role_messages."""
    # Set up role 'coder' on channel 'chan_1'
    db_manager.set_channel_role("discord", "chan_1", "coder")

    session1 = Session(id="sess_1", title="Sess 1", created_at=time.time(), updated_at=time.time())
    db_manager.create_session(session1)
    db_manager.set_active_session_for_channel("discord", "chan_1", "sess_1")

    # Save messages for sess_1 (role 'coder')
    msg1 = Message(
        id="m1",
        session_id="sess_1",
        chatbot_id="discord",
        channel_id="chan_1",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="I love python coding",
        timestamp=time.time(),
        status=MessageStatus.PROCESSED,
    )
    msg2 = Message(
        id="m2",
        session_id="sess_1",
        chatbot_id="discord",
        channel_id="chan_1",
        sender="assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.THOUGHT,
        content="Thinking about python",
        timestamp=time.time() + 1,
        status=MessageStatus.RESPONDED,
    )
    msg3 = Message(
        id="m3",
        session_id="sess_1",
        chatbot_id="discord",
        channel_id="chan_1",
        sender="assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content="Here is python code",
        timestamp=time.time() + 2,
        status=MessageStatus.DELIVERED,
    )
    db_manager.save_message(msg1)
    db_manager.save_message(msg2)
    db_manager.save_message(msg3)

    # Set up role 'helper' on channel 'chan_2'
    db_manager.set_channel_role("discord", "chan_2", "helper")
    session2 = Session(id="sess_2", title="Sess 2", created_at=time.time(), updated_at=time.time())
    db_manager.create_session(session2)
    db_manager.set_active_session_for_channel("discord", "chan_2", "sess_2")

    msg4 = Message(
        id="m4",
        session_id="sess_2",
        chatbot_id="discord",
        channel_id="chan_2",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Help me with python please",
        timestamp=time.time(),
        status=MessageStatus.PROCESSED,
    )
    db_manager.save_message(msg4)

    # Insert memories
    db_manager.upsert_agent_memory("memo", "mem1", "Python Tip", "Python is great", "default")
    db_manager.upsert_agent_memory("memo", "mem2", "Coder Tip", "Write python code", "coder")
    db_manager.upsert_agent_memory("memo", "mem3", "Helper Tip", "Help python users", "helper")

    # Search messages for 'coder'
    coder_msgs = db_manager.search_role_messages("coder", "python")
    assert len(coder_msgs) == 2
    assert {m.id for m in coder_msgs} == {"m1", "m3"}

    # Search memories for 'coder'
    coder_mems = db_manager.search_role_memories("coder", "python")
    assert len(coder_mems) == 1
    assert {m["key"] for m in coder_mems} == {"mem2"}


def test_search_role_data_wildcard_and_filters(db_manager):
    """Tests search_role_memories and search_role_messages with wildcard and filters."""
    # Set up role 'coder' on channel 'chan_1'
    db_manager.set_channel_role("discord", "chan_1", "coder")

    session1 = Session(id="sess_1", title="Sess 1", created_at=1700000000, updated_at=1700000000)
    db_manager.create_session(session1)
    db_manager.set_active_session_for_channel("discord", "chan_1", "sess_1")

    # Save messages with specific timestamps
    # 2026-06-15 12:00:00 UTC (1781534400.0)
    base_ts = 1781534400.0

    msg1 = Message(
        id="m1",
        session_id="sess_1",
        chatbot_id="discord",
        channel_id="chan_1",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="First python msg",
        timestamp=base_ts,
        status=MessageStatus.PROCESSED,
    )
    msg2 = Message(
        id="m2",
        session_id="sess_1",
        chatbot_id="discord",
        channel_id="chan_1",
        sender="assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content="Second python msg",
        timestamp=base_ts + 3600,
        status=MessageStatus.RESPONDED,  # +1 hr
    )
    msg3 = Message(
        id="m3",
        session_id="sess_1",
        chatbot_id="discord",
        channel_id="chan_1",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Third java msg",
        timestamp=base_ts + 7200,
        status=MessageStatus.PROCESSED,  # +2 hr
    )
    db_manager.save_message(msg1)
    db_manager.save_message(msg2)
    db_manager.save_message(msg3)

    # Insert memories
    db_manager.upsert_agent_memory("memo", "mem1", "Python Tip", "Python is great", "default")
    db_manager.upsert_agent_memory("memo", "mem2", "Java Tip", "Java is verbose", "coder")

    # We need to set explicit updated_at for memories
    with db_manager.connection_provider.connection() as conn:
        with conn:
            conn.execute("UPDATE agent_memories SET updated_at = ? WHERE key = 'mem1'", (base_ts,))
            conn.execute("UPDATE agent_memories SET updated_at = ? WHERE key = 'mem2'", (base_ts + 3600,))

    # Test 1: Wildcard content search for messages (returns all messages)
    msgs = db_manager.search_role_messages("coder", "*")
    assert len(msgs) == 3
    assert {m.id for m in msgs} == {"m1", "m2", "m3"}

    msgs_empty = db_manager.search_role_messages("coder", "")
    assert len(msgs_empty) == 3
    assert {m.id for m in msgs_empty} == {"m1", "m2", "m3"}

    # Test 2: Wildcard content search for memories (should NOT include default)
    mems = db_manager.search_role_memories("coder", "*")
    assert len(mems) == 1
    assert {m["key"] for m in mems} == {"mem2"}

    # Test 3: Time range filtering for messages
    msgs_time = db_manager.search_role_messages("coder", "*", start_time=base_ts, end_time=base_ts + 4000)
    assert len(msgs_time) == 2
    assert {m.id for m in msgs_time} == {"m1", "m2"}

    # Test 4: Time range filtering for memories
    mems_time = db_manager.search_role_memories("coder", "*", start_time=base_ts + 1000, end_time=base_ts + 5000)
    assert len(mems_time) == 1
    assert mems_time[0]["key"] == "mem2"

    # Test 5: Limit filtering
    msgs_limit = db_manager.search_role_messages("coder", "*", limit=2)
    assert len(msgs_limit) == 2
    # Should return latest first
    assert msgs_limit[0].id == "m3"
    assert msgs_limit[1].id == "m2"


def test_thread_session_role_inheritance(db_manager):
    """Tests that a thread session created with a specific inherited role is not
    overwritten to 'default' when set_active_session_for_channel is called.
    """
    # 1. Parent channel has role 'asuka'
    db_manager.set_channel_role("discord", "parent_chan", "asuka")

    # 2. Create a session initialized with role 'asuka'
    session = Session(
        id="sess_thread_asuka",
        title="Thread Session with Asuka",
        created_at=time.time(),
        updated_at=time.time(),
        role_name="asuka",
    )
    db_manager.create_session(session)

    # 3. Call set_active_session_for_channel for the thread channel.
    # Note that there are NO messages in this session yet, so querying parent channel via messages would fail.
    db_manager.set_active_session_for_channel(
        chatbot_id="discord",
        channel_id="thread_chan",
        session_id="sess_thread_asuka",
    )

    # 4. Verify that the session's role remains 'asuka' and was not overwritten by 'default'
    retrieved = db_manager.get_session("sess_thread_asuka")
    assert retrieved is not None
    assert retrieved.role_name == "asuka"


def test_agent_memory_semantic_search(db_manager, monkeypatch):
    """Tests semantic search on agent memories by mocking embeddings."""
    embeddings_map = {
        "Python Tip\nPython is great": [1.0, 0.0],
        "Java Tip\nJava is verbose": [0.0, 1.0],
        "Weather Tip\nIt is raining today": [0.5, 0.5],
        "Find python tips": [0.9, 0.1],
    }

    def pad_vector(v):
        return v + [0.0] * (384 - len(v))

    padded_map = {k: pad_vector(v) for k, v in embeddings_map.items()}

    def mock_get_embedding(text: str) -> list[float]:
        for key_text, vec in padded_map.items():
            if text in key_text or key_text in text:
                return vec
        return pad_vector([0.0, 0.0])

    def mock_get_embeddings(texts: list[str]) -> list[list[float]]:
        return [mock_get_embedding(t) for t in texts]

    monkeypatch.setattr("kesoku.utils.embedding.get_embedding", mock_get_embedding)
    monkeypatch.setattr("kesoku.utils.embedding.get_embeddings", mock_get_embeddings)

    db_manager.upsert_agent_memory("memo", "mem_py", "Python Tip", "Python is great", "coder")
    db_manager.upsert_agent_memory("memo", "mem_java", "Java Tip", "Java is verbose", "coder")
    db_manager.upsert_agent_memory("memo", "mem_weather", "Weather Tip", "It is raining today", "coder")

    results = db_manager.search_role_memories_semantic(role="coder", query_text="Find python tips", limit=10)

    assert len(results) == 3
    assert results[0]["key"] == "mem_py"
    assert results[1]["key"] == "mem_weather"
    assert results[2]["key"] == "mem_java"

    assert "similarity_score" in results[0]
    assert results[0]["similarity_score"] > results[1]["similarity_score"]
    assert results[1]["similarity_score"] > results[2]["similarity_score"]

