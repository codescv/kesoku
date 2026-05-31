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
    db_manager.set_active_session_for_channel(
        chatbot_id="discord", channel_id="123456", session_id="session_mapping"
    )

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

    # Save
    db_manager.save_message(msg1)
    db_manager.save_message(msg2)

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
        category="learnings",
        key="python_pref",
        title="Python Preference",
        content="User prefers type-annotated Python code.",
        role="default",
    )
    db_manager.upsert_agent_memory(
        category="learnings",
        key="rust_pref",
        title="Rust Preference",
        content="User likes cargo clean.",
        role="coder",
    )

    # Get Specific
    mem = db_manager.get_agent_memory(category="learnings", key="python_pref", role="default")
    assert mem is not None
    assert mem["title"] == "Python Preference"
    assert mem["content"] == "User prefers type-annotated Python code."

    # List filtered by category & role (should include default)
    mems = db_manager.get_agent_memories(category="learnings", role="coder")
    assert len(mems) == 2  # Should fetch both 'coder' and 'default' memories

    # Delete
    db_manager.delete_agent_memory(category="learnings", key="python_pref", role="default")
    assert db_manager.get_agent_memory(category="learnings", key="python_pref", role="default") is None


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
