"""Unit tests for the CrossSessionContext schema and CAS locking mechanisms."""

import time

import pytest

from kesoku.db import DatabaseManager


@pytest.mark.asyncio
async def test_cross_session_context_crud(tmp_path) -> None:
    """Test basic CRUD operations on the SQLite cross_session_contexts table."""
    temp_db = str(tmp_path / "test_context_crud.db")
    db = DatabaseManager(temp_db)
    db.init_tables()

    role = "programmer"

    # 1. Initial fetch should be None
    ctx = db.get_cross_session_context(role)
    assert ctx is None

    # 2. Upsert a new context
    db.upsert_cross_session_context(role, "Active working on refactoring.")

    # 3. Retrieve and verify
    ctx = db.get_cross_session_context(role)
    assert ctx is not None
    assert ctx.role == role
    assert ctx.content == "Active working on refactoring."
    assert ctx.status == "idle"
    assert isinstance(ctx.updated_at, float)


@pytest.mark.asyncio
async def test_cross_session_context_atomic_locking(tmp_path) -> None:
    """Test atomic Compare-And-Swap (CAS) locking behavior."""
    temp_db = str(tmp_path / "test_context_locking.db")
    db = DatabaseManager(temp_db)
    db.init_tables()

    role = "assistant"
    db.upsert_cross_session_context(role, "Initial context content.")

    # 1. First lock claim should succeed (idle -> updating)
    success1 = db.claim_cross_session_context_for_update(role)
    assert success1 is True

    # Verify status is updated to updating in the DB
    ctx = db.get_cross_session_context(role)
    assert ctx is not None
    assert ctx.status == "updating"

    # 2. Second lock claim should fail since status is already 'updating'
    success2 = db.claim_cross_session_context_for_update(role)
    assert success2 is False

    # 3. Releasing the lock should set status to 'idle' and update the content
    db.release_cross_session_context_lock(role, "Consolidated new summary content.")
    ctx = db.get_cross_session_context(role)
    assert ctx is not None
    assert ctx.content == "Consolidated new summary content."
    assert ctx.status == "idle"

    # 4. Now claiming lock should succeed again
    success3 = db.claim_cross_session_context_for_update(role)
    assert success3 is True


@pytest.mark.asyncio
async def test_cross_session_context_lock_expiry(tmp_path) -> None:
    """Test that claim_cross_session_context_for_update heals stale locks (>300s)."""
    temp_db = str(tmp_path / "test_context_expiry.db")
    db = DatabaseManager(temp_db)
    db.init_tables()

    role = "waifu"
    db.upsert_cross_session_context(role, "Base content.")

    # Manually simulate an active lock by updating the DB with status='updating'
    # but back-date the updated_at timestamp to 400 seconds ago (stale)
    conn = db._get_connection()
    try:
        with conn:
            conn.execute(
                """
                UPDATE cross_session_contexts
                SET status = 'updating', updated_at = ?
                WHERE role = ?
                """,
                (time.time() - 400, role),
            )
    finally:
        conn.close()

    # Verify manually set stale state
    ctx_before = db.get_cross_session_context(role)
    assert ctx_before is not None
    assert ctx_before.status == "updating"

    # Claim lock should self-heal the stale lock and return True (success)
    success = db.claim_cross_session_context_for_update(role)
    assert success is True

    # Verify lock status in database is updating with fresh timestamp
    ctx_after = db.get_cross_session_context(role)
    assert ctx_after is not None
    assert ctx_after.status == "updating"
    assert time.time() - ctx_after.updated_at < 10.0


@pytest.mark.asyncio
async def test_cross_session_context_startup_recovery(tmp_path) -> None:
    """Test that recover_orphaned_context_locks correctly resets stale locks on boot."""
    temp_db = str(tmp_path / "test_context_recovery.db")
    db = DatabaseManager(temp_db)
    db.init_tables()

    # Insert multiple roles with active locks
    db.upsert_cross_session_context("roleA", "Context A")
    db.upsert_cross_session_context("roleB", "Context B")

    # Manually set both status to 'updating' in DB to simulate service crash
    conn = db._get_connection()
    try:
        with conn:
            conn.execute("UPDATE cross_session_contexts SET status = 'updating'")
    finally:
        conn.close()

    # Verify they are locked
    assert db.get_cross_session_context("roleA").status == "updating"
    assert db.get_cross_session_context("roleB").status == "updating"

    # Perform startup recovery
    recovered_count = db.recover_orphaned_context_locks()
    assert recovered_count == 2

    # Verify all are reset to idle
    assert db.get_cross_session_context("roleA").status == "idle"
    assert db.get_cross_session_context("roleB").status == "idle"


@pytest.mark.asyncio
async def test_get_role_messages_since(tmp_path) -> None:
    """Test retrieving role conversational messages with filters and time threshold."""
    from kesoku.constants import MessageRole, MessageStatus, MessageType
    from kesoku.db import Message, Session

    temp_db = str(tmp_path / "test_role_messages.db")
    db = DatabaseManager(temp_db)
    db.init_tables()

    role = "asuka"

    # 1. Create session mappings and channel roles
    db.create_session(Session(id="sess_asuka1", title="Asuka Session 1"))
    db.create_session(Session(id="sess_asuka2", title="Asuka Session 2"))
    db.set_active_session_for_channel("cli", "chan_asuka1", "sess_asuka1")
    db.set_active_session_for_channel("cli", "chan_asuka2", "sess_asuka2")
    db.set_channel_role("cli", "chan_asuka1", role)
    db.set_channel_role("cli", "chan_asuka2", role)

    # 2. Save user and assistant messages
    msg_user1 = Message(
        id="msg_u1",
        session_id="sess_asuka1",
        chatbot_id="cli",
        channel_id="chan_asuka1",
        sender="User",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Hello Asuka!",
        timestamp=100.0,
        status=MessageStatus.RESPONDED,
    )
    msg_asuka1 = Message(
        id="msg_a1",
        session_id="sess_asuka1",
        chatbot_id="cli",
        channel_id="chan_asuka1",
        sender="Asuka",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content="Humph, why are you talking to me?",
        timestamp=105.0,
        status=MessageStatus.RESPONDED,
    )
    # Stale tool message (should be filtered out)
    msg_tool = Message(
        id="msg_t1",
        session_id="sess_asuka1",
        chatbot_id="cli",
        channel_id="chan_asuka1",
        sender="System",
        role=MessageRole.TOOL,
        type=MessageType.TOOL_CALL,
        content="read_file()",
        timestamp=106.0,
        status=MessageStatus.RESPONDED,
    )
    # Message in a different session
    msg_user2 = Message(
        id="msg_u2",
        session_id="sess_asuka2",
        chatbot_id="cli",
        channel_id="chan_asuka2",
        sender="User",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Are you there?",
        timestamp=200.0,
        status=MessageStatus.RESPONDED,
    )

    db.save_message(msg_user1)
    db.save_message(msg_asuka1)
    db.save_message(msg_tool)
    db.save_message(msg_user2)

    # 3. Fetch since timestamp = 50
    res_all = db.get_role_messages_since(role, since_timestamp=50.0)
    # Should have msg_u1, msg_a1, msg_u2 (tool is filtered out!)
    assert len(res_all) == 3
    assert res_all[0].id == "msg_u1"
    assert res_all[1].id == "msg_a1"
    assert res_all[2].id == "msg_u2"

    # 4. Fetch since timestamp = 150
    res_late = db.get_role_messages_since(role, since_timestamp=150.0)
    assert len(res_late) == 1
    assert res_late[0].id == "msg_u2"

    # 5. Fetch since timestamp = 50 but exclude session 'sess_asuka1'
    res_exclude = db.get_role_messages_since(role, since_timestamp=50.0, exclude_session_id="sess_asuka1")
    assert len(res_exclude) == 1
    assert res_exclude[0].id == "msg_u2"


@pytest.mark.asyncio
async def test_get_role_messages_since_default_unbound_channels(tmp_path) -> None:
    """Test retrieving 'default' role messages for unbound channels without channel_roles entries."""
    from kesoku.constants import MessageRole, MessageStatus, MessageType
    from kesoku.db import Message, Session

    temp_db = str(tmp_path / "test_default_unbound.db")
    db = DatabaseManager(temp_db)
    db.init_tables()

    # 1. Create session mapping for a default channel (do NOT set role in channel_roles!)
    db.create_session(Session(id="sess_def", title="Default Session"))
    db.set_active_session_for_channel("cli", "chan_def", "sess_def")

    # 2. Save default user message (unbound channel)
    msg_user = Message(
        id="msg_du",
        session_id="sess_def",
        chatbot_id="cli",
        channel_id="chan_def",
        sender="User",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Standard unbound message.",
        timestamp=100.0,
        status=MessageStatus.RESPONDED,
    )
    db.save_message(msg_user)

    # 3. Fetch messages for role 'default' since timestamp = 50
    res = db.get_role_messages_since("default", since_timestamp=50.0)
    # Should successfully find the unbound channel message since its role defaults to 'default'!
    assert len(res) == 1
    assert res[0].id == "msg_du"

