"""Unit tests for DatabaseManager and models in kesoku.db."""

import os
import time

import pytest

from kesoku.agent.compressor import HistoryCompressor
from kesoku.agent.history import segment_logical_turns
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import AsyncDatabaseManager, DatabaseManager, Message, Session, SummaryNode


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

    # Update title
    time.sleep(0.01)
    db_manager.update_session_title("session_abc", "New Custom Title")
    updated_sess = db_manager.get_session("session_abc")
    assert updated_sess.title == "New Custom Title"
    assert updated_sess.updated_at > session1.updated_at

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

    # Verify messages are saved cleanly without per-message embedding overhead
    saved_msg1 = db_manager.get_messages_by_filters(filters={"id": "msg_1"})[0]
    saved_msg2 = db_manager.get_messages_by_filters(filters={"id": "msg_2"})[0]
    saved_thought = db_manager.get_messages_by_filters(filters={"id": "msg_thought"})[0]
    assert saved_msg1.embedding is None
    assert saved_msg2.embedding is None
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


def test_memory_files_and_chunks_crud(db_manager):
    """Tests upserting, querying, cascade deleting, and clearing role memory files and chunks."""
    now = time.time()
    emb_bytes = b"\x00\x00\x80?" * 4

    file_id_1 = db_manager.upsert_memory_file(
        role="coder",
        rel_path="topics/python.md",
        description="Python style guide",
        content_hash="hash_v1",
        mtime=now - 60,
        updated_at=now - 60,
        chunks=[
            (0, "# topics/python.md\nPython style guide", emb_bytes),
            (1, "# topics/python.md (Python style guide)\n- Use uv run", emb_bytes),
        ],
    )
    assert file_id_1 > 0

    # Upsert another file for 'coder' and one for 'writer'
    file_id_2 = db_manager.upsert_memory_file(
        role="coder",
        rel_path="projects/kesoku.md",
        description="Kesoku architecture",
        content_hash="hash_k1",
        mtime=now,
        updated_at=now,
        chunks=[(0, "# projects/kesoku.md\nKesoku architecture", emb_bytes)],
    )
    db_manager.upsert_memory_file(
        role="writer",
        rel_path="pets/neon.md",
        description="Neon the cat",
        content_hash="hash_n1",
        mtime=now,
        updated_at=now,
        chunks=[(0, "# pets/neon.md\nNeon the cat", emb_bytes)],
    )

    coder_files = db_manager.get_memory_files_by_role("coder")
    assert len(coder_files) == 2
    assert coder_files[0]["rel_path"] == "projects/kesoku.md"
    assert coder_files[1]["rel_path"] == "topics/python.md"

    coder_chunks = db_manager.get_memory_chunks_for_role("coder")
    assert len(coder_chunks) == 3

    # Updating an existing file replaces its old chunks cleanly
    db_manager.upsert_memory_file(
        role="coder",
        rel_path="topics/python.md",
        description="Updated Python style guide",
        content_hash="hash_v2",
        mtime=now + 10,
        updated_at=now + 10,
        chunks=[(0, "# topics/python.md\nUpdated Python style guide", emb_bytes)],
    )
    coder_chunks_after_update = db_manager.get_memory_chunks_for_role("coder")
    assert len(coder_chunks_after_update) == 2

    # Deleting a file by ID cascades to its chunks
    db_manager.delete_memory_files_by_ids([file_id_2])
    assert len(db_manager.get_memory_files_by_role("coder")) == 1
    assert len(db_manager.get_memory_chunks_for_role("coder")) == 1

    # Clearing role memory index removes all files/chunks for that role without touching other roles
    db_manager.clear_role_memory_index("coder")
    assert db_manager.get_memory_files_by_role("coder") == []
    assert db_manager.get_memory_chunks_for_role("coder") == []
    assert len(db_manager.get_memory_files_by_role("writer")) == 1


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


def test_get_channel_role_with_inheritance_after_ui_button_click(db_manager):
    """Ensure a UI button message lacking parent_channel_id does not break parent role inheritance."""
    db_manager.set_channel_role("discord_asuka", "parent_chan_1", "asuka")
    now = time.time()
    session = Session(
        id="111e43df",
        title="Beginner Japanese Lesson 20",
        created_at=now - 100,
        updated_at=now,
        role_name="asuka",
    )
    db_manager.create_session(session)

    # First user message from Discord thread includes parent_channel_id
    msg1 = Message(
        id="m1",
        session_id="111e43df",
        chatbot_id="discord_asuka",
        channel_id="thread_chan_1",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Hello",
        timestamp=now - 50,
        metadata={"parent_channel_id": "parent_chan_1"},
    )
    db_manager.save_message(msg1)

    # Later user message simulated by UI button click lacked parent_channel_id
    msg2 = Message(
        id="m2",
        session_id="111e43df",
        chatbot_id="discord_asuka",
        channel_id="thread_chan_1",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Next segment",
        timestamp=now,
        metadata={"discord_message_id": "12345"},
    )
    db_manager.save_message(msg2)

    assert (
        db_manager.get_channel_role_with_inheritance("discord_asuka", "thread_chan_1", "111e43df")
        == "asuka"
    )
    assert (
        db_manager.get_channel_role_with_inheritance("discord_asuka", "thread_chan_1", None)
        == "asuka"
    )


@pytest.mark.asyncio
async def test_get_session_history_active_only_and_limit(db_manager: DatabaseManager) -> None:
    """Verify active_only drops compressed ASSISTANT/TOOL rows while preserving turn structure and limit slicing."""
    now = 1_700_000_000.0
    sid = "sess_active_only"
    db_manager.create_session(Session(id=sid, title="Active Only Test", created_at=now, updated_at=now))

    messages = [
        # Turn 1 (uncompressed protected head)
        Message(
            id="u1",
            session_id=sid,
            chatbot_id="cli",
            channel_id="ch",
            sender="user",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="Turn 1 user prompt",
            timestamp=now + 1,
            status=MessageStatus.RESPONDED,
        ),
        Message(
            id="a1_thought",
            session_id=sid,
            chatbot_id="cli",
            channel_id="ch",
            sender="assistant",
            role=MessageRole.ASSISTANT,
            type=MessageType.THOUGHT,
            content="Turn 1 thinking",
            timestamp=now + 2,
            status=MessageStatus.DELIVERED,
            parent_id="u1",
        ),
        Message(
            id="a1",
            session_id=sid,
            chatbot_id="cli",
            channel_id="ch",
            sender="assistant",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content="Turn 1 reply",
            timestamp=now + 3,
            status=MessageStatus.DELIVERED,
            parent_id="u1",
        ),
        # Turn 2 (compressed -> node_1: includes USER, TOOL_CALL, TOOL_RESULT, SYSTEM hook nudge, THOUGHT, TEXT)
        Message(
            id="u2",
            session_id=sid,
            chatbot_id="cli",
            channel_id="ch",
            sender="user",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="Turn 2 user prompt",
            timestamp=now + 10,
            status=MessageStatus.RESPONDED,
        ),
        Message(
            id="tc2",
            session_id=sid,
            chatbot_id="cli",
            channel_id="ch",
            sender="read_file",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_CALL,
            content="Calling read_file",
            timestamp=now + 11,
            status=MessageStatus.RESPONDED,
            parent_id="u2",
        ),
        Message(
            id="tr2",
            session_id=sid,
            chatbot_id="cli",
            channel_id="ch",
            sender="read_file",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_RESULT,
            content="x" * 5000,
            timestamp=now + 12,
            status=MessageStatus.RESPONDED,
            parent_id="tc2",
        ),
        Message(
            id="h2",
            session_id=sid,
            chatbot_id="cli",
            channel_id="ch",
            sender="Hook",
            role=MessageRole.SYSTEM,
            type=MessageType.TEXT,
            content="Continue verification",
            timestamp=now + 13,
            status=MessageStatus.RESPONDED,
            parent_id="u2",
        ),
        Message(
            id="a2_thought",
            session_id=sid,
            chatbot_id="cli",
            channel_id="ch",
            sender="assistant",
            role=MessageRole.ASSISTANT,
            type=MessageType.THOUGHT,
            content="Turn 2 thinking",
            timestamp=now + 14,
            status=MessageStatus.DELIVERED,
            parent_id="u2",
        ),
        Message(
            id="a2",
            session_id=sid,
            chatbot_id="cli",
            channel_id="ch",
            sender="assistant",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content="Turn 2 bulky reply",
            timestamp=now + 15,
            status=MessageStatus.DELIVERED,
            parent_id="u2",
        ),
        # Non-conversational rename event must remain excluded in both modes
        Message(
            id="ren1",
            session_id=sid,
            chatbot_id="cli",
            channel_id="ch",
            sender="system",
            role=MessageRole.SYSTEM,
            type=MessageType.SESSION_RENAME,
            content="Renamed session",
            timestamp=now + 16,
            status=MessageStatus.DELIVERED,
        ),
        # Turn 3 (uncompressed active/recent turn)
        Message(
            id="u3",
            session_id=sid,
            chatbot_id="cli",
            channel_id="ch",
            sender="user",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="Turn 3 user prompt",
            timestamp=now + 20,
            status=MessageStatus.RESPONDED,
        ),
        Message(
            id="a3",
            session_id=sid,
            chatbot_id="cli",
            channel_id="ch",
            sender="assistant",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content="Turn 3 reply",
            timestamp=now + 21,
            status=MessageStatus.DELIVERED,
            parent_id="u3",
        ),
    ]
    for msg in messages:
        db_manager.save_message(msg)

    node = SummaryNode(
        id="node_1",
        session_id=sid,
        level=0,
        summary="Summary of turn 2",
        start_timestamp=now + 10,
        end_timestamp=now + 15,
        token_count=12,
        source_token_count=1500,
        parent_id=None,
        created_at=now + 16,
    )
    db_manager.insert_summary_node(node)
    db_manager.update_messages_summary_node(["u2", "tc2", "tr2", "h2", "a2_thought", "a2"], "node_1")

    # 1. active_only=False (default) returns all non-rename rows unchanged
    full_default = db_manager.get_session_history(sid, limit=0)
    full_explicit = db_manager.get_session_history(sid, limit=0, active_only=False)
    assert [m.id for m in full_default] == [m.id for m in full_explicit]
    assert [m.id for m in full_default] == [
        "u1",
        "a1_thought",
        "a1",
        "u2",
        "tc2",
        "tr2",
        "h2",
        "a2_thought",
        "a2",
        "u3",
        "a3",
    ]

    # 2. active_only=True drops Turn 2's ASSISTANT/TOOL rows while keeping USER/SYSTEM rows (with summary_node_id)
    active = db_manager.get_session_history(sid, limit=0, active_only=True)
    assert [m.id for m in active] == ["u1", "a1_thought", "a1", "u2", "h2", "u3", "a3"]
    by_id = {m.id: m for m in active}
    assert by_id["u2"].summary_node_id == "node_1"
    assert by_id["h2"].summary_node_id == "node_1"
    assert by_id["u1"].summary_node_id is None
    assert by_id["u3"].summary_node_id is None

    # 3. Logical turn segmentation & <recent_user_messages> block match full history identically
    full_turns = segment_logical_turns(full_default)
    active_turns = segment_logical_turns(active)
    assert len(active_turns) == len(full_turns) == 3
    full_flags = [any(m.summary_node_id is not None for m in t) for t in full_turns]
    active_flags = [any(m.summary_node_id is not None for m in t) for t in active_turns]
    assert active_flags == full_flags == [False, True, False]
    assert HistoryCompressor.build_recent_user_messages_block(
        active_turns, 5
    ) == HistoryCompressor.build_recent_user_messages_block(full_turns, 5)
    assert "Turn 2 user prompt" in HistoryCompressor.build_recent_user_messages_block(active_turns, 5)

    # 4. limit slices AFTER active_only filtering ([-limit:])
    limited_active = db_manager.get_session_history(sid, limit=4, active_only=True)
    assert [m.id for m in limited_active] == ["u2", "h2", "u3", "a3"]

    # 5. AsyncDatabaseManager forwards active_only accurately
    async_db = AsyncDatabaseManager(db_manager)
    async_active = await async_db.get_session_history(sid, limit=0, order="grouped", active_only=True)
    assert [m.id for m in async_active] == ["u1", "a1_thought", "a1", "u2", "h2", "u3", "a3"]







