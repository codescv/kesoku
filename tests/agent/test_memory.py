"""Unit tests for the SQLite Category-based and Role-bound Agent Memory System."""

import asyncio
import time

import pytest

from kesoku.agent.tools import (
    ToolContext,
    delete_memory,
    list_memories,
    memory_search,
    update_memory,
    view_memory,
    view_message,
)
from kesoku.agent.tools.memory import memory_grep
from kesoku.config import KesokuConfig, WorkspaceConfig
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager, Message, Session


@pytest.mark.asyncio
async def test_database_memory_crud(tmp_path) -> None:
    """Test basic CRUD operations on the SQLite agent_memories table directly."""
    temp_db = str(tmp_path / "test_memory_crud.db")
    db = DatabaseManager(temp_db)
    db.init_tables()

    # 1. Upsert default progress memory
    db.upsert_agent_memory(
        category="progress",
        key="standard_japanese",
        title="标准日本语",
        content="学习完了第22课",
        role="default",
    )

    # 2. Upsert role-specific memo memory
    db.upsert_agent_memory(
        category="memo",
        key="proxy_setting",
        title="代理配置",
        content="联网需要配置环境变量 HTTP_PROXY",
        role="asuka",
    )

    # 3. Fetch specific memory
    mem1 = db.get_agent_memory(category="progress", key="standard_japanese", role="default")
    assert mem1 is not None
    assert mem1["title"] == "标准日本语"
    assert mem1["content"] == "学习完了第22课"
    assert mem1["role"] == "default"

    mem2 = db.get_agent_memory(category="memo", key="proxy_setting", role="asuka")
    assert mem2 is not None
    assert mem2["title"] == "代理配置"
    assert mem2["role"] == "asuka"

    # 4. Fetch multiple memories (listing / filtering)
    # Getting memories filtered by category
    progress_mems = db.get_agent_memories(category="progress")
    assert len(progress_mems) == 1
    assert progress_mems[0]["key"] == "standard_japanese"

    # Query with role='asuka' (should fetch only asuka's memories)
    all_asuka_mems = db.get_agent_memories(role="asuka")
    assert len(all_asuka_mems) == 1
    assert all_asuka_mems[0]["key"] == "proxy_setting"

    # Query with role='tifa' (should fetch only tifa memories, which is empty)
    all_tifa_mems = db.get_agent_memories(role="tifa")
    assert len(all_tifa_mems) == 0

    # 5. Deleting memory
    db.delete_agent_memory(category="memo", key="proxy_setting", role="asuka")
    deleted_mem = db.get_agent_memory(category="memo", key="proxy_setting", role="asuka")
    assert deleted_mem is None


@pytest.mark.asyncio
async def test_memory_tools_execution(tmp_path) -> None:
    """Test agent-accessible memory tools execution under category/role sandboxes."""
    from kesoku.gateway.gateway import Gateway

    temp_db = str(tmp_path / "test_memory_tools.db")
    DatabaseManager(temp_db).init_tables()

    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    # Setup a context with active gateway
    ctx = ToolContext(
        session_id="sess_mem_test",
        session_workspace="test_ws",
        gateway=gw,
    )

    # 1. Attempt to write to a category not in allowed allowlist
    res = await update_memory(
        category="forbidden_category",
        key="my_key",
        title="Invalid Title",
        content="Content",
        role="default",
        context=ctx,
    )
    assert "Write Denied: Category 'forbidden_category' is not recognized" in res
    assert "You can only use the following configured categories:" in res

    # 3. Verify that updating with an invalid key is rejected
    res_fail = await update_memory(
        category="progress",
        key="  Standard Japanese Book  ",
        title="《标日》学习进度",
        content="已学完第23课",
        role="asuka",
        context=ctx,
    )
    assert "Error: Invalid Key" in res_fail

    # 4. Normal update on a role-isolated category with a strictly valid key
    res = await update_memory(
        category="memo",
        key="funny_asuka_event",
        title="Asuka Day 1",
        content="Asuka was tsundere today",
        role="asuka",
        context=ctx,
    )
    assert "Memory successfully saved!" in res
    assert "Key: `funny_asuka_event`" in res

    # 4. Testing list_memories tool with roleplay segregation
    # Writing a default memo memory
    await update_memory(
        category="memo",
        key="funny_general_event",
        title="General Fun",
        content="Someone made a joke today",
        role="default",
        context=ctx,
    )

    # List memories as Tifa
    list_tifa = await list_memories(category="memo", role="tifa", context=ctx)
    # Tifa should NOT see default memories or Asuka's memories
    assert "funny_general_event" not in list_tifa
    assert "funny_asuka_event" not in list_tifa

    # List memories as Asuka
    list_asuka = await list_memories(category="memo", role="asuka", context=ctx)
    # Asuka can see only Asuka-specific memories
    assert "funny_general_event" not in list_asuka
    assert "funny_asuka_event" in list_asuka

    # 5. Testing view_memory tool with dynamic in-memory Markdown aggregation (key=None)
    view_asuka_all = await view_memory(category="memo", key=None, role="asuka", context=ctx)
    assert "# Category: memo (scope: asuka)" in view_asuka_all
    assert "## Asuka Day 1 (key: `funny_asuka_event`, scope: `asuka`)" in view_asuka_all
    assert "Asuka was tsundere today" in view_asuka_all
    assert "General Fun" not in view_asuka_all

    # 6. Testing delete_memory tool
    delete_res = await delete_memory(category="memo", key="funny_asuka_event", role="asuka", context=ctx)
    assert "Memory successfully deleted" in delete_res

    # Verify deletion
    list_asuka_after = await list_memories(category="memo", role="asuka", context=ctx)
    assert "funny_asuka_event" not in list_asuka_after


@pytest.mark.asyncio
async def test_category_role_routing(tmp_path) -> None:
    """Test that memories are routed to default or active role scope according to the rules."""

    from kesoku.constants import MessageRole, MessageStatus
    from kesoku.db import Message
    from kesoku.gateway.gateway import Gateway

    temp_db = str(tmp_path / "test_routing.db")
    DatabaseManager(temp_db).init_tables()

    from kesoku.config import get_config

    cfg = get_config()
    cfg.workspace.db_path = temp_db
    gw = Gateway(context=KesokuContext(config=cfg))

    # Bind the channel to a specific role 'tifa'
    await gw.db.set_channel_role("discord", "chan_123", "tifa")

    # Create a mock user message in channel 'chan_123' to simulate the active channel context
    msg = Message(
        id="msg_abc",
        session_id="sess_123",
        chatbot_id="discord",
        channel_id="chan_123",
        sender="User",
        role=MessageRole.USER,
        content="Hello",
        status=MessageStatus.RESPONDED,
    )
    await gw.post(msg)

    ctx = ToolContext(
        session_id="sess_123",
        session_workspace="test_ws",
        original_msg_id="msg_abc",
        gateway=gw,
    )

    # 1. Update standard 'progress' memory - it should go to 'default' role scope
    res = await update_memory(
        category="progress",
        key="milestone",
        title="Milestone",
        content="Completed Task",
        role="tifa",  # Passed explicitly
        context=ctx,
    )
    assert "Scope: `tifa`" in res



    # 2b. Update 'memo' memory - it should also go to 'tifa' active role scope
    res_memo = await update_memory(
        category="memo",
        key="interesting_event",
        title="Interesting",
        content="An interesting thing happened",
        role="default",  # Passed explicitly but should be overridden
        context=ctx,
    )


@pytest.mark.asyncio
async def test_memory_length_limit(tmp_path) -> None:
    """Test that update_memory enforces content character limits properly."""
    from kesoku.gateway.gateway import Gateway

    temp_db = str(tmp_path / "test_memory_limit.db")
    DatabaseManager(temp_db).init_tables()

    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    ctx = ToolContext(
        session_id="sess_limit_test",
        session_workspace="test_ws",
        gateway=gw,
    )

    # 1. Valid content length (<= 500 characters)
    res = await update_memory(
        category="memo",
        key="valid_len",
        title="Valid Title",
        content="A" * 500,
        context=ctx,
    )
    assert "Memory successfully saved!" in res

    # 2. Exceeding content length (> 500 characters)
    res_fail = await update_memory(
        category="memo",
        key="invalid_len",
        title="Invalid Title",
        content="A" * 501,
        context=ctx,
    )
    assert "Error: Content length (501 characters) exceeds the maximum limit of 500 characters" in res_fail

    # 3. Exceeding content length with import of MAX_MEMORY_CONTENT_LENGTH
    from kesoku.agent.tools import MAX_MEMORY_CONTENT_LENGTH

    res_fail_constant = await update_memory(
        category="memo",
        key="invalid_len_constant",
        title="Invalid Title",
        content="A" * (MAX_MEMORY_CONTENT_LENGTH + 1),
        context=ctx,
    )
    expected_err = (
        f"Error: Content length ({MAX_MEMORY_CONTENT_LENGTH + 1} characters) "
        f"exceeds the maximum limit of {MAX_MEMORY_CONTENT_LENGTH} characters"
    )
    assert expected_err in res_fail_constant


@pytest.mark.asyncio
async def test_memory_ordering(tmp_path) -> None:
    """Test that get_agent_memories returns records ordered by updated_at DESC."""
    temp_db = str(tmp_path / "test_memory_ordering.db")
    db = DatabaseManager(temp_db)
    db.init_tables()

    # Insert memories with sleep in between to ensure different updated_at timestamps
    db.upsert_agent_memory("progress", "key_first", "First Title", "Content A")
    await asyncio.sleep(0.01)
    db.upsert_agent_memory("progress", "key_second", "Second Title", "Content B")
    await asyncio.sleep(0.01)
    db.upsert_agent_memory("progress", "key_third", "Third Title", "Content C")

    mems = db.get_agent_memories(category="progress")
    assert len(mems) == 3
    # Should be newest first (key_third, then key_second, then key_first)
    assert mems[0]["key"] == "key_third"
    assert mems[1]["key"] == "key_second"
    assert mems[2]["key"] == "key_first"


@pytest.mark.asyncio
async def test_user_preferences_deprecation(tmp_path) -> None:
    """Verify user_preferences memory category is deprecated and rejected."""
    from kesoku.gateway.gateway import Gateway

    temp_db = str(tmp_path / "test_pref.db")
    DatabaseManager(temp_db).init_tables()

    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    ctx = ToolContext(
        session_id="sess_pref",
        session_workspace="test_ws",
        gateway=gw,
    )

    res = await update_memory(
        category="user_preferences",
        key="dont_use_codeblocks",
        title="Code Block Preference",
        content="Avoid markdown code blocks",
        context=ctx,
    )
    assert "Write Denied: Category 'user_preferences' is not recognized" in res


@pytest.mark.asyncio
async def test_update_memory_overwrite_prevention(tmp_path) -> None:
    """Test that update_memory prevents accidental overwrites using old_content."""
    from kesoku.gateway.gateway import Gateway

    temp_db = str(tmp_path / "test_overwrite.db")
    DatabaseManager(temp_db).init_tables()

    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    ctx = ToolContext(
        session_id="sess_overwrite_test",
        session_workspace="test_ws",
        gateway=gw,
    )

    # 1. Initial write (new key) - should succeed without old_content
    res = await update_memory(
        category="progress",
        key="my_task",
        title="My Task",
        content="Initial State",
        context=ctx,
    )
    assert "Memory successfully saved!" in res

    # 2. Update existing key without old_content - should fail
    res_fail = await update_memory(
        category="progress",
        key="my_task",
        title="My Task",
        content="New State",
        context=ctx,
    )
    assert "Write Denied: Memory already exists" in res_fail
    assert "you MUST provide the `old_content` parameter" in res_fail

    # 3. Update existing key with incorrect old_content - should fail
    res_fail_wrong = await update_memory(
        category="progress",
        key="my_task",
        title="My Task",
        content="New State",
        old_content="Wrong Old State",
        context=ctx,
    )
    assert "Write Denied: The provided `old_content` does not match" in res_fail_wrong

    # 4. Update existing key with correct old_content - should succeed
    res_success = await update_memory(
        category="progress",
        key="my_task",
        title="My Task",
        content="New State",
        old_content="Initial State",
        context=ctx,
    )
    assert "Memory successfully saved!" in res_success

    # Verify the update
    val = await view_memory(category="progress", key="my_task", context=ctx)
    assert "New State" in val


@pytest.mark.asyncio
async def test_memo_category(tmp_path) -> None:
    """Test that 'memo' category is allowed and routes to 'default' role scope."""
    from kesoku.gateway.gateway import Gateway

    temp_db = str(tmp_path / "test_memo.db")
    DatabaseManager(temp_db).init_tables()

    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    ctx = ToolContext(
        session_id="sess_memo_test",
        session_workspace="test_ws",
        gateway=gw,
    )

    # 'memo' should be allowed by default now
    res = await update_memory(
        category="memo",
        key="daily_event",
        title="Interesting Event",
        content="Met a friendly cat",
        context=ctx,
    )
    assert "Memory successfully saved!" in res
    assert "Category: `memo`" in res
    assert "Scope: `default`" in res


@pytest.mark.asyncio
async def test_memory_grep_tool(tmp_path) -> None:
    """Test memory_grep tool finds matching memories and messages for the active role."""
    from kesoku.gateway.gateway import Gateway

    temp_db = str(tmp_path / "test_grep_tool.db")
    db = DatabaseManager(temp_db)
    db.init_tables()

    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    # Set up role 'coder' on channel 'chan_1'
    await gw.db.set_channel_role("discord", "chan_1", "coder")
    session1 = Session(id="sess_1", title="Sess 1", created_at=time.time(), updated_at=time.time())
    await gw.db.create_session(session1)
    await gw.db.set_active_session_for_channel("discord", "chan_1", "sess_1")

    # Create a mock user message to simulate active channel context
    msg_context = Message(
        id="msg_ctx",
        session_id="sess_1",
        chatbot_id="discord",
        channel_id="chan_1",
        sender="User",
        role=MessageRole.USER,
        content="Context message",
        status=MessageStatus.RESPONDED,
    )
    await gw.post(msg_context)

    ctx = ToolContext(
        session_id="sess_1",
        session_workspace="test_ws",
        original_msg_id="msg_ctx",
        gateway=gw,
    )

    # Save messages
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
    await gw.post(msg1)
    await gw.post(msg2)
    await gw.post(msg3)

    # Insert memories
    await gw.db.upsert_agent_memory("memo", "mem1", "Python Tip", "Python is great", "default")
    await gw.db.upsert_agent_memory("memo", "mem2", "Coder Tip", "Write python code", "coder")
    await gw.db.upsert_agent_memory("memo", "mem3", "Helper Tip", "Help python users", "helper")

    # Run memory_grep
    res = await memory_grep(query="python", context=ctx)

    # Assertions
    assert "Search Results for 'python' (Role: coder)" in res
    assert "Matching Memories" in res
    assert "Python Tip" not in res  # mem1 (default)
    assert "Coder Tip" in res  # mem2 (coder)
    assert "Helper Tip" not in res  # mem3 (helper)

    assert "Matching Messages" in res
    assert "I love python coding" in res  # m1
    assert "Here is python code" in res  # m3
    assert "Thinking about python" not in res  # m2 (thought)

@pytest.mark.asyncio
async def test_memory_grep_tool_wildcard_and_filters(tmp_path) -> None:
    """Test memory_grep tool with wildcard query and time/limit filters."""
    from kesoku.gateway.gateway import Gateway

    temp_db = str(tmp_path / "test_grep_filters.db")
    db = DatabaseManager(temp_db)
    db.init_tables()

    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    # Set up role 'coder' on channel 'chan_1'
    await gw.db.set_channel_role("discord", "chan_1", "coder")
    session1 = Session(id="sess_1", title="Sess 1", created_at=time.time(), updated_at=time.time())
    await gw.db.create_session(session1)
    await gw.db.set_active_session_for_channel("discord", "chan_1", "sess_1")

    # Create a mock user message to simulate active channel context
    msg_context = Message(
        id="msg_ctx",
        session_id="sess_1",
        chatbot_id="discord",
        channel_id="chan_1",
        sender="User",
        role=MessageRole.USER,
        content="Context message",
        status=MessageStatus.RESPONDED,
    )
    await gw.post(msg_context)

    # Base timestamp: 2026-06-15 12:00:00 UTC (1781534400.0)
    base_ts = 1781534400.0

    # Set msg_ctx timestamp to be older (Sunday, one day before Monday base_ts)
    with gw.db.sync_db.connection_provider.connection() as conn:
        with conn:
            conn.execute("UPDATE messages SET timestamp = ? WHERE id = 'msg_ctx'", (base_ts - 86400,))

    ctx = ToolContext(
        session_id="sess_1",
        session_workspace="test_ws",
        original_msg_id="msg_ctx",
        gateway=gw,
    )

    # Save messages with specific timestamps
    msg1 = Message(
        id="m1",
        session_id="sess_1",
        chatbot_id="discord",
        channel_id="chan_1",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="First message on Monday",
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
        content="Second message on Monday",
        timestamp=base_ts + 3600,
        status=MessageStatus.RESPONDED,
    )
    msg3 = Message(
        id="m3",
        session_id="sess_1",
        chatbot_id="discord",
        channel_id="chan_1",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Third message on Tuesday",
        timestamp=base_ts + 86400,
        status=MessageStatus.PROCESSED,
    )

    # We bypass gw.post because gw.post sets timestamp=time.time()
    # Let's save directly to DB
    gw.db.sync_db.save_message(msg1)
    gw.db.sync_db.save_message(msg2)
    gw.db.sync_db.save_message(msg3)

    # Insert memories
    await gw.db.upsert_agent_memory("memo", "mem1", "Python Tip", "Python is great", "default")
    await gw.db.upsert_agent_memory("memo", "mem2", "Java Tip", "Java is verbose", "coder")

    # We need to set explicit updated_at for memories
    with gw.db.sync_db.connection_provider.connection() as conn:
        with conn:
            conn.execute("UPDATE agent_memories SET updated_at = ? WHERE key = 'mem1'", (base_ts,))
            conn.execute("UPDATE agent_memories SET updated_at = ? WHERE key = 'mem2'", (base_ts + 3600,))

    # Test 1: Wildcard search (returns messages only, no memories)
    res_wildcard = await memory_grep(query="*", context=ctx)
    assert "Search Results for '*'" in res_wildcard
    assert "Matching Messages" in res_wildcard
    assert "First message on Monday" in res_wildcard
    assert "Second message on Monday" in res_wildcard
    assert "Third message on Tuesday" in res_wildcard
    assert "Matching Memories" not in res_wildcard  # Wildcard should skip memories

    # Test 2: Wildcard search with date filters (Monday only)
    res_date = await memory_grep(
        query="*",
        start_time="2026-06-15T00:00:00",
        end_time="2026-06-15T23:59:59",
        context=ctx,
    )
    assert "Search Results for '*'" in res_date
    assert "First message on Monday" in res_date
    assert "Second message on Monday" in res_date
    assert "Third message on Tuesday" not in res_date  # Tuesday msg should be excluded

    # Test 3: Normal query with date filters (should filter both)
    res_normal = await memory_grep(
        query="message",
        start_time="2026-06-15T00:00:00",
        end_time="2026-06-15T23:59:59",
        context=ctx,
    )
    assert "Search Results for 'message'" in res_normal
    assert "First message on Monday" in res_normal
    assert "Second message on Monday" in res_normal
    assert "Third message on Tuesday" not in res_normal

    # Test 4: Limit filter
    res_limit = await memory_grep(
        query="*",
        limit=2,
        context=ctx,
    )
    assert "Third message on Tuesday" in res_limit
    assert "Second message on Monday" in res_limit
    assert "First message on Monday" not in res_limit


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
async def test_memory_search_tool(tmp_path, monkeypatch) -> None:
    """Test memory_search tool semantically finds matching memories for the active role."""
    from kesoku.gateway.gateway import Gateway

    temp_db = str(tmp_path / "test_search_tool.db")
    db = DatabaseManager(temp_db)
    db.init_tables()

    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    # Set up role 'coder' on channel 'chan_1'
    await gw.db.set_channel_role("discord", "chan_1", "coder")
    session1 = Session(id="sess_1", title="Sess 1", created_at=time.time(), updated_at=time.time())
    await gw.db.create_session(session1)
    await gw.db.set_active_session_for_channel("discord", "chan_1", "sess_1")

    # Create a mock user message to simulate active channel context
    msg_context = Message(
        id="msg_ctx",
        session_id="sess_1",
        chatbot_id="discord",
        channel_id="chan_1",
        sender="User",
        role=MessageRole.USER,
        content="Context message",
        status=MessageStatus.RESPONDED,
    )
    await gw.post(msg_context)

    ctx = ToolContext(
        session_id="sess_1",
        session_workspace="test_ws",
        original_msg_id="msg_ctx",
        gateway=gw,
    )

    # Setup mocked embeddings
    embeddings_map = {
        "Python Tip\nPython is great": [1.0, 0.0],
        "Java Tip\nJava is verbose": [0.0, 1.0],
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

    # Insert memories (will compute and store embeddings)
    await gw.db.upsert_agent_memory("memo", "mem1", "Python Tip", "Python is great", "coder")
    await gw.db.upsert_agent_memory("memo", "mem2", "Java Tip", "Java is verbose", "coder")

    # Run memory_search
    res = await memory_search(query="Find python tips", context=ctx)

    # Assertions
    assert "Semantic Search Results for 'Find python tips' (Role: coder)" in res
    assert "Matching Memories" in res
    assert "mem1" in res
    assert "mem2" in res

    pos_mem1 = res.index("mem1")
    pos_mem2 = res.index("mem2")
    assert pos_mem1 < pos_mem2
    assert "score: 0.9939" in res
    assert "score: 0.1104" in res


@pytest.mark.asyncio
async def test_memory_search_hybrid_and_boosting(tmp_path, monkeypatch) -> None:
    """Test memory_search hybrid exact matching and score boosting + text truncation."""
    import array

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
        "Hello controller world": [0.0, 1.0],
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

    with db.connection_provider.connection() as conn:
        with conn:
            sql = (
                "INSERT INTO messages ("
                "  id, session_id, chatbot_id, channel_id, sender, role, type, "
                "  content, metadata, timestamp, status, embedding"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
            conn.execute(sql, (
                "m_sem", "sess_1", "discord", "chan_1", "user", "user", "text",
                "Game Console Joystick", "{}", 1.0, "processed",
                array.array("f", pad_vector([1.0, 0.0])).tobytes()
            ))
            conn.execute(sql, (
                "m_exact", "sess_1", "discord", "chan_1", "user", "user", "text",
                "Hello controller world", "{}", 2.0, "processed", None
            ))
            conn.execute(sql, (
                "m_both", "sess_1", "discord", "chan_1", "user", "user", "text",
                "How controller is built", "{}", 3.0, "processed",
                array.array("f", pad_vector([0.5, 0.5])).tobytes()
            ))
            long_content = "controller " + "a" * 600
            conn.execute(sql, (
                "m_long", "sess_1", "discord", "chan_1", "user", "user", "text",
                long_content, "{}", 4.0, "processed", None
            ))

    ctx = ToolContext(
        session_id="sess_1",
        session_workspace="test_ws",
        gateway=gw,
    )

    res = await memory_search(query="controller", limit=10, context=ctx)

    assert "m_both" in res
    assert "m_exact" in res
    assert "m_sem" in res
    assert "m_long" in res

    pos_both = res.index("m_both")
    pos_exact = res.index("m_exact")
    pos_sem = res.index("m_sem")

    assert pos_both < pos_exact
    assert pos_exact < pos_sem

    assert "Truncated for Brevity" in res
    assert len(res) < 1500


@pytest.mark.asyncio
async def test_memory_search_wildcard_and_time_filters(tmp_path, monkeypatch) -> None:
    """Test memory_search supporting wildcard queries and time range filtering."""
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

    res_wildcard = await memory_search(query="*", limit=10, context=ctx)
    assert "Search Results for '*'" in res_wildcard
    assert "m3" in res_wildcard
    assert "m2" in res_wildcard
    assert "m1" in res_wildcard

    pos_m3 = res_wildcard.index("m3")
    pos_m2 = res_wildcard.index("m2")
    pos_m1 = res_wildcard.index("m1")
    assert pos_m3 < pos_m2 < pos_m1
    assert "score:" not in res_wildcard

    res_time = await memory_search(
        query="*",
        start_time="1500",
        end_time="2500",
        limit=10,
        context=ctx
    )
    assert "m2" in res_time
    assert "m1" not in res_time
    assert "m3" not in res_time



