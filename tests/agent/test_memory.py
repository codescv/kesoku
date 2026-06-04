"""Unit tests for the SQLite Category-based and Role-bound Agent Memory System."""

import asyncio

import pytest

from kesoku.agent.tools import ToolContext, delete_memory, list_memories, update_memory, view_memory
from kesoku.config import KesokuConfig, WorkspaceConfig
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager, Message


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

    # 2. Upsert role-specific learnings memory
    db.upsert_agent_memory(
        category="learnings",
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

    mem2 = db.get_agent_memory(category="learnings", key="proxy_setting", role="asuka")
    assert mem2 is not None
    assert mem2["title"] == "代理配置"
    assert mem2["role"] == "asuka"

    # 4. Fetch multiple memories (listing / filtering)
    # Getting memories filtered by category
    progress_mems = db.get_agent_memories(category="progress")
    assert len(progress_mems) == 1
    assert progress_mems[0]["key"] == "standard_japanese"

    # Query with role='asuka' (should fetch both default + asuka's memories)
    all_asuka_mems = db.get_agent_memories(role="asuka")
    # Both default progress and asuka specific learnings are in the system
    assert len(all_asuka_mems) == 2

    # Query with role='tifa' (should fetch only default memories, and tifa specific ones if they existed)
    all_tifa_mems = db.get_agent_memories(role="tifa")
    assert len(all_tifa_mems) == 1
    assert all_tifa_mems[0]["key"] == "standard_japanese"

    # 5. Deleting memory
    db.delete_agent_memory(category="learnings", key="proxy_setting", role="asuka")
    deleted_mem = db.get_agent_memory(category="learnings", key="proxy_setting", role="asuka")
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
        category="user_preferences",
        key="funny_asuka_event",
        title="Asuka Day 1",
        content="Asuka was tsundere today",
        role="asuka",
        context=ctx,
    )
    assert "Memory successfully saved!" in res
    assert "Key: `funny_asuka_event`" in res

    # 4. Testing list_memories tool with roleplay segregation
    # Writing a default user_preferences memory
    await update_memory(
        category="user_preferences",
        key="funny_general_event",
        title="General Fun",
        content="Someone made a joke today",
        role="default",
        context=ctx,
    )

    # List memories as Tifa
    list_tifa = await list_memories(category="user_preferences", role="tifa", context=ctx)
    # Tifa can see default memories but NOT Asuka's memories
    assert "funny_general_event" in list_tifa
    assert "funny_asuka_event" not in list_tifa

    # List memories as Asuka
    list_asuka = await list_memories(category="user_preferences", role="asuka", context=ctx)
    # Asuka can see both default and Asuka-specific memories
    assert "funny_general_event" in list_asuka
    assert "funny_asuka_event" in list_asuka

    # 5. Testing view_memory tool with dynamic in-memory Markdown aggregation (key=None)
    view_asuka_all = await view_memory(category="user_preferences", key=None, role="asuka", context=ctx)
    assert "# Category: user_preferences (scope: asuka)" in view_asuka_all
    assert "## General Fun (key: `funny_general_event`, scope: `default`)" in view_asuka_all
    assert "## Asuka Day 1 (key: `funny_asuka_event`, scope: `asuka`)" in view_asuka_all
    assert "Someone made a joke today" in view_asuka_all
    assert "Asuka was tsundere today" in view_asuka_all

    # 6. Testing delete_memory tool
    delete_res = await delete_memory(category="user_preferences", key="funny_asuka_event", role="asuka", context=ctx)
    assert "Memory successfully deleted" in delete_res

    # Verify deletion
    list_asuka_after = await list_memories(category="user_preferences", role="asuka", context=ctx)
    assert "funny_asuka_event" not in list_asuka_after


@pytest.mark.asyncio
async def test_category_role_routing_and_play_role(tmp_path) -> None:
    """Test that memories are routed to default or active role scope according to the rules,
    and play_role tool executes successfully.
    """
    import asyncio

    from kesoku.agent.tools import play_role
    from kesoku.constants import MessageRole, MessageStatus
    from kesoku.db import Message
    from kesoku.gateway.gateway import Gateway

    temp_db = str(tmp_path / "test_routing.db")
    DatabaseManager(temp_db).init_tables()

    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
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
    assert "Scope: `default`" in res

    # 2. Update 'user_preferences' memory - it should go to 'tifa' active role scope
    res = await update_memory(
        category="user_preferences",
        key="funny_event",
        title="Funny",
        content="A funny thing happened",
        role="default",  # Passed explicitly but should be overridden
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
    assert "Scope: `tifa`" in res_memo

    # 3. Test play_role tool to switch role to asuka
    # First let's ensure asuka and tifa directories exist under workspace roles_dir
    roles_dir = str(tmp_path / "roles")
    cfg.workspace.roles_dir = roles_dir

    def write_intro():
        import os

        os.makedirs(os.path.join(roles_dir, "asuka"), exist_ok=True)
        with open(os.path.join(roles_dir, "asuka", "intro.md"), "w") as f:
            f.write("Asuka Intro")

    await asyncio.to_thread(write_intro)

    play_res = await play_role("asuka", context=ctx)
    assert "Persona Switched Successfully!" in play_res
    assert "asuka" in play_res

    # Verify that database role is updated
    current_role = await gw.db.get_channel_role("discord", "chan_123")
    assert current_role == "asuka"


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
        category="user_preferences",
        key="valid_len",
        title="Valid Title",
        content="A" * 500,
        context=ctx,
    )
    assert "Memory successfully saved!" in res

    # 2. Exceeding content length (> 500 characters)
    res_fail = await update_memory(
        category="user_preferences",
        key="invalid_len",
        title="Invalid Title",
        content="A" * 501,
        context=ctx,
    )
    assert "Error: Content length (501 characters) exceeds the maximum limit of 500 characters" in res_fail

    # 3. Exceeding content length with import of MAX_MEMORY_CONTENT_LENGTH
    from kesoku.agent.tools import MAX_MEMORY_CONTENT_LENGTH

    res_fail_constant = await update_memory(
        category="user_preferences",
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
async def test_user_preferences_memory_behavior(tmp_path) -> None:
    """Verify user_preferences memory is allowed, behaves with correct role scopes and respects limits."""
    from kesoku.gateway.gateway import Gateway

    temp_db = str(tmp_path / "test_pref.db")
    DatabaseManager(temp_db).init_tables()

    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    gw = Gateway(context=KesokuContext(config=cfg))

    # Bind active channel role for the active context message
    await gw.db.set_channel_role("discord", "chan_pref", "tifa")

    # Create a mock user message in channel 'chan_pref' to simulate active context
    msg = Message(
        id="msg_pref",
        session_id="sess_pref",
        chatbot_id="discord",
        channel_id="chan_pref",
        sender="User",
        role="user",
        content="Preferences request",
        status="responded",
    )
    await gw.post(msg)

    ctx = ToolContext(
        session_id="sess_pref",
        session_workspace="test_ws",
        original_msg_id="msg_pref",
        gateway=gw,
    )

    # 1. Update 'user_preferences' memory - it should dynamically route to 'tifa' role scope
    res = await update_memory(
        category="user_preferences",
        key="dont_use_codeblocks",
        title="Code Block Preference",
        content="Avoid markdown code blocks",
        role="default",  # Passed explicitly but should be overridden by channel active role 'tifa'
        context=ctx,
    )
    assert "Memory successfully saved!" in res
    assert "Scope: `tifa`" in res

    # 2. Max content length limit (MAX_MEMORY_CONTENT_LENGTH is 500, let's check it enforces it)
    res_fail = await update_memory(
        category="user_preferences",
        key="pref_too_long",
        title="Too Long Preference",
        content="A" * 501,
        context=ctx,
    )
    assert "Error: Content length (501 characters) exceeds the maximum limit of 500 characters" in res_fail


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
