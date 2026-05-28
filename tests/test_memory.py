"""Unit tests for the SQLite Category-based and Role-bound Agent Memory System."""

import pytest

from kesoku.agent.tools import ToolContext, delete_memory, list_memories, update_memory, view_memory
from kesoku.config import KesokuConfig, WorkspaceConfig
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager


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
    res = update_memory(
        category="forbidden_category",
        key="my_key",
        title="Invalid Title",
        content="Content",
        role="default",
        create_category=False,
        context=ctx,
    )
    assert "Write Denied: Category 'forbidden_category' is not recognized" in res

    # 2. Force override creating a new category with user permission set
    res = update_memory(
        category="forbidden_category",
        key="my_key",
        title="Invalid Title",
        content="Content",
        role="default",
        create_category=True,
        context=ctx,
    )
    assert "Memory successfully saved!" in res
    assert "Category: `forbidden_category`" in res
    assert "Key: `my_key`" in res

    # 3. Verify that updating with an invalid key is rejected
    res_fail = update_memory(
        category="progress",
        key="  Standard Japanese Book  ",
        title="《标日》学习进度",
        content="已学完第23课",
        role="asuka",
        create_category=False,
        context=ctx,
    )
    assert "Error: Invalid Key" in res_fail

    # 4. Normal update on a role-isolated category with a strictly valid key
    res = update_memory(
        category="notes",
        key="funny_asuka_event",
        title="Asuka Day 1",
        content="Asuka was tsundere today",
        role="asuka",
        create_category=False,
        context=ctx,
    )
    assert "Memory successfully saved!" in res
    assert "Key: `funny_asuka_event`" in res

    # 4. Testing list_memories tool with roleplay segregation
    # Writing a default notes memory
    update_memory(
        category="notes",
        key="funny_general_event",
        title="General Fun",
        content="Someone made a joke today",
        role="default",
        create_category=False,
        context=ctx,
    )

    # List memories as Tifa
    list_tifa = list_memories(category="notes", role="tifa", context=ctx)
    # Tifa can see default memories but NOT Asuka's memories
    assert "funny_general_event" in list_tifa
    assert "funny_asuka_event" not in list_tifa

    # List memories as Asuka
    list_asuka = list_memories(category="notes", role="asuka", context=ctx)
    # Asuka can see both default and Asuka-specific memories
    assert "funny_general_event" in list_asuka
    assert "funny_asuka_event" in list_asuka

    # 5. Testing view_memory tool with dynamic in-memory Markdown aggregation (key=None)
    view_asuka_all = view_memory(category="notes", key=None, role="asuka", context=ctx)
    assert "# Category: notes (scope: asuka)" in view_asuka_all
    assert "## General Fun (key: `funny_general_event`, scope: `default`)" in view_asuka_all
    assert "## Asuka Day 1 (key: `funny_asuka_event`, scope: `asuka`)" in view_asuka_all
    assert "Someone made a joke today" in view_asuka_all
    assert "Asuka was tsundere today" in view_asuka_all

    # 6. Testing delete_memory tool
    delete_res = delete_memory(category="notes", key="funny_asuka_event", role="asuka", context=ctx)
    assert "Memory successfully deleted" in delete_res

    # Verify deletion
    list_asuka_after = list_memories(category="notes", role="asuka", context=ctx)
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
    await gw.set_channel_role("discord", "chan_123", "tifa")

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
    res = update_memory(
        category="progress",
        key="milestone",
        title="Milestone",
        content="Completed Task",
        role="tifa",  # Passed explicitly
        context=ctx,
    )
    assert "Scope: `default`" in res

    # 2. Update 'notes' memory - it should go to 'tifa' active role scope
    res = update_memory(
        category="notes",
        key="funny_event",
        title="Funny",
        content="A funny thing happened",
        role="default",  # Passed explicitly but should be overridden
        context=ctx,
    )
    assert "Scope: `tifa`" in res

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
    current_role = await gw.get_channel_role("discord", "chan_123")
    assert current_role == "asuka"
