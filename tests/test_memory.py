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

    # 1. Upsert global progress memory
    db.upsert_agent_memory(
        category="progress",
        key="standard_japanese",
        title="标准日本语",
        content="学习完了第22课",
        role="global",
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
    mem1 = db.get_agent_memory(category="progress", key="standard_japanese", role="global")
    assert mem1 is not None
    assert mem1["title"] == "标准日本语"
    assert mem1["content"] == "学习完了第22课"
    assert mem1["role"] == "global"

    mem2 = db.get_agent_memory(category="learnings", key="proxy_setting", role="asuka")
    assert mem2 is not None
    assert mem2["title"] == "代理配置"
    assert mem2["role"] == "asuka"

    # 4. Fetch multiple memories (listing / filtering)
    # Getting memories filtered by category
    progress_mems = db.get_agent_memories(category="progress")
    assert len(progress_mems) == 1
    assert progress_mems[0]["key"] == "standard_japanese"

    # Query with role='asuka' (should fetch both global + asuka's memories)
    all_asuka_mems = db.get_agent_memories(role="asuka")
    # Both global progress and asuka specific learnings are in the system
    assert len(all_asuka_mems) == 2

    # Query with role='tifa' (should fetch only global memories, and tifa specific ones if they existed)
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
        role="global",
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
        role="global",
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

    # 4. Normal update on a core category with a strictly valid key
    res = update_memory(
        category="progress",
        key="standard_japanese_book",
        title="《标日》学习进度",
        content="已学完第23课",
        role="asuka",
        create_category=False,
        context=ctx,
    )
    assert "Memory successfully saved!" in res
    assert "Key: `standard_japanese_book`" in res

    # 4. Testing list_memories tool with roleplay segregation
    # Writing a global progress memory to test co-existence
    update_memory(
        category="progress",
        key="brief_intelligence",
        title="《智能简史》读书笔记",
        content="读到第10章",
        role="global",
        create_category=False,
        context=ctx,
    )

    # List memories as Tifa
    list_tifa = list_memories(category="progress", role="tifa", context=ctx)
    # Tifa can only see global memories
    assert "brief_intelligence" in list_tifa
    assert "standard_japanese_book" not in list_tifa

    # List memories as Asuka
    list_asuka = list_memories(category="progress", role="asuka", context=ctx)
    # Asuka can see both global and Asuka-specific memories
    assert "brief_intelligence" in list_asuka
    assert "standard_japanese_book" in list_asuka

    # 5. Testing view_memory tool with dynamic in-memory Markdown aggregation (key=None)
    view_asuka_all = view_memory(category="progress", key=None, role="asuka", context=ctx)
    assert "# Category: progress (scope: asuka)" in view_asuka_all
    assert "## 《智能简史》读书笔记 (key: `brief_intelligence`, scope: `global`)" in view_asuka_all
    assert "## 《标日》学习进度 (key: `standard_japanese_book`, scope: `asuka`)" in view_asuka_all
    assert "读到第10章" in view_asuka_all
    assert "已学完第23课" in view_asuka_all

    # 6. Testing delete_memory tool
    delete_res = delete_memory(category="progress", key="standard_japanese_book", role="asuka", context=ctx)
    assert "Memory successfully deleted" in delete_res

    # Verify deletion
    list_asuka_after = list_memories(category="progress", role="asuka", context=ctx)
    assert "standard_japanese_book" not in list_asuka_after
