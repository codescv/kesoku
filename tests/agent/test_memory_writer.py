"""Unit tests for `MemoryWriter` atomic writes, dedup, contradiction/supersede, and `update_file`."""

import pytest

from kesoku.agent.memory_store import parse_memory_frontmatter
from kesoku.agent.memory_writer import MemoryWriter
from kesoku.agent.tools.file import update_file
from kesoku.agent.tools.registry import ToolContext
from kesoku.config import KesokuConfig, WorkspaceConfig
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager, Session
from kesoku.gateway.gateway import Gateway


def test_memory_writer_auto_section_dedup_tags_and_conflict_reapply(tmp_path, monkeypatch) -> None:
    """Verify section auto-creation, auto `#session`/`#time` tags, Jaccard dedup, and hash-conflict re-apply."""
    monkeypatch.setattr("kesoku.utils.embedding.get_embeddings", lambda texts: [[0.5] * 384 for _ in texts])

    db = DatabaseManager(str(tmp_path / "writer.db"))
    db.init_tables()
    mem_dir = tmp_path / "roles" / "coder" / "memories"
    mem_dir.mkdir(parents=True)

    # Pre-LLM snapshot when `user/profile.md` did not exist yet
    snaps = MemoryWriter.snapshot_files(str(mem_dir), ["user/profile.md"])
    assert snaps["user/profile.md"] == ""

    # Simulate a concurrent in-turn edit creating `user/profile.md` BEFORE `apply_ops` acquires `.memory.lock`
    (mem_dir / "user").mkdir(parents=True)
    (mem_dir / "user" / "profile.md").write_text(
        "---\ndescription: Core user profile\nvisibility: PINNED\n---\n## 饮食\n- 不吃香菜 #direct #time:2026-10-01\n",
        encoding="utf-8",
    )

    ops = [
        {
            "op": "upsert_bullet",
            "file": "user/profile.md",
            "section": "饮食",
            "text": "用户有乳糖不耐受，喝拿铁要换燕麦奶",
            "tags": ["#direct"],
            "pinned": True,
        },
        # Near-duplicate bullet (high CJK bigram Jaccard) -> deduplicated automatically
        {
            "op": "upsert_bullet",
            "file": "user/profile.md",
            "section": "饮食",
            "text": "用户有乳糖不耐受，喝拿铁需要换燕麦奶",
            "tags": ["#inferred"],
            "pinned": True,
        },
        # Missing section `## 出行` -> auto-created without dropping the op (`C4`)
        {
            "op": "upsert_bullet",
            "file": "user/profile.md",
            "section": "出行",
            "text": "偏好靠窗座位",
            "tags": ["#direct"],
            "pinned": True,
        },
    ]

    res = MemoryWriter.apply_ops(
        ops,
        role="coder",
        memories_dir=str(mem_dir),
        db=db,
        session_id="62646c19abcdef",
        event_date="2026-11-10",
        snapshots=snaps,
    )

    # Ops were re-applied on fresh disk content rather than dropped!
    assert res.reapplied_after_conflict == ["user/profile.md"]
    assert res.ops_dropped == 0
    content = (mem_dir / "user" / "profile.md").read_text(encoding="utf-8")
    # Concurrent edit preserved AND new bullets added with provenance tags
    assert "不吃香菜" in content
    assert content.count("乳糖不耐受") == 1
    assert "#session:62646c19" in content
    assert "#time:2026-11-10" in content
    assert "## 出行" in content
    assert "偏好靠窗座位" in content


def test_memory_writer_contradiction_supersede_and_retract_rules(tmp_path, monkeypatch) -> None:
    """Verify `#direct > #inferred`, newer `#time` wins, `#superseded` history archive, and `retract`."""
    monkeypatch.setattr("kesoku.utils.embedding.get_embeddings", lambda texts: [[0.2] * 384 for _ in texts])

    mem_dir = tmp_path / "roles" / "tutor" / "memories"
    mem_dir.mkdir(parents=True)
    target = mem_dir / "residence.md"
    target.write_text(
        "---\ndescription: User residence\n---\n"
        "## 居住地\n"
        "- 常住在大阪梅田 #direct #time:2026-10-01 #session:11112222\n"
        "- 周末常去奈良散步 #inferred #time:2026-10-05 #session:11112222\n",
        encoding="utf-8",
    )

    # 1. `#inferred` attempting to replace an existing `#direct` bullet MUST lose (`#direct > #inferred`)
    MemoryWriter.apply_ops(
        [
            {
                "op": "upsert_bullet",
                "file": "residence.md",
                "section": "居住地",
                "text": "可能住在神户",
                "replaces": "常住在大阪梅田",
                "tags": ["#inferred"],
            }
        ],
        role="tutor",
        memories_dir=str(mem_dir),
        event_date="2026-11-01",
    )
    after_inferred = target.read_text(encoding="utf-8")
    assert "常住在大阪梅田" in after_inferred
    assert "可能住在神户" not in after_inferred

    # 2. Newer `#direct` replacing older `#direct` supersedes it and moves old bullet to `## 历史` with `#superseded`
    MemoryWriter.apply_ops(
        [
            {
                "op": "upsert_bullet",
                "file": "residence.md",
                "section": "居住地",
                "text": "已搬家到东京目黑区",
                "replaces": "常住在大阪梅田",
                "tags": ["#direct"],
            },
            {
                "op": "retract",
                "file": "residence.md",
                "match": "周末常去奈良散步",
            },
            {
                "op": "set_description",
                "file": "residence.md",
                "description": "User current Tokyo residence and move history",
            },
        ],
        role="tutor",
        memories_dir=str(mem_dir),
        session_id="99887766",
        event_date="2026-11-11",
    )

    updated = target.read_text(encoding="utf-8")
    fm, body = parse_memory_frontmatter(updated)
    assert fm.description == "User current Tokyo residence and move history"
    assert "已搬家到东京目黑区" in body
    assert "周末常去奈良散步" not in body
    assert "## 历史" in body
    hist_part = body.split("## 历史", 1)[1]
    assert "常住在大阪梅田" in hist_part
    assert "#superseded" in hist_part


def test_memory_writer_pinned_allowlist_rejected_tags_and_12kib_cap(tmp_path, monkeypatch) -> None:
    """Verify non-allowlisted `pinned=True` & `rejected_tags` drop, and `>12 KiB` reports `files_over_cap`."""
    monkeypatch.setattr("kesoku.utils.embedding.get_embeddings", lambda texts: [[0.1] * 384 for _ in texts])

    mem_dir = tmp_path / "roles" / "tutor" / "memories"
    mem_dir.mkdir(parents=True)

    # Pre-seed an ~8.1 KiB file so one more bullet crosses max_file_bytes=8192
    padding_lines = "\n".join(f"- 长期技术笔记条目 {i:04d}：包含详细架构约束与测试参数说明 #direct" for i in range(110))
    (mem_dir / "big_topic.md").write_text(
        f"---\ndescription: Large topic file\n---\n## 笔记\n{padding_lines}\n",
        encoding="utf-8",
    )

    res = MemoryWriter.apply_ops(
        [
            # Rejected tag `#fiction` -> dropped
            {
                "op": "upsert_bullet",
                "file": "big_topic.md",
                "section": "笔记",
                "text": "虚构设定里的魔法学院",
                "tags": ["#fiction"],
            },
            # Non-allowlisted `pinned=True` -> dropped
            {
                "op": "upsert_bullet",
                "file": "big_topic.md",
                "section": "笔记",
                "text": "试图把普通主题设为置顶",
                "tags": ["#direct"],
                "pinned": True,
            },
            # Valid bullet pushing `big_topic.md` over cap -> written atomically, zero `.overflow.md`
            {
                "op": "upsert_bullet",
                "file": "big_topic.md",
                "section": "笔记",
                "text": "新增一条真实架构决策记录：SQLite 使用 WAL 模式并限制单写者队列以消除锁竞争",
                "tags": ["#direct"],
            },
        ],
        role="tutor",
        memories_dir=str(mem_dir),
        pinned_files=["user/profile.md"],
        rejected_tags=["#fiction"],
        max_file_bytes=8192,
    )

    assert res.ops_dropped == 2
    assert res.ops_applied == 1
    assert res.files_over_cap == ["big_topic.md"]
    assert not list(mem_dir.rglob("*.overflow.md"))


@pytest.mark.asyncio
async def test_update_file_on_memory_dir_enforces_lock_pinned_allowlist_and_return_contract(
    tmp_path, monkeypatch
) -> None:
    """Verify `update_file` on `$MEMORY_DIR` keeps exact return strings, syncs DB, and guards `PINNED`."""
    monkeypatch.setattr("kesoku.utils.embedding.get_embeddings", lambda texts: [[0.3] * 384 for _ in texts])

    db_path = str(tmp_path / "file_tool.db")
    db = DatabaseManager(db_path)
    db.init_tables()
    roles_dir = tmp_path / "roles"
    cfg = KesokuConfig(
        workspace=WorkspaceConfig(
            db_path=db_path,
            roles_dir=str(roles_dir),
            sessions_dir=str(tmp_path / "sessions"),
        )
    )
    cfg.agent_working_dir = str(tmp_path)
    gw = Gateway(context=KesokuContext(config=cfg, db=db))
    await gw.db.set_channel_role("cli", "ch1", "coder")
    await gw.db.create_session(Session(id="s_edit", title="Edit", created_at=1.0, updated_at=1.0, role_name="coder"))
    await gw.db.set_active_session_for_channel("cli", "ch1", "s_edit")

    ctx = ToolContext(
        session_id="s_edit",
        session_workspace="s_edit",
        chatbot_id="cli",
        channel_id="ch1",
        gateway=gw,
        kesoku_context=gw.context,
    )

    with monkeypatch.context() as m:
        m.setattr("kesoku.agent.tools.file.get_config", lambda: cfg)
        m.setattr("kesoku.utils.path.get_config", lambda: cfg)

        # 1. Create allow-listed `$MEMORY_DIR/user/profile.md` with `visibility: PINNED` -> exact return contract
        out_ok = update_file(
            file_name="$MEMORY_DIR/user/profile.md",
            old_content="",
            new_content="---\ndescription: User profile\nvisibility: PINNED\n---\n- 偏好使用 uv run pytest\n",
            context=ctx,
        )
        assert out_ok.startswith("Success: Created file '$MEMORY_DIR/user/profile.md'")
        indexed = db.get_memory_files_by_role("coder", visibility="PINNED")
        assert [r["rel_path"] for r in indexed] == ["user/profile.md"]

        # 2. Attempting `visibility: PINNED` on non-allow-listed file is rejected cleanly
        out_bad = update_file(
            file_name="$MEMORY_DIR/random_notes.md",
            old_content="",
            new_content="---\ndescription: Unauthorized pinned\nvisibility: PINNED\n---\n- 内容\n",
            context=ctx,
        )
        assert "Error:" in out_bad
        assert "visibility: PINNED" in out_bad
