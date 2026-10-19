"""Unit tests for the Markdown Role Memory Store, Progressive Disclosure, and memory_search tool."""

import time

import pytest

import kesoku.agent.tools
import kesoku.agent.tools.memory
from kesoku.agent.history import TRUNCATED_ARG_TOOLS
from kesoku.agent.memory_store import (
    MemoryMatch,
    chunk_memory_markdown,
    format_ambient_memory_block,
    parse_memory_frontmatter,
    render_context_query,
    search_role_memories,
    sync_role_memories,
)
from kesoku.agent.tools import (
    ToolContext,
    default_registry,
    memory_search,
)
from kesoku.config import KesokuConfig, WorkspaceConfig
from kesoku.constants import MessageRole, MessageType
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager, Message, Session


def test_parse_memory_frontmatter() -> None:
    """Verify frontmatter description extraction and body fallbacks."""
    raw_with_fm = (
        "---\n"
        "description: Neon the cat adoption date and favorite snacks\n"
        "---\n\n"
        "# Pet Profile\n"
        "- Adopted on 2026-05-01\n"
    )
    desc, body = parse_memory_frontmatter(raw_with_fm)
    assert desc == "Neon the cat adoption date and favorite snacks"
    assert body.startswith("# Pet Profile")

    # Multi-line YAML block scalar
    raw_multiline = (
        "---\n"
        "description: >-\n"
        "  Multi-line description\n"
        "  for testing\n"
        "---\n"
        "Body text"
    )
    desc_multi, body_multi = parse_memory_frontmatter(raw_multiline)
    assert desc_multi == "Multi-line description for testing"
    assert body_multi == "Body text"

    # Fallback to first Markdown heading when frontmatter is absent
    raw_no_fm = "# Kamakura Trip Notes\n\nVisited Enoshima aquarium."
    desc_no_fm, body_no_fm = parse_memory_frontmatter(raw_no_fm)
    assert desc_no_fm == "Kamakura Trip Notes"
    assert "Visited Enoshima aquarium." in body_no_fm


def test_chunk_memory_markdown_breadcrumbs() -> None:
    """Verify structural Markdown chunking preserves ancestor heading breadcrumbs and ignores code comments."""
    body = (
        "# Architecture\n"
        "Top-level intro.\n\n"
        "## Database\n"
        "Uses SQLite WAL mode.\n"
        "```python\n"
        "# Not a markdown heading\n"
        "conn.execute('PRAGMA journal_mode=WAL;')\n"
        "```\n\n"
        "### Indexing\n"
        "Memory chunks table.\n"
    )
    chunks = chunk_memory_markdown("projects/kesoku.md", "Kesoku architecture overview", body)
    assert chunks[0] == "# projects/kesoku.md\nKesoku architecture overview"
    assert any("# Architecture\n## Database" in c and "# Not a markdown heading" in c for c in chunks)
    assert any("# Architecture\n## Database\n### Indexing\nMemory chunks table." in c for c in chunks)


def test_sync_role_memories_incremental_and_pruning(tmp_path, monkeypatch) -> None:
    """Verify incremental mtime/hash syncing and deleted file pruning in SQLite."""
    db_path = str(tmp_path / "test_sync.db")
    db = DatabaseManager(db_path)
    db.init_tables()

    embed_calls = 0

    def mock_get_embeddings(texts: list[str]) -> list[list[float]]:
        nonlocal embed_calls
        embed_calls += 1
        return [[1.0] + [0.0] * 383 for _ in texts]

    monkeypatch.setattr("kesoku.utils.embedding.get_embeddings", mock_get_embeddings)

    mem_dir = tmp_path / "roles" / "writer" / "memories"
    mem_dir.mkdir(parents=True)
    f1 = mem_dir / "pets.md"
    f1.write_text("---\ndescription: Cat Neon\n---\n- Likes tuna\n", encoding="utf-8")

    # 1. Initial sync indexes f1
    updated1 = sync_role_memories(db, "writer", str(mem_dir))
    assert updated1 == 1
    assert embed_calls == 1
    files_in_db = db.get_memory_files_by_role("writer")
    assert len(files_in_db) == 1
    assert files_in_db[0]["rel_path"] == "pets.md"

    # 2. Second sync without file changes skips re-embedding
    updated2 = sync_role_memories(db, "writer", str(mem_dir))
    assert updated2 == 0
    assert embed_calls == 1

    # 3. Deleting the file prunes it from memory_files and memory_chunks
    f1.unlink()
    updated3 = sync_role_memories(db, "writer", str(mem_dir))
    assert updated3 == 0
    assert db.get_memory_files_by_role("writer") == []
    assert db.get_memory_chunks_for_role("writer") == []


def test_search_role_memories_hybrid_and_role_isolation(tmp_path, monkeypatch) -> None:
    """Verify MAX(chunk_similarity) + lexical_boost ranking and strict role isolation."""
    db_path = str(tmp_path / "test_search_mem.db")
    db = DatabaseManager(db_path)
    db.init_tables()

    def pad(v: list[float]) -> list[float]:
        return v + [0.0] * (384 - len(v))

    def mock_get_embedding(text: str) -> list[float]:
        if text == "controller":
            return pad([1.0, 0.0])
        if "controller" in text.lower() or "joystick" in text.lower():
            return pad([0.8, 0.6])
        return pad([0.0, 1.0])

    monkeypatch.setattr("kesoku.utils.embedding.get_embedding", mock_get_embedding)
    monkeypatch.setattr(
        "kesoku.utils.embedding.get_embeddings",
        lambda texts: [mock_get_embedding(t) for t in texts],
    )

    coder_mem = tmp_path / "roles" / "coder" / "memories"
    coder_mem.mkdir(parents=True)
    # File 1: semantic match + exact lexical match ("controller")
    (coder_mem / "hardware.md").write_text(
        "---\ndescription: Custom controller build log\n---\n- Built a custom controller\n",
        encoding="utf-8",
    )
    # File 2: semantic match only ("joystick", no literal "controller")
    (coder_mem / "arcade.md").write_text(
        "---\ndescription: Arcade joystick notes\n---\n- Sanwa joystick parts\n",
        encoding="utf-8",
    )

    # Another role's memory that should never leak into 'coder'
    writer_mem = tmp_path / "roles" / "writer" / "memories"
    writer_mem.mkdir(parents=True)
    (writer_mem / "secret.md").write_text(
        "---\ndescription: Writer controller secret\n---\n- Secret controller\n",
        encoding="utf-8",
    )
    sync_role_memories(db, "writer", str(writer_mem))

    matches = search_role_memories(
        db=db,
        role="coder",
        query="controller",
        limit=5,
        min_score=0.5,
        memories_dir=str(coder_mem),
    )
    assert len(matches) == 2
    assert matches[0].rel_path == "hardware.md"
    assert matches[1].rel_path == "arcade.md"
    # Lexical boost gives hardware.md a higher score than arcade.md despite identical cosine similarity
    assert matches[0].score > matches[1].score
    assert all(m.rel_path != "secret.md" for m in matches)


def test_render_context_query_and_ambient_block() -> None:
    """Verify multi-turn context query extraction and <memory> progressive disclosure formatting."""
    m1 = Message(
        id="u1",
        session_id="s1",
        chatbot_id="cli",
        channel_id="c1",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="<instructions>\nIgnore me\n</instructions>\n<current_message>\nOld turn 1\n</current_message>",
        timestamp=100.0,
    )
    m2 = Message(
        id="u2",
        session_id="s1",
        chatbot_id="cli",
        channel_id="c1",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="<memory>\nOld mem\n</memory>\n<current_message>\nLet's check Neon's snacks\n</current_message>",
        timestamp=200.0,
    )
    m3 = Message(
        id="a2",
        session_id="s1",
        chatbot_id="cli",
        channel_id="c1",
        sender="Writer",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content="Sure! Let me check.\n[question: Which one? || Tuna | Salmon]",
        timestamp=210.0,
    )
    m4 = Message(
        id="u3",
        session_id="s1",
        chatbot_id="cli",
        channel_id="c1",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="What did she eat last week?",
        timestamp=300.0,
    )

    q = render_context_query([m1, m2, m3, m4], m4, max_turns=2)
    assert "Old turn 1" not in q
    assert "Ignore me" not in q
    assert "User: Let's check Neon's snacks" in q
    assert "Assistant: Sure! Let me check." in q
    assert "Tuna | Salmon" not in q
    assert "User: What did she eat last week?" in q

    block = format_ambient_memory_block(
        [
            MemoryMatch(
                rel_path="pets/neon.md",
                memory_path="$MEMORY_DIR/pets/neon.md",
                description="Neon the cat profile and favorite snacks",
                score=0.88,
                mtime=1700000000.0,
            )
        ]
    )
    assert block.startswith("<memory>\n")
    assert '$MEMORY_DIR/pets/neon.md' in block
    assert "description: Neon the cat profile and favorite snacks" in block
    assert "read_file" in block
    assert block.endswith("</memory>\n")


@pytest.mark.asyncio
async def test_memory_search_tool_and_legacy_tools_removed(tmp_path, monkeypatch) -> None:
    """Verify memory_search tool works end-to-end and chat_search/view_message are removed."""
    from kesoku.gateway.gateway import Gateway

    names = [f.__name__ for f in default_registry.get_tools_list()]
    assert "memory_search" in names
    assert "update_file" in names
    assert "chat_search" not in names
    assert "view_message" not in names
    assert "update_session_notes" not in names
    assert not hasattr(kesoku.agent.tools, "chat_search")
    assert not hasattr(kesoku.agent.tools, "view_message")

    monkeypatch.setattr("kesoku.utils.embedding.get_embedding", lambda text: [1.0] + [0.0] * 383)
    monkeypatch.setattr(
        "kesoku.utils.embedding.get_embeddings",
        lambda texts: [[1.0] + [0.0] * 383 for _ in texts],
    )

    temp_db = str(tmp_path / "test_tool.db")
    roles_dir = tmp_path / "roles"
    mem_dir = roles_dir / "coder" / "memories"
    mem_dir.mkdir(parents=True)
    (mem_dir / "python_rules.md").write_text(
        "---\ndescription: Always use uv run pytest\n---\n- Run tests with uv\n",
        encoding="utf-8",
    )

    db = DatabaseManager(temp_db)
    db.init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db, roles_dir=str(roles_dir)))
    gw = Gateway(context=KesokuContext(config=cfg))

    await gw.db.set_channel_role("discord", "chan_1", "coder")
    session1 = Session(
        id="sess_1",
        title="Sess 1",
        created_at=time.time(),
        updated_at=time.time(),
        role_name="coder",
    )
    await gw.db.create_session(session1)
    await gw.db.set_active_session_for_channel("discord", "chan_1", "sess_1")

    ctx = ToolContext(
        session_id="sess_1",
        session_workspace="test_ws",
        chatbot_id="discord",
        channel_id="chan_1",
        gateway=gw,
    )

    with monkeypatch.context() as m:
        m.setattr("kesoku.utils.path.get_config", lambda: cfg)
        res = await memory_search(query="pytest", limit=5, context=ctx)

    assert "Memory Search Results for 'pytest'" in res
    assert "$MEMORY_DIR/python_rules.md" in res
    assert "Always use uv run pytest" in res
    assert "read_file" in res


def test_legacy_notes_tool_still_truncated_in_history() -> None:
    """Old DB records that called update_session_notes keep having their arguments truncated."""
    assert "update_session_notes" in TRUNCATED_ARG_TOOLS
    assert "update_file" in TRUNCATED_ARG_TOOLS
