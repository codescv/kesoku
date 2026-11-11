"""Unit tests for Tier 0a `<memory_pinned>`, `PinnedMemoryCache`, and `allow_sync=False` (`H1`, `H4`)."""

import os

from kesoku.agent.memory_store import (
    PinnedMemoryCache,
    bump_memory_index_version,
    format_pinned_block,
    load_pinned_memories,
    parse_memory_frontmatter,
    search_role_memories,
    sync_role_memories,
)
from kesoku.db import DatabaseManager


def test_parse_memory_frontmatter_visibility_and_cues() -> None:
    """Verify `MemoryFrontmatter` extracts `description`, `visibility`, and `cues` while keeping `str` compat."""
    raw = (
        "---\n"
        "description: Core user diet and residence profile\n"
        "visibility: PINNED\n"
        "cues:\n"
        "  - 乳糖不耐\n"
        "  - 住址\n"
        "---\n"
        "## 饮食禁忌\n"
        "- 用户乳糖不耐受，喝咖啡选燕麦奶 #direct #time:2026-11-01\n"
    )
    fm, body = parse_memory_frontmatter(raw)
    assert fm == "Core user diet and residence profile"
    assert fm.description == "Core user diet and residence profile"
    assert fm.visibility == "PINNED"
    assert fm.cues == ["乳糖不耐", "住址"]
    assert "乳糖不耐受" in body


def test_pinned_cache_hit_and_version_bump_invalidation(tmp_path, monkeypatch) -> None:
    """Verify `PinnedMemoryCache` serves cached `<memory_pinned>` with 0 disk reads until version bumps (`H1`)."""
    mem_dir = tmp_path / "roles" / "tutor" / "memories"
    user_dir = mem_dir / "user"
    user_dir.mkdir(parents=True)
    profile = user_dir / "profile.md"
    profile.write_text(
        "---\ndescription: User profile\nvisibility: PINNED\n---\n## 偏好\n- 乳糖不耐受 #direct\n",
        encoding="utf-8",
    )

    cache = PinnedMemoryCache()
    block_1 = cache.get_block("tutor", memories_dir=str(mem_dir), max_chars=2500)
    assert "<memory_pinned" in block_1
    assert "乳糖不耐受" in block_1

    # Rewrite file on disk WITHOUT bumping version -> cache returns identical cached string without disk read
    profile.write_text(
        "---\ndescription: User profile\nvisibility: PINNED\n---\n## 偏好\n- 乳糖不耐受 #direct\n- 常住东京 #direct\n",
        encoding="utf-8",
    )
    block_cached = cache.get_block("tutor", memories_dir=str(mem_dir), max_chars=2500)
    assert block_cached == block_1
    assert "常住东京" not in block_cached

    # Bump version -> next `get_block` reloads fresh content
    bump_memory_index_version("tutor")
    block_refreshed = cache.get_block("tutor", memories_dir=str(mem_dir), max_chars=2500)
    assert "常住东京" in block_refreshed


def test_format_pinned_block_line_boundary_truncation_and_callback() -> None:
    """Verify `<memory_pinned>` caps at `max_chars` (`<= 2500`), truncates cleanly on newline, and fires callback."""
    bullets = "\n".join(f"- 核心档案条目编号 {i:02d}：需要长期保留的重要饮食与行程偏好信息 #direct" for i in range(60))
    triggered: list[str] = []

    block = format_pinned_block(
        [("user/profile.md", f"## 档案\n{bullets}")],
        max_chars=600,
        on_over_budget=triggered.append,
    )
    assert len(block) <= 600
    assert block.startswith("<memory_pinned ")
    assert block.endswith("</memory_pinned>\n")
    assert "…[pinned memory truncated]" in block
    assert triggered == ["user/profile.md"]


def test_pinned_excluded_from_ambient_search_and_allow_sync_false_skips_walk(tmp_path, monkeypatch) -> None:
    """Verify `PINNED` exclusion in `search_role_memories` and `allow_sync=False` skipping `os.walk`."""
    monkeypatch.setattr("kesoku.utils.embedding.get_embedding", lambda text: [1.0] + [0.0] * 383)
    monkeypatch.setattr(
        "kesoku.utils.embedding.get_embeddings",
        lambda texts: [[1.0] + [0.0] * 383 for _ in texts],
    )

    db = DatabaseManager(str(tmp_path / "pinned.db"))
    db.init_tables()

    mem_dir = tmp_path / "roles" / "tutor" / "memories"
    (mem_dir / "user").mkdir(parents=True)
    (mem_dir / "user" / "profile.md").write_text(
        "---\ndescription: Pinned user profile\nvisibility: PINNED\n---\n- 乳糖不耐受\n",
        encoding="utf-8",
    )
    (mem_dir / "lessons.md").write_text(
        "---\ndescription: Lesson progress\nvisibility: DEFAULT\n---\n- 标日第12课已完成\n",
        encoding="utf-8",
    )
    (mem_dir / "hidden.md").write_text(
        "---\ndescription: Archived notes\nvisibility: HIDDEN\n---\n- 旧记录\n",
        encoding="utf-8",
    )

    sync_role_memories(db, "tutor", str(mem_dir))

    # `load_pinned_memories` returns only PINNED
    pinned = load_pinned_memories("tutor", memories_dir=str(mem_dir), db=db)
    assert [p[0] for p in pinned] == ["user/profile.md"]

    # Pre-turn `allow_sync=False, include_pinned=False`: never calls `os.walk`, excludes PINNED & HIDDEN
    def _forbid_walk(*args, **kwargs):
        raise AssertionError("os.walk must not run when allow_sync=False")

    monkeypatch.setattr(os, "walk", _forbid_walk)

    ambient_hits = search_role_memories(
        db,
        "tutor",
        "乳糖不耐 标日",
        limit=10,
        min_score=0.1,
        memories_dir=str(mem_dir),
        include_pinned=False,
        allow_sync=False,
    )
    assert [m.rel_path for m in ambient_hits] == ["lessons.md"]

    # Explicit tool search `include_pinned=True` surfaces PINNED + DEFAULT, still excluding HIDDEN
    explicit_hits = search_role_memories(
        db,
        "tutor",
        "*",
        limit=10,
        memories_dir=str(mem_dir),
        include_pinned=True,
        allow_sync=False,
    )
    assert {m.rel_path for m in explicit_hits} == {"user/profile.md", "lessons.md"}
