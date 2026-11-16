"""Tests for the user profile store and its managed memory block (`kesoku.agent.user_profile`)."""

import json

import pytest

from kesoku.agent import user_profile as up
from kesoku.agent.memory_store import load_pinned_memories
from kesoku.db.manager import DatabaseManager


@pytest.fixture
def db(tmp_path):
    mgr = DatabaseManager(str(tmp_path / "k.db"))
    mgr.init_tables()
    return mgr


def test_render_block_and_marker_neutralization():
    block = up.render_profile_block("Sam <!-- x -->", "line one\nline two")
    assert block.startswith(up.BLOCK_START)
    assert block.rstrip().endswith(up.BLOCK_END)
    assert "称呼：Sam <!- x ->" in block
    assert "简介：line one" in block and "  line two" in block
    assert block.count("<!--") == 2
    assert up.render_profile_block("", "  ") == ""


def test_upsert_into_new_file_creates_pinned_frontmatter():
    text = up.upsert_managed_block("", up.render_profile_block("Sam", ""))
    assert text.startswith("---\n")
    assert "visibility: PINNED" in text
    assert up.BLOCK_START in text


def test_upsert_preserves_text_outside_block():
    original = (
        '---\ndescription: "mine"\nvisibility: PINNED\n---\n\n## 饮食\n- likes tea\n\n## 历史\n- old fact\n'
    )
    first = up.upsert_managed_block(original, up.render_profile_block("Sam", "bio"))
    assert first.index(up.BLOCK_START) < first.index("## 饮食")
    assert "- likes tea" in first and "- old fact" in first
    assert first.startswith('---\ndescription: "mine"\nvisibility: PINNED\n---\n')

    second = up.upsert_managed_block(first, up.render_profile_block("Robin", "bio"))
    assert "Robin" in second and "Sam" not in second
    assert second.replace(up.render_profile_block("Robin", "bio"), "") == first.replace(
        up.render_profile_block("Sam", "bio"), ""
    )

    removed = up.upsert_managed_block(second, "")
    assert up.BLOCK_START not in removed
    assert "- likes tea" in removed


def test_upsert_without_frontmatter_prepends():
    out = up.upsert_managed_block("hand written\n", up.render_profile_block("Sam", ""))
    assert out.startswith(up.BLOCK_START)
    assert out.endswith("hand written\n")


def test_sync_profile_to_roles_writes_pinned_and_invalidates(tmp_path, db, monkeypatch):
    roles = tmp_path / "roles"
    for rid in ("alice", "helper_bot"):
        (roles / rid / "memories").mkdir(parents=True)
    existing = roles / "helper_bot" / "memories" / "user" / "profile.md"
    existing.parent.mkdir(parents=True)
    existing.write_text('---\ndescription: "d"\nvisibility: PINNED\n---\n\n## 偏好\n- keep me\n', encoding="utf-8")
    invalidated = []
    monkeypatch.setattr(up.default_pinned_cache, "invalidate", lambda role=None: invalidated.append(role))

    changed = up.sync_profile_to_roles(db, str(roles), ["alice", "helper_bot", "../escape"], "Sam", "Engineer")

    assert sorted(changed) == ["alice", "helper_bot"]
    assert sorted(invalidated) == ["alice", "helper_bot"]
    pinned = load_pinned_memories("alice", str(roles / "alice" / "memories"))
    assert pinned and "称呼：Sam" in pinned[0][1]
    kept = existing.read_text(encoding="utf-8")
    assert "- keep me" in kept and "简介：Engineer" in kept
    assert db.get_memory_files_by_role("alice")

    # Idempotent: nothing changes on a second identical sync.
    assert up.sync_profile_to_roles(db, str(roles), ["alice", "helper_bot"], "Sam", "Engineer") == []


def test_profile_store_defaults_and_validation(tmp_path):
    store = up.UserProfileStore(str(tmp_path / "awd"))
    assert store.load()["character_dir"] == up.DEFAULT_CHARACTER_DIR
    assert store.public_view()["avatar_ref"] is None

    store.save({"nickname": "Sam", "bio": "b", "cameo_opt_in": True, "character_dir": "../x", "avatar_rev": "2"})
    loaded = store.load()
    assert loaded["character_dir"] == up.DEFAULT_CHARACTER_DIR
    assert loaded["avatar_rev"] == 2
    assert loaded["cameo_opt_in"] is True

    updated = store.update(nickname="Robin")
    assert updated["nickname"] == "Robin"
    assert json.loads(store.path and open(store.path, encoding="utf-8").read())["nickname"] == "Robin"


# ---------------------------------------------------------------- v0.3.6 import + cross-channel sync

USER_MD = (
    "# USER PROFILE\n"
    "- Male\n"
    "- Discord handle: @bottlemerchant\n"
    "- Nickname: 小张 (In Japanese: 小張(シャオジャン) — pass kana)\n"
    "  - In Japanese (text and TTS) the name is always 小張(シャオジャン).\n"
    "- Hobbie: Guitar, Gym\n"
    "- AI Research Engineer\n"
    "\n"
    "# Files\n"
    "- Blog drafts: `blog/`.\n"
    "\n"
    "# Time & Location\n"
    "- travels often.\n"
)
MEMORY_PROFILE = (
    '---\ndescription: "用户核心档案"\nvisibility: PINNED\n---\n\n'
    "## 学习\n"
    "- 日语水平与学习进度：已完成《新版标准日本语》初级第 21 课 #direct #session:bbccb454 #time:2026-11-12\n\n"
    "## 健康与饮食\n"
    "- 乳糖不耐受，完全不吃辣 #direct #session:ff5f9f8f #time:2026-11-09\n\n"
    "## 历史\n"
    "- 旧事实 #time:2025-01-01\n"
)


def _workspace(tmp_path):
    from PIL import Image

    awd = tmp_path / "band"
    (awd / "instructions").mkdir(parents=True)
    (awd / "instructions" / "User.md").write_text(USER_MD, encoding="utf-8")
    mem = awd / "roles" / "tutor" / "memories" / "user"
    mem.mkdir(parents=True)
    (mem / "profile.md").write_text(MEMORY_PROFILE, encoding="utf-8")
    (awd / "roles" / "plain" / "memories").mkdir(parents=True)
    cdir = awd / "characters" / "xiaozhang"
    (cdir / "images").mkdir(parents=True)
    Image.new("RGB", (64, 80), (200, 10, 10)).save(cdir / "images" / "xiaozhang.jpeg", "JPEG")
    (cdir / "characters.json").write_text(
        json.dumps([{"name": "Xiao Zhang", "draftable": False, "aliases": ["小张", "小張"],
                     "image": "characters/xiaozhang/images/xiaozhang.jpeg"}], ensure_ascii=False),
        encoding="utf-8",
    )
    return awd


def test_import_from_workspace_reads_user_md_memories_and_avatar(tmp_path):
    awd = _workspace(tmp_path)
    store = up.UserProfileStore(str(awd))
    assert store.needs_import()
    got = store.import_from_workspace(str(awd), str(awd / "roles"))
    assert got["nickname"] == "小张"
    lines = got["bio"].splitlines()
    assert lines[:4] == [
        "- Male", "- Discord handle: @bottlemerchant", "- Hobbie: Guitar, Gym", "- AI Research Engineer"
    ]
    assert "- 日语水平与学习进度：已完成《新版标准日本语》初级第 21 课" in lines
    assert "- 乳糖不耐受，完全不吃辣" in lines
    assert not any("#session" in ln or "旧事实" in ln or "Nickname" in ln for ln in lines)
    assert got["avatar_imported"] is True and got["character_dir"] == "xiaozhang"
    assert set(got["sources"]) == {"instructions/User.md", "roles/tutor/memories/user/profile.md"}

    saved = store.apply_import(got)
    assert saved["nickname"] == "小张" and saved["character_dir"] == "xiaozhang" and saved["avatar_rev"] == 1
    assert store.public_view()["avatar_ref"] == "user/avatar.webp"
    assert not store.needs_import()


def test_sync_writes_user_md_in_place_without_duplicating_memory(tmp_path, db):
    awd = _workspace(tmp_path)
    store = up.UserProfileStore(str(awd))
    bio = store.import_from_workspace(str(awd), str(awd / "roles"))["bio"] + "\n- 喜欢拉面"
    up.sync_profile_to_roles(db, str(awd / "roles"), ["tutor", "plain"], "小张", bio)

    md = (awd / "instructions" / "User.md").read_text(encoding="utf-8")
    head = md.split("# Files", 1)[0]
    assert md.endswith(USER_MD[USER_MD.index("\n# Files") :])  # other sections verbatim
    assert "- Nickname: 小张 (In Japanese: 小張(シャオジャン) — pass kana)\n  - In Japanese (text and TTS)" in head
    assert head.index("- Discord handle") < head.index("- Nickname") < head.index("- Hobbie")
    assert "- 喜欢拉面" in head and head.count("- Male") == 1
    # Memory-owned facts stay only in the role memory (the memory writer keeps updating them).
    assert "乳糖不耐受" not in md and "日语水平" not in md
    tutor = (awd / "roles" / "tutor" / "memories" / "user" / "profile.md").read_text(encoding="utf-8")
    assert up.BLOCK_START not in tutor and tutor.count("乳糖不耐受") == 1
    assert not (awd / "roles" / "plain" / "memories" / "user" / "profile.md").exists()

    # Renaming keeps the parenthetical note; a second identical sync is a no-op.
    up.sync_profile_to_roles(db, str(awd / "roles"), ["tutor", "plain"], "阿张", bio)
    md2 = (awd / "instructions" / "User.md").read_text(encoding="utf-8")
    assert "- Nickname: 阿张 (In Japanese: 小張(シャオジャン) — pass kana)" in md2
    assert up.sync_profile_to_roles(db, str(awd / "roles"), ["tutor", "plain"], "阿张", bio) == []
    assert (awd / "instructions" / "User.md").read_text(encoding="utf-8") == md2


def test_sync_without_user_md_dedupes_block_against_existing_bullets(tmp_path, db):
    roles = tmp_path / "roles"
    mem = roles / "tutor" / "memories" / "user"
    mem.mkdir(parents=True)
    (mem / "profile.md").write_text(MEMORY_PROFILE, encoding="utf-8")
    bio = "- 乳糖不耐受，完全不吃辣\n- 喜欢拉面"
    changed = up.sync_profile_to_roles(db, str(roles), ["tutor"], "Sam", bio, instructions_user_md="")
    assert changed == ["tutor"]
    text = (mem / "profile.md").read_text(encoding="utf-8")
    assert text.count("乳糖不耐受") == 1
    assert "- 简介：喜欢拉面" in text and "称呼：Sam" in text


def test_shared_user_profile_md_ai_memory_writer_and_cross_role_visibility(tmp_path, db, monkeypatch):
    """Verify `$AWD/user/profile.md` is shared across roles for `MemoryWriter` and `load_pinned_memories`."""
    from kesoku.agent.memory_store import default_pinned_cache, resolve_shared_profile_path
    from kesoku.agent.memory_writer import MemoryWriter

    monkeypatch.setattr("kesoku.utils.embedding.get_embeddings", lambda texts: [[0.4] * 384 for _ in texts])

    awd = tmp_path / "workspace"
    tutor_mem = awd / "roles" / "tutor" / "memories"
    coder_mem = awd / "roles" / "coder" / "memories"
    tutor_mem.mkdir(parents=True)
    coder_mem.mkdir(parents=True)

    store = up.UserProfileStore(str(awd))
    body = store.write_profile_md("## 基本信息\n- 称呼：小张 #direct\n")
    assert "称呼：小张" in body
    assert store.read_profile_md() == "## 基本信息\n- 称呼：小张 #direct"
    assert store.public_view()["profile_md"] == "## 基本信息\n- 称呼：小张 #direct"

    shared_expected = str(awd / "user" / "profile.md")
    assert resolve_shared_profile_path(str(tutor_mem)) == shared_expected
    assert resolve_shared_profile_path(str(coder_mem)) == shared_expected
    assert resolve_shared_profile_path(str(tmp_path / "isolated_memories")) is None

    # Both roles see the shared PINNED profile even though `roles/*/memories/user/profile.md` does not exist
    assert not (tutor_mem / "user" / "profile.md").exists()
    assert not (coder_mem / "user" / "profile.md").exists()
    assert load_pinned_memories("tutor", str(tutor_mem)) == [("user/profile.md", "## 基本信息\n- 称呼：小张 #direct")]
    assert load_pinned_memories("coder", str(coder_mem)) == [("user/profile.md", "## 基本信息\n- 称呼：小张 #direct")]

    block_coder_before = default_pinned_cache.get_block("coder", memories_dir=str(coder_mem))
    assert "称呼：小张" in block_coder_before

    # AI MemoryWriter writes to `user/profile.md` from role `tutor` -> updates shared `$AWD/user/profile.md`
    snaps = MemoryWriter.snapshot_files(str(tutor_mem), ["user/profile.md"])
    assert snaps["user/profile.md"] != ""
    res = MemoryWriter.apply_ops(
        [
            {
                "op": "upsert_bullet",
                "file": "user/profile.md",
                "section": "健康与饮食",
                "text": "乳糖不耐受，避免鲜奶与奶酪",
                "tags": ["#direct"],
                "pinned": True,
            }
        ],
        role="tutor",
        memories_dir=str(tutor_mem),
        db=db,
        session_id="abcd1234",
        event_date="2026-11-16",
        snapshots=snaps,
    )
    assert res.ops_applied == 1
    assert res.touched_files == ["user/profile.md"]
    assert not (tutor_mem / "user" / "profile.md").exists()
    assert (awd / "user" / ".memory.lock").exists()

    # Role `coder` immediately sees the AI update in its `<memory_pinned>` cache and `UserProfileStore`
    block_coder_after = default_pinned_cache.get_block("coder", memories_dir=str(coder_mem))
    assert "乳糖不耐受，避免鲜奶与奶酪" in block_coder_after
    assert "乳糖不耐受，避免鲜奶与奶酪" in (store.read_profile_md() or "")
