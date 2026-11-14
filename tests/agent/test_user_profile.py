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
