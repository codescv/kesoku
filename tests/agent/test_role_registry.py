"""Tests for the role registry (`kesoku.agent.role_registry`)."""

import json
import os

from kesoku.agent.role_registry import RoleRegistry, is_valid_role_id, slugify_role_id


def _mk_role(roles_dir, role_id, intro="", role_json=None):
    d = roles_dir / role_id
    d.mkdir(parents=True)
    (d / "intro.md").write_text(intro, encoding="utf-8")
    if role_json is not None:
        (d / "role.json").write_text(json.dumps(role_json), encoding="utf-8")
    return d


def test_synthesizes_role_json_from_intro_heading(tmp_path):
    roles = tmp_path / "roles"
    d = _mk_role(roles, "helper_bot", intro="# Name\nHelper Bot 🤖\n\n# Persona\nKind.\n")
    reg = RoleRegistry(str(roles))

    summary = reg.get("helper_bot")

    assert summary["display_name"] == "Helper Bot 🤖"
    assert summary["builtin"] is True
    assert summary["avatar_rev"] == 0
    assert summary["avatar_ref"] is None
    persisted = json.loads((d / "role.json").read_text(encoding="utf-8"))
    assert persisted["role_id"] == "helper_bot"
    assert persisted["builtin"] is True


def test_synthesizes_from_plain_first_line_and_falls_back_to_dir_name(tmp_path):
    roles = tmp_path / "roles"
    _mk_role(roles, "alice", intro="Alice the archivist\nmore text\n")
    _mk_role(roles, "bob", intro="")
    reg = RoleRegistry(str(roles))

    assert reg.get("alice")["display_name"] == "Alice the archivist"
    assert reg.get("bob")["display_name"] == "bob"


def test_existing_role_json_is_respected(tmp_path):
    roles = tmp_path / "roles"
    _mk_role(
        roles,
        "carol",
        intro="# Name\nIgnored\n",
        role_json={"display_name": "Carol", "tagline": "hi", "builtin": False, "avatar_rev": 3, "voice_preset": "v1"},
    )
    (roles / "carol" / "avatar.webp").write_bytes(b"x")
    reg = RoleRegistry(str(roles))

    s = reg.get("carol")

    assert s["display_name"] == "Carol"
    assert s["builtin"] is False
    assert s["avatar_rev"] == 3
    assert s["voice_preset"] == "v1"
    assert s["avatar_ref"] == "roles/carol/avatar.webp"


def test_list_skips_trash_invalid_names_and_non_roles(tmp_path):
    roles = tmp_path / "roles"
    _mk_role(roles, "alice", intro="A")
    _mk_role(roles, "zed_custom", intro="Z", role_json={"display_name": "Zed", "builtin": False})
    (roles / ".trash" / "old-1").mkdir(parents=True)
    (roles / "Bad-Name").mkdir()
    (roles / "Bad-Name" / "intro.md").write_text("x")
    (roles / "empty_dir").mkdir()
    reg = RoleRegistry(str(roles))

    ids = [r["role_id"] for r in reg.list_roles()]

    assert ids == ["alice", "zed_custom"]  # builtin first, then custom


def test_role_dir_rejects_traversal_and_symlink_escape(tmp_path):
    roles = tmp_path / "roles"
    roles.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "intro.md").write_text("x")
    os.symlink(outside, roles / "evil")
    reg = RoleRegistry(str(roles))

    assert reg.role_dir("../outside") is None
    assert reg.role_dir("evil") is None
    assert reg.get("evil") is None
    assert "evil" not in reg.list_role_ids()


def test_update_and_bump_avatar_rev(tmp_path):
    roles = tmp_path / "roles"
    _mk_role(roles, "alice", intro="Alice")
    reg = RoleRegistry(str(roles))

    reg.update_meta("alice", tagline="new tagline")
    meta = reg.bump_avatar_rev("alice", avatar_ai=True)
    reg.bump_avatar_rev("alice", avatar_ai=True)

    assert meta["avatar_rev"] == 1
    assert reg.get("alice")["avatar_rev"] == 2
    assert reg.get("alice")["tagline"] == "new tagline"
    assert reg.reload()[0]["avatar_rev"] == 2


def test_slugify_and_unique_role_id(tmp_path):
    roles = tmp_path / "roles"
    _mk_role(roles, "night_owl", intro="x")
    reg = RoleRegistry(str(roles))

    assert slugify_role_id("Night Owl!") == "night_owl"
    assert slugify_role_id("小助手") == "role"
    assert slugify_role_id("Café Ünïcode") == "cafe_unicode"
    assert reg.unique_role_id("Night Owl") == "night_owl_2"
    long_id = reg.unique_role_id("x" * 80)
    assert is_valid_role_id(long_id) and len(long_id) == 32
    assert not is_valid_role_id("a")
    assert not is_valid_role_id("../etc")
