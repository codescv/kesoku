"""Unit tests for Kesoku Skill System and tool integration."""

import inspect
import pathlib
from typing import Any
from unittest.mock import MagicMock

import pytest

from kesoku.agent.skills import SkillManager, parse_skill_markdown
from kesoku.agent.tools import list_skills, use_skill
from kesoku.config import KesokuConfig, WorkspaceConfig
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager
from kesoku.gateway.gateway import Gateway


def test_parse_skill_markdown(tmp_path: pathlib.Path) -> None:
    """Test successfully parsing YAML frontmatter and body from a skill markdown file."""
    skill_file = tmp_path / "SKILL.md"
    content = (
        "---\n"
        "name: test-skill\n"
        "description: A test skill for unit tests\n"
        "version: 2.1.0\n"
        "required_permissions: [run_shell_command]\n"
        "metadata:\n"
        "  tags: [test, demo]\n"
        "  platforms: [linux, darwin]\n"
        "---\n"
        "\n"
        "# Instructions\n"
        "Run `echo hello`."
    )
    skill_file.write_text(content, encoding="utf-8")

    manifest, body = parse_skill_markdown(str(skill_file))
    assert manifest.name == "test-skill"
    assert manifest.description == "A test skill for unit tests"
    assert manifest.version == "2.1.0"
    assert manifest.required_permissions == ["run_shell_command"]
    assert manifest.metadata.tags == ["test", "demo"]
    assert manifest.metadata.platforms == ["linux", "darwin"]
    assert body == "# Instructions\nRun `echo hello`."


def test_parse_skill_markdown_malformed(tmp_path: pathlib.Path) -> None:
    """Test parse_skill_markdown raises ValueError on malformed frontmatter."""
    skill_file = tmp_path / "SKILL.md"
    # Missing closing ---
    content = "---\nname: malformed\n# Instructions"
    skill_file.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="No YAML frontmatter found"):
        parse_skill_markdown(str(skill_file))


def test_skill_manager_platform_filtering(tmp_path: pathlib.Path) -> None:
    """Test SkillManager platform filtering behavior in list_skills and get_skill."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    # Skill 1: Cross-platform (omitted platforms)
    s1 = skills_dir / "skill-all"
    s1.mkdir()
    (s1 / "SKILL.md").write_text("---\nname: skill-all\ndescription: All OS\n---\n# Body", encoding="utf-8")

    # Skill 2: Explicitly empty platforms (excluded everywhere)
    s2 = skills_dir / "skill-none"
    s2.mkdir()
    (s2 / "SKILL.md").write_text(
        "---\nname: skill-none\ndescription: No OS\nmetadata:\n  platforms: []\n---\n# Body",
        encoding="utf-8",
    )

    # Skill 3: Linux only
    s3 = skills_dir / "skill-linux"
    s3.mkdir()
    (s3 / "SKILL.md").write_text(
        "---\nname: skill-linux\ndescription: Linux OS\nmetadata:\n  platforms: [linux]\n---\n# Body",
        encoding="utf-8",
    )

    # Skill 4: Windows only
    s4 = skills_dir / "skill-windows"
    s4.mkdir()
    (s4 / "SKILL.md").write_text(
        "---\nname: skill-windows\ndescription: Windows OS\nmetadata:\n  platforms: [windows]\n---\n# Body",
        encoding="utf-8",
    )

    manager = SkillManager(skills_dir=str(skills_dir))

    # Mocking host_platform as linux
    manager._get_current_platform = lambda: "linux"  # type: ignore
    linux_skills = manager.list_skills()
    names = [s["name"] for s in linux_skills]
    assert "skill-all" in names
    assert "skill-linux" in names
    assert "skill-none" not in names
    assert "skill-windows" not in names

    # Verify get_skill raises ValueError when trying to access unsupported platform skill
    with pytest.raises(ValueError, match="not supported on current OS"):
        manager.get_skill("skill-windows")


def test_skill_manager_path_traversal(tmp_path: pathlib.Path) -> None:
    """Test SkillManager strictly rejects directory traversal attempts."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    manager = SkillManager(skills_dir=str(skills_dir))

    with pytest.raises(KeyError, match="Contains prohibited characters"):
        manager.get_skill("../passwd")


def test_skill_tool_wrappers(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test list_skills and use_skill tool functions."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    s1 = skills_dir / "demo-skill"
    s1.mkdir()
    (s1 / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: A demo tool skill\n---\n# Step 1\nTest CLI.",
        encoding="utf-8",
    )

    # Monkeypatch global skill_manager in tools module
    test_manager = SkillManager(skills_dir=str(skills_dir))
    monkeypatch.setattr("kesoku.agent.tools.skill_manager", test_manager)

    res_list = list_skills()
    assert "demo-skill" in res_list
    assert "A demo tool skill" in res_list

    res_use = use_skill("demo-skill")
    assert "# Skill: demo-skill" in res_use
    assert "SKILL_DIR=" in res_use
    assert str(s1) in res_use
    assert "# Step 1\nTest CLI." in res_use


# ---------------------------------------------------------------------------
# Role-scoped skills: <roles_dir>/<role>/skills/<name>/SKILL.md
# ---------------------------------------------------------------------------


def _make_skill(base: pathlib.Path, name: str, description: str, body: str = "# Body") -> pathlib.Path:
    """Create <base>/<name>/SKILL.md and return the skill directory."""
    d = base / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}", encoding="utf-8")
    return d


@pytest.fixture
def role_skill_tree(tmp_path: pathlib.Path) -> tuple[SkillManager, pathlib.Path]:
    """Global skills (shared-skill, clash) + role_a skills (private-a, clash) + role_b skills (private-b)."""
    skills_dir = tmp_path / "skills"
    roles_dir = tmp_path / "roles"
    _make_skill(skills_dir, "shared-skill", "Global shared skill")
    _make_skill(skills_dir, "clash", "GLOBAL clash version", "# Global clash body")
    _make_skill(roles_dir / "role_a" / "skills", "private-a", "Role A private skill", "# Private A body")
    _make_skill(roles_dir / "role_a" / "skills", "clash", "ROLE_A clash version", "# Role A clash body")
    _make_skill(roles_dir / "role_b" / "skills", "private-b", "Role B private skill")
    (roles_dir / "role_c").mkdir(parents=True)  # role without a skills dir
    return SkillManager(skills_dir=str(skills_dir), roles_dir=str(roles_dir)), tmp_path


def _names(skills: list[dict]) -> list[str]:
    """Skill names from list_skills output."""
    return [s["name"] for s in skills]


def test_role_skills_listed_only_for_that_role(role_skill_tree: tuple[SkillManager, pathlib.Path]) -> None:
    """A role sees global + its own skills; other roles and role=None never see them."""
    manager, _ = role_skill_tree
    a = _names(manager.list_skills(role="role_a"))
    assert "private-a" in a and "shared-skill" in a and "private-b" not in a
    b = _names(manager.list_skills(role="role_b"))
    assert "private-b" in b and "private-a" not in b
    none = _names(manager.list_skills())
    assert "private-a" not in none and "private-b" not in none and "shared-skill" in none


def test_role_skill_wins_on_name_collision(role_skill_tree: tuple[SkillManager, pathlib.Path]) -> None:
    """On a name clash the role-scoped skill shadows the global one (listed once)."""
    manager, tmp = role_skill_tree
    listed = [s for s in manager.list_skills(role="role_a") if s["name"] == "clash"]
    assert len(listed) == 1 and listed[0]["description"] == "ROLE_A clash version"
    _, content = manager.get_skill("clash", role="role_a")
    assert "# Role A clash body" in content
    assert str(tmp / "roles" / "role_a" / "skills" / "clash") in content
    _, global_content = manager.get_skill("clash")
    assert "# Global clash body" in global_content
    _, b_content = manager.get_skill("clash", role="role_b")
    assert "# Global clash body" in b_content


def test_get_skill_dir_role_and_global(role_skill_tree: tuple[SkillManager, pathlib.Path]) -> None:
    """get_skill_dir resolves the role path first and falls back to the global dir."""
    manager, tmp = role_skill_tree
    role_dir = (tmp / "roles" / "role_a" / "skills" / "private-a").resolve()
    assert manager.get_skill_dir("private-a", role="role_a") == str(role_dir)
    assert manager.get_skill_dir("shared-skill", role="role_a") == str((tmp / "skills" / "shared-skill").resolve())


def test_role_private_skill_not_loadable_by_other_role(role_skill_tree: tuple[SkillManager, pathlib.Path]) -> None:
    """get_skill for another role's private skill fails like a missing skill."""
    manager, _ = role_skill_tree
    with pytest.raises(KeyError):
        manager.get_skill("private-a", role="role_b")
    with pytest.raises(KeyError):
        manager.get_skill("private-a")


def test_role_without_skills_dir_or_unknown_role(role_skill_tree: tuple[SkillManager, pathlib.Path]) -> None:
    """A role without skills/ (or an unknown role) just gets the global skills."""
    manager, _ = role_skill_tree
    global_names = _names(manager.list_skills())
    assert _names(manager.list_skills(role="role_c")) == global_names
    assert _names(manager.list_skills(role="no_such_role")) == global_names
    _, content = manager.get_skill("shared-skill", role="no_such_role")
    assert "Body" in content


@pytest.mark.parametrize("bad_role", ["../evil", "..", "role_a/../../evil", "/abs/path", "role a"])
def test_role_name_traversal_never_escapes(role_skill_tree: tuple[SkillManager, pathlib.Path], bad_role: str) -> None:
    """Malicious role names never expose skills outside <roles_dir>/<role>/skills."""
    manager, tmp = role_skill_tree
    _make_skill(tmp / "evil" / "skills", "evil-skill", "Escaped skill")
    try:
        names = _names(manager.list_skills(role=bad_role))
    except (KeyError, ValueError):
        names = []
    assert "evil-skill" not in names
    with pytest.raises((KeyError, ValueError)):
        manager.get_skill("evil-skill", role=bad_role)


def test_role_skill_symlink_escape_rejected(role_skill_tree: tuple[SkillManager, pathlib.Path]) -> None:
    """A symlinked role skill pointing outside the role skills dir fails the canonical boundary check."""
    manager, tmp = role_skill_tree
    outside = _make_skill(tmp / "outside", "linked", "Outside skill")
    (tmp / "roles" / "role_a" / "skills" / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(KeyError):
        manager.get_skill("linked", role="role_a")


async def _role_context(tmp_path: pathlib.Path, role_name: str) -> MagicMock:
    """ToolContext stand-in backed by a real gateway/DB session whose role_name is role_name."""
    db_path = str(tmp_path / f"ctx_{role_name}.db")
    DatabaseManager(db_path).init_tables()
    gw = Gateway(context=KesokuContext(config=KesokuConfig(workspace=WorkspaceConfig(db_path=db_path))))
    await gw.create_session(f"sess_{role_name}", title="t", role=role_name)
    ctx = MagicMock()
    ctx.gateway = gw
    ctx.session_id = f"sess_{role_name}"
    ctx.original_msg_id = None
    return ctx


async def _call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call a tool wrapper that may be sync or async."""
    res = fn(*args, **kwargs)
    if inspect.isawaitable(res):
        res = await res
    return res


async def test_skill_tool_wrappers_resolve_role_from_context(
    role_skill_tree: tuple[SkillManager, pathlib.Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """list_skills/use_skill tools expose the active session role's private skills only."""
    manager, tmp = role_skill_tree
    monkeypatch.setattr("kesoku.agent.tools.skill_manager", manager)
    monkeypatch.setattr("kesoku.agent.tools.memory.skill_manager", manager, raising=False)
    ctx_a = await _role_context(tmp, "role_a")
    ctx_b = await _role_context(tmp, "role_b")
    listed_a = await _call(list_skills, context=ctx_a)
    assert "private-a" in listed_a and "shared-skill" in listed_a and "private-b" not in listed_a
    used = await _call(use_skill, "private-a", context=ctx_a)
    assert "# Private A body" in used
    denied = await _call(use_skill, "private-a", context=ctx_b)
    assert "Failed to load skill" in denied
    listed_none = await _call(list_skills)
    assert "private-a" not in listed_none
