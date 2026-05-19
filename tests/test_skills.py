"""Unit tests for Kesoku Skill System and tool integration."""

import pathlib

import pytest

from kesoku.agent.skills import SkillManager, parse_skill_markdown
from kesoku.agent.tools import list_skills, use_skill


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
