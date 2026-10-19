from kesoku.agent.repo_rules import RepoRulesManager


def test_repo_rules_discovery(tmp_path):
    awd = tmp_path / "awd"
    awd.mkdir()

    subdir = awd / "sub"
    subdir.mkdir()

    staging = tmp_path / "staging"
    staging.mkdir()

    rule1 = subdir / "AGENTS.md"
    rule1.write_text("Rule 1")

    manager = RepoRulesManager(
        staging_dir=str(staging),
        agent_working_dir=str(awd),
        max_files=3,
        max_chars=4000,
        rule_filenames=["AGENTS.md"]
    )

    # Discover from a file inside subdir
    target_file = subdir / "some_code.py"
    added = manager.discover([str(target_file)])
    assert len(added) == 1
    assert added[0] == str(rule1)

    # Render
    content = manager.render()
    assert "sub/AGENTS.md" in content
    assert "Rule 1" in content

    # Discover again (already discovered)
    added2 = manager.discover([str(target_file)])
    assert len(added2) == 0

def test_repo_rules_max_files(tmp_path):
    awd = tmp_path / "awd"
    awd.mkdir()
    staging = tmp_path / "staging"
    staging.mkdir()

    for i in range(5):
        d = awd / f"dir{i}"
        d.mkdir()
        (d / "AGENTS.md").write_text(f"Rule {i}")

    manager = RepoRulesManager(
        staging_dir=str(staging),
        agent_working_dir=str(awd),
        max_files=2,
        rule_filenames=["AGENTS.md"]
    )

    added = manager.discover([str(awd / f"dir{i}" / "file.py") for i in range(5)])
    assert len(added) == 2
    assert len(manager.discovered_files) == 2

