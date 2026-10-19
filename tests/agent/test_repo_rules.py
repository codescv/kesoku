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


def test_stop_reminder_write_detection(tmp_path):
    """Verify K2 write detection: notes.md in STAGING_DIR and read-only shell commands do not trigger writes."""
    from kesoku.agent.turn_executor import TurnExecutor

    assert not TurnExecutor._is_shell_write_command("pytest -q 2>&1 | tail -20")
    assert not TurnExecutor._is_shell_write_command("git diff HEAD 2>/dev/null")
    assert not TurnExecutor._is_shell_write_command("git status && ruff check src/")
    assert not TurnExecutor._is_shell_write_command("sed -n '1,20p' foo.py")
    assert TurnExecutor._is_shell_write_command("echo 'x = 1' > repo/x.py")
    assert TurnExecutor._is_shell_write_command("sed -i 's/a/b/' repo/x.py")
    assert TurnExecutor._is_shell_write_command("git apply fix.patch")
    assert TurnExecutor._is_shell_write_command("cat patch | tee repo/x.py")

    awd = tmp_path / "awd"
    awd.mkdir()
    staging = tmp_path / "staging"
    staging.mkdir()

    resolved_notes = TurnExecutor._resolve_touched_path("$STAGING_DIR/notes.md", str(awd), str(staging))
    assert resolved_notes == str(staging / "notes.md")
    resolved_repo = TurnExecutor._resolve_touched_path("repo/x.py", str(awd), str(staging))
    assert resolved_repo == str(awd / "repo" / "x.py")


def test_staging_artifacts_excludes_repo_rules_and_subagent(tmp_path):
    """Verify STAGING_DIR snapshot excludes repo_rules.json and subagent_* logs."""
    from kesoku.agent.state_snapshot import render_staging_artifacts

    (tmp_path / "repo_rules.json").write_text('["/awd/AGENTS.md"]', encoding="utf-8")
    (tmp_path / "subagent_1_abc.log").write_text("subagent log", encoding="utf-8")
    (tmp_path / "keep_artifact.txt").write_text("important artifact", encoding="utf-8")

    rendered = render_staging_artifacts(str(tmp_path))
    assert "keep_artifact.txt" in rendered
    assert "repo_rules.json" not in rendered
    assert "subagent_1_abc.log" not in rendered


