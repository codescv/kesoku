"""Tests for fixture materialisation, config profiles, isolation of held-out material, and test restoration."""

import os
import re
import subprocess
import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest

from benchmarks.kesoku_bench.harness import fixture as fx
from benchmarks.kesoku_bench.harness.taskspec import TASKS_DIR, load_task
from kesoku.agent.skills import parse_skill_markdown
from kesoku.config import KesokuConfig


@pytest.mark.parametrize("profile", ["default", "compaction_forced", "compaction_small"])
def test_profiles_validate(profile: str) -> None:
    cfg = KesokuConfig.model_validate(fx.render_config_dict(profile))
    a = cfg.agent
    assert a.compression_strategy == "watermark"
    assert a.protect_tail_min_tokens < a.context_low_watermark < a.context_high_watermark
    assert a.protect_front_turns == 1
    assert "chat_search" in a.disabled_tools and "web_search" in a.disabled_tools
    assert a.raw_llm_logs is True and a.auto_context_search is False
    assert cfg.gemini.thinking_level == "medium"
    assert cfg.gemini.context_caching_mode == "implicit"
    assert any("jetski" in p for p in cfg.shell.forbidden_patterns)


def test_profile_specifics() -> None:
    default = KesokuConfig.model_validate(fx.render_config_dict("default")).agent
    forced = KesokuConfig.model_validate(fx.render_config_dict("compaction_forced")).agent
    small = KesokuConfig.model_validate(fx.render_config_dict("compaction_small")).agent
    assert default.protect_tail_turns == 4 and default.context_high_watermark == 64000
    assert forced.protect_tail_turns == 2 and forced.context_high_watermark == 64000
    assert small.context_high_watermark == 8000 and small.watermark_chunk_tokens == 2500


def test_overrides_and_model_options() -> None:
    data = fx.render_config_dict("compaction_forced", {"shell.background_threshold_seconds": 8.0},
                                 thinking_level="low", model_name="gemini-x")
    cfg = KesokuConfig.model_validate(data)
    assert cfg.shell.background_threshold_seconds == 8.0
    assert cfg.gemini.thinking_level == "low"
    assert cfg.gemini.model_name == "gemini-x" and cfg.gemini.context_model_name == "gemini-x"
    assert "profiles" not in data


def test_unknown_profile() -> None:
    with pytest.raises(ValueError, match="unknown config profile"):
        fx.render_config_dict("nope")


def test_forbidden_patterns_block_expected_commands() -> None:
    patterns = fx.render_config_dict("default")["shell"]["forbidden_patterns"]
    blocked = ["/google/bin/releases/jetski-devs/tools/cli run", "pytest -q", "cd x && python a.py",
               "ls ../../holdout", "cat /x/kesoku_bench/tasks", "find / -name gold"]
    allowed = ["uv run pytest -q", "uv run python -m pkg", "git diff", "kill 1234", "ls tests"]
    for cmd in blocked:
        assert any(re.search(p, cmd) for p in patterns), cmd
    for cmd in allowed:
        assert not any(re.search(p, cmd) for p in patterns), cmd


def test_substitute() -> None:
    assert fx.substitute("port {PORT} at {AWD}/repo", 1234, "/w") == "port 1234 at /w/repo"


def test_sanitized_env_drops_venv() -> None:
    env = fx.sanitized_env({"VIRTUAL_ENV": "/v", "PATH": "/v/bin:/usr/bin", "PYTHONPATH": "x", "HOME": "/h"})
    assert "VIRTUAL_ENV" not in env and "PYTHONPATH" not in env
    assert env["PATH"] == "/usr/bin" and env["HOME"] == "/h"


def test_free_port() -> None:
    assert 0 < fx.free_port() < 65536


def test_materialize_layout_and_isolation(make_task: Callable[..., Path], tmp_path: Path) -> None:
    spec = load_task(make_task())
    work_root = tmp_path / "work"
    holdout_root = tmp_path / "holdout"
    f = fx.materialize(spec, work_root, "run1", holdout_root, prewarm=False, port=4321)
    assert f.ws_dir == work_root.resolve() / "run1" / spec.id / "ws"
    assert (f.ws_dir / "config.toml").is_file()
    assert (f.ws_dir / "roles" / "eval_engineer" / "intro.md").is_file()
    assert (f.ws_dir / "roles" / "eval_engineer" / "preferences.md").is_file()
    assert (f.repo_dir / "calc.py").is_file()
    assert (f.ws_dir / "sessions").is_dir() and (f.ws_dir / "skills").is_dir()
    assert f.port == 4321
    # held-out tests are not anywhere under the AWD
    assert not list(f.ws_dir.rglob("test_holdout_*.py"))
    assert not (f.repo_dir / "tests" / "test_holdout_calc.py").exists()
    assert work_root.resolve() not in f.holdout_dir.parents
    assert (f.pristine_tests_dir / "test_basic.py").is_file()
    # no path or symlink under ws/ resolves into the benchmark tasks directory or the task source
    for p in f.ws_dir.rglob("*"):
        real = p.resolve()
        assert TASKS_DIR not in real.parents and spec.task_dir not in real.parents, p
        assert not p.is_symlink(), p
    # git repo with one commit
    log = subprocess.run(["git", "log", "--oneline"], cwd=f.repo_dir, capture_output=True, text=True, check=True)
    assert "fixture" in log.stdout
    # config validates and the AWD is the ws dir
    with open(f.config_path, "rb") as fh:
        cfg = KesokuConfig.model_validate(tomllib.load(fh))
    assert cfg.workspace.db_path == "kesoku.db"


def test_materialize_rejects_holdout_inside_work_root(make_task: Callable[..., Path], tmp_path: Path) -> None:
    spec = load_task(make_task())
    with pytest.raises(ValueError, match="holdout_root"):
        fx.materialize(spec, tmp_path / "work", "r", tmp_path / "work" / "h", prewarm=False)


def test_materialize_profile_override_and_task_overrides(make_task: Callable[..., Path], tmp_path: Path) -> None:
    task_dir = make_task(profile="compaction_forced",
                         extra='config_overrides = { "shell.background_threshold_seconds" = 8.0 }')
    spec = load_task(task_dir)
    f = fx.materialize(spec, tmp_path / "w", "r", tmp_path / "h", profile="compaction_small", prewarm=False)
    with open(f.config_path, "rb") as fh:
        cfg = KesokuConfig.model_validate(tomllib.load(fh))
    assert cfg.agent.context_high_watermark == 8000
    assert cfg.shell.background_threshold_seconds == 8.0
    assert f.profile == "compaction_small"


def test_suite_d_skills_copied(make_task: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_dir = make_task(task_id="D9_synthetic", suite="D")
    (task_dir / "skills" / "target-skill").mkdir(parents=True)
    (task_dir / "skills" / "target-skill" / "SKILL.md").write_text("---\nname: target-skill\n---\nbody\n")
    distractors = tmp_path / "distractors"
    for name in ("d-one", "d-two"):
        (distractors / name).mkdir(parents=True)
        (distractors / name / "SKILL.md").write_text(f"---\nname: {name}\n---\nbody\n")
    monkeypatch.setattr(fx, "DISTRACTOR_SKILLS_DIR", distractors)
    spec = load_task(task_dir)
    installed = fx.copy_skills(spec, tmp_path / "skills_out", distractors)
    assert installed == ["target-skill", "d-one", "d-two"]
    assert (tmp_path / "skills_out" / "d-two" / "SKILL.md").is_file()


def test_non_d_suites_get_no_skills(make_task: Callable[..., Path], tmp_path: Path) -> None:
    spec = load_task(make_task())
    assert fx.copy_skills(spec, tmp_path / "s") == []


def test_real_distractor_skills_parse() -> None:
    """The shared distractor skills are valid SKILL.md files with frontmatter (11 expected)."""
    dirs = sorted(p for p in fx.DISTRACTOR_SKILLS_DIR.iterdir() if p.is_dir())
    assert len(dirs) == 11
    for d in dirs:
        manifest, body = parse_skill_markdown(str(d / "SKILL.md"))
        assert manifest.name == d.name
        assert 200 <= len(body) <= 1200
        assert [p for p in (d / "scripts").iterdir() if p.is_file()]


def test_restore_visible_tests_by_path(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    pristine = tmp_path / "pristine"
    (repo / "tests").mkdir(parents=True)
    pristine.mkdir()
    (pristine / "test_a.py").write_text("orig")
    (pristine / "sub").mkdir()
    (pristine / "sub" / "test_b.py").write_text("orig-b")
    (repo / "tests" / "test_a.py").write_text("tampered")
    (repo / "tests" / "test_new.py").write_text("agent authored")
    restored = fx.restore_visible_tests(repo, pristine)
    assert sorted(restored) == ["tests/sub/test_b.py", "tests/test_a.py"]
    assert (repo / "tests" / "test_a.py").read_text() == "orig"
    assert (repo / "tests" / "test_new.py").read_text() == "agent authored"


def test_materialize_grading_tests_and_overlay_gold(make_task: Callable[..., Path], tmp_path: Path) -> None:
    task_dir = make_task()
    (task_dir / "holdout" / "ref").mkdir()
    (task_dir / "holdout" / "ref" / "ref_impl.py").write_text("X = 1\n")
    (task_dir / "gold" / "apply.py").write_text(
        "import pathlib, sys\npathlib.Path(sys.argv[1], 'APPLIED').write_text('ok')\n")
    spec = load_task(task_dir)
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    copied = fx.materialize_grading_tests(spec, repo, staging_dir=tmp_path / "stage")
    assert sorted(copied) == ["tests/ref_impl.py", "tests/test_holdout_calc.py"]
    assert (tmp_path / "stage" / "tests" / "test_holdout_calc.py").is_file()
    fx.overlay_gold(spec, repo)
    assert "a + b" in (repo / "calc.py").read_text()
    assert not (repo / "apply.py").exists()
    assert (repo / "APPLIED").read_text() == "ok"


def test_uv_sync_prewarm(make_task: Callable[..., Path], tmp_path: Path) -> None:
    spec = load_task(make_task())
    f = fx.materialize(spec, tmp_path / "w", "r", tmp_path / "h", prewarm=True)
    assert f.extras.get("uv_sync_rc") == 0, f.extras
    assert (f.repo_dir / ".venv").is_dir()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=f.repo_dir, capture_output=True, text=True)
    assert status.stdout.strip() == "", status.stdout


def test_default_holdout_root_outside_tmp_work(tmp_path: Path) -> None:
    root = fx.default_holdout_root()
    try:
        assert root.name.startswith("kb-holdout-")
        assert Path("/tmp/kesoku-bench") not in root.parents
    finally:
        os.rmdir(root)
