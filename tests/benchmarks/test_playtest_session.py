"""Unit tests for `kesoku.testing` and `PlaytestSession`."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import kesoku.testing as kt
from kesoku.agent.llm import BaseLLM, LLMResponse, MockLLM, ToolCallRequest
from kesoku.config import KesokuConfig
from kesoku.testing import (
    BenchSession,
    PlaytestChatbot,
    PlaytestSession,
    TurnOutcome,
    extract_tool_events,
    render_config_toml,
    sanitized_env,
)


def _write_fake_gm_script(ws: Path) -> Path:
    """Create a deterministic mock `skills/gm/gm.py` inside `ws` for testing `run_setup_gm` and tool calls."""
    gm_dir = ws / "skills" / "gm"
    gm_dir.mkdir(parents=True, exist_ok=True)
    (gm_dir / "SKILL.md").write_text("---\nname: gm\ndescription: mock gm\n---\nMock GM skill.\n", encoding="utf-8")
    gm_py = gm_dir / "gm.py"
    gm_py.write_text(
        """import json
import os
import pathlib
import sys

argv = sys.argv[1:]
staging = pathlib.Path(os.environ["STAGING_DIR"])
game_dir = staging / "game"
game_dir.mkdir(parents=True, exist_ok=True)
ctx_dir = staging / "context"
ctx_dir.mkdir(parents=True, exist_ok=True)

if argv == ["fail"]:
    print(json.dumps({"ok": False, "error": "bad subcommand"}))
    sys.exit(2)

scene_path = game_dir / "scene.json"
scene = json.loads(scene_path.read_text(encoding="utf-8")) if scene_path.is_file() else {"image_pending": False}
if argv[:2] == ["game", "start"]:
    scene["mode"] = argv[2] if len(argv) > 2 else "trpg"
    scene["location"] = "tavern"
    (game_dir / "party.json").write_text(json.dumps({"members": ["hero"]}), encoding="utf-8")
    (game_dir / "ledger.json").write_text(json.dumps({"gold": 100}), encoding="utf-8")
elif argv[:2] == ["scene", "flag-image"]:
    scene["image_pending"] = True
elif argv[:2] == ["scene", "clear-image"]:
    scene["image_pending"] = False
scene_path.write_text(json.dumps(scene), encoding="utf-8")
(ctx_dir / "20-scene.md").write_text(f"location: {scene.get('location', 'none')}\\n", encoding="utf-8")

print(json.dumps({
    "ok": True,
    "argv": argv,
    "user_turn": os.environ.get("USER_TURN"),
    "awd": os.environ.get("AWD"),
    "staging_dir": os.environ.get("STAGING_DIR"),
    "dry_run": os.environ.get("AI_IMAGE_DRY_RUN"),
    "deferred": os.environ.get("KESOKU_DEFERRED_MEDIA"),
}))
""",
        encoding="utf-8",
    )
    return gm_py


def _write_role(ws: Path, role: str = "default") -> None:
    """Create minimal role prompt files under `<ws>/roles/<role>/`."""
    role_dir = ws / "roles" / role
    role_dir.mkdir(parents=True, exist_ok=True)
    (role_dir / "intro.md").write_text("You are a helpful test assistant.\n", encoding="utf-8")
    (role_dir / "preferences.md").write_text("Keep answers concise.\n", encoding="utf-8")


def _factory(llm: BaseLLM) -> Any:
    return lambda _cfg: llm


def test_public_exports_and_standalone_src_import(tmp_path: Path) -> None:
    """`kesoku.testing` exports all public symbols and imports cleanly with ONLY `src/` on `PYTHONPATH`."""
    assert BenchSession is kt.BenchSession
    assert PlaytestSession is kt.PlaytestSession
    assert TurnOutcome is kt.TurnOutcome
    assert extract_tool_events is kt.extract_tool_events
    assert render_config_toml is kt.render_config_toml
    assert sanitized_env is kt.sanitized_env

    src_dir = Path(kt.__file__).resolve().parents[2]
    env = sanitized_env()
    env["PYTHONPATH"] = str(src_dir)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; assert 'benchmarks' not in sys.modules; "
            "from kesoku.testing import BenchSession, PlaytestSession, TurnOutcome, "
            "extract_tool_events, render_config_toml, sanitized_env; "
            "assert render_config_toml('default')",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.asyncio
async def test_playtest_session_env_isolation_and_setup_gm(tmp_path: Path) -> None:
    """`PlaytestSession` defaults to `role='default'`, isolates `env` in `cfg.shell.env`, and runs `setup_gm`."""
    ws = tmp_path / "ws"
    _write_role(ws, "default")
    _write_fake_gm_script(ws)

    sentinel_key = "PLAYTEST_UNIT_ISOLATED_MARKER"
    os.environ.pop(sentinel_key, None)

    llm = MockLLM(
        responses=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        name="run_shell_command",
                        arguments={"command": "uv run skills/gm/gm.py scene flag-image"},
                    )
                ],
            ),
            LLMResponse(content="Scene updated.", tool_calls=[]),
        ]
    )

    session = PlaytestSession(
        workspace=ws,
        model="gemini-3.8-flash",
        thinking_level="low",
        env={
            sentinel_key: "isolated-value",
            "AI_IMAGE_DRY_RUN": "1",
            "TTS_DRY_RUN": "1",
            "KESOKU_DEFERRED_MEDIA": "1",
        },
        llm_factory=_factory(llm),
        idle_stable_s=0.05,
        poll_interval_s=0.02,
    )
    assert session.role == "default"

    async with session:
        # Global os.environ is untouched across parallel workers.
        assert sentinel_key not in os.environ
        assert session.cfg is not None
        assert session.cfg.agent.memory_extraction_enabled is False
        assert session.cfg.agent.auto_context_search is False
        assert session.cfg.gemini.model_name == "gemini-3.8-flash"
        assert session.cfg.gemini.thinking_level == "low"
        assert session.cfg.shell.env[sentinel_key] == "isolated-value"

        # Direct setup_gm runs with USER_TURN=0, STAGING_DIR, and AWD without invoking the LLM.
        setup_out = session.run_setup_gm([["game", "start", "trpg"]])
        assert len(setup_out) == 1
        assert setup_out[0]["ok"] is True
        assert setup_out[0]["user_turn"] == "0"
        assert Path(setup_out[0]["awd"]).resolve() == ws.resolve()
        assert Path(setup_out[0]["staging_dir"]).resolve() == session.staging_dir
        assert setup_out[0]["dry_run"] == "1"
        assert session.ledger.totals()["llm_calls"] == 0

        # Also verify awaiting `run_setup_gm` and `snapshot` works identically.
        awaited_out = await session.run_setup_gm([["scene", "clear-image"]])
        assert awaited_out[0]["ok"] is True

        pre_snap = await session.snapshot()
        assert pre_snap["turn"] == 0
        assert pre_snap["scene_json"]["mode"] == "trpg"
        assert pre_snap["party_json"] == {"members": ["hero"]}
        assert pre_snap["ledger_json"] == {"gold": 100}
        assert pre_snap["image_pending"] is False
        assert pre_snap["system_prompt_tokens"] > 0

        # Non-zero setup_gm command raises RuntimeError.
        with pytest.raises(RuntimeError, match="exit code 2"):
            session.run_setup_gm([["fail"]])

        # Turn 1 runs through Agent + TurnExecutor + shell.py with USER_TURN=1 and KESOKU_DEFERRED_MEDIA=1.
        outcome = await session.send_user_message("Trigger an image flag")
        assert outcome.status == "processed"
        assert outcome.final_text == "Scene updated."

        snap = session.snapshot()
        assert snap["turn"] == 1
        assert snap["final_text"] == "Scene updated."
        assert snap["image_pending"] is True
        assert len(snap["tool_events"]) == 1
        ev = snap["tool_events"][0]
        assert ev["name"] == "run_shell_command"
        assert ev["is_error"] is False
        assert ev["exit_code"] == 0
        assert '"user_turn": "1"' in ev["result"]
        assert '"deferred": "1"' in ev["result"]
        assert '"dry_run": "1"' in ev["result"]

    assert sentinel_key not in os.environ


@pytest.mark.asyncio
async def test_playtest_session_snapshot_turns_media_log_and_deferred_pickup(tmp_path: Path) -> None:
    """`snapshot(turn=...)` filters tool_events & `.playtest_media.jsonl` per turn and picks up deferred media."""
    ws = tmp_path / "ws"
    _write_role(ws, "custom_role")
    _write_fake_gm_script(ws)

    # Enable streaming in config.toml (mirroring production config) to verify MeteredLLM stream forwarding.
    cfg_toml = render_config_toml("default", {"gemini.stream": True})
    (ws / "config.toml").write_text(cfg_toml, encoding="utf-8")

    # Script that simulates a two-phase deferred image producer (`KESOKU_DEFERRED_MEDIA=1`).
    producer = ws / "skills" / "fake_media.py"
    producer.write_text(
        """import json, os, pathlib, subprocess, sys, time
staging = pathlib.Path(os.environ["STAGING_DIR"])
turn = int(os.environ.get("USER_TURN", "0"))
out = staging / f"shot_{turn}.png"
pending = pathlib.Path(f"{out}.pending.json")
if "--worker" in sys.argv:
    time.sleep(0.08)
    part = pathlib.Path(f"{out}.part")
    part.write_bytes(b"\\x89PNG\\r\\n\\x1a\\n")
    os.replace(part, out)
    with open(staging / ".playtest_media.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "image", "turn": turn, "output": str(out)}) + "\\n")
    pending.unlink(missing_ok=True)
    sys.exit(0)

pending.write_text(json.dumps({
    "v": 1, "kind": "deferred_media", "target": str(out),
    "job_id": f"j_{turn}", "created_at": time.time(), "timeout_sec": 30, "pid": 0,
}), encoding="utf-8")
proc = subprocess.Popen([sys.executable, __file__, "--worker"], env=os.environ.copy())
pending.write_text(json.dumps({
    "v": 1, "kind": "deferred_media", "target": str(out),
    "job_id": f"j_{turn}", "created_at": time.time(), "timeout_sec": 30, "pid": proc.pid,
}), encoding="utf-8")
print("Deferred started")
""",
        encoding="utf-8",
    )

    llm = MockLLM(
        responses=[
            # Turn 1: one failing gm command, then self-correction + deferred image.
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(name="run_shell_command", arguments={"command": "uv run skills/gm/gm.py fail"})
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        name="run_shell_command",
                        arguments={"command": "uv run skills/fake_media.py"},
                    )
                ],
            ),
            LLMResponse(content="Turn 1 reply [file: shot_1.png]", tool_calls=[]),
            # Turn 2: second deferred image.
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        name="run_shell_command",
                        arguments={"command": "uv run skills/fake_media.py"},
                    )
                ],
            ),
            LLMResponse(content="Turn 2 reply [file: shot_2.png]", tool_calls=[]),
        ]
    )

    async with PlaytestSession(
        workspace=ws,
        role="custom_role",
        model="claude-sonnet-5-5",
        thinking_level="minimal",
        env={"KESOKU_DEFERRED_MEDIA": "1"},
        setup_gm=[["game", "start", "trpg"]],
        llm_factory=_factory(llm),
        idle_stable_s=0.05,
        poll_interval_s=0.02,
    ) as session:
        assert session.cfg is not None
        assert session.cfg.agent.llm == "claude"
        assert session.cfg.claude.model_name == "claude-sonnet-5-5"
        assert session.cfg.claude.thinking_level == "low"
        assert session.setup_gm_outputs[0]["ok"] is True

        out1 = await session.send_user_message("first turn")
        assert out1.status == "processed"
        snap1 = session.snapshot(turn=1)
        assert snap1["turn"] == 1
        assert snap1["final_text"] == "Turn 1 reply [file: shot_1.png]"
        assert [e["exit_code"] for e in snap1["tool_events"]] == [2, 0]
        assert [e["is_error"] for e in snap1["tool_events"]] == [True, False]
        assert "shot_1.png" in snap1["staging_files"]
        assert not any(f.endswith(".pending.json") for f in snap1["staging_files"])
        assert len(snap1["media_log"]) == 1 and snap1["media_log"][0]["turn"] == 1

        out2 = await session.send_user_message("second turn")
        assert out2.status == "processed"
        snap2 = session.snapshot(turn=2)
        assert snap2["turn"] == 2
        assert snap2["final_text"] == "Turn 2 reply [file: shot_2.png]"
        assert len(snap2["tool_events"]) == 1 and snap2["tool_events"][0]["exit_code"] == 0
        assert len(snap2["media_log"]) == 1 and snap2["media_log"][0]["turn"] == 2

        full_snap = session.snapshot()
        assert full_snap["turn"] == 2
        assert len(full_snap["tool_events"]) == 3
        assert [m["turn"] for m in full_snap["media_log"]] == [1, 2]
        assert isinstance(session.bot, PlaytestChatbot)
        assert len(session.bot.delivered_files) == 2


def test_kesoku_config_validates_playtest_template() -> None:
    """Default `kesoku.testing` template validates cleanly into `KesokuConfig`."""
    data = kt.render_config_dict("default", model_name="gemini-3.8-flash", thinking_level="low")
    cfg = KesokuConfig.model_validate(data)
    assert cfg.gemini.model_name == "gemini-3.8-flash"
    assert cfg.gemini.thinking_level == "low"
