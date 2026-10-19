from unittest.mock import MagicMock

import pytest

from kesoku.agent.hooks import run_post_tool_use_hooks, run_stop_hooks
from kesoku.config import HooksConfig, PostToolUseHook, StopHook
from kesoku.db import Message


@pytest.mark.asyncio
async def test_post_tool_hooks(tmp_path):
    call = MagicMock()
    call.name = "update_file"
    call.arguments = {"file_name": "test.txt"}

    msg = Message(
        session_id="s", chatbot_id="c", channel_id="c", sender="test",
        role="assistant", type="text", content="Result OK"
    )

    cfg = HooksConfig(
        post_tool_use=[
            PostToolUseHook(name="h1", command="echo hook output", tools=["update_file"], cwd="", append_on="always"),
            PostToolUseHook(name="h2", command="echo skip", tools=["read_file"], cwd=""),
        ],
        stop=[]
    )

    awd = str(tmp_path)
    await run_post_tool_use_hooks(call, msg, cfg, awd, "/staging", {})
    assert "[hook h1 exit 0]" in msg.content
    assert "hook output" in msg.content
    assert "skip" not in msg.content

@pytest.mark.asyncio
async def test_stop_hooks(tmp_path):
    cfg = HooksConfig(
        stop=[
            StopHook(name="s1", command="echo stop hook 1", when_tools_used=["run_shell_command"], cwd=""),
            StopHook(name="s2", command="echo stop hook 2", when_tools_used=["update_file"], cwd="")
        ]
    )

    awd = str(tmp_path)
    outcomes = await run_stop_hooks({"run_shell_command"}, cfg, awd, {})

    assert len(outcomes) == 1
    assert outcomes[0].name == "s1"
    assert outcomes[0].exit_code == 0
    assert "stop hook 1" in outcomes[0].output


@pytest.mark.asyncio
async def test_post_tool_hook_file_pattern_and_append_on_nonzero(tmp_path):
    """PostToolUseHook with file_pattern only runs on matching files and respects append_on='nonzero'."""
    call_py = MagicMock()
    call_py.name = "update_file"
    call_py.arguments = {"file_name": "pkg/module.py"}

    call_md = MagicMock()
    call_md.name = "update_file"
    call_md.arguments = {"file_name": "README.md"}

    call_shell = MagicMock()
    call_shell.name = "run_shell_command"
    call_shell.arguments = {"command": "ls"}

    cfg = HooksConfig(
        post_tool_use=[
            PostToolUseHook(
                name="py_lint",
                command="echo 'lint error' && exit 1",
                tools=["update_file", "run_shell_command"],
                file_pattern=r"\.py$",
                append_on="nonzero",
            ),
        ],
        stop=[],
    )

    awd = str(tmp_path)
    msg_py = Message(
        session_id="s", chatbot_id="c", channel_id="c", sender="test",
        role="tool", type="tool_result", content="Updated", metadata={"tool_result": "Updated"},
    )
    await run_post_tool_use_hooks(call_py, msg_py, cfg, awd, "/staging", {})
    assert "[hook py_lint exit 1]" in msg_py.content
    assert "[hook py_lint exit 1]" in msg_py.metadata["tool_result"]

    # Non-matching extension is skipped
    msg_md = Message(
        session_id="s", chatbot_id="c", channel_id="c", sender="test",
        role="tool", type="tool_result", content="Updated", metadata={"tool_result": "Updated"},
    )
    await run_post_tool_use_hooks(call_md, msg_md, cfg, awd, "/staging", {})
    assert "[hook py_lint" not in msg_md.content

    # Tool without file_name does not match file_pattern="*.py"
    msg_sh = Message(
        session_id="s", chatbot_id="c", channel_id="c", sender="test",
        role="tool", type="tool_result", content="Listed", metadata={"tool_result": "Listed"},
    )
    await run_post_tool_use_hooks(call_shell, msg_sh, cfg, awd, "/staging", {})
    assert "[hook py_lint" not in msg_sh.content

