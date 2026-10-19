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
