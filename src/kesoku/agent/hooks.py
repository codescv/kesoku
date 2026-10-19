import asyncio
import os
import re
import shlex
from typing import Any

from pydantic import BaseModel

from kesoku.config import HooksConfig
from kesoku.db import Message
from kesoku.logger import setup_logger
from kesoku.utils.text import truncate_middle

logger = setup_logger(__name__)

class HookOutcome(BaseModel):
    """Result of a hook execution."""
    name: str
    exit_code: int
    output: str

async def run_post_tool_use_hooks(

    call: Any,
    result_msg: Message,
    cfg: HooksConfig,
    agent_working_dir: str,
    staging_dir: str,
    env: dict[str, str]
) -> None:
    """Run post-tool use hooks."""
    # call: CallRecord
    tool_name = call.name
    # Extract file path if update_file
    file_path = ""
    if tool_name in ("update_file", "read_file") and "file_name" in call.arguments:
        file_path = call.arguments["file_name"]
        if not os.path.isabs(file_path):  # noqa: ASYNC240
            file_path = os.path.abspath(os.path.join(agent_working_dir, file_path))  # noqa: ASYNC240

    for hook in cfg.post_tool_use:
        if tool_name not in hook.tools:
            continue
        if hook.file_pattern and file_path:
            if not re.search(hook.file_pattern, file_path):
                continue

        # Prepare command
        cmd = hook.command.replace("{file}", shlex.quote(file_path)).replace("{awd}", shlex.quote(agent_working_dir)).replace("{staging_dir}", shlex.quote(staging_dir))  # noqa: E501

        # Prepare cwd
        cwd = agent_working_dir
        if hook.cwd == "" and file_path:
            cwd = os.path.dirname(file_path)  # noqa: ASYNC240
        elif hook.cwd:
            cwd = os.path.abspath(os.path.join(agent_working_dir, hook.cwd))  # noqa: ASYNC240

        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
                env=env,
                preexec_fn=os.setsid
            )

            try:
                stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=hook.timeout_seconds)
                output = stdout_bytes.decode(errors="replace")
                exit_code = proc.returncode
            except TimeoutError:
                try:
                    os.killpg(proc.pid, 15)
                except ProcessLookupError:
                    pass
                output = "Timeout"
                exit_code = -1

            if hook.append_on == "always" or (hook.append_on == "nonzero" and exit_code != 0):
                formatted = f"\n\n[hook {hook.name} exit {exit_code}]\n{truncate_middle(output, hook.max_output_chars)}"  # noqa: E501
                result_msg.content += formatted
                if result_msg.metadata and "tool_result" in result_msg.metadata:
                    result_msg.metadata["tool_result"] += formatted

        except Exception as e:
            err = f"\n\n[hook {hook.name} failed to run: {e}]"
            result_msg.content += err
            if result_msg.metadata and "tool_result" in result_msg.metadata:
                result_msg.metadata["tool_result"] += err

async def run_stop_hooks(

    turn_tools_used: set[str],
    cfg: HooksConfig,
    agent_working_dir: str,
    env: dict[str, str]
) -> list[HookOutcome]:
    """Run stop hooks."""
    outcomes = []
    for hook in cfg.stop:
        if not any(tool in turn_tools_used for tool in hook.when_tools_used):
            continue

        cmd = hook.command
        cwd = os.path.abspath(os.path.join(agent_working_dir, hook.cwd)) if hook.cwd else agent_working_dir  # noqa: ASYNC240

        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
                env=env,
                preexec_fn=os.setsid
            )

            try:
                stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=hook.timeout_seconds)
                output = stdout_bytes.decode(errors="replace")
                exit_code = proc.returncode
            except TimeoutError:
                try:
                    os.killpg(proc.pid, 15)
                except ProcessLookupError:
                    pass
                output = "Timeout"
                exit_code = -1

            outcomes.append(HookOutcome(name=hook.name, exit_code=exit_code, output=output))
        except Exception as e:
            outcomes.append(HookOutcome(name=hook.name, exit_code=-1, output=str(e)))

    return outcomes
