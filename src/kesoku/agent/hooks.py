"""Post-tool-use and stop hook execution for Kesoku AI Agent."""

import asyncio
import os
import re
import shlex
import signal
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


async def _terminate_hook_proc(proc: asyncio.subprocess.Process) -> None:
    """Terminate a timed-out hook process group cleanly."""
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
        await asyncio.wait_for(proc.wait(), timeout=2.0)
    except Exception:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            await proc.wait()
        except Exception:
            pass


def _build_hook_env(agent_working_dir: str, staging_dir: str, extra_env: dict[str, str] | None) -> dict[str, str]:
    """Build subprocess environment with AWD and STAGING_DIR set."""
    run_env = os.environ.copy()
    if extra_env:
        for k, v in extra_env.items():
            run_env[str(k)] = str(v)
    if agent_working_dir:
        run_env["AWD"] = agent_working_dir
    if staging_dir:
        run_env["STAGING_DIR"] = staging_dir
    return run_env


async def run_post_tool_use_hooks(
    call: Any,
    result_msg: Message,
    cfg: HooksConfig,
    agent_working_dir: str,
    staging_dir: str,
    env: dict[str, str],
) -> None:
    """Run post-tool-use hooks matching the tool call and append output when configured."""
    if not cfg.post_tool_use:
        return
    tool_name = call.name
    file_path = ""
    if tool_name in ("update_file", "read_file") and "file_name" in call.arguments:
        file_path = str(call.arguments["file_name"])
        if not os.path.isabs(file_path):  # noqa: ASYNC240
            file_path = os.path.abspath(os.path.join(agent_working_dir, file_path))  # noqa: ASYNC240

    run_env = _build_hook_env(agent_working_dir, staging_dir, env)

    for hook in cfg.post_tool_use:
        if tool_name not in hook.tools:
            continue
        if hook.file_pattern:
            if not file_path or not re.search(hook.file_pattern, file_path):
                continue

        cmd = (
            hook.command.replace("{file}", shlex.quote(file_path))
            .replace("{awd}", shlex.quote(agent_working_dir or ""))
            .replace("{staging_dir}", shlex.quote(staging_dir or ""))
        )

        cwd = agent_working_dir
        if hook.cwd == "" and file_path:
            candidate_cwd = os.path.dirname(file_path)  # noqa: ASYNC240
            if os.path.isdir(candidate_cwd):  # noqa: ASYNC240
                cwd = candidate_cwd
        elif hook.cwd:
            cwd = os.path.abspath(os.path.join(agent_working_dir, hook.cwd))  # noqa: ASYNC240

        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
                env=run_env,
                preexec_fn=os.setsid if os.name == "posix" else None,
            )

            try:
                stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=hook.timeout_seconds)
                output = stdout_bytes.decode(errors="replace")
                exit_code = proc.returncode if proc.returncode is not None else 0
            except TimeoutError:
                await _terminate_hook_proc(proc)
                output = f"Timed out after {hook.timeout_seconds}s"
                exit_code = -1

            if hook.append_on == "always" or (hook.append_on == "nonzero" and exit_code != 0):
                trimmed = truncate_middle(output, hook.max_output_chars)
                formatted = f"\n\n[hook {hook.name} exit {exit_code}]\n{trimmed}"
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
    env: dict[str, str],
    staging_dir: str = "",
) -> list[HookOutcome]:
    """Run configured stop hooks for the current turn and return their outcomes."""
    outcomes: list[HookOutcome] = []
    if not cfg.stop:
        return outcomes

    run_env = _build_hook_env(agent_working_dir, staging_dir, env)

    for hook in cfg.stop:
        if not any(tool in turn_tools_used for tool in hook.when_tools_used):
            continue

        cmd = hook.command
        cwd = (
            os.path.abspath(os.path.join(agent_working_dir, hook.cwd))  # noqa: ASYNC240
            if hook.cwd
            else agent_working_dir
        )

        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
                env=run_env,
                preexec_fn=os.setsid if os.name == "posix" else None,
            )

            try:
                stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=hook.timeout_seconds)
                output = stdout_bytes.decode(errors="replace")
                exit_code = proc.returncode if proc.returncode is not None else 0
            except TimeoutError:
                await _terminate_hook_proc(proc)
                output = f"Timed out after {hook.timeout_seconds}s"
                exit_code = -1

            outcomes.append(HookOutcome(name=hook.name, exit_code=exit_code, output=output))
        except Exception as e:
            outcomes.append(HookOutcome(name=hook.name, exit_code=-1, output=str(e)))

    return outcomes
