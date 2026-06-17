"""Shell execution tools and background process monitoring for Kesoku AI Agent."""

import asyncio
import logging
import os
import re
import shlex
import signal
import time
from typing import Any

from kesoku.agent.tools.registry import ToolContext, default_registry
from kesoku.config import get_config
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.utils.async_fs import (
    async_exists,
    async_read_text_file,
    async_write_text_file,
)
from kesoku.utils.path import PathResolver

logger = logging.getLogger(__name__)

MAX_TOOL_OUTPUT_LENGTH = 30000


class ActiveJobsRegistry:
    """Registry of active background subprocesses to prevent zombie processes."""

    def __init__(self) -> None:
        """Initialize the registry with an empty dict and lock."""
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        job_id: str,
        session_id: str,
        process: asyncio.subprocess.Process,
        log_filepath: str,
    ) -> None:
        """Register a new active background process.

        Args:
            job_id: The background job identifier.
            session_id: The session ID the job belongs to.
            process: The asyncio subprocess instance.
            log_filepath: Path to the log file.
        """
        async with self._lock:
            self._jobs[job_id] = {
                "session_id": session_id,
                "process": process,
                "log_filepath": log_filepath,
                "creation_time": time.time(),
            }
            logger.info(f"Registered background job '{job_id}' for session '{session_id}'")

    async def unregister(self, job_id: str) -> None:
        """Unregister a background job.

        Args:
            job_id: The job identifier to remove.
        """
        async with self._lock:
            self._jobs.pop(job_id, None)
            logger.info(f"Unregistered background job '{job_id}'")

    async def stop_all_for_session(self, session_id: str) -> None:
        """Terminate and clean up all active background jobs for a session.

        Args:
            session_id: The session ID to clean up.
        """
        async with self._lock:
            target_job_ids = [jid for jid, job in self._jobs.items() if job["session_id"] == session_id]

            for jid in target_job_ids:
                job = self._jobs[jid]
                proc = job["process"]
                logger.info(f"Force terminating background job '{jid}' for session '{session_id}'")
                try:
                    # Terminate the process group to kill child processes as well
                    if os.name == "posix":
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        except Exception:
                            proc.terminate()
                    else:
                        proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except Exception as e:
                    logger.debug(f"Failed to gracefully terminate process group {proc.pid}: {e}")
                    try:
                        if os.name == "posix":
                            try:
                                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                            except Exception:
                                proc.kill()
                        else:
                            proc.kill()
                        await proc.wait()
                    except Exception as ex:
                        logger.debug(f"Failed to SIGKILL process group {proc.pid}: {ex}")
                self._jobs.pop(jid, None)


async def stream_to_file(
    stream: asyncio.StreamReader | None,
    file_path: str,
    mode: str = "a",
) -> None:
    """Asynchronously streams a StreamReader stream into a target log file.

    Args:
        stream: The asyncio StreamReader instance.
        file_path: Absolute path to the target file.
        mode: File open mode ('w' or 'a').
    """
    if stream is None:
        return
    try:
        if mode == "w":
            await async_write_text_file(file_path, "")

        def _append_line(text: str) -> None:
            with open(file_path, "a", encoding="utf-8", errors="replace") as f:
                f.write(text)
                f.flush()

        while True:
            line = await stream.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace")
            await asyncio.to_thread(_append_line, decoded)
    except Exception as e:
        logger.error(f"Failed to stream output to log file {file_path}: {e}")


async def monitor_background_job(
    job_id: str,
    proc: asyncio.subprocess.Process,
    log_filepath_stdout: str,
    log_filepath_stderr: str,
    context: ToolContext,
    original_msg_id: str,
    stdout_task: asyncio.Task[None],
    stderr_task: asyncio.Task[None],
) -> None:
    """Monitors a background shell subprocess, posts results to gateway, and posts a system alert to wake up LLM."""
    from kesoku.gateway.gateway import Gateway

    try:
        # Wait for process and stream tasks to finish
        return_code = await proc.wait()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

        # Read the final output logs
        stdout = ""
        stderr = ""
        if await async_exists(log_filepath_stdout):
            try:
                stdout = await async_read_text_file(log_filepath_stdout)
            except Exception as e:
                stdout = f"Failed to read stdout log: {e}"
        if await async_exists(log_filepath_stderr):
            try:
                stderr = await async_read_text_file(log_filepath_stderr)
            except Exception as e:
                stderr = f"Failed to read stderr log: {e}"

        output = f"=== STDOUT ===\n{stdout}\n=== STDERR ===\n{stderr}"

        # Cap output to avoid blowing up context window
        truncated_output = output
        if len(output) > MAX_TOOL_OUTPUT_LENGTH:
            truncated_output = (
                f"Output truncated (total length {len(output)} bytes).\n"
                f"Stdout saved to: `{log_filepath_stdout}`.\n"
                f"Stderr saved to: `{log_filepath_stderr}`.\n\n"
                f"Preview:\n{output[: MAX_TOOL_OUTPUT_LENGTH // 2]}...\n{output[-MAX_TOOL_OUTPUT_LENGTH // 2 :]}"
            )

        # Post special System wakeup alert to Gateway
        gw = context.gateway or Gateway()

        # Resolve the target chatbot_id and channel_id from the original message in database
        chatbot_id = "system"
        channel_id = "system"
        if original_msg_id:
            try:
                orig_msgs = await gw.db.get_messages_by_filters({"id": original_msg_id})
                if orig_msgs:
                    chatbot_id = orig_msgs[0].chatbot_id
                    channel_id = orig_msgs[0].channel_id
            except Exception as e:
                logger.warning(f"Failed to resolve original message {original_msg_id} platform details: {e}")

        status_str = "successfully" if return_code == 0 else f"with error code {return_code}"
        content = (
            f"[System Alert] Background Job `{job_id}` has finished executing {status_str}.\n"
            f"Stdout path: `{log_filepath_stdout}`\n"
            f"Stderr path: `{log_filepath_stderr}`\n\n"
            f"{truncated_output}"
        )

        wakeup_msg = Message(
            session_id=context.session_id,
            chatbot_id=chatbot_id,
            channel_id=channel_id,
            sender="System",
            role=MessageRole.SYSTEM,
            type=MessageType.TEXT,
            content=content,
            status=MessageStatus.PENDING_AGENT,  # This triggers LLM wakeup
            parent_id=original_msg_id,
        )
        await gw.post(wakeup_msg)
        logger.info(f"Background Job '{job_id}' finished. Posted wakeup message to session '{context.session_id}'")

    except Exception as ex:
        logger.error(f"Error in monitor_background_job for '{job_id}': {ex}", exc_info=True)
    finally:
        if context.active_jobs:
            await context.active_jobs.unregister(job_id)


class ShellCommandError(RuntimeError):
    """Raised when a shell command execution fails or times out."""

    pass


@default_registry.register
async def run_shell_command(
    command: str,
    cwd: str | None = None,
    background_threshold_seconds: float | None = None,
    max_output_chars: int = 1000,
    context: ToolContext | None = None,
) -> str:
    """Execute a CLI shell command within a target directory, defaulting to the AWD (Agent Working Directory).

    The command is executed within the specified `cwd` directory. If `cwd` is not provided or is empty,
    it defaults to the Agent Working Directory (AWD). If `cwd` is a relative path, it is resolved relative to the AWD.
    All temporary scripts, data files, or build artifacts should be created in the session staging directory
    (which can be found in the system prompt STAGING_DIR instructions).

    Tips:
    - **IMPORTANT**: Because you have an output token limit (4096) when calling the LLM,
      if you use this tool to write a file, make sure you split into multiple commands,
      and write at most 4000 characters per command.
    - NEVER run commands that are more than 5000 characters just to be safe.
    - If you have a command that is very long, only emit 1 tool call to avoid token limit exceed error.
    - Tool output is capped at `max_output_chars` (default 1000, max 30000) characters to save tokens.
      If you need more output, increase the `max_output_chars` parameter, or filter/redirect the command output.
    - You are encouraged to combine commands using '|', '&&', ';' etc to save turns.

    Args:
        command: The command string to execute (e.g., 'uv run pytest' or 'echo hello').
        cwd: Optional target working directory for executing the command.
            If relative, it's relative to AWD. Defaults to AWD.
        background_threshold_seconds: Optional foreground timeout limit (in seconds) override.
            If the command takes longer than this limit, it transitions to background execution.
            If not provided, defaults to configuration setting.
        max_output_chars: Maximum output characters to return. Defaults to 1000.
            Output exceeding this length is truncated and full output saved to a staging file.
        context: Optional tool execution context.

    Returns:
        Command execution stdout and stderr, or truncated preview if output exceeds max_output_chars.
    """
    config = get_config()
    if not config.shell.enabled:
        return "Execution denied: The shell command tool is disabled in configuration."

    # Security Check 1: Prohibited blocklist patterns
    for pattern in config.shell.blocklist_patterns:
        if re.search(pattern, command):
            return f"Execution denied: command matches prohibited blocklist pattern '{pattern}'."

    # Security Check 2: Permitted allowlist patterns
    if config.shell.mode == "allowlist":
        matched = False
        for pattern in config.shell.allowlist_patterns:
            if re.search(pattern, command):
                matched = True
                break
        if not matched:
            return "Execution denied: command does not match any permitted allowlist pattern."

    if not context:
        return (
            "Execution denied: ToolContext is missing. "
            "Shell commands must be executed within an active session context."
        )

    # Resolve target working directory
    exec_dir = PathResolver.resolve(cwd)
    os.makedirs(exec_dir, exist_ok=True)

    # Prepare environment variables
    env = os.environ.copy()
    if config.shell.env:
        env.update(config.shell.env)

    # Inject AWD and STAGING_DIR environment variables
    env["AWD"] = PathResolver.get_awd()
    if context and context.session_workspace:
        env["STAGING_DIR"] = PathResolver.get_session_staging_dir(context.session_workspace)

    # Prepare unique background job identifiers and log paths
    job_id = f"job_{int(time.time())}"
    log_filename_stdout = f"background_{job_id}.stdout"
    log_filename_stderr = f"background_{job_id}.stderr"
    folder_name = context.session_workspace
    session_staging_dir = PathResolver.get_session_staging_dir(folder_name)
    log_filepath_stdout = os.path.join(session_staging_dir, log_filename_stdout)
    log_filepath_stderr = os.path.join(session_staging_dir, log_filename_stderr)

    logger.info(f"Executing shell command in '{exec_dir}': {command}")

    # Configure process group on POSIX to allow clean SIGTERM/SIGKILL propagation
    preexec_fn = None
    if os.name == "posix":
        preexec_fn = os.setsid

    proc = None
    try:
        if config.shell.use_shell:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=exec_dir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=preexec_fn,
            )
        else:
            tokens = shlex.split(command)
            proc = await asyncio.create_subprocess_exec(
                *tokens,
                cwd=exec_dir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=preexec_fn,
            )

        # Start background streaming of stdout/stderr to their respective log files
        stdout_task = asyncio.create_task(stream_to_file(proc.stdout, log_filepath_stdout, "w"))
        stderr_task = asyncio.create_task(stream_to_file(proc.stderr, log_filepath_stderr, "w"))

        # Wait up to configured foreground threshold limit
        threshold = (
            background_threshold_seconds
            if background_threshold_seconds is not None
            else getattr(config.shell, "background_threshold_seconds", 300.0)
        )
        try:
            await asyncio.wait_for(proc.wait(), timeout=threshold)
            # Process finished successfully within threshold. Ensure all pipe reads complete.
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        except TimeoutError:
            # Command exceeded foreground threshold. Transition to background job.
            if context.active_jobs:
                await context.active_jobs.register(job_id, context.session_id, proc, log_filepath_stdout)

            # Spin off background monitor task
            asyncio.create_task(
                monitor_background_job(
                    job_id=job_id,
                    proc=proc,
                    log_filepath_stdout=log_filepath_stdout,
                    log_filepath_stderr=log_filepath_stderr,
                    context=context,
                    original_msg_id=context.original_msg_id or "",
                    stdout_task=stdout_task,
                    stderr_task=stderr_task,
                )
            )

            return (
                f"The command took longer than {threshold} seconds and has been transitioned to background execution.\n"
                f"Background Job ID: `{job_id}`\n"
                f"Stdout path: `{log_filepath_stdout}`\n"
                f"Stderr path: `{log_filepath_stderr}`\n\n"
                "Please inform the user that the task is running in the background. "
                "You will be automatically notified and your turn resumed when it finishes. "
                "You must stop your current turn now."
            )

    except asyncio.CancelledError:
        if proc:
            try:
                if os.name == "posix":
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except Exception:
                        proc.terminate()
                else:
                    proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except Exception:
                try:
                    if os.name == "posix":
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except Exception:
                            proc.kill()
                    else:
                        proc.kill()
                    await proc.wait()
                except Exception:
                    pass
        logger.info(f"Shell command execution cancelled: {command}")
        raise
    except Exception as ex:
        if not isinstance(ex, asyncio.CancelledError):
            logger.error(f"Failed to execute command '{command}': {ex}")
            raise ShellCommandError(f"Error executing command: {ex}") from ex
        raise

    # Read the written log file contents
    stdout = ""
    stderr = ""
    if await async_exists(log_filepath_stdout):
        try:
            stdout = await async_read_text_file(log_filepath_stdout)
        except Exception as e:
            stdout = f"Failed to read stdout logs: {e}"
    if await async_exists(log_filepath_stderr):
        try:
            stderr = await async_read_text_file(log_filepath_stderr)
        except Exception as e:
            stderr = f"Failed to read stderr logs: {e}"

    out_str = f"=== STDOUT ===\n{stdout}\n=== STDERR ===\n{stderr}"

    # Unified truncation logic
    effective_max = min(max_output_chars, MAX_TOOL_OUTPUT_LENGTH)
    final_output = out_str
    if len(out_str) > effective_max:
        timestamp = int(time.time())
        output_filename = f"cmd_output_{timestamp}.txt"
        output_filepath = os.path.join(session_staging_dir, output_filename)
        try:
            await async_write_text_file(output_filepath, out_str)
            final_output = (
                f"Output truncated (total length {len(out_str)} bytes). "
                f"Full output saved to session workspace file: `{output_filepath}`.\n"
                f"You can use tool `run_shell_command` (e.g., `cat {output_filename}`) "
                f"on this path to examine the full output.\n\n"
                f"Preview (first {effective_max} chars):\n{out_str[:effective_max]}\n\n"
                f"[Output truncated. If you need to view more output, you can set the 'max_output_chars' "
                f"parameter to a larger value (up to 30000), or filter the command output "
                f"(e.g., using grep/awk/head/tail/etc.).]"
            )
        except Exception as ex:
            logger.error(f"Failed to save truncated output to '{output_filepath}': {ex}")
            final_output = (
                f"Output truncated (total length {len(out_str)} bytes). "
                f"Preview:\n{out_str[:effective_max]}\n\n"
                f"[Output truncated. If you need to view more output, you can set the 'max_output_chars' "
                f"parameter to a larger value (up to 30000), or filter the command output "
                f"(e.g., using grep/awk/head/tail/etc.).]"
            )

    if proc.returncode != 0:
        raise ShellCommandError(f"Command failed with exit code {proc.returncode}.\n{final_output}")

    return final_output
