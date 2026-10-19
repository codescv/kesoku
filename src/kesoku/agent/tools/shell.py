"""Shell execution tools and background process monitoring for Kesoku AI Agent."""

import asyncio
import logging
import os
import re
import shlex
import signal
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any

from kesoku.agent.state_snapshot import (
    JOB_COMMAND_MAX_CHARS,
    FinishedJob,
    JobInfo,
    JobsSnapshot,
    sorted_running,
)
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
from kesoku.utils.text import truncate_middle

logger = logging.getLogger(__name__)

MAX_TOOL_OUTPUT_LENGTH = 30000


def _process_group_id(pid: int) -> int:
    """Return the process group of `pid`, falling back to `pid` itself (non-POSIX, or the process is gone)."""
    if os.name != "posix":
        return pid
    try:
        return os.getpgid(pid)
    except (OSError, AttributeError):
        return pid






@dataclass
class KillResult:
    """Result of killing a background job."""
    success: bool
    message: str
    exit_code: int | None = None

class ActiveJobsRegistry:
    """Registry of active background subprocesses to prevent zombie processes.

    Besides the running jobs, the registry keeps a small per-session history of finished jobs (at most
    `FINISHED_PER_SESSION` per session, for at most `MAX_FINISHED_SESSIONS` sessions, least recently updated
    evicted first) so the per-turn `running_jobs` snapshot can report recent outcomes. That history is in-memory
    only and does not survive a service restart; a session's history is dropped by `stop_all_for_session`.
    """

    FINISHED_PER_SESSION = 5
    MAX_FINISHED_SESSIONS = 200

    def __init__(self) -> None:
        """Initialize the registry with an empty dict and lock."""
        self._jobs: dict[str, dict[str, Any]] = {}
        # session_id -> recently finished jobs; ordered by last update for LRU eviction.
        self._finished: OrderedDict[str, deque[FinishedJob]] = OrderedDict()
        self._killed_by_agent: set[str] = set()
        self._lock = asyncio.Lock()

    async def register(
        self,
        job_id: str,
        session_id: str,
        process: asyncio.subprocess.Process,
        log_filepath: str,
        *,
        command: str = "",
        stderr_filepath: str = "",
    ) -> None:
        """Register a new active background process.

        Args:
            job_id: The background job identifier.
            session_id: The session ID the job belongs to.
            process: The asyncio subprocess instance.
            log_filepath: Path to the stdout log file.
            command: The shell command (stored truncated to `JOB_COMMAND_MAX_CHARS`).
            stderr_filepath: Path to the stderr log file.
        """
        pid = process.pid
        # Jobs are started under setsid, so the group id is captured once here (the leader may exit later).
        pgid = _process_group_id(pid)
        async with self._lock:
            self._jobs[job_id] = {
                "session_id": session_id,
                "process": process,
                "log_filepath": log_filepath,
                "stderr_filepath": stderr_filepath,
                "command": command[:JOB_COMMAND_MAX_CHARS],
                "pid": pid,
                "pgid": pgid,
                "creation_time": time.time(),
            }
            logger.info(f"Registered background job '{job_id}' for session '{session_id}' (pgid {pgid})")

    def _record_finished_locked(self, session_id: str, finished: FinishedJob) -> None:
        """Append a finished job to the session's bounded history (caller holds the lock)."""
        history = self._finished.get(session_id)
        if history is None:
            history = deque(maxlen=self.FINISHED_PER_SESSION)
            self._finished[session_id] = history
        history.append(finished)
        self._finished.move_to_end(session_id)
        while len(self._finished) > self.MAX_FINISHED_SESSIONS:
            self._finished.popitem(last=False)


    async def kill_job(self, job_id: str, session_id: str) -> KillResult:
        """Terminate a background job's process group (SIGTERM -> 3s -> SIGKILL) without holding the registry lock."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                # Check if it's already in finished list
                history = self._finished.get(session_id, [])
                for fj in history:
                    if fj.job_id == job_id:
                        return KillResult(
                            True,
                            f"Job {job_id} already finished with exit {fj.return_code}.",
                            fj.return_code,
                        )
                return KillResult(False, f"Job id '{job_id}' is unknown or expired.")

            if job["session_id"] != session_id:
                return KillResult(False, f"Job id '{job_id}' is unknown or expired.")

            # (1) under lock: verify ownership, set killed_by_agent, move to finished
            job["killed_by_agent"] = True
            self._killed_by_agent.add(job_id)
            proc = job["process"]
            pgid = job["pgid"]
            stdout_path = job.get("log_filepath", "")
            stderr_path = job.get("stderr_filepath", "")

            # Pop from active, add to finished with pending return code
            del self._jobs[job_id]
            fj = FinishedJob(
                job_id=job_id,
                command=job["command"],
                return_code=None,
                finished_at=time.time(),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                reason="killed",
            )
            self._record_finished_locked(session_id, fj)

        # (2) release lock, kill
        try:
            if os.name == "posix":
                os.killpg(pgid, signal.SIGTERM)
            else:
                proc.terminate()
        except ProcessLookupError:
            pass

        rc = None
        action_taken = "SIGTERM"
        try:
            rc = await asyncio.wait_for(proc.wait(), timeout=3.0)
        except TimeoutError:
            try:
                if os.name == "posix":
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    proc.kill()
            except ProcessLookupError:
                pass
            action_taken = "SIGKILL"
            rc = await proc.wait()

        # (4) under lock: fill in return_code
        async with self._lock:
            history = self._finished.get(session_id, [])
            for i, f_job in enumerate(history):
                if f_job.job_id == job_id:
                    history[i] = FinishedJob(
                        job_id=f_job.job_id,
                        command=f_job.command,
                        return_code=rc,
                        finished_at=f_job.finished_at,
                        stdout_path=f_job.stdout_path,
                        stderr_path=f_job.stderr_path,
                        reason="killed",
                    )
                    break

        return KillResult(True, f"Killed job {job_id} (pgid {pgid}, {action_taken}, exit {rc}).", rc)

    def was_killed_by_agent(self, job_id: str, session_id: str | None = None) -> bool:
        """Check if the job was killed by the agent."""
        if job_id in self._killed_by_agent:
            return True
        if session_id is not None:
            history = self._finished.get(session_id, [])
            for fj in history:
                if fj.job_id == job_id and fj.reason == "killed":
                    return True
        else:
            for history in self._finished.values():
                for fj in history:
                    if fj.job_id == job_id and fj.reason == "killed":
                        return True
        return False

    async def unregister(self, job_id: str, return_code: int | None = None) -> None:
        """Unregister a background job and record it in the session's finished-job history.

        Unknown job ids (e.g. already removed by `stop_all_for_session`) are ignored.

        Args:
            job_id: The job identifier to remove.
            return_code: The process exit code, if known.
        """
        async with self._lock:
            job = self._jobs.pop(job_id, None)
            if job is None:
                return
            self._record_finished_locked(
                job["session_id"],
                FinishedJob(
                    job_id=job_id,
                    command=job.get("command", ""),
                    return_code=return_code,
                    finished_at=time.time(),
                    stdout_path=job.get("log_filepath", ""),
                    stderr_path=job.get("stderr_filepath", ""),
                ),
            )
            logger.info(f"Unregistered background job '{job_id}' (return code {return_code})")

    async def snapshot(self, session_id: str) -> JobsSnapshot:
        """Take a point-in-time snapshot of a session's running and recently finished jobs.

        Args:
            session_id: The session ID.

        Returns:
            Running jobs ordered by start time and finished jobs ordered oldest first.
        """
        async with self._lock:
            running = [
                JobInfo(
                    job_id=jid,
                    command=job.get("command", ""),
                    pid=job["pid"] if "pid" in job else job["process"].pid,
                    pgid=job["pgid"] if "pgid" in job else job["process"].pid,
                    started_at=job["creation_time"],
                    stdout_path=job.get("log_filepath", ""),
                    stderr_path=job.get("stderr_filepath", ""),
                )
                for jid, job in self._jobs.items()
                if job["session_id"] == session_id
            ]
            finished = list(self._finished.get(session_id, ()))
        return JobsSnapshot(running=sorted_running(running), finished=finished)

    async def stop_all_for_session(self, session_id: str) -> None:
        """Terminate and clean up all active background jobs for a session.

        The session's finished-job history is dropped as well (worker stop, `/clear`, session delete), so the
        stopped jobs are not reported by later snapshots.

        Args:
            session_id: The session ID to clean up.
        """
        async with self._lock:
            self._finished.pop(session_id, None)
            target_job_ids = [jid for jid, job in self._jobs.items() if job["session_id"] == session_id]

            for jid in target_job_ids:
                self._killed_by_agent.add(jid)
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

        if context.active_jobs and context.active_jobs.was_killed_by_agent(job_id, context.session_id):
            logger.info(f"Background Job '{job_id}' was killed by agent; suppressing wakeup alert.")
            return

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
            f"Stderr path: `{log_filepath_stderr}`"
        )
        if len(output) <= MAX_TOOL_OUTPUT_LENGTH:
            content += f"\n\n{output}"

        wakeup_msg = Message(
            session_id=context.session_id,
            chatbot_id=chatbot_id,
            channel_id=channel_id,
            sender="System",
            role=MessageRole.SYSTEM,
            type=MessageType.TEXT,
            content=content,
            status=MessageStatus.PENDING_AGENT,  # This triggers LLM wakeup
        )
        await gw.post(wakeup_msg)
        logger.info(f"Background Job '{job_id}' finished. Posted wakeup message to session '{context.session_id}'")

    except Exception as ex:
        logger.error(f"Error in monitor_background_job for '{job_id}': {ex}", exc_info=True)
    finally:
        if context.active_jobs:
            await context.active_jobs.unregister(job_id, return_code=proc.returncode)


class ShellCommandError(RuntimeError):
    """Raised when a shell command execution fails or times out."""

    pass


@default_registry.register
async def run_shell_command(
    command: str,
    cwd: str | None = None,
    background_threshold_seconds: float | None = None,
    max_output_chars: int = 5000,
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
    - Tool output is capped at `max_output_chars` (default 5000, max 30000) characters to save tokens.
      If you need more output, increase the `max_output_chars` parameter, or filter/redirect the command output.
    - You are encouraged to combine commands using '|', '&&', ';' etc to save turns.

    Args:
        command: The command string to execute (e.g., 'uv run pytest' or 'echo hello').
        cwd: Optional target working directory for executing the command.
            If relative, it's relative to AWD. Defaults to AWD.
        background_threshold_seconds: Optional foreground timeout limit (in seconds) override.
            If the command takes longer than this limit, it transitions to background execution.
            If not provided, defaults to configuration setting.
        max_output_chars: Maximum output characters to return. Defaults to 5000.
            Output exceeding this length is truncated in the middle (head and tail kept) and the full output is
            saved to a staging file.
        context: Optional tool execution context.

    Returns:
        Command execution stdout and stderr, or truncated preview if output exceeds max_output_chars.
    """
    config = getattr(getattr(context, "kesoku_context", None), "config", None) or get_config()
    if not config.shell.enabled:
        return "Execution denied: The shell command tool is disabled in configuration."

    # Security Check: Forbidden patterns with custom error messages
    for pattern, error_msg in config.shell.forbidden_patterns.items():
        if re.search(pattern, command):
            return error_msg

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
    job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"
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
        is_foreground_only = bool(getattr(context, "foreground_only", False))
        if is_foreground_only:
            subagent_timeout = float(getattr(config.agent, "subagent_shell_timeout_seconds", 120))
            threshold = (
                min(float(background_threshold_seconds), subagent_timeout)
                if background_threshold_seconds is not None
                else subagent_timeout
            )
        else:
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
            if is_foreground_only:
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
                await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
                return (
                    f"Command killed after {int(threshold)} s "
                    "(sub-agent commands cannot run in the background)"
                )

            # Command exceeded foreground threshold. Transition to background job.
            if context.active_jobs:
                await context.active_jobs.register(
                    job_id,
                    context.session_id,
                    proc,
                    log_filepath_stdout,
                    command=command,
                    stderr_filepath=log_filepath_stderr,
                )

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
                f"The command took longer than {threshold} seconds and has been transitioned to background execution.\n"  # noqa: E501
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
        output_filename = f"cmd_output_{timestamp}_{uuid.uuid4().hex[:8]}.txt"
        output_filepath = os.path.join(session_staging_dir, output_filename)
        # Middle-out preview: the head (headers, first error) and the tail (summary) both survive.
        placeholder = (
            f"\n\n... <{len(out_str) - effective_max} of {len(out_str)} chars omitted from the middle; "
            f"full output in `{output_filename}`> ...\n\n"
            if effective_max >= 1000
            else "\n... <truncated> ...\n"
        )
        preview = truncate_middle(out_str, effective_max, placeholder)
        preview_label = f"Preview (first ~45% and last ~55% of {effective_max} chars):"
        try:
            await async_write_text_file(output_filepath, out_str)
            final_output = (
                f"Output truncated (total length {len(out_str)} bytes). "
                f"Full output saved to session workspace file: `{output_filepath}`.\n"
                f'Page through it with `read_file("{output_filepath}", start_line=1)` (or re-run with a larger '
                f"`max_output_chars`, max {MAX_TOOL_OUTPUT_LENGTH}).\n\n"
                f"{preview_label}\n{preview}"
            )
        except Exception as ex:
            logger.error(f"Failed to save truncated output to '{output_filepath}': {ex}")
            final_output = f"Output truncated (total length {len(out_str)} bytes). {preview_label}\n{preview}"

    if proc.returncode != 0:
        raise ShellCommandError(f"Command failed with exit code {proc.returncode}.\n{final_output}")

    return final_output


@default_registry.register
async def manage_job(action: str, job_id: str | None = None, tail_chars: int = 4000,
                     context: ToolContext | None = None) -> str:
    """Inspect or stop background jobs of this session started by run_shell_command.

    action="list": all running and recently finished jobs. action="status": liveness, exit code and the head+tail of
    stdout/stderr (tail_chars per stream, max 10000) of one job. action="kill": terminate the job's whole process
    group (SIGTERM, then SIGKILL after 3 s) and drop it from the registry. job_id is required for status/kill.
    """
    if not context:
        return "Error: missing context."

    if action not in ("list", "status", "kill"):
        return "Error: action must be one of list, status, kill."

    tail_chars = max(200, min(tail_chars, 10000))
    session_id = context.session_id

    if action == "list":
        snapshot = await context.active_jobs.snapshot(session_id)
        if snapshot.is_empty:
            return "No background jobs in this session."

        lines = []
        if snapshot.running:
            lines.append(f"Running ({len(snapshot.running)}):")
            for j in snapshot.running:
                lines.append(f"- {j.job_id} | pgid {j.pgid} | started {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(j.started_at))} | {j.command}")  # noqa: E501
                lines.append(f"  Logs: {j.stdout_path} / {j.stderr_path}")
        if snapshot.finished:
            lines.append(f"Finished recently ({len(snapshot.finished)}):")
            for j in snapshot.finished:
                exit_str = "stopped" if j.reason == "stopped" else f"exit {j.return_code}" if j.return_code is not None else "exit killed"  # noqa: E501
                lines.append(f"- {j.job_id} | {exit_str} | finished {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(j.finished_at))} | {j.command}")  # noqa: E501
        return "\n".join(lines)

    elif action in ("status", "kill"):
        if not job_id:
            snapshot = await context.active_jobs.snapshot(session_id)
            running_ids = [j.job_id for j in snapshot.running]
            return f"Error: missing job_id. Running jobs: {', '.join(running_ids) if running_ids else 'None'}"

        if action == "kill":
            res = await context.active_jobs.kill_job(job_id, session_id)
            snapshot = await context.active_jobs.snapshot(session_id)
            log_str = ""
            for rj in snapshot.running:
                if rj.job_id == job_id:
                    log_str = f" Logs: {rj.stdout_path} / {rj.stderr_path}"
                    break
            if not log_str:
                for fj in snapshot.finished:
                    if fj.job_id == job_id:
                        log_str = f" Logs: {fj.stdout_path} / {fj.stderr_path}"
                        break
            return res.message + log_str

        else: # status
            snapshot = await context.active_jobs.snapshot(session_id)
            job = None
            is_running = True
            for rj in snapshot.running:
                if rj.job_id == job_id:
                    job = rj
                    break

            if not job:
                is_running = False
                for fj in snapshot.finished:
                    if fj.job_id == job_id:
                        job = fj
                        break

            if not job:
                known = [j.job_id for j in snapshot.running + snapshot.finished]
                return f"Job id '{job_id}' is unknown or expired (only the last 5 finished jobs are kept). Known: {', '.join(known) if known else 'none'}"  # noqa: E501

            lines = []
            if is_running:
                lines.append(f"Job {job.job_id}: RUNNING since {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.started_at))} (pgid {job.pgid})")  # noqa: E501
            else:
                exit_str = "stopped" if job.reason == "stopped" else f"exit {job.return_code}" if job.return_code is not None else "exit killed"  # noqa: E501
                lines.append(f"Job {job.job_id}: FINISHED {exit_str} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.finished_at))}")  # noqa: E501

            stdout_path = job.stdout_path
            stderr_path = job.stderr_path

            async def _read_log(path: str) -> str:
                if not path or not await async_exists(path):
                    return "(no output yet)"
                try:
                    content = await async_read_text_file(path)
                    return truncate_middle(content, tail_chars)
                except Exception as e:
                    return f"(error reading log: {e})"

            out = await _read_log(stdout_path)
            err = await _read_log(stderr_path)
            lines.append(f"--- stdout ({len(out)} chars) ---")
            lines.append(out)
            lines.append(f"--- stderr ({len(err)} chars) ---")
            lines.append(err)

            return "\n".join(lines)
