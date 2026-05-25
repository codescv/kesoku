"""Tool registry and MVP skills for Kesoku AI Agent."""
# ruff: noqa: ASYNC230, ASYNC240

import asyncio
import functools
import inspect
import os
import re
import shlex
import time
from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from kesoku.agent.skills import SkillManager
from kesoku.config import get_config
from kesoku.logger import setup_logger

logger = setup_logger(__name__)

MAX_TOOL_OUTPUT_LENGTH = 3000
TIMEOUT_SECONDS = 1800


class ToolContext(BaseModel):
    """Contextual session metadata injected into executing tools."""

    model_config = {"arbitrary_types_allowed": True}

    session_id: str = Field(..., description="Unique session identifier")
    session_workspace: str = Field(..., description="Relative folder name for the session workspace")
    original_msg_id: str | None = Field(None, description="ID of the message initiating the turn")
    active_jobs: Any = Field(None, exclude=True)


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
        import signal

        async with self._lock:
            target_job_ids = [
                jid for jid, job in self._jobs.items()
                if job["session_id"] == session_id
            ]

            for jid in target_job_ids:
                job = self._jobs[jid]
                proc = job["process"]
                logger.info(f"Force terminating background job '{jid}' for session '{session_id}'")
                try:
                    # Terminate the process group to kill child processes as well
                    if os.name == "posix":
                        import os as local_os
                        try:
                            local_os.killpg(local_os.getpgid(proc.pid), signal.SIGTERM)
                        except Exception:
                            proc.terminate()
                    else:
                        proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except Exception as e:
                    logger.debug(f"Failed to gracefully terminate process group {proc.pid}: {e}")
                    try:
                        if os.name == "posix":
                            import os as local_os
                            try:
                                local_os.killpg(local_os.getpgid(proc.pid), signal.SIGKILL)
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
        with open(file_path, mode, encoding="utf-8", errors="replace") as f:
            while True:
                line = await stream.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace")
                f.write(decoded)
                f.flush()
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
    """Monitors a background shell subprocess, posts results to gateway, and posts a system alert to wake up LLM.

    Args:
        job_id: Unique identifier for the background job.
        proc: The asyncio Process instance.
        log_filepath_stdout: The path to the stdout log file on disk.
        log_filepath_stderr: The path to the stderr log file on disk.
        context: The ToolContext containing session ID.
        original_msg_id: The ID of the original message that triggered this command.
        stdout_task: The background stdout streaming task.
        stderr_task: The background stderr streaming task.
    """
    from kesoku.constants import MessageRole, MessageStatus, MessageType
    from kesoku.db import Message
    from kesoku.gateway.gateway import Gateway

    try:
        # Wait for process and stream tasks to finish
        return_code = await proc.wait()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

        # Read the final output logs
        stdout = ""
        stderr = ""
        if os.path.exists(log_filepath_stdout):
            try:
                with open(log_filepath_stdout, encoding="utf-8", errors="replace") as f:
                    stdout = f.read()
            except Exception as e:
                stdout = f"Failed to read stdout log: {e}"
        if os.path.exists(log_filepath_stderr):
            try:
                with open(log_filepath_stderr, encoding="utf-8", errors="replace") as f:
                    stderr = f.read()
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
                f"Preview:\n{output[:MAX_TOOL_OUTPUT_LENGTH // 2]}...\n{output[-MAX_TOOL_OUTPUT_LENGTH // 2:]}"
            )

        # Post special System wakeup alert to Gateway
        gw = Gateway()

        status_str = "successfully" if return_code == 0 else f"with error code {return_code}"
        content = (
            f"[System Alert] Background Job `{job_id}` has finished executing {status_str}.\n"
            f"Stdout path: `{log_filepath_stdout}`\n"
            f"Stderr path: `{log_filepath_stderr}`\n\n"
            f"{truncated_output}"
        )

        wakeup_msg = Message(
            session_id=context.session_id,
            chatbot_id="system",
            channel_id="system",
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


def _create_schema_func(func: Callable) -> Callable:
    """Create a wrapper function with context parameter removed from its signature for LLM schema generation."""
    sig = inspect.signature(func)
    if "context" not in sig.parameters:
        return func
    new_params = [p for p in sig.parameters.values() if p.name != "context"]
    new_sig = sig.replace(parameters=new_params)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    wrapper.__signature__ = new_sig  # type: ignore
    return wrapper


class ToolRegistry:
    """Maintains a registry of callable Python functions exposed as LLM tools."""

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self._tools: dict[str, Callable] = {}
        self._schema_tools: dict[str, Callable] = {}

    def register(self, func: Callable) -> Callable:
        """Register a function as a tool. Can be used as a decorator.

        Args:
            func: The Python function to register. Must have type hints and docstrings.

        Returns:
            The registered function unchanged.
        """
        self._tools[func.__name__] = func
        self._schema_tools[func.__name__] = _create_schema_func(func)
        logger.info(f"Registered tool: {func.__name__}")
        return func

    def get_tools_list(self) -> list[Callable]:
        """Retrieve the list of registered tool callables formatted for LLM schema generation.

        Returns:
            A list of callable functions with context arguments stripped.
        """
        return list(self._schema_tools.values())

    def get_tool(self, name: str) -> Callable:
        """Retrieve a specific tool function by name for execution.

        Args:
            name: Name of the tool function.

        Returns:
            The callable function.

        Raises:
            KeyError: If the tool is not registered.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered.")
        return self._tools[name]


# Default global registry instance
default_registry = ToolRegistry()


class WebSearchTool:
    """Tool for executing Google Search queries via Gemini API grounding."""

    def __init__(self, client: genai.Client | None = None) -> None:
        """Initialize WebSearchTool.

        Args:
            client: Optional pre-configured genai.Client (useful for dependency injection and unit tests).
        """
        self._client = client

    def _get_client(self) -> genai.Client:
        """Retrieve or initialize the Google GenAI client lazily."""
        if self._client is not None:
            return self._client

        config = get_config().gemini
        if config.auth_mode == "vertex":
            logger.info(
                f"Initializing WebSearchTool Gemini client in Vertex AI mode "
                f"(Project: {config.project_id}, Region: {config.location})"
            )
            return genai.Client(
                vertexai=True,
                project=config.project_id,
                location=config.location,
            )
        else:
            key = config.api_key or os.getenv("GEMINI_API_KEY")
            if not key:
                logger.warning("GEMINI_API_KEY is not set. WebSearchTool calls may fail if not authenticated.")
            return genai.Client(api_key=key)

    async def web_search(self, query: str, context: ToolContext | None = None) -> str:
        """Search the web for current information on a given topic using Google Search grounding.

        Args:
            query: The search query string.
            context: Optional tool execution context.

        Returns:
            Search results summary with grounding sources.
        """
        logger.info(f"Executing web search for query: '{query}'")
        try:
            client = self._get_client()
            config = get_config().gemini

            generate_config = types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])

            res = await client.aio.models.generate_content(
                model=config.model_name,
                contents=query,
                config=generate_config,
            )
        except Exception as e:
            logger.error(f"Web search API call failed: {e}")
            return f"Web search failed: {e}"

        text_content = res.text or ""
        sources: list[str] = []

        if (
            res.candidates
            and res.candidates[0].grounding_metadata
            and res.candidates[0].grounding_metadata.grounding_chunks
        ):
            seen_urls = set()
            for chunk in res.candidates[0].grounding_metadata.grounding_chunks:
                web_chunk = getattr(chunk, "web", None)
                if web_chunk and getattr(web_chunk, "uri", None) and web_chunk.uri not in seen_urls:
                    seen_urls.add(web_chunk.uri)
                    title = getattr(web_chunk, "title", None) or getattr(web_chunk, "domain", None) or "Web Source"
                    sources.append(f"- {title}: {web_chunk.uri}")

        if sources:
            sources_str = "\n".join(sources)
            return f"{text_content}\n\nSources:\n{sources_str}"

        return text_content


web_search_tool = WebSearchTool()


@default_registry.register
async def web_search(query: str, context: ToolContext | None = None) -> str:
    """Search the web for current information on a given topic.

    Args:
        query: The search query string.
        context: Optional tool execution context.

    Returns:
        Search results summary with grounding sources.
    """
    return await web_search_tool.web_search(query, context)


class ShellCommandError(RuntimeError):
    """Raised when a shell command execution fails or times out."""

    pass


@default_registry.register
async def run_shell_command(
    command: str,
    cwd: str | None = None,
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
    - Tool output is capped at 3000 characters to save tokens. If you need to view
      a large file, read it by chunks of lines.
    - You are encouraged to combine commands using '|', '&&', ';' etc to save turns.
    - If you think the output of the command is long or not important, redirect it to
      a file in /tmp or filter the output (e.g. `ls -R | grep .py`)

    Args:
        command: The command string to execute (e.g., 'uv run pytest' or 'echo hello').
        cwd: Optional target working directory for executing the command.
            If relative, it's relative to AWD. Defaults to AWD.
        context: Optional tool execution context.

    Returns:
        Command execution stdout and stderr, or a file reference if output exceeds 1000 characters.
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
    if cwd:
        if os.path.isabs(cwd):
            exec_dir = os.path.realpath(cwd)
        else:
            awd = config.agent_working_dir or os.getcwd()
            exec_dir = os.path.realpath(os.path.join(awd, cwd))
    else:
        exec_dir = os.path.realpath(config.agent_working_dir or os.getcwd())

    os.makedirs(exec_dir, exist_ok=True)

    # Prepare environment variables
    env = os.environ.copy()
    if config.shell.env:
        env.update(config.shell.env)

    # Inject AWD and STAGING_DIR environment variables
    env["AWD"] = os.path.realpath(config.agent_working_dir or os.getcwd())
    if context and context.session_workspace:
        env["STAGING_DIR"] = os.path.realpath(os.path.join(config.workspace.sessions_dir, context.session_workspace))

    # Prepare unique background job identifiers and log paths
    job_id = f"job_{int(time.time())}"
    log_filename_stdout = f"background_{job_id}.stdout"
    log_filename_stderr = f"background_{job_id}.stderr"
    folder_name = context.session_workspace
    session_staging_dir = os.path.realpath(os.path.join(config.workspace.sessions_dir, folder_name))
    os.makedirs(session_staging_dir, exist_ok=True)
    log_filepath_stdout = os.path.join(session_staging_dir, log_filename_stdout)
    log_filepath_stderr = os.path.join(session_staging_dir, log_filename_stderr)

    logger.info(f"Executing shell command in '{exec_dir}': {command}")

    # Configure process group on POSIX to allow clean SIGTERM/SIGKILL propagation
    preexec_fn = None
    if os.name == "posix":
        import os as local_os
        preexec_fn = local_os.setsid

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
        threshold = getattr(config.shell, "background_threshold_seconds", 300.0)
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
                    import os as local_os
                    import signal
                    try:
                        local_os.killpg(local_os.getpgid(proc.pid), signal.SIGTERM)
                    except Exception:
                        proc.terminate()
                else:
                    proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except Exception:
                try:
                    if os.name == "posix":
                        import os as local_os
                        import signal
                        try:
                            local_os.killpg(local_os.getpgid(proc.pid), signal.SIGKILL)
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
    if os.path.exists(log_filepath_stdout):
        try:
            with open(log_filepath_stdout, encoding="utf-8", errors="replace") as f:
                stdout = f.read()
        except Exception as e:
            stdout = f"Failed to read stdout logs: {e}"
    if os.path.exists(log_filepath_stderr):
        try:
            with open(log_filepath_stderr, encoding="utf-8", errors="replace") as f:
                stderr = f.read()
        except Exception as e:
            stderr = f"Failed to read stderr logs: {e}"

    out_str = f"=== STDOUT ===\n{stdout}\n=== STDERR ===\n{stderr}"

    # Unified truncation logic
    final_output = out_str
    if len(out_str) > MAX_TOOL_OUTPUT_LENGTH:
        timestamp = int(time.time())
        output_filename = f"cmd_output_{timestamp}.txt"
        output_filepath = os.path.join(session_staging_dir, output_filename)
        try:
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(out_str)
            preview_len = MAX_TOOL_OUTPUT_LENGTH // 2
            final_output = (
                f"Output truncated (total length {len(out_str)} bytes). "
                f"Full output saved to session workspace file: `{output_filepath}`.\n"
                f"You can use tool `run_shell_command` (e.g., `cat {output_filename}`) "
                f"on this path to examine the full output.\n\n"
                f"Preview:\n{out_str[:preview_len]}...\n{out_str[-preview_len:]}"
            )
        except Exception as ex:
            logger.error(f"Failed to save truncated output to '{output_filepath}': {ex}")
            final_output = (
                f"Output truncated (total length {len(out_str)} bytes). "
                f"Preview:\n{out_str[:MAX_TOOL_OUTPUT_LENGTH]}"
            )

    if proc.returncode != 0:
        raise ShellCommandError(f"Command failed with exit code {proc.returncode}.\n{final_output}")

    return final_output


skill_manager = SkillManager()


@default_registry.register
def list_skills(context: ToolContext | None = None) -> str:
    """List all valid skills in skills_dir supported on the current host operating system.

    Args:
        context: Optional tool execution context.

    Returns:
        Formatted summary of available skills.
    """
    skills = skill_manager.list_skills()
    if not skills:
        return "No skills available or supported on this platform."
    lines = ["=== Available Skills ==="]
    for s in skills:
        lines.append(f"- {s['name']} (v{s['version']}): {s['description']}")
    return "\n".join(lines)


@default_registry.register
def use_skill(skill_name: str, context: ToolContext | None = None) -> str:
    """Retrieve the complete instructions and absolute directory path for a specific skill.

    Args:
        skill_name: Name of the skill.
        context: Optional tool execution context.

    Returns:
        Complete markdown instructions and absolute path header for the skill.
    """
    try:
        _, content = skill_manager.get_skill(skill_name)
        return content
    except Exception as e:
        return f"Failed to load skill '{skill_name}': {e}"
