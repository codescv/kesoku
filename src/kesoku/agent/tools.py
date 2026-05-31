"""Tool registry and MVP skills for Kesoku AI Agent."""

import asyncio
import functools
import inspect
import os
import re
import shlex
import signal
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kesoku.gateway.gateway import Gateway

    GatewayType = Gateway | None
    ActiveJobsRegistryType = "ActiveJobsRegistry | None"
else:
    GatewayType = Any
    ActiveJobsRegistryType = Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from kesoku.agent.skills import SkillManager
from kesoku.config import get_config
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.logger import setup_logger
from kesoku.utils.async_fs import (
    async_exists,
    async_get_subdirectories,
    async_isdir,
    async_read_text_file,
    async_realpath,
    async_write_text_file,
)

logger = setup_logger(__name__)

MAX_TOOL_OUTPUT_LENGTH = 3000
TIMEOUT_SECONDS = 1800
MAX_MEMORY_CONTENT_LENGTH = 500


class ToolContext(BaseModel):
    """Contextual session metadata injected into executing tools."""

    model_config = {"arbitrary_types_allowed": True}

    session_id: str = Field(..., description="Unique session identifier")
    session_workspace: str = Field(..., description="Relative folder name for the session workspace")
    original_msg_id: str | None = Field(None, description="ID of the message initiating the turn")
    active_jobs: ActiveJobsRegistryType = Field(None, exclude=True)
    transitioned_to_session: str | None = Field(None, description="New session ID if history was compacted")
    gateway: GatewayType = Field(None, exclude=True)


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
    background_threshold_seconds: float | None = None,
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
        background_threshold_seconds: Optional foreground timeout limit (in seconds) override.
            If the command takes longer than this limit, it transitions to background execution.
            If not provided, defaults to configuration setting.
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
            exec_dir = await async_realpath(cwd)
        else:
            awd = config.agent_working_dir or os.getcwd()
            exec_dir = await async_realpath(os.path.join(awd, cwd))
    else:
        exec_dir = await async_realpath(config.agent_working_dir or os.getcwd())

    os.makedirs(exec_dir, exist_ok=True)

    # Prepare environment variables
    env = os.environ.copy()
    if config.shell.env:
        env.update(config.shell.env)

    # Inject AWD and STAGING_DIR environment variables
    env["AWD"] = await async_realpath(config.agent_working_dir or os.getcwd())
    if context and context.session_workspace:
        env["STAGING_DIR"] = await async_realpath(
            os.path.join(config.workspace.sessions_dir, context.session_workspace)
        )

    # Prepare unique background job identifiers and log paths
    job_id = f"job_{int(time.time())}"
    log_filename_stdout = f"background_{job_id}.stdout"
    log_filename_stderr = f"background_{job_id}.stderr"
    folder_name = context.session_workspace
    session_staging_dir = await async_realpath(os.path.join(config.workspace.sessions_dir, folder_name))
    os.makedirs(session_staging_dir, exist_ok=True)
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
    final_output = out_str
    if len(out_str) > MAX_TOOL_OUTPUT_LENGTH:
        timestamp = int(time.time())
        output_filename = f"cmd_output_{timestamp}.txt"
        output_filepath = os.path.join(session_staging_dir, output_filename)
        try:
            await async_write_text_file(output_filepath, out_str)
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
                f"Output truncated (total length {len(out_str)} bytes). Preview:\n{out_str[:MAX_TOOL_OUTPUT_LENGTH]}"
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


@default_registry.register
async def compact_history(summary: str, context: ToolContext) -> str:
    """Compact the active conversation history and transition this channel to a clean new session.

    Use this tool when your context token usage is high or you are close to your context limit,
    to summarize the conversation history so far and start fresh.

    Args:
        summary: The comprehensive chronological summary of the conversation so far.
                 You MUST follow this strict Markdown format:
                 ### 1. Tasks
                 - **Completed**: [List of tasks successfully resolved so far, milestones]
                 - **In Progress**: [List of tasks currently being worked on]

                 ### 2. Critical States & Facts
                 - **Key Facts**: [Important facts mentioned by the user, interesting happenings]
                 - **Files**: [The absolute path of important directories:
                               your intermediate files, important files you discovered, etc]

                 ### 3. User Preferences
                 - **Custom Rules**: [User-defined preferences, language choice, coding styles, or formatting rules]

                 ### 4. Key Commands & Executions
                 - **Important Commands**: [List of most commonly used or successfully executed shell
                                            commands, avoid re-exploring them in the new session]
        context: The tool execution context (injected automatically).

    Returns:
        Confirmation message indicating compaction status.
    """
    logger.info(f"Initiating history compaction for session '{context.session_id}'")

    gw = context.gateway
    if not gw:
        raise ValueError("Gateway instance must be injected in ToolContext to compact history.")

    # 1. Retrieve the old session to copy its configuration
    old_session = await gw.db.get_session(context.session_id)
    if not old_session:
        raise RuntimeError(f"Active session '{context.session_id}' not found.")

    # 2. Find the real initiating user message and platform identifiers
    old_history = await gw.db.get_session_history(context.session_id, limit=100)

    # A. Find the real user message for summary/content copying (skipping System triggers)
    initiating_msg = None
    for msg in reversed(old_history):
        is_cron = msg.metadata.get("is_cronjob") or msg.metadata.get("wechat_cronjob") or msg.sender == "Cronjob"
        if msg.role == MessageRole.USER and (msg.sender != "System" or is_cron):
            initiating_msg = msg
            break

    # B. Find the latest user/system message to resolve external channel identifiers reliably
    meta_msg = None
    for msg in reversed(old_history):
        if msg.role == MessageRole.USER:
            meta_msg = msg
            break
    if not meta_msg:
        meta_msg = old_history[-1] if old_history else None

    if not meta_msg:
        raise RuntimeError("Could not find any historical message for external channel mapping.")

    chatbot_id = meta_msg.chatbot_id
    channel_id = meta_msg.channel_id
    sender_name = initiating_msg.sender if initiating_msg else "System"

    # 3. Create the new session record
    new_session_title = f"Compacted {old_session.title[:15]}"
    new_session = await gw.create_session(
        title=new_session_title,
        system_prompt=old_session.system_prompt,
        chatbot_id=chatbot_id,
        channel_id=channel_id,
    )
    new_session_id = new_session.id

    # 4. Retrieve the latest raw user message content (empty if no real user prompt exists)
    latest_user_msg_content = initiating_msg.content if initiating_msg else ""
    if latest_user_msg_content and "\n\n[Context Monitor:" in latest_user_msg_content:
        latest_user_msg_content = latest_user_msg_content.split("\n\n[Context Monitor:")[0]

    # Check if the assistant has already completed and responded to this initiating user message in old_history
    is_turn_completed = False
    completed_assistant_reply = None
    if initiating_msg:
        initiating_idx = old_history.index(initiating_msg)
        for msg in old_history[initiating_idx + 1 :]:
            if msg.role == MessageRole.ASSISTANT and msg.type == MessageType.TEXT:
                completed_assistant_reply = msg
                break
        is_turn_completed = completed_assistant_reply is not None

    # 5. Construct the compacted starting message
    compacted_content = f"[Conversation History Summary]\n{summary}\n\n[Latest User Message]\n{latest_user_msg_content}"

    # 6. Post the compacted message under the new session ID to the channel
    # If the turn was already completed, save it as PROCESSED so it doesn't trigger a new worker run
    compacted_msg_status = MessageStatus.PROCESSED if is_turn_completed else MessageStatus.PENDING_AGENT
    compacted_msg = Message(
        session_id=new_session_id,
        chatbot_id=chatbot_id,
        channel_id=channel_id,
        sender=sender_name,
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content=compacted_content,
        status=compacted_msg_status,
        metadata=dict(initiating_msg.metadata),
    )

    # 7. Post a system notification assistant message to inform the user of the background transition
    notification_msg = Message(
        session_id=new_session_id,
        chatbot_id=chatbot_id,
        channel_id=channel_id,
        sender="Notification",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content="🔄 Conversation history has been automatically compacted to optimize response speed.",
        status=MessageStatus.PENDING,
    )

    # Save messages to the database
    await gw.post(compacted_msg)
    await gw.post(notification_msg)

    # 8. If the previous turn was already completed, also copy the assistant's final response to the new session
    if is_turn_completed and completed_assistant_reply:
        # Strip turn_metrics from copied metadata to avoid stats inconsistency in the new session
        copied_metadata = dict(completed_assistant_reply.metadata)
        copied_metadata.pop("turn_metrics", None)

        copied_reply = Message(
            session_id=new_session_id,
            chatbot_id=chatbot_id,
            channel_id=channel_id,
            sender=completed_assistant_reply.sender,
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content=completed_assistant_reply.content,
            status=MessageStatus.DELIVERED,
            metadata=copied_metadata,
        )
        await gw.post(copied_reply)

    # 8. Record transition in context so TurnExecutor knows to abort cleanly
    context.transitioned_to_session = new_session_id

    logger.info(f"Compacted history successfully. Transitioned '{context.session_id}' -> '{new_session_id}'")
    return f"Conversation history successfully compacted and transitioned to new session: {new_session_id}."


# Memory System Helpers and Tools
async def get_allowed_categories(db: Any) -> set[str]:
    """Retrieves the set of all currently permitted or existing memory categories."""
    categories = {"learnings", "progress", "user_preferences"}
    try:
        memories = await db.get_agent_memories()
        for m in memories:
            categories.add(m["category"])
    except Exception as e:
        logger.warning(f"Failed to fetch existing categories from database: {e}")
    return categories


def sanitize_key(input_key: str) -> str:
    """Sanitizes key by lowercasing, stripping, and replacing invalid characters with underscores."""
    clean_key = re.sub(r"[^a-z0-9_]", "_", input_key.lower().strip())
    clean_key = re.sub(r"_+", "_", clean_key).strip("_")
    return clean_key


def validate_key(key: str) -> bool:
    """Verifies if the key strictly contains only lowercase letters, underscores, and numbers."""
    return bool(re.match(r"^[a-z0-9_]+$", key))


async def _resolve_memory_role(category: str, role_param: str | None, context: ToolContext | None) -> str:
    """Resolve the correct role scope based on the memory category and context rules."""
    category = category.strip().lower()

    # Rule 1: Standard categories ALWAYS use "default" role
    if category in {"progress", "learnings"}:
        return "default"

    # Rule 2: user_preferences category uses current channel's active role
    if category in {"user_preferences"}:
        if context and context.gateway and context.original_msg_id:
            db = context.gateway.db
            try:
                msg_list = await db.get_messages_by_filters({"id": context.original_msg_id})
                if msg_list:
                    msg = msg_list[0]
                    return await db.get_channel_role_with_inheritance(
                        msg.chatbot_id, msg.channel_id, context.session_id
                    )
            except Exception as e:
                logger.warning(f"Failed to resolve active role for memory category {category}: {e}")
        # Fallback
        return role_param if role_param else "default"

    # For other categories, return the passed role parameter
    return role_param if role_param else "default"


@default_registry.register
async def list_memories(
    category: str,
    role: str | None = None,
    context: ToolContext | None = None,
) -> str:
    """List all active memory keys and titles under the specified category.

    Args:
        category: The memory category (e.g., 'progress', 'learnings', 'user_preferences').
        role: Optional roleplay persona scope (defaults to None for implicit channel/default persona).
        context: Injected tool execution context.

    Returns:
        A clean list of active keys, titles, and their last updated timestamps.
    """
    if not context:
        return "Error: ToolContext is missing."

    resolved_role = await _resolve_memory_role(category, role, context)
    db = context.gateway.db
    try:
        memories = await db.get_agent_memories(category=category, role=resolved_role)
        if not memories:
            return f"No memories found in category '{category}' for role scope '{resolved_role}'."

        lines = [f"=== Memories in '{category}' (scope: {resolved_role}) ==="]
        for m in memories:
            updated_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m["updated_at"]))
            lines.append(f'- key: `{m["key"]}` | title: "{m["title"]}" | updated: {updated_str} | scope: {m["role"]}')
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Failed to list memories: {e}", exc_info=True)
        return f"Error listing memories: {e}"


@default_registry.register
async def view_memory(
    category: str,
    key: str | None = None,
    role: str | None = None,
    context: ToolContext | None = None,
) -> str:
    """Retrieve detailed content for a specific memory key, or dynamically render all memories in a category.

    If `key` is provided, returns the content of that specific record.
    If `key` is None (or omitted), dynamically aggregates all entries inside that category under the implicit
    correct role/default scope and formats them into a beautiful, readable Markdown block.

    Args:
        category: The memory category (e.g., 'progress', 'learnings', 'user_preferences').
        key: Optional unique snake_case key. If omitted, renders all entries.
        role: Optional roleplay persona scope (defaults to None for implicit channel/default persona).
        context: Injected tool execution context.

    Returns:
        The content of the specific memory entry, or a compiled Markdown text block of all entries.
    """
    if not context:
        return "Error: ToolContext is missing."

    resolved_role = await _resolve_memory_role(category, role, context)
    db = context.gateway.db
    try:
        if key:
            if not validate_key(key):
                return (
                    f"Error: Invalid Key '{key}'.\n"
                    "Memory keys must strictly contain only lowercase letters, "
                    "underscores, and numbers (regex: ^[a-z0-9_]+$)."
                )
            sanitized = sanitize_key(key)
            mem = await db.get_agent_memory(category=category, key=sanitized, role=resolved_role)
            if not mem:
                return f"No memory found for category='{category}', key='{sanitized}', role='{resolved_role}'."
            return f"=== Memory: {mem['title']} (key: {mem['key']}, scope: {mem['role']}) ===\n{mem['content']}"

        memories = await db.get_agent_memories(category=category, role=resolved_role)
        if not memories:
            return f"No memories found in category '{category}' for role scope '{resolved_role}'."

        lines = [f"# Category: {category} (scope: {resolved_role})"]
        for m in memories:
            lines.append(f"\n## {m['title']} (key: `{m['key']}`, scope: `{m['role']}`)")
            lines.append(m["content"].strip())
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Failed to view memory: {e}", exc_info=True)
        return f"Error viewing memory: {e}"


@default_registry.register
async def update_memory(
    category: str,
    key: str,
    title: str,
    content: str,
    role: str | None = None,
    create_category: bool = False,
    context: ToolContext | None = None,
) -> str:
    """Insert or atomically replace a memory record in the SQLite database.

    This executes a transactionally isolated UPSERT SQL statement, completely protecting other keys.

    Note: To prevent memory bloat, each memory entry's content MUST be kept concise.

    Args:
        category: The memory category (e.g., 'progress', 'learnings', 'user_preferences').
        key: The unique snake_case key.
        title: Human-readable label or title for the entry.
        content: Detailed markdown or JSON content payload (must be <= 200 characters).
        role: Optional roleplay persona scope (defaults to None for implicit channel/default persona).
        create_category: True if permission was granted to initialize a new category.
        context: Injected tool execution context.

    Returns:
        A success message confirming the sanitized key, title, and role scope.
    """
    if not context:
        return "Error: ToolContext is missing."

    resolved_role = await _resolve_memory_role(category, role, context)
    db = context.gateway.db
    try:
        allowed = await get_allowed_categories(db)
        if category not in allowed and not create_category:
            return (
                f"Write Denied: Category '{category}' is not recognized.\n"
                f"Permitted categories: {sorted(list(allowed))}.\n"
                "If you need to create a new category, you MUST ask the user for explicit permission "
                "first, and then invoke this tool with `create_category=True`."
            )

        if not validate_key(key):
            return (
                f"Error: Invalid Key '{key}'.\n"
                "Memory keys must strictly contain only lowercase letters, "
                "underscores, and numbers (regex: ^[a-z0-9_]+$)."
            )

        if len(content) > MAX_MEMORY_CONTENT_LENGTH:
            return (
                f"Error: Content length ({len(content)} characters) exceeds the maximum limit "
                f"of {MAX_MEMORY_CONTENT_LENGTH} characters. Please keep the memory extremely concise and try again."
            )

        sanitized_key = sanitize_key(key)
        await db.upsert_agent_memory(
            category=category,
            key=sanitized_key,
            title=title,
            content=content,
            role=resolved_role,
        )
        return (
            f"Memory successfully saved!\n"
            f"  Category: `{category}`\n"
            f"  Key: `{sanitized_key}`\n"
            f'  Title: "{title}"\n'
            f"  Scope: `{resolved_role}`"
        )
    except Exception as e:
        logger.error(f"Failed to update memory: {e}", exc_info=True)
        return f"Error updating memory: {e}"


@default_registry.register
async def delete_memory(
    category: str,
    key: str,
    role: str | None = None,
    context: ToolContext | None = None,
) -> str:
    """Atomically delete a specific memory entry under a category and implicit correct role scope.

    Args:
        category: The memory category (e.g., 'progress', 'learnings', 'user_preferences').
        key: The unique snake_case key to delete.
        role: Optional roleplay persona scope (defaults to None for implicit channel/default persona).
        context: Injected tool execution context.

    Returns:
        A confirmation message.
    """
    if not context:
        return "Error: ToolContext is missing."

    resolved_role = await _resolve_memory_role(category, role, context)
    db = context.gateway.db
    try:
        if not validate_key(key):
            return (
                f"Error: Invalid Key '{key}'.\n"
                "Memory keys must strictly contain only lowercase letters, "
                "underscores, and numbers (regex: ^[a-z0-9_]+$)."
            )

        sanitized = sanitize_key(key)
        mem = await db.get_agent_memory(category=category, key=sanitized, role=resolved_role)
        if not mem:
            return (
                f"No memory entry found for category='{category}', key='{sanitized}', role='{resolved_role}' to delete."
            )

        await db.delete_agent_memory(category=category, key=sanitized, role=resolved_role)
        return f"Memory successfully deleted: category='{category}', key='{sanitized}', role='{resolved_role}'."
    except Exception as e:
        logger.error(f"Failed to delete memory: {e}", exc_info=True)
        return f"Error deleting memory: {e}"


@default_registry.register
async def play_role(role: str, context: ToolContext) -> str:
    """Switch the active roleplay persona for the current channel.

    Switching the role automatically loads and injects the role's profile instructions (intro.md)
    into the system prompt for the active session.

    Args:
        role: The name of the character role to play (e.g. 'default', 'tifa', 'asuka').
        context: The tool execution context (injected automatically).

    Returns:
        A status message confirming the role switch.
    """
    role = role.strip()
    cfg = get_config()

    # 1. Validate role directory exists
    roles_dir = cfg.workspace.roles_dir
    role_dir = os.path.join(roles_dir, role)
    if not await async_exists(role_dir) or not await async_isdir(role_dir):
        # Get available roles
        available_roles = await async_get_subdirectories(roles_dir)
        if not available_roles:
            available_roles = ["default"]
        return (
            f"⚠️ **Error:** Persona `{role}` does not exist.\n"
            f"✨ **Available Personas:** {', '.join(f'`{r}`' for r in sorted(available_roles))}"
        )

    # 2. Query original message to find chatbot_id and channel_id
    db = context.gateway.db
    msg_list = await db.get_messages_by_filters({"id": context.original_msg_id})
    if not msg_list:
        return "⚠️ **Error:** Failed to resolve current channel mapping for this tool call."

    msg = msg_list[0]
    chatbot_id = msg.chatbot_id
    channel_id = msg.channel_id

    # 3. Save binding in database
    await db.set_channel_role(chatbot_id, channel_id, role)

    # 4. Rebuild active session's system prompt
    session = await context.gateway.db.get_session(context.session_id)
    if session:
        from kesoku.agent.prompt import build_sys_prompt

        new_sys_prompt = build_sys_prompt(session=session)
        await context.gateway.db.update_session_system_prompt(context.session_id, new_sys_prompt)

    return (
        f"🎭 **Persona Switched Successfully!**\n"
        f"Character role has been set to **`{role}`** for this channel.\n"
        f"The instructions from `{role}/intro.md` have been successfully injected into your system prompt. "
        f"Please adopt this persona immediately."
    )


@default_registry.register
async def view_cross_session_memory(context: ToolContext) -> str:
    """Retrieve a cross-session narrative timeline of recent events.

    This includes conversations and milestones that occurred in other active
    threads/channels for the current active persona role.

    Use this tool when the user refers to external discussions or events that
    you do not have in your current local conversation history.

    Args:
        context: Injected tool execution context (automatically resolved).

    Returns:
        A formatted Markdown string containing the cross-session event
        timeline and any recent un-consolidated conversations.
    """
    if not context or not context.gateway:
        return "Error: ToolContext or Gateway is missing."



    # 1. Resolve active role using helper
    active_role = await _resolve_memory_role(category="user_preferences", role_param=None, context=context)

    try:
        # 2. Query consolidated timeline and recent messages from Gateway
        stored_ctx, new_messages = await context.gateway.db.get_cross_session_memory_updates(
            role=active_role,
            exclude_session_id=context.session_id,
        )
        timeline_content = stored_ctx.content if stored_ctx else ""

        lines = [f"=== Cross-Session Event Timeline (Role: {active_role}) ==="]
        if timeline_content.strip():
            lines.append(timeline_content.strip())
        else:
            lines.append("No historical timeline events recorded yet.")

        if new_messages:
            lines.append("\n=== Recent Activity in other threads (Un-consolidated) ===")
            for m in new_messages:
                time_str = time.strftime("%m-%d %H:%M", time.localtime(m.timestamp))
                lines.append(f"- [{time_str}] {m.sender}: {m.content}")

        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Failed to view cross-session memory: {e}", exc_info=True)
        return f"Error retrieving cross-session memory: {e}"

