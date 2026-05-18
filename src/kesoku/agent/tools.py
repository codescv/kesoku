"""Tool registry and MVP skills for Kesoku AI Agent."""

import functools
import inspect
import os
import re
import shlex
import subprocess
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

MAX_OUTPUT_LENGTH = 1000
TIMEOUT_SECONDS = 1800


class ToolContext(BaseModel):
    """Contextual session metadata injected into executing tools."""

    session_id: str = Field(..., description="Unique session identifier")
    session_workspace: str = Field(..., description="Relative folder name for the session workspace")


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

    def web_search(self, query: str, context: ToolContext | None = None) -> str:
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

            generate_config = types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )

            res = client.models.generate_content(
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
def web_search(query: str, context: ToolContext | None = None) -> str:
    """Search the web for current information on a given topic.

    Args:
        query: The search query string.
        context: Optional tool execution context.

    Returns:
        Search results summary with grounding sources.
    """
    return web_search_tool.web_search(query, context)


@default_registry.register
def run_shell_command(command: str, context: ToolContext | None = None) -> str:
    """Execute a CLI shell command within a dedicated per-session staging directory.

    The command is executed inside an isolated staging directory specific to the current session
    (e.g., sessions/<YYMMDD-HH-MM>_<session_title>_<session_id>). All temporary scripts, data files, or build artifacts
    must be created within this directory.
    If a user task requires executing commands in another location (such as the project root repository),
    you must explicitly chain a 'cd' command (e.g., 'cd /path/to/repo && git status').

    Args:
        command: The command string to execute (e.g., 'uv run pytest' or 'echo hello').
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

    # Resolve session staging directory
    folder_name = context.session_workspace
    session_staging_dir = os.path.realpath(os.path.join(config.workspace.sessions_dir, folder_name))
    os.makedirs(session_staging_dir, exist_ok=True)

    # Prepare environment variables
    env = os.environ.copy()
    if config.shell.env:
        env.update(config.shell.env)

    logger.info(f"Executing shell command in '{session_staging_dir}': {command}")

    try:
        if config.shell.use_shell:
            res = subprocess.run(
                command,
                shell=True,
                cwd=session_staging_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
        else:
            tokens = shlex.split(command)
            res = subprocess.run(
                tokens,
                shell=False,
                cwd=session_staging_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
    except subprocess.TimeoutExpired as e:
        out = e.output.decode("utf-8", errors="replace") if isinstance(e.output, bytes) else (e.output or "")
        err = (
            e.stderr.decode("utf-8", errors="replace")
            if getattr(e, "stderr", None) and isinstance(e.stderr, bytes)
            else (getattr(e, "stderr", None) or "")
        )
        return f"Command timed out after {TIMEOUT_SECONDS} seconds.\nSTDOUT:\n{out}\nSTDERR:\n{err}"
    except Exception as ex:
        logger.error(f"Failed to execute command '{command}': {ex}")
        return f"Error executing command: {ex}"

    out_str = f"=== STDOUT ===\n{res.stdout}\n=== STDERR ===\n{res.stderr}"
    if len(out_str) > MAX_OUTPUT_LENGTH:
        timestamp = int(time.time())
        output_filename = f"cmd_output_{timestamp}.txt"
        output_filepath = os.path.join(session_staging_dir, output_filename)
        try:
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(out_str)
            preview_len = MAX_OUTPUT_LENGTH // 2
            return (
                f"Output truncated (total length {len(out_str)} bytes). "
                f"Full output saved to session workspace file: `{output_filepath}`.\n"
                f"You can use tool `run_shell_command` (e.g., `cat {output_filename}`) "
                f"on this path to examine the full output.\n\n"
                f"Preview:\n{out_str[:preview_len]}...\n{out_str[-preview_len:]}"
            )
        except Exception as ex:
            logger.error(f"Failed to save truncated output to '{output_filepath}': {ex}")
            return f"Output truncated (total length {len(out_str)} bytes). Preview:\n{out_str[:MAX_OUTPUT_LENGTH]}"

    return out_str


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
