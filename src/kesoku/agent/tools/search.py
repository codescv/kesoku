"""Web search tools for Kesoku AI Agent using Google Search grounding."""

import asyncio
import logging
import os
import threading
import warnings
from typing import Any

from google import genai
from google.genai import types

from kesoku.agent.tools.registry import ToolContext, default_registry
from kesoku.config import GeminiConfig, get_config

logger = logging.getLogger(__name__)

# Hard upper bound for one grounded search call (the main model has been observed to take up to ~87 s).
WEB_SEARCH_TIMEOUT_SECONDS = 30.0
# Search grounding needs little reasoning; keep thinking low for latency/cost.
WEB_SEARCH_THINKING_LEVEL = "low"


def _resolve_search_model(config: GeminiConfig) -> str:
    """Use the lighter context/background model when configured, else the main model."""
    return config.context_model_name or config.model_name


def _build_thinking_config(config: GeminiConfig) -> Any | None:
    """Build a low `ThinkingConfig`, mirroring how `GeminiLLM` resolves thinking levels.

    Returns None when thinking is disabled in config (`thinking_level=None` means "model default", typically a
    model without thinking-level support) or the installed SDK has no `ThinkingConfig`.
    """
    if config.thinking_level is None or not hasattr(types, "ThinkingConfig"):
        return None
    level = "minimal" if config.thinking_level == "minimal" else WEB_SEARCH_THINKING_LEVEL
    resolved = getattr(types.ThinkingLevel, level.upper(), level.upper()) if hasattr(types, "ThinkingLevel") else None
    if resolved is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*is not a valid ThinkingLevel.*", category=UserWarning)
            return types.ThinkingConfig(thinking_level=resolved)
    except Exception as e:  # Older SDKs without `thinking_level`.
        logger.debug(f"WebSearchTool: thinking_level unsupported by SDK ({e}); using model default.")
        return None


def _is_thinking_unsupported_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "thinking" in msg and ("400" in msg or "invalid" in msg or "not supported" in msg or "unsupported" in msg)


class WebSearchTool:
    """Tool for executing Google Search queries via Gemini API grounding."""

    def __init__(self, client: genai.Client | None = None) -> None:
        """Initialize WebSearchTool.

        Args:
            client: Optional pre-configured genai.Client (useful for dependency injection and unit tests).
        """
        self._client = client
        self._client_lock = threading.Lock()

    def _get_client(self) -> genai.Client:
        """Retrieve or lazily initialize (once, thread-safe) the cached Google GenAI client."""
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                self._client = self._create_client()
            return self._client

    @staticmethod
    def _create_client() -> genai.Client:
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
        key = config.api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            logger.warning("GEMINI_API_KEY is not set. WebSearchTool calls may fail if not authenticated.")
        return genai.Client(api_key=key)

    async def _generate(self, client: genai.Client, config: GeminiConfig, query: str) -> Any:
        model = _resolve_search_model(config)
        tools = [types.Tool(google_search=types.GoogleSearch())]
        thinking = _build_thinking_config(config)
        generate_config = types.GenerateContentConfig(tools=tools)
        if thinking is not None:
            generate_config.thinking_config = thinking
        try:
            return await client.aio.models.generate_content(model=model, contents=query, config=generate_config)
        except Exception as e:
            if thinking is None or not _is_thinking_unsupported_error(e):
                raise
            logger.info(f"WebSearchTool: model '{model}' rejected thinking_config ({e}); retrying without it.")
            return await client.aio.models.generate_content(
                model=model, contents=query, config=types.GenerateContentConfig(tools=tools)
            )

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
            res = await asyncio.wait_for(
                self._generate(client, config, query), timeout=WEB_SEARCH_TIMEOUT_SECONDS
            )
        except TimeoutError:
            logger.warning(f"Web search timed out after {WEB_SEARCH_TIMEOUT_SECONDS:.0f}s for query: '{query}'")
            return (
                f"Web search timed out after {WEB_SEARCH_TIMEOUT_SECONDS:.0f} seconds. "
                "Try a shorter or more specific query, or continue without web results."
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
