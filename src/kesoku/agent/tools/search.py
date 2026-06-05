"""Web search tools for Kesoku AI Agent using Google Search grounding."""

import logging
import os

from google import genai
from google.genai import types

from kesoku.agent.tools.registry import ToolContext, default_registry
from kesoku.config import get_config

logger = logging.getLogger(__name__)


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
