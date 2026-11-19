"""Web search tools for Kesoku AI Agent using Google Search grounding."""

import asyncio
import logging
import os
import threading
import time
import warnings
from collections import OrderedDict
from typing import Any

from google import genai
from google.genai import types

from kesoku.agent.tools.registry import ToolContext, default_registry
from kesoku.config import GeminiConfig, get_config

logger = logging.getLogger(__name__)

# Hard upper bound for one grounded search call (the main model has been observed to take up to ~87 s).
WEB_SEARCH_TIMEOUT_SECONDS = 30.0
# Prefer minimal thinking for lowest latency; automatically falls back to low / None when unsupported.
WEB_SEARCH_THINKING_LEVEL = "minimal"
WEB_SEARCH_FALLBACK_THINKING_LEVEL = "low"
# Cap grounded summary output so the search sub-model returns compact facts instead of a multi-thousand-word essay.
WEB_SEARCH_MAX_OUTPUT_TOKENS = 800
# Short in-process TTL cache to avoid redundant remote grounding calls on retries or repeated follow-ups.
WEB_SEARCH_CACHE_TTL_SECONDS = 600.0
WEB_SEARCH_CACHE_MAX_ENTRIES = 128

WEB_SEARCH_SYSTEM_INSTRUCTION = (
    "You are a concise web search extractor serving an upstream AI assistant. "
    "Use Google Search to answer the query directly with compact bullet points containing "
    "key facts, exact names/commands/package IDs, numbers, and dates. "
    "Match the query's language. Omit greetings, disclaimers, generic background, and filler."
)


def _resolve_search_model(config: GeminiConfig) -> str:
    """Use the dedicated search model when configured, else context model, else main model."""
    return config.search_model_name or config.context_model_name or config.model_name


def _normalize_cache_key(model: str, query: str) -> tuple[str, str]:
    """Build a canonical cache key scoped by `(model, normalized_query)`."""
    return (model, " ".join((query or "").strip().lower().split()))


def _build_thinking_config(config: GeminiConfig, level: str | None = None) -> Any | None:
    """Build a `ThinkingConfig` for the requested thinking level.

    Returns None when thinking is disabled in config (`thinking_level=None` means "model default", typically a
    model without thinking-level support) or the installed SDK has no `ThinkingConfig`.
    """
    if config.thinking_level is None or not hasattr(types, "ThinkingConfig"):
        return None
    target_level = level or WEB_SEARCH_THINKING_LEVEL
    resolved = (
        getattr(types.ThinkingLevel, target_level.upper(), target_level.upper())
        if hasattr(types, "ThinkingLevel")
        else None
    )
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


def _is_minimal_thinking_unsupported_error(err: Exception) -> bool:
    msg = str(err).lower()
    return _is_thinking_unsupported_error(err) and "minimal" in msg


class WebSearchTool:
    """Tool for executing Google Search queries via Gemini API grounding."""

    def __init__(
        self,
        client: genai.Client | None = None,
        cache_ttl_seconds: float = WEB_SEARCH_CACHE_TTL_SECONDS,
        cache_max_entries: int = WEB_SEARCH_CACHE_MAX_ENTRIES,
    ) -> None:
        """Initialize WebSearchTool.

        Args:
            client: Optional pre-configured genai.Client (useful for dependency injection and unit tests).
            cache_ttl_seconds: TTL in seconds for caching successful search results (0 disables caching).
            cache_max_entries: Maximum number of cached queries retained in LRU order.
        """
        self._client = client
        self._client_lock = threading.Lock()
        self._cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self._cache_max_entries = max(0, int(cache_max_entries))
        self._cache: OrderedDict[tuple[str, str], tuple[float, str]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._minimal_unsupported_models: set[str] = set()

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

    def _cache_get(self, key: tuple[str, str]) -> str | None:
        if self._cache_ttl_seconds <= 0 or self._cache_max_entries <= 0:
            return None
        now = time.monotonic()
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if now >= expires_at:
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return value

    def _cache_put(self, key: tuple[str, str], value: str) -> None:
        if self._cache_ttl_seconds <= 0 or self._cache_max_entries <= 0 or not value:
            return
        expires_at = time.monotonic() + self._cache_ttl_seconds
        with self._cache_lock:
            self._cache[key] = (expires_at, value)
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_max_entries:
                self._cache.popitem(last=False)

    def _build_generate_config(self, thinking: Any | None) -> types.GenerateContentConfig:
        kwargs: dict[str, Any] = {
            "tools": [types.Tool(google_search=types.GoogleSearch())],
            "system_instruction": WEB_SEARCH_SYSTEM_INSTRUCTION,
            "max_output_tokens": WEB_SEARCH_MAX_OUTPUT_TOKENS,
        }
        if thinking is not None:
            kwargs["thinking_config"] = thinking
        return types.GenerateContentConfig(**kwargs)

    async def _generate(self, client: genai.Client, config: GeminiConfig, query: str) -> Any:
        model = _resolve_search_model(config)
        initial_level = (
            WEB_SEARCH_FALLBACK_THINKING_LEVEL
            if model in self._minimal_unsupported_models
            else WEB_SEARCH_THINKING_LEVEL
        )
        thinking = _build_thinking_config(config, level=initial_level)
        generate_config = self._build_generate_config(thinking)
        try:
            return await client.aio.models.generate_content(model=model, contents=query, config=generate_config)
        except Exception as e:
            if thinking is None or not _is_thinking_unsupported_error(e):
                raise
            if initial_level == WEB_SEARCH_THINKING_LEVEL and _is_minimal_thinking_unsupported_error(e):
                self._minimal_unsupported_models.add(model)
                logger.info(
                    f"WebSearchTool: model '{model}' rejected minimal thinking ({e}); "
                    f"retrying with {WEB_SEARCH_FALLBACK_THINKING_LEVEL}."
                )
                fallback_thinking = _build_thinking_config(config, level=WEB_SEARCH_FALLBACK_THINKING_LEVEL)
                try:
                    return await client.aio.models.generate_content(
                        model=model,
                        contents=query,
                        config=self._build_generate_config(fallback_thinking),
                    )
                except Exception as e2:
                    if fallback_thinking is None or not _is_thinking_unsupported_error(e2):
                        raise
                    logger.info(
                        f"WebSearchTool: model '{model}' also rejected {WEB_SEARCH_FALLBACK_THINKING_LEVEL} "
                        f"thinking ({e2}); retrying without thinking_config."
                    )
                    return await client.aio.models.generate_content(
                        model=model,
                        contents=query,
                        config=self._build_generate_config(None),
                    )

            logger.info(f"WebSearchTool: model '{model}' rejected thinking_config ({e}); retrying without it.")
            return await client.aio.models.generate_content(
                model=model,
                contents=query,
                config=self._build_generate_config(None),
            )

    async def web_search(self, query: str, context: ToolContext | None = None) -> str:
        """Search the web for current information on a given topic using Google Search grounding.

        Args:
            query: The search query string.
            context: Optional tool execution context.

        Returns:
            Search results summary with grounding sources.
        """
        config = get_config().gemini
        model = _resolve_search_model(config)
        cache_key = _normalize_cache_key(model, query)
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.info(f"Returning cached web search result for query: '{query}' (model={model})")
            return cached

        logger.info(f"Executing web search for query: '{query}' (model={model})")
        try:
            client = self._get_client()
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
            formatted = f"{text_content}\n\nSources:\n{sources_str}"
        else:
            formatted = text_content

        self._cache_put(cache_key, formatted)
        return formatted


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
