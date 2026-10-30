"""Runtime context container for Kesoku, encapsulating all configuration and persistence interfaces.

This avoids global mutable state and supports dependency injection.
"""


from typing import Any

from kesoku.agent.llm import BaseLLM, get_llm
from kesoku.agent.tools import ActiveJobsRegistry, ToolRegistry, default_registry
from kesoku.config import KesokuConfig, get_config
from kesoku.db import AsyncDatabaseManager, DatabaseManager


class KesokuContext:
    """Encapsulates instance-specific settings, database adapters, and tool registries."""

    def __init__(
        self,
        config: KesokuConfig | None = None,
        db: DatabaseManager | None = None,
        tool_registry: ToolRegistry | None = None,
        llm: BaseLLM | None = None,
    ) -> None:
        """Initialize the KesokuContext container.

        Args:
            config: Root configuration instance. If None, defaults to lazily getting global configuration.
            db: Database persistence manager instance. If None, initialized from db_path in workspace settings.
            tool_registry: Tool registry. If None, defaults to default_registry.
            llm: Optional LLM client instance (useful for mock injection in tests).
        """
        self._config = config
        self.sync_db: DatabaseManager = db or DatabaseManager(self.config.workspace.db_path)
        self.db: AsyncDatabaseManager = AsyncDatabaseManager(self.sync_db)
        self.tool_registry: ToolRegistry = tool_registry or default_registry
        self._llm = llm
        self._llm_cache: dict[tuple[str, str | None, str | None, bool], BaseLLM] = {}
        self._llm_cache_cfg: KesokuConfig | None = None
        self.active_jobs = ActiveJobsRegistry()

    @property
    def config(self) -> KesokuConfig:
        """Get the configuration container. If None was supplied at init, dynamically queries get_config()."""
        if self._config is not None:
            return self._config
        return get_config()

    def get_llm(
        self,
        provider: str | None = None,
        use_context_compression: bool = False,
        model_name: str | None = None,
        thinking_level: str | None = None,
    ) -> BaseLLM:
        """Dynamically resolve and build (or reuse cached) an LLM provider instance at execution time.

        Args:
            provider: Optional provider name (e.g. 'gemini', 'claude'). If None, resolves from config.
            use_context_compression: Whether to resolve using context compression settings.
            model_name: Optional model identifier override for the resolved provider.
            thinking_level: Optional thinking/reasoning level override for the resolved provider.

        Returns:
            An instance of BaseLLM.
        """
        if self._llm is not None and provider is None and model_name is None and thinking_level is None:
            return self._llm

        cfg = self.config
        if cfg is not self._llm_cache_cfg:
            self._llm_cache.clear()
            self._llm_cache_cfg = cfg

        target_provider = provider or (
            cfg.agent.context_llm
            if use_context_compression and cfg.agent.context_llm
            else cfg.agent.llm
        )
        cache_key = (target_provider.lower(), model_name, thinking_level, use_context_compression)
        cached = self._llm_cache.get(cache_key)
        if cached is not None:
            return cached

        llm_kwargs: dict[str, Any] = {
            "provider": target_provider,
            "config": cfg,
            "use_context_compression": use_context_compression,
        }
        if model_name is not None:
            llm_kwargs["model_name"] = model_name
        if thinking_level is not None:
            llm_kwargs["thinking_level"] = thinking_level

        instance = get_llm(**llm_kwargs)
        self._llm_cache[cache_key] = instance
        return instance
