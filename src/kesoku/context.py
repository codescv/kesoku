"""Runtime context container for Kesoku, encapsulating all configuration and persistence interfaces.

This avoids global mutable state and supports dependency injection.
"""

from kesoku.agent.llm import BaseLLM, get_llm
from kesoku.agent.tools import ActiveJobsRegistry, ToolRegistry, default_registry
from kesoku.config import KesokuConfig, get_config
from kesoku.db import DatabaseManager


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
        self.db: DatabaseManager = db or DatabaseManager(self.config.workspace.db_path)
        self.tool_registry: ToolRegistry = tool_registry or default_registry
        self._llm = llm
        self.active_jobs = ActiveJobsRegistry()

    @property
    def config(self) -> KesokuConfig:
        """Get the configuration container. If None was supplied at init, dynamically queries get_config()."""
        if self._config is not None:
            return self._config
        return get_config()

    def get_llm(self, provider: str | None = None) -> BaseLLM:
        """Dynamically resolve and build an LLM provider instance at execution time.

        Args:
            provider: Optional provider name (e.g. 'gemini', 'claude'). If None, resolves from config.

        Returns:
            An instance of BaseLLM.
        """
        if self._llm is not None and provider is None:
            return self._llm

        target_provider = provider or self.config.agent.llm
        return get_llm(provider=target_provider, config=self.config)
