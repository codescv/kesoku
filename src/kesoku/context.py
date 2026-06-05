"""Runtime context container for Kesoku, encapsulating all configuration and persistence interfaces.

This avoids global mutable state and supports dependency injection.
"""

import os

from openlcm import LCMEngine

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
        self.active_jobs = ActiveJobsRegistry()

        # Resolve OpenLCM SQLite DB path relative to kesoku.db directory
        self.lcm_db_path = os.path.join(os.path.dirname(self.config.workspace.db_path), "lcm.db")
        self._lcm_engines: dict[str, LCMEngine] = {}

    def get_lcm_engine(self, session_id: str, context_length: int = 1048576) -> LCMEngine:
        """Create and bind a session-specific LCMEngine instance to prevent concurrent clobbering."""
        if session_id not in self._lcm_engines:
            async def lcm_summarize_fn(prompt: str, max_tokens: int) -> str:
                model_client = self.get_llm()
                res = await model_client.generate(prompt=prompt)
                return res.content

            engine = LCMEngine(
                summarize_fn=lcm_summarize_fn,
                db_path=self.lcm_db_path,
            )
            engine.bind_session(session_id=session_id, context_length=context_length)
            self._lcm_engines[session_id] = engine
        else:
            engine = self._lcm_engines[session_id]
            if context_length > 0 and context_length != engine.context_length:
                engine.set_context_length(context_length)

        return engine

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
