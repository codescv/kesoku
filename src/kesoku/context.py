"""Runtime context container for Kesoku, encapsulating all configuration and persistence interfaces.

This avoids global mutable state and supports dependency injection.
"""

import asyncio
import os

from kesoku.agent.llm import BaseLLM, get_llm
from kesoku.agent.tools import ActiveJobsRegistry, ToolRegistry, default_registry
from kesoku.config import KesokuConfig, get_config
from kesoku.db import AsyncDatabaseManager, DatabaseManager
from kesoku.db.embeddings import EmbeddingStore


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

        # Resolve SQLite DB path for embeddings relative to kesoku.db directory
        self.embedding_db_path = os.path.join(os.path.dirname(self.config.workspace.db_path), "context_embeddings.db")

        # Ensure LiteLLM can locate the correct GCP project and location for Vertex AI
        if self.config.agent.embedding_model.startswith("vertex_ai/"):
            if self.config.gemini.project_id:
                os.environ["VERTEX_PROJECT"] = self.config.gemini.project_id
            if self.config.gemini.location:
                os.environ["VERTEX_LOCATION"] = self.config.gemini.location

        # Initialize global EmbeddingStore for general indexing
        self.embedding_store = EmbeddingStore(
            db_path=self.embedding_db_path,
            embedding_model=self.config.agent.embedding_model,
        )

        if self.embedding_store.enabled:
            self._register_embedding_listeners()

    def _register_embedding_listeners(self) -> None:
        """Register listeners on database manager to update embedding index in real-time."""
        from kesoku.db.models import Message

        async def on_message_saved(msg: Message) -> None:
            if (msg.role in ("user", "assistant")) and (msg.type == "text"):
                # Query session to get its role_name for role-based index separation
                session = await self.db.get_session(msg.session_id)
                session_role = (session.role_name if session else None) or "default"
                content_type = f"kesoku_message:{session_role}"
                asyncio.create_task(
                    self.embedding_store.embed(
                        content_type=content_type,
                        content_id=msg.id,
                        text=msg.content,
                    )
                )

        async def on_memory_upserted(category: str, key: str, title: str, content: str, role: str) -> None:
            content_type = f"kesoku_memory:{role}"
            content_id = f"{category}:{key}"
            text_to_index = f"{key}: {title}\n{content}"
            asyncio.create_task(
                self.embedding_store.embed(
                    content_type=content_type,
                    content_id=content_id,
                    text=text_to_index,
                )
            )

        async def on_memory_deleted(category: str, key: str, role: str) -> None:
            content_type = f"kesoku_memory:{role}"
            content_id = f"{category}:{key}"
            # delete is synchronous but run in thread lock
            self.embedding_store.delete(content_type=content_type, content_id=content_id)

        if hasattr(self.db, "register_on_message_saved"):
            self.db.register_on_message_saved(on_message_saved)
        if hasattr(self.db, "register_on_memory_upserted"):
            self.db.register_on_memory_upserted(on_memory_upserted)
        if hasattr(self.db, "register_on_memory_deleted"):
            self.db.register_on_memory_deleted(on_memory_deleted)

    @property
    def config(self) -> KesokuConfig:
        """Get the configuration container. If None was supplied at init, dynamically queries get_config()."""
        if self._config is not None:
            return self._config
        return get_config()

    def get_llm(self, provider: str | None = None, use_context_compression: bool = False) -> BaseLLM:
        """Dynamically resolve and build an LLM provider instance at execution time.

        Args:
            provider: Optional provider name (e.g. 'gemini', 'claude'). If None, resolves from config.
            use_context_compression: Whether to resolve using context compression settings.

        Returns:
            An instance of BaseLLM.
        """
        if self._llm is not None and provider is None:
            return self._llm

        target_provider = provider or (
            self.config.agent.context_llm
            if use_context_compression and self.config.agent.context_llm
            else self.config.agent.llm
        )
        return get_llm(provider=target_provider, config=self.config, use_context_compression=use_context_compression)
