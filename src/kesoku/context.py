"""Runtime context container for Kesoku, encapsulating all configuration and persistence interfaces.

This avoids global mutable state and supports dependency injection.
"""

import asyncio
import logging
import os
from typing import Any

from openlcm import LCMEngine
from openlcm.core.embeddings import EmbeddingStore

from kesoku.agent.llm import BaseLLM, get_llm
from kesoku.agent.tools import ActiveJobsRegistry, ToolRegistry, default_registry
from kesoku.config import KesokuConfig, get_config
from kesoku.db import AsyncDatabaseManager, DatabaseManager


def _apply_embedding_monkey_patch() -> None:
    if getattr(EmbeddingStore, "_patched_by_kesoku", False):
        return

    def _patched_init_db(self: EmbeddingStore) -> None:
        import logging
        import sqlite3

        from openlcm.core.db_bootstrap import run_versioned_migrations

        try:
            import sqlite_vec

            self._conn = sqlite3.connect(str(self.db_path), timeout=5.0, check_same_thread=False)
            try:
                self._conn.enable_load_extension(True)
            except AttributeError:
                pass
            self._conn.execute("PRAGMA journal_mode=WAL")
            sqlite_vec.load(self._conn)

            # Ensure tables are bootstrapped
            run_versioned_migrations(self._conn)
            self._conn.commit()

            self._enabled = True
        except Exception as exc:
            logging.getLogger(__name__).debug("EmbeddingStore disabled: %s", exc)
            self._enabled = False
            self._conn = None

    EmbeddingStore._init_db = _patched_init_db

    async def _patched_get_embedding(
        self: EmbeddingStore, text: str, task_type: str | None = None
    ) -> list[float] | None:
        if not self._enabled or not text:
            return None
        try:
            import litellm

            actual_task_type = task_type or "RETRIEVAL_DOCUMENT"
            kwargs = {}
            if self.embedding_model.startswith("vertex_ai/"):
                kwargs["task_type"] = actual_task_type
            resp = await litellm.aembedding(model=self.embedding_model, input=[text[:8000]], **kwargs)
            vec = resp.data[0]["embedding"]
            if self._dim == 0:
                self._dim = len(vec)
            return vec
        except Exception as exc:
            logging.getLogger(__name__).debug("Embedding call failed: %s", exc)
            return None

    EmbeddingStore._get_embedding = _patched_get_embedding

    async def _patched_search(
        self: EmbeddingStore,
        query_text: str,
        *,
        content_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if not self._enabled or not self._conn or not query_text:
            return []
        query_vec = await self._get_embedding(query_text, task_type="RETRIEVAL_QUERY")
        if query_vec is None:
            return []

        import logging
        import struct

        query_blob = struct.pack(f"{len(query_vec)}f", *query_vec)

        try:
            where = "AND content_type = ?" if content_type else ""
            args: list[Any] = [query_blob]
            if content_type:
                args.append(content_type)
            args.append(limit)

            rows = self._conn.execute(
                f"""
                SELECT content_type, content_id,
                       vec_distance_cosine(embedding, ?) AS distance
                FROM lcm_embeddings
                WHERE 1=1 {where}
                ORDER BY distance ASC
                LIMIT ?
                """,
                args,
            ).fetchall()
            return [
                {
                    "content_type": r[0],
                    "content_id": r[1],
                    "score": round(1.0 - float(r[2]), 4),
                    "distance": round(float(r[2]), 4),
                }
                for r in rows
            ]
        except Exception as exc:
            logging.getLogger(__name__).debug("Embedding search failed: %s", exc)
            return []

    EmbeddingStore.search = _patched_search
    EmbeddingStore._patched_by_kesoku = True


_apply_embedding_monkey_patch()


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

        # Initialize global EmbeddingStore for general indexing
        self.embedding_store = EmbeddingStore(
            db_path=self.lcm_db_path,
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

    def get_lcm_engine(self, session_id: str, context_length: int = 1048576) -> LCMEngine:
        """Create and bind a session-specific LCMEngine instance to prevent concurrent clobbering."""
        if session_id not in self._lcm_engines:

            async def lcm_summarize_fn(prompt: str, max_tokens: int) -> str:
                model_client = self.get_llm(use_lcm=True)
                res = await model_client.generate(prompt=prompt)
                return res.content

            engine = LCMEngine(
                summarize_fn=lcm_summarize_fn,
                db_path=self.lcm_db_path,
                embedding_model=self.config.agent.embedding_model,
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

    def get_llm(self, provider: str | None = None, use_lcm: bool = False) -> BaseLLM:
        """Dynamically resolve and build an LLM provider instance at execution time.

        Args:
            provider: Optional provider name (e.g. 'gemini', 'claude'). If None, resolves from config.
            use_lcm: Whether to resolve using LCM settings.

        Returns:
            An instance of BaseLLM.
        """
        if self._llm is not None and provider is None:
            return self._llm

        target_provider = provider or (
            self.config.agent.lcm_llm if use_lcm and self.config.agent.lcm_llm else self.config.agent.llm
        )
        return get_llm(provider=target_provider, config=self.config, use_lcm=use_lcm)
