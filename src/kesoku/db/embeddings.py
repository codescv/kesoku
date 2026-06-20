"""Native SQLite vector embedding store for Kesoku, utilizing sqlite-vec and litellm."""

import logging
import sqlite3
import struct
from typing import Any

import litellm
import sqlite_vec

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """Manages generation, storage, and semantic querying of vector embeddings."""

    def __init__(self, db_path: str, embedding_model: str) -> None:
        """Initialize the EmbeddingStore instance.

        Args:
            db_path: Path to the SQLite database file to use for vector storage.
            embedding_model: The model name to pass to litellm.aembedding
                (e.g. 'vertex_ai/text-multilingual-embedding-002').
        """
        self.db_path = db_path
        self.embedding_model = embedding_model
        self._conn: sqlite3.Connection | None = None
        self.enabled = False
        self._init_db()

    def _init_db(self) -> None:
        """Connect to SQLite database, load sqlite-vec extension, and bootstrap tables."""
        try:
            self._conn = sqlite3.connect(str(self.db_path), timeout=5.0, check_same_thread=False)
            try:
                self._conn.enable_load_extension(True)
            except AttributeError:
                pass
            self._conn.execute("PRAGMA journal_mode=WAL")
            sqlite_vec.load(self._conn)

            # Ensure tables are bootstrapped
            with self._conn:
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS lcm_embeddings (
                        content_type TEXT NOT NULL,
                        content_id TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        PRIMARY KEY (content_type, content_id)
                    );
                    """
                )
            self.enabled = True
            logger.info("EmbeddingStore initialized successfully.")
        except Exception as exc:
            logger.debug("EmbeddingStore disabled or failed to initialize: %s", exc)
            self.enabled = False
            self._conn = None

    async def _get_embedding(self, text: str, task_type: str | None = None) -> list[float] | None:
        """Query LLM provider asynchronously for text vector embedding."""
        if not self.enabled or not text:
            return None
        try:
            actual_task_type = task_type or "RETRIEVAL_DOCUMENT"
            kwargs = {}
            if self.embedding_model.startswith("vertex_ai/"):
                kwargs["task_type"] = actual_task_type

            resp = await litellm.aembedding(
                model=self.embedding_model,
                input=[text[:8000]],
                **kwargs
            )
            vec = resp.data[0]["embedding"]
            return vec
        except Exception as exc:
            logger.debug("Embedding API call failed: %s", exc)
            return None

    async def embed(self, content_type: str, content_id: str, text: str) -> None:
        """Generate and save embedding for a given content item."""
        if not self.enabled or not self._conn or not text:
            return
        vec = await self._get_embedding(text)
        if vec is None:
            return

        vec_blob = struct.pack(f"{len(vec)}f", *vec)
        try:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO lcm_embeddings (content_type, content_id, embedding)
                    VALUES (?, ?, ?)
                    """,
                    (content_type, content_id, vec_blob),
                )
        except Exception as exc:
            logger.debug("Failed to insert embedding into database: %s", exc)

    def delete(self, content_type: str, content_id: str) -> None:
        """Remove embedding from storage by content coordinates."""
        if not self.enabled or not self._conn:
            return
        try:
            with self._conn:
                self._conn.execute(
                    "DELETE FROM lcm_embeddings WHERE content_type = ? AND content_id = ?",
                    (content_type, content_id),
                )
        except Exception as exc:
            logger.debug("Failed to delete embedding from database: %s", exc)

    async def search(
        self,
        query_text: str,
        *,
        content_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Perform cosine similarity search using sqlite-vec against stored embeddings."""
        if not self.enabled or not self._conn or not query_text:
            return []
        query_vec = await self._get_embedding(query_text, task_type="RETRIEVAL_QUERY")
        if query_vec is None:
            return []

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
            logger.debug("Embedding search failed: %s", exc)
            return []
