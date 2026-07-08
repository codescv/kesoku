"""Embedding generation utilities for Kesoku using fastembed."""

import array
import logging

import numpy as np
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

_model_instance: TextEmbedding | None = None


def get_embedding_model() -> TextEmbedding:
    """Lazy load and return the TextEmbedding model instance."""
    global _model_instance
    if _model_instance is None:
        logger.info("Initializing fastembed TextEmbedding model...")
        _model_instance = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        logger.info("fastembed TextEmbedding model initialized.")
    return _model_instance


def get_embedding(text: str) -> list[float]:
    """Generate a single text embedding.

    Args:
        text: Input string.

    Returns:
        A list of float numbers representing the embedding.
    """
    model = get_embedding_model()
    embeddings = list(model.embed([text]))
    return list(embeddings[0])


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts.

    Args:
        texts: Input list of strings.

    Returns:
        List of float lists.
    """
    model = get_embedding_model()
    embeddings = list(model.embed(texts))
    return [list(emb) for emb in embeddings]


def vector_to_bytes(vector: list[float]) -> bytes:
    """Convert float list vector to raw bytes for SQL BLOB storage.

    Args:
        vector: A list of float numbers.

    Returns:
        Raw bytes representing the vector.
    """
    return array.array("f", vector).tobytes()


def bytes_to_vector(data: bytes) -> list[float]:
    """Convert raw bytes back to float list vector.

    Args:
        data: Raw bytes of the vector.

    Returns:
        A list of floats.
    """
    return list(array.array("f", data))


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        v1: First vector.
        v2: Second vector.

    Returns:
        Cosine similarity score as float.
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))
