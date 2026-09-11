"""Tests for the batched cosine-similarity scoring used by semantic search."""

import array

import numpy as np
import pytest

from kesoku.db.manager import _batch_semantic_scores
from kesoku.utils import embedding as embedding_utils


class FakeRow:
    """Minimal stand-in for a sqlite3.Row exposing a chunk_embedding column."""

    def __init__(self, blob: bytes | None) -> None:
        """Store the embedding blob.

        Args:
            blob: Raw float32 embedding bytes, or None.
        """
        self._blob = blob

    def __getitem__(self, key: str) -> bytes | None:
        """Return the column value.

        Args:
            key: Column name; only 'chunk_embedding' is supported.

        Returns:
            The stored blob.

        Raises:
            KeyError: If an unknown column is requested.
        """
        if key != "chunk_embedding":
            raise KeyError(key)
        return self._blob


def _blob(vector: list[float]) -> bytes:
    """Encode a vector the same way the database stores it.

    Args:
        vector: Float values to encode.

    Returns:
        Raw float32 bytes.
    """
    return array.array("f", vector).tobytes()


def test_batch_scores_match_scalar_cosine() -> None:
    """Batched scoring must agree with the original per-row cosine implementation."""
    query = [1.0, 0.0, 2.0, -1.0]
    vectors = [
        [1.0, 0.0, 2.0, -1.0],  # identical
        [-1.0, 0.0, -2.0, 1.0],  # opposite
        [0.0, 1.0, 0.0, 0.0],  # orthogonal
        [0.5, 0.25, 1.0, -0.5],
    ]
    rows = [FakeRow(_blob(v)) for v in vectors]

    query_vec = np.asarray(query, dtype=np.float32)
    scores = _batch_semantic_scores(rows, query_vec, float(np.linalg.norm(query_vec)))

    expected = [embedding_utils.cosine_similarity(query, v) for v in vectors]
    for got, want in zip(scores, expected, strict=True):
        assert got == pytest.approx(want, abs=1e-6)


def test_batch_scores_skip_unscoreable_rows() -> None:
    """Missing, wrong-dimension and zero embeddings score 0.0 without failing the batch."""
    query = [1.0, 0.0, 2.0, -1.0]
    rows = [
        FakeRow(None),
        FakeRow(_blob([1.0, 2.0])),  # wrong dimensionality
        FakeRow(_blob([0.0, 0.0, 0.0, 0.0])),  # zero vector
        FakeRow(_blob(query)),
    ]

    query_vec = np.asarray(query, dtype=np.float32)
    scores = _batch_semantic_scores(rows, query_vec, float(np.linalg.norm(query_vec)))

    assert scores[0] == 0.0
    assert scores[1] == 0.0
    assert scores[2] == 0.0
    assert scores[3] == pytest.approx(1.0, abs=1e-6)


def test_batch_scores_without_query_vector() -> None:
    """When the query embedding is unavailable every row scores 0.0."""
    rows = [FakeRow(_blob([1.0, 2.0, 3.0])), FakeRow(None)]

    assert _batch_semantic_scores(rows, None, 0.0) == [0.0, 0.0]
    assert _batch_semantic_scores([], None, 0.0) == []
