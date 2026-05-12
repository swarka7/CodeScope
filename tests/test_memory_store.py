from __future__ import annotations

import pytest

from codescope.models.code_chunk import CodeChunk
from codescope.vectorstore.memory_store import MemoryStore, cosine_similarity


def _chunk(name: str) -> CodeChunk:
    return CodeChunk(
        id=name,
        file_path="file.py",
        chunk_type="function",
        name=name,
        parent=None,
        start_line=1,
        end_line=1,
        source_code="pass\n",
        imports=[],
        dependencies=[],
    )


def test_cosine_similarity_basic_cases() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)


def test_cosine_similarity_dimension_mismatch() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 0.0])


def test_memory_store_returns_top_k() -> None:
    store = MemoryStore()
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.7, 0.7],
    ]
    store.add(chunks, embeddings)

    results = store.search([1.0, 0.0], top_k=2)

    assert [r.chunk.name for r in results] == ["a", "c"]
    assert results[0].score >= results[1].score

