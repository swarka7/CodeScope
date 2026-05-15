from __future__ import annotations

import pytest

from codescope.models.code_chunk import CodeChunk
from codescope.vectorstore.memory_store import MemoryStore, cosine_similarity


def _chunk(
    name: str,
    *,
    id: str | None = None,
    file_path: str = "file.py",
    start_line: int = 1,
) -> CodeChunk:
    return CodeChunk(
        id=id or name,
        file_path=file_path,
        chunk_type="function",
        name=name,
        parent=None,
        start_line=start_line,
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


def test_memory_store_orders_tied_scores_deterministically() -> None:
    store = MemoryStore()
    chunks = [
        _chunk("late", id="c", file_path="b.py", start_line=20),
        _chunk("first", id="b", file_path="a.py", start_line=10),
        _chunk("second", id="a", file_path="a.py", start_line=10),
        _chunk("middle", id="d", file_path="a.py", start_line=20),
    ]
    store.add(chunks, [[1.0, 0.0] for _ in chunks])

    results = store.search([1.0, 0.0], top_k=10)

    assert [r.chunk.name for r in results] == ["second", "first", "middle", "late"]
