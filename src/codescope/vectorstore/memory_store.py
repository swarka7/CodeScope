from __future__ import annotations

from dataclasses import dataclass

from codescope.models.code_chunk import CodeChunk


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk: CodeChunk
    score: float


class MemoryStore:
    """A minimal in-memory vector store for semantic search."""

    def __init__(self) -> None:
        self._items: list[tuple[CodeChunk, list[float]]] = []

    def add(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self._items.append((chunk, embedding))

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            return []

        scored = [
            SearchResult(chunk=chunk, score=cosine_similarity(query_embedding, embedding))
            for chunk, embedding in self._items
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension")

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / ((norm_a**0.5) * (norm_b**0.5))

