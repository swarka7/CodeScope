from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from codescope.graph.dependency_graph import DependencyGraph
from codescope.models.code_chunk import CodeChunk
from codescope.vectorstore.memory_store import SearchResult


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    kind: Literal["semantic", "related"]
    chunk: CodeChunk
    score: float | None


@dataclass(frozen=True, slots=True)
class ScoredRelatedChunk:
    chunk: CodeChunk
    score: int


def enrich_with_related(
    *,
    query: str,
    semantic_results: list[SearchResult],
    graph: DependencyGraph,
    max_semantic_sources: int = 2,
    max_related: int = 5,
    min_related_score: int = 2,
) -> list[RetrievalResult]:
    enriched: list[RetrievalResult] = [
        RetrievalResult(kind="semantic", chunk=result.chunk, score=result.score)
        for result in semantic_results
    ]

    if max_semantic_sources <= 0 or max_related <= 0:
        return enriched

    seen_chunk_ids = {result.chunk.id for result in semantic_results}
    allow_test_chunks = _query_mentions_tests(query)
    allow_infra_chunks = _query_mentions_infrastructure(query)

    best_by_id: dict[str, ScoredRelatedChunk] = {}
    ordered_ids: list[str] = []

    for semantic in semantic_results[:max_semantic_sources]:
        for dependency_name, related in graph.related_candidates(semantic.chunk):
            if related.id in seen_chunk_ids:
                continue
            if (not allow_test_chunks) and _is_test_chunk(related):
                continue

            score = score_related_chunk(
                query=query,
                semantic_source=semantic.chunk,
                dependency_name=dependency_name,
                related_chunk=related,
                allow_infra_chunks=allow_infra_chunks,
            )

            if related.id not in best_by_id:
                ordered_ids.append(related.id)
                best_by_id[related.id] = ScoredRelatedChunk(chunk=related, score=score)
            else:
                existing = best_by_id[related.id]
                if score > existing.score:
                    best_by_id[related.id] = ScoredRelatedChunk(chunk=related, score=score)

    scored = [best_by_id[chunk_id] for chunk_id in ordered_ids]
    scored = [item for item in scored if item.score >= min_related_score]
    scored.sort(key=lambda item: item.score, reverse=True)

    for item in scored[:max_related]:
        if item.chunk.id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(item.chunk.id)
        enriched.append(RetrievalResult(kind="related", chunk=item.chunk, score=None))

    return enriched


def score_related_chunk(
    *,
    query: str,
    semantic_source: CodeChunk,
    dependency_name: str,
    related_chunk: CodeChunk,
    allow_infra_chunks: bool,
) -> int:
    score = 0

    if related_chunk.file_path == semantic_source.file_path:
        score += 3

    if dependency_name == related_chunk.name:
        score += 2

    if semantic_source.parent and related_chunk.parent == semantic_source.parent:
        score += 1

    if _is_test_chunk(related_chunk) and (not _query_mentions_tests(query)):
        score -= 3

    if _is_infrastructure_chunk(related_chunk) and (not allow_infra_chunks):
        score -= 2

    return score


def _is_test_chunk(chunk: CodeChunk) -> bool:
    path = chunk.file_path.replace("\\", "/").lower().strip("/")
    path_wrapped = f"/{path}/"
    if "/tests/" in path_wrapped:
        return True

    file_name = path.rsplit("/", 1)[-1]
    if file_name in {"conftest.py"}:
        return True
    if file_name.startswith("test_"):
        return True
    if file_name.endswith("_test.py"):
        return True

    return False


def _query_mentions_tests(query: str) -> bool:
    keywords = {"test", "tests", "testing", "pytest"}
    tokens: list[str] = []
    current: list[str] = []

    for ch in query.lower():
        if ch.isalnum():
            current.append(ch)
            continue
        if current:
            tokens.append("".join(current))
            current = []

    if current:
        tokens.append("".join(current))

    return any(token in keywords for token in tokens)


def _is_infrastructure_chunk(chunk: CodeChunk) -> bool:
    path = chunk.file_path.replace("\\", "/").lower().strip("/")
    path_wrapped = f"/{path}/"
    return (
        "/codescope/indexing/" in path_wrapped
        or "/codescope/vectorstore/" in path_wrapped
        or "/codescope/embeddings/" in path_wrapped
        or "/codescope/parser/" in path_wrapped
    )


def _query_mentions_infrastructure(query: str) -> bool:
    keywords = {
        "index",
        "indexing",
        "indexed",
        "embed",
        "embedding",
        "embeddings",
        "vector",
        "vectors",
        "chunk",
        "chunks",
        "metadata",
        "store",
    }
    tokens: list[str] = []
    current: list[str] = []

    for ch in query.lower():
        if ch.isalnum():
            current.append(ch)
            continue
        if current:
            tokens.append("".join(current))
            current = []

    if current:
        tokens.append("".join(current))

    return any(token in keywords for token in tokens)
