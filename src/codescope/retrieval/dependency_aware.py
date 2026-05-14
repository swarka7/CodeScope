from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


@dataclass(frozen=True, slots=True)
class ScoredTraversalChunk:
    chunk: CodeChunk
    score: int
    depth: int


def enrich_with_related(
    *,
    query: str,
    semantic_results: list[SearchResult],
    graph: DependencyGraph,
    max_semantic_sources: int = 2,
    max_related: int = 5,
    min_related_score: int = 2,
    max_depth: int = 2,
    per_hop_limit: int | None = 10,
) -> list[RetrievalResult]:
    enriched: list[RetrievalResult] = [
        RetrievalResult(kind="semantic", chunk=result.chunk, score=result.score)
        for result in semantic_results
    ]

    if max_semantic_sources <= 0 or max_related <= 0 or max_depth <= 0:
        return enriched

    seen_chunk_ids = {result.chunk.id for result in semantic_results}
    allow_test_chunks = _query_mentions_tests(query)
    allow_infra_chunks = _query_mentions_infrastructure(query)

    traversal = _collect_related_chunks(
        query=query,
        semantic_results=semantic_results[:max_semantic_sources],
        graph=graph,
        allow_test_chunks=allow_test_chunks,
        allow_infra_chunks=allow_infra_chunks,
        max_depth=max_depth,
        per_hop_limit=per_hop_limit,
        already_seen_ids=seen_chunk_ids,
    )

    traversal = [item for item in traversal if item.score >= min_related_score]
    traversal.sort(key=_related_sort_key)

    for item in traversal[:max_related]:
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
    elif related_chunk.parent and dependency_name == f"{related_chunk.parent}.{related_chunk.name}":
        score += 1

    if semantic_source.parent and related_chunk.parent == semantic_source.parent:
        score += 1

    if _is_test_chunk(related_chunk) and (not _query_mentions_tests(query)):
        score -= 3

    if _is_infrastructure_chunk(related_chunk) and (not allow_infra_chunks):
        score -= 2

    return score


def score_traversed_related_chunk(
    *,
    query: str,
    semantic_source: CodeChunk,
    hop_source: CodeChunk,
    dependency_name: str,
    related_chunk: CodeChunk,
    allow_infra_chunks: bool,
    depth: int,
) -> int:
    score = score_related_chunk(
        query=query,
        semantic_source=semantic_source,
        dependency_name=dependency_name,
        related_chunk=related_chunk,
        allow_infra_chunks=allow_infra_chunks,
    )

    if related_chunk.file_path != semantic_source.file_path and _same_module(
        related_chunk, semantic_source
    ):
        score += 1

    if related_chunk.file_path != hop_source.file_path and _same_module(related_chunk, hop_source):
        score += 1

    score += max(0, 3 - depth)  # depth=1 -> +2, depth=2 -> +1, deeper -> +0

    return score


def _collect_related_chunks(
    *,
    query: str,
    semantic_results: list[SearchResult],
    graph: DependencyGraph,
    allow_test_chunks: bool,
    allow_infra_chunks: bool,
    max_depth: int,
    per_hop_limit: int | None,
    already_seen_ids: set[str],
) -> list[ScoredTraversalChunk]:
    best_by_id: dict[str, ScoredTraversalChunk] = {}
    ordered_ids: list[str] = []

    for semantic in semantic_results:
        frontier: list[CodeChunk] = [semantic.chunk]
        visited: set[str] = {semantic.chunk.id}

        for depth in range(1, max_depth + 1):
            candidates = _expand_frontier(
                query=query,
                semantic_source=semantic.chunk,
                frontier=frontier,
                graph=graph,
                allow_test_chunks=allow_test_chunks,
                allow_infra_chunks=allow_infra_chunks,
                depth=depth,
                already_seen_ids=already_seen_ids,
                visited=visited,
            )

            if not candidates:
                break

            candidates.sort(key=lambda item: (-item.score, item.chunk.file_path, item.chunk.id))

            if per_hop_limit is not None:
                candidates = candidates[: max(per_hop_limit, 0)]

            frontier = []
            for item in candidates:
                visited.add(item.chunk.id)
                frontier.append(item.chunk)
                _update_best(best_by_id, ordered_ids, item)

    return [best_by_id[chunk_id] for chunk_id in ordered_ids]


def _expand_frontier(
    *,
    query: str,
    semantic_source: CodeChunk,
    frontier: list[CodeChunk],
    graph: DependencyGraph,
    allow_test_chunks: bool,
    allow_infra_chunks: bool,
    depth: int,
    already_seen_ids: set[str],
    visited: set[str],
) -> list[ScoredTraversalChunk]:
    best_by_id: dict[str, ScoredTraversalChunk] = {}

    for hop_source in frontier:
        for dependency_name, related in graph.related_candidates(hop_source):
            if related.id in already_seen_ids:
                continue
            if related.id in visited:
                continue
            if (not allow_test_chunks) and _is_test_chunk(related):
                continue

            score = score_traversed_related_chunk(
                query=query,
                semantic_source=semantic_source,
                hop_source=hop_source,
                dependency_name=dependency_name,
                related_chunk=related,
                allow_infra_chunks=allow_infra_chunks,
                depth=depth,
            )

            item = ScoredTraversalChunk(chunk=related, score=score, depth=depth)
            existing = best_by_id.get(related.id)
            if existing is None or item.score > existing.score:
                best_by_id[related.id] = item

    return list(best_by_id.values())


def _update_best(
    best_by_id: dict[str, ScoredTraversalChunk], ordered_ids: list[str], item: ScoredTraversalChunk
) -> None:
    existing = best_by_id.get(item.chunk.id)
    if existing is None:
        ordered_ids.append(item.chunk.id)
        best_by_id[item.chunk.id] = item
        return

    if item.score > existing.score:
        best_by_id[item.chunk.id] = item
        return

    if item.score == existing.score and item.depth < existing.depth:
        best_by_id[item.chunk.id] = item


def _related_sort_key(item: ScoredTraversalChunk) -> tuple[int, int, str, str, str, str]:
    return (
        -item.score,
        item.depth,
        item.chunk.file_path,
        item.chunk.chunk_type,
        item.chunk.parent or "",
        item.chunk.name,
    )


def _same_module(a: CodeChunk, b: CodeChunk) -> bool:
    return _module_dir(a.file_path) == _module_dir(b.file_path)


def _module_dir(file_path: str) -> str:
    path = file_path.replace("\\", "/")
    try:
        return Path(path).parent.as_posix()
    except (OSError, RuntimeError):
        return path.rsplit("/", 1)[0] if "/" in path else ""


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
