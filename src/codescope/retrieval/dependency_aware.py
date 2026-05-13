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


def enrich_with_related(
    *,
    query: str,
    semantic_results: list[SearchResult],
    graph: DependencyGraph,
    max_semantic_sources: int = 2,
    max_related: int = 5,
) -> list[RetrievalResult]:
    enriched: list[RetrievalResult] = [
        RetrievalResult(kind="semantic", chunk=result.chunk, score=result.score)
        for result in semantic_results
    ]

    if max_semantic_sources <= 0 or max_related <= 0:
        return enriched

    seen_chunk_ids = {result.chunk.id for result in semantic_results}
    allow_test_chunks = _query_mentions_tests(query)
    related_added = 0

    for semantic in semantic_results[:max_semantic_sources]:
        if related_added >= max_related:
            break

        same_file: list[CodeChunk] = []
        other_files: list[CodeChunk] = []

        for related in graph.related_chunks(semantic.chunk):
            if related.id in seen_chunk_ids:
                continue
            if (not allow_test_chunks) and _is_test_chunk(related):
                continue

            if related.file_path == semantic.chunk.file_path:
                same_file.append(related)
            else:
                other_files.append(related)

        for related in [*same_file, *other_files]:
            if related_added >= max_related:
                break
            if related.id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(related.id)
            related_added += 1
            enriched.append(RetrievalResult(kind="related", chunk=related, score=None))

    return enriched


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
