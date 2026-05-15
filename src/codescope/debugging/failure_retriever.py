from __future__ import annotations

from pathlib import Path

from codescope.debugging.call_graph_context import expand_failure_call_path_context
from codescope.debugging.failure_scoring import (
    FILE_LINE_RE,
    MAX_SOURCE_HINTS,
    MAX_TRACEBACK_SYMBOLS,
    PYTEST_LOCATION_RE,
    FailureSignals,
    extract_message_symbols,
    extract_traceback_hints,
    failure_result_sort_key,
    hint_to_normalized_path,
    rerank_semantic_results_for_failure,
    score_failure_chunk,
    select_semantic_results_for_failure,
)
from codescope.embeddings.embedder import Embedder
from codescope.graph.dependency_graph import DependencyGraph
from codescope.indexing.index_compatibility import check_index_compatibility
from codescope.indexing.index_store import IndexStore
from codescope.models.code_chunk import CodeChunk
from codescope.models.test_failure import TestFailure
from codescope.retrieval.dependency_aware import RetrievalResult, enrich_with_related
from codescope.vectorstore.memory_store import MemoryStore, SearchResult


class FailureRetriever:
    """Retrieves likely relevant code chunks for a given test failure."""

    _MAX_MESSAGE_CHARS = 400
    _MAX_TRACEBACK_EXCERPT_CHARS = 800
    _MAX_TRACEBACK_EXCERPT_LINES = 14
    _MAX_TRACEBACK_SYMBOLS = MAX_TRACEBACK_SYMBOLS
    _MAX_SOURCE_HINTS = MAX_SOURCE_HINTS

    def __init__(
        self,
        repo_path: Path,
        *,
        index_store: IndexStore | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._repo_path = Path(repo_path)
        self._index_store = index_store or IndexStore(self._repo_path)
        self._embedder = embedder or Embedder()

    @staticmethod
    def build_failure_query(failure: TestFailure) -> str:
        """Build a high-signal, deterministic retrieval query from a test failure."""
        test_name = failure.test_name.strip()
        error_type = (failure.error_type or "").strip()
        message = failure.message.strip()
        traceback_text = failure.traceback.strip()

        if len(message) > FailureRetriever._MAX_MESSAGE_CHARS:
            message = message[: FailureRetriever._MAX_MESSAGE_CHARS].rstrip() + "..."

        symbols, source_hints = extract_traceback_hints(
            traceback_text,
            max_symbols=FailureRetriever._MAX_TRACEBACK_SYMBOLS,
            max_source_hints=FailureRetriever._MAX_SOURCE_HINTS,
        )

        if failure.file_path.strip():
            location = failure.file_path.strip()
            if failure.line_number is not None:
                location = f"{location}:{failure.line_number}"
            source_hints = FailureRetriever._merge_unique([location], source_hints)

        excerpt = FailureRetriever._build_traceback_excerpt(
            traceback_text,
            max_lines=FailureRetriever._MAX_TRACEBACK_EXCERPT_LINES,
            max_chars=FailureRetriever._MAX_TRACEBACK_EXCERPT_CHARS,
        )

        parts: list[str] = []
        parts.append("Test:")
        parts.append(test_name or "<unknown>")

        parts.append("")
        parts.append("Error:")
        parts.append(error_type or "<unknown>")

        parts.append("")
        parts.append("Message:")
        parts.append(message or "<none>")

        parts.append("")
        parts.append("Traceback symbols:")
        parts.extend(symbols if symbols else ["<none>"])

        parts.append("")
        parts.append("Source hints:")
        parts.extend(source_hints if source_hints else ["<none>"])

        if excerpt:
            parts.append("")
            parts.append("Traceback excerpt:")
            parts.append(excerpt)

        return "\n".join(parts).strip() + "\n"

    @staticmethod
    def build_query(failure: TestFailure) -> str:
        return FailureRetriever.build_failure_query(failure)

    @staticmethod
    def _merge_unique(left: list[str], right: list[str]) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for raw in [*left, *right]:
            value = raw.strip()
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(value)
        return merged

    @staticmethod
    def _build_traceback_excerpt(traceback_text: str, *, max_lines: int, max_chars: int) -> str:
        if not traceback_text:
            return ""

        lines = [line.rstrip() for line in traceback_text.splitlines() if line.strip()]
        if not lines:
            return ""

        filtered: list[str] = []
        for line in lines:
            stripped = line.strip()
            if FILE_LINE_RE.search(stripped):
                continue
            if PYTEST_LOCATION_RE.search(stripped):
                continue
            filtered.append(line)

        if not filtered:
            return ""

        tail = filtered[-max_lines:]
        excerpt = "\n".join(tail)
        if len(excerpt) <= max_chars:
            return excerpt

        truncated = excerpt[: max_chars].rstrip()
        return truncated + "..."

    @staticmethod
    def _extract_traceback_hints(
        traceback_text: str, *, max_symbols: int, max_source_hints: int
    ) -> tuple[list[str], list[str]]:
        return extract_traceback_hints(
            traceback_text,
            max_symbols=max_symbols,
            max_source_hints=max_source_hints,
        )

    @staticmethod
    def _hint_to_normalized_path(hint: str) -> str:
        return hint_to_normalized_path(hint)

    @staticmethod
    def _extract_message_symbols(message: str) -> set[str]:
        return extract_message_symbols(message)

    @staticmethod
    def score_failure_chunk(
        *,
        chunk: CodeChunk,
        base_score: float,
        failure: TestFailure,
        signals: FailureSignals | None = None,
    ) -> float:
        return score_failure_chunk(
            chunk=chunk,
            base_score=base_score,
            failure=failure,
            signals=signals,
        )

    @staticmethod
    def rerank_semantic_results_for_failure(
        *, failure: TestFailure, semantic_results: list[SearchResult]
    ) -> list[SearchResult]:
        return rerank_semantic_results_for_failure(
            failure=failure,
            semantic_results=semantic_results,
        )

    def retrieve(self, failure: TestFailure, *, top_k: int = 5) -> list[RetrievalResult]:
        compatibility = check_index_compatibility(
            index_store=self._index_store,
            embedding_model_name=self._embedder.model_name,
        )
        if not compatibility.compatible:
            raise ValueError(compatibility.message)

        chunks, embeddings, _metadata = self._index_store.load()

        query = self.build_failure_query(failure)
        query_embedding = self._embedder.embed_text(query)

        store = MemoryStore()
        store.add(chunks, embeddings)
        candidate_k = max(top_k, top_k * 8, 40)
        semantic_candidates = store.search(query_embedding, top_k=candidate_k)
        semantic_ranked = self.rerank_semantic_results_for_failure(
            failure=failure, semantic_results=semantic_candidates
        )
        graph = DependencyGraph(chunks)

        call_path_context = expand_failure_call_path_context(
            failure=failure,
            seed_results=semantic_ranked,
            graph=graph,
        )
        if call_path_context:
            semantic_ranked = _merge_search_results(semantic_ranked, call_path_context)

        semantic = self.select_semantic_results_for_failure(
            failure=failure,
            ranked_results=semantic_ranked,
            top_k=top_k,
        )
        return enrich_with_related(query=query, semantic_results=semantic, graph=graph)

    @staticmethod
    def select_semantic_results_for_failure(
        *, failure: TestFailure, ranked_results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        return select_semantic_results_for_failure(
            failure=failure,
            ranked_results=ranked_results,
            top_k=top_k,
        )


def _merge_search_results(
    semantic_results: list[SearchResult], call_path_results: list[SearchResult]
) -> list[SearchResult]:
    best_by_id: dict[str, SearchResult] = {result.chunk.id: result for result in semantic_results}

    for candidate in call_path_results:
        existing = best_by_id.get(candidate.chunk.id)
        if existing is None:
            best_by_id[candidate.chunk.id] = candidate
            continue

        reasons = _merge_reasons(existing.reasons, candidate.reasons)
        if candidate.score > existing.score:
            best_by_id[candidate.chunk.id] = SearchResult(
                chunk=candidate.chunk,
                score=candidate.score,
                reasons=reasons,
            )
        elif reasons != existing.reasons:
            best_by_id[candidate.chunk.id] = SearchResult(
                chunk=existing.chunk,
                score=existing.score,
                reasons=reasons,
            )

    merged = list(best_by_id.values())
    merged.sort(key=lambda result: failure_result_sort_key(result.score, result))
    return merged


def _merge_reasons(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    merged: list[str] = []
    for reason in [*left, *right]:
        key = reason.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(reason)
    return tuple(merged)
