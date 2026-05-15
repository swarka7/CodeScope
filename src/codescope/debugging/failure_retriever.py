from __future__ import annotations

import re
from pathlib import Path

from codescope.debugging.call_graph_context import expand_failure_call_path_context
from codescope.debugging.failure_signals import (
    VALIDATION_WORDS,
    FailureSignals,
    calls_relevant_validation_helper,
    calls_validation_helper,
    chunk_signal_text,
    chunk_signal_tokens,
    contains_expected_exception,
    contains_raise_statement,
    defines_expected_exception,
    extract_failure_signals,
    has_validation_name,
    raises_expected_exception,
    relevant_exception_symbol_matches,
)
from codescope.embeddings.embedder import Embedder
from codescope.graph.dependency_graph import DependencyGraph
from codescope.indexing.index_compatibility import check_index_compatibility
from codescope.indexing.index_store import IndexStore
from codescope.models.code_chunk import CodeChunk
from codescope.models.test_failure import TestFailure
from codescope.retrieval.dependency_aware import RetrievalResult, enrich_with_related
from codescope.utils.path_utils import is_test_path, normalize_path
from codescope.vectorstore.memory_store import MemoryStore, SearchResult


class FailureRetriever:
    """Retrieves likely relevant code chunks for a given test failure."""

    _MAX_MESSAGE_CHARS = 400
    _MAX_TRACEBACK_EXCERPT_CHARS = 800
    _MAX_TRACEBACK_EXCERPT_LINES = 14
    _MAX_TRACEBACK_SYMBOLS = 12
    _MAX_SOURCE_HINTS = 10

    _FILE_LINE_RE = re.compile(
        r"""File\s+["'](?P<path>[^"']+?\.py)["'],\s+line\s+(?P<line>\d+)(?:,\s+in\s+(?P<func>[A-Za-z_]\w*))?"""
    )
    _PYTEST_LOCATION_RE = re.compile(r"""(?P<path>\S+?\.py):(?P<line>\d+):""")
    _IN_FUNCTION_RE = re.compile(r"""\bin\s+(?P<func>[A-Za-z_]\w*)\b""")
    _CALL_SYMBOL_RE = re.compile(r"""\b(?P<name>[A-Za-z_]\w*)\s*\(""")
    _HINT_PATH_RE = re.compile(r"""^(?P<path>.+?\.py)(?::(?P<line>\d+))?$""")

    _TEST_CHUNK_PENALTY = 0.35
    _NON_TEST_CHUNK_BOOST = 0.10
    _MESSAGE_SYMBOL_BOOST = 0.60
    _TRACEBACK_SYMBOL_BOOST = 0.35
    _SAME_FILE_HINT_BOOST = 0.25
    _SAME_DIR_HINT_BOOST = 0.15
    _EXPECTED_EXCEPTION_BOOST = 1.50
    _EXPECTED_EXCEPTION_DEFINITION_BOOST = 3.00
    _RAISE_EXPECTED_EXCEPTION_BOOST = 2.50
    _EXPECTED_EXCEPTION_VALIDATION_TEXT_BOOST = 1.00
    _EXCEPTION_SYMBOL_BOOST = 0.20
    _MAX_EXCEPTION_SYMBOL_BOOST = 0.80
    _BEHAVIORAL_WORD_BOOST = 0.12
    _MAX_BEHAVIORAL_WORD_BOOST = 0.48
    _OPERATION_WORD_BOOST = 0.10
    _MAX_OPERATION_WORD_BOOST = 0.40
    _DOMAIN_WORD_BOOST = 0.05
    _MAX_DOMAIN_WORD_BOOST = 0.25
    _DID_NOT_RAISE_GUARD_NAME_BOOST = 0.90
    _DID_NOT_RAISE_RAISE_BOOST = 0.50
    _DID_NOT_RAISE_VALIDATION_CALL_BOOST = 0.80
    _DID_NOT_RAISE_GENERIC_VALIDATION_CALL_BOOST = 0.15
    _DID_NOT_RAISE_VALIDATION_TEXT_BOOST = 0.25
    _DID_NOT_RAISE_GENERIC_VALIDATION_TEXT_BOOST = 0.10
    _DID_NOT_RAISE_GENERIC_GUARD_NAME_BOOST = 0.20
    _DID_NOT_RAISE_GENERIC_RAISE_BOOST = 0.10
    _DID_NOT_RAISE_SPECIFIC_CHUNK_BOOST = 0.20
    _DID_NOT_RAISE_GENERIC_SIGNAL_CAP = 0.15
    _DID_NOT_RAISE_NON_EXCEPTION_CLASS_SCALE = 0.35

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

        symbols, source_hints = FailureRetriever._extract_traceback_hints(
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
            if FailureRetriever._FILE_LINE_RE.search(stripped):
                continue
            if FailureRetriever._PYTEST_LOCATION_RE.search(stripped):
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
        if not traceback_text:
            return ([], [])

        noisy_symbols = {
            "assert",
            "repr",
            "str",
            "int",
            "float",
            "bool",
            "list",
            "dict",
            "set",
            "tuple",
            "len",
            "range",
            "print",
        }

        symbols: list[str] = []
        source_hints: list[str] = []
        seen_symbols: set[str] = set()
        seen_hints: set[str] = set()

        def add_symbol(symbol: str) -> None:
            if len(symbols) >= max_symbols:
                return
            normalized = symbol.strip()
            if not normalized:
                return
            key = normalized.lower()
            if key in noisy_symbols:
                return
            if key in seen_symbols:
                return
            seen_symbols.add(key)
            symbols.append(normalized)

        def add_hint(path: str, line_number: int | None) -> None:
            if len(source_hints) >= max_source_hints:
                return
            normalized_path = path.strip().strip("\"'")
            if not normalized_path:
                return
            hint = normalized_path
            if line_number is not None:
                hint = f"{hint}:{line_number}"
            key = hint.replace("\\", "/").lower()
            if key in seen_hints:
                return
            seen_hints.add(key)
            source_hints.append(hint)

        for raw in traceback_text.splitlines():
            line = raw.strip()
            if not line:
                continue

            file_match = FailureRetriever._FILE_LINE_RE.search(line)
            if file_match:
                add_hint(file_match.group("path"), int(file_match.group("line")))
                func = file_match.group("func")
                if func:
                    add_symbol(func)

            loc_match = FailureRetriever._PYTEST_LOCATION_RE.search(line)
            if loc_match:
                add_hint(loc_match.group("path"), int(loc_match.group("line")))

            func_match = FailureRetriever._IN_FUNCTION_RE.search(line)
            if func_match:
                add_symbol(func_match.group("func"))

            for call in FailureRetriever._CALL_SYMBOL_RE.finditer(line):
                add_symbol(call.group("name"))

            if len(symbols) >= max_symbols and len(source_hints) >= max_source_hints:
                break

        return (symbols, source_hints)

    @staticmethod
    def _hint_to_normalized_path(hint: str) -> str:
        match = FailureRetriever._HINT_PATH_RE.match(hint.strip())
        if match is None:
            return ""
        return normalize_path(match.group("path"))

    @staticmethod
    def _extract_message_symbols(message: str) -> set[str]:
        symbols: set[str] = set()
        for match in FailureRetriever._CALL_SYMBOL_RE.finditer(message):
            symbols.add(match.group("name"))
        return symbols

    @staticmethod
    def score_failure_chunk(
        *,
        chunk: CodeChunk,
        base_score: float,
        failure: TestFailure,
        signals: FailureSignals | None = None,
    ) -> float:
        """Heuristic reranking score used during failure-aware retrieval.

        This is intentionally simple: it adjusts the semantic similarity score with
        lightweight signals (test vs source, symbol matches, path hints) to prefer
        likely implementation code over test code.
        """
        score = float(base_score)
        failure_signals = signals or extract_failure_signals(failure)

        message_symbols = FailureRetriever._extract_message_symbols(failure.message)
        traceback_symbols, source_hints = FailureRetriever._extract_traceback_hints(
            failure.traceback,
            max_symbols=FailureRetriever._MAX_TRACEBACK_SYMBOLS,
            max_source_hints=FailureRetriever._MAX_SOURCE_HINTS,
        )
        traceback_symbol_set = {symbol for symbol in traceback_symbols}

        if is_test_path(chunk.file_path):
            score -= FailureRetriever._TEST_CHUNK_PENALTY
        else:
            score += FailureRetriever._NON_TEST_CHUNK_BOOST

        if chunk.name in message_symbols:
            score += FailureRetriever._MESSAGE_SYMBOL_BOOST

        if chunk.name in traceback_symbol_set:
            score += FailureRetriever._TRACEBACK_SYMBOL_BOOST

        normalized_chunk_path = normalize_path(chunk.file_path)
        hint_files = [FailureRetriever._hint_to_normalized_path(hint) for hint in source_hints]
        hint_files = [hint for hint in hint_files if hint]

        non_test_hint_files = [hint for hint in hint_files if not is_test_path(hint)]
        hint_dirs = {hint.rsplit("/", 1)[0] for hint in non_test_hint_files if "/" in hint}

        for hint_file in non_test_hint_files:
            if normalized_chunk_path == hint_file or normalized_chunk_path.endswith(
                "/" + hint_file
            ):
                score += FailureRetriever._SAME_FILE_HINT_BOOST
                break

        for hint_dir in hint_dirs:
            if normalized_chunk_path.startswith(hint_dir + "/") or normalized_chunk_path.endswith(
                "/" + hint_dir
            ):
                score += FailureRetriever._SAME_DIR_HINT_BOOST
                break

        signal_score = FailureRetriever._score_structured_failure_signals(
            chunk=chunk, signals=failure_signals
        )
        if is_test_path(chunk.file_path):
            signal_score *= 0.25
        score += signal_score

        return score

    @staticmethod
    def _score_structured_failure_signals(*, chunk: CodeChunk, signals: FailureSignals) -> float:
        text = chunk_signal_text(chunk)
        text_lower = text.lower()
        source_lower = chunk.source_code.lower()
        tokens = chunk_signal_tokens(chunk)
        score = 0.0

        defines_exception = defines_expected_exception(chunk, signals)
        expected_exception_in_text = contains_expected_exception(text_lower, signals)
        expected_exception_in_source = contains_expected_exception(source_lower, signals)
        raises_exception = raises_expected_exception(chunk.source_code, signals)
        has_guard_name = has_validation_name(chunk.name)
        calls_guard = calls_validation_helper(chunk)
        calls_relevant_guard = calls_relevant_validation_helper(chunk, signals)
        has_validation_text = bool(tokens & VALIDATION_WORDS)
        relevant_exception_matches = relevant_exception_symbol_matches(tokens, signals)

        if expected_exception_in_text:
            score += FailureRetriever._EXPECTED_EXCEPTION_BOOST

        if defines_exception:
            score += FailureRetriever._EXPECTED_EXCEPTION_DEFINITION_BOOST

        exception_matches = set(signals.exception_symbols) & tokens
        weak_signal_score = min(
            FailureRetriever._MAX_EXCEPTION_SYMBOL_BOOST,
            len(exception_matches) * FailureRetriever._EXCEPTION_SYMBOL_BOOST,
        )

        behavioral_matches = set(signals.behavioral_words) & tokens
        weak_signal_score += min(
            FailureRetriever._MAX_BEHAVIORAL_WORD_BOOST,
            len(behavioral_matches) * FailureRetriever._BEHAVIORAL_WORD_BOOST,
        )

        operation_matches = set(signals.operation_words) & tokens
        weak_signal_score += min(
            FailureRetriever._MAX_OPERATION_WORD_BOOST,
            len(operation_matches) * FailureRetriever._OPERATION_WORD_BOOST,
        )

        domain_matches = set(signals.domain_words) & tokens
        weak_signal_score += min(
            FailureRetriever._MAX_DOMAIN_WORD_BOOST,
            len(domain_matches) * FailureRetriever._DOMAIN_WORD_BOOST,
        )

        if not signals.did_not_raise:
            score += weak_signal_score
            return score

        if raises_exception:
            score += FailureRetriever._RAISE_EXPECTED_EXCEPTION_BOOST
        elif contains_raise_statement(source_lower) and relevant_exception_matches:
            score += FailureRetriever._DID_NOT_RAISE_RAISE_BOOST
        elif contains_raise_statement(source_lower):
            score += FailureRetriever._DID_NOT_RAISE_GENERIC_RAISE_BOOST

        if expected_exception_in_source and has_validation_text:
            score += FailureRetriever._EXPECTED_EXCEPTION_VALIDATION_TEXT_BOOST

        if has_guard_name and (expected_exception_in_source or relevant_exception_matches):
            score += FailureRetriever._DID_NOT_RAISE_GUARD_NAME_BOOST
        elif has_guard_name:
            score += FailureRetriever._DID_NOT_RAISE_GENERIC_GUARD_NAME_BOOST

        if calls_relevant_guard:
            score += FailureRetriever._DID_NOT_RAISE_VALIDATION_CALL_BOOST
        elif calls_guard:
            score += FailureRetriever._DID_NOT_RAISE_GENERIC_VALIDATION_CALL_BOOST

        if has_validation_text and (expected_exception_in_source or relevant_exception_matches):
            score += FailureRetriever._DID_NOT_RAISE_VALIDATION_TEXT_BOOST
        elif has_validation_text:
            score += FailureRetriever._DID_NOT_RAISE_GENERIC_VALIDATION_TEXT_BOOST

        has_strong_validation_signal = (
            defines_exception
            or raises_exception
            or expected_exception_in_source
            or (has_guard_name and bool(relevant_exception_matches))
            or calls_relevant_guard
        )
        if not has_strong_validation_signal:
            weak_signal_score = min(
                weak_signal_score,
                FailureRetriever._DID_NOT_RAISE_GENERIC_SIGNAL_CAP,
            )

        score += weak_signal_score

        if has_strong_validation_signal and chunk.chunk_type in {"function", "method"}:
            score += FailureRetriever._DID_NOT_RAISE_SPECIFIC_CHUNK_BOOST

        if chunk.chunk_type == "class" and not defines_exception:
            score *= FailureRetriever._DID_NOT_RAISE_NON_EXCEPTION_CLASS_SCALE

        return score

    @staticmethod
    def rerank_semantic_results_for_failure(
        *, failure: TestFailure, semantic_results: list[SearchResult]
    ) -> list[SearchResult]:
        signals = extract_failure_signals(failure)
        scored: list[tuple[float, SearchResult]] = []
        for result in semantic_results:
            adjusted = FailureRetriever.score_failure_chunk(
                chunk=result.chunk,
                base_score=result.score,
                failure=failure,
                signals=signals,
            )
            scored.append((adjusted, result))

        scored.sort(key=lambda item: _failure_result_sort_key(item[0], item[1]))
        return [
            SearchResult(chunk=item[1].chunk, score=item[0], reasons=item[1].reasons)
            for item in scored
        ]

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

        semantic = semantic_ranked[:top_k]
        return enrich_with_related(query=query, semantic_results=semantic, graph=graph)


def _failure_result_sort_key(
    score: float, result: SearchResult
) -> tuple[float, bool, str, int, str]:
    chunk = result.chunk
    return (
        -score,
        is_test_path(chunk.file_path),
        normalize_path(chunk.file_path),
        chunk.start_line,
        chunk.id,
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
    merged.sort(key=lambda result: _failure_result_sort_key(result.score, result))
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
