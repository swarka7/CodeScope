from __future__ import annotations

from codescope.debugging.failure_signals import (
    VALIDATION_WORDS,
    FailureSignals,
    calls_relevant_validation_helper,
    calls_validation_helper,
    chunk_signal_tokens,
    contains_expected_exception,
    defines_expected_exception,
    extract_failure_signals,
    has_validation_name,
    raises_expected_exception,
    relevant_exception_symbol_matches,
)
from codescope.graph.dependency_graph import DependencyGraph
from codescope.models.code_chunk import CodeChunk
from codescope.models.test_failure import TestFailure
from codescope.utils.path_utils import is_test_path, normalize_path
from codescope.vectorstore.memory_store import SearchResult


def expand_failure_call_path_context(
    *,
    failure: TestFailure,
    seed_results: list[SearchResult],
    graph: DependencyGraph,
    max_seed_chunks: int = 4,
    max_depth: int = 2,
    per_hop_limit: int = 6,
    max_candidates: int = 10,
) -> list[SearchResult]:
    if max_seed_chunks <= 0 or max_depth <= 0 or max_candidates <= 0:
        return []

    signals = extract_failure_signals(failure)
    seeds = _select_seed_results(seed_results, max_seed_chunks=max_seed_chunks)
    if not seeds:
        return []

    top_seed_id = seeds[0].chunk.id
    best_by_id: dict[str, SearchResult] = {}
    visited: set[str] = {seed.chunk.id for seed in seeds}

    for seed in seeds:
        frontier: list[SearchResult] = [seed]
        for depth in range(1, max_depth + 1):
            next_frontier: list[SearchResult] = []
            for source in frontier:
                expanded = _expand_source(
                    failure=failure,
                    signals=signals,
                    source=source,
                    graph=graph,
                    depth=depth,
                    top_seed_id=top_seed_id,
                    visited=visited,
                )
                expanded.sort(key=_call_path_sort_key)
                for result in expanded[:per_hop_limit]:
                    visited.add(result.chunk.id)
                    _update_best(best_by_id, result)
                    next_frontier.append(result)
            if not next_frontier:
                break
            frontier = next_frontier

    results = list(best_by_id.values())
    results.sort(key=_call_path_sort_key)
    return results[:max_candidates]


def _select_seed_results(
    seed_results: list[SearchResult], *, max_seed_chunks: int
) -> list[SearchResult]:
    seeds: list[SearchResult] = []
    for result in seed_results:
        if is_test_path(result.chunk.file_path):
            continue
        if result.chunk.chunk_type not in {"function", "method"}:
            continue
        seeds.append(result)
        if len(seeds) >= max_seed_chunks:
            break
    return seeds


def _expand_source(
    *,
    failure: TestFailure,
    signals: FailureSignals,
    source: SearchResult,
    graph: DependencyGraph,
    depth: int,
    top_seed_id: str,
    visited: set[str],
) -> list[SearchResult]:
    results: list[SearchResult] = []

    for dependency_name, candidate in graph.related_candidates(source.chunk):
        if candidate.id in visited:
            continue
        if candidate.id == source.chunk.id:
            continue
        if is_test_path(candidate.file_path):
            continue
        if not _is_call_path_candidate(candidate, signals=signals, depth=depth):
            continue

        reasons = _call_path_reasons(
            failure=failure,
            source=source.chunk,
            candidate=candidate,
            dependency_name=dependency_name,
            top_seed_id=top_seed_id,
        )
        score = _call_path_score(
            source_score=source.score,
            candidate=candidate,
            signals=signals,
            depth=depth,
            reasons=reasons,
        )
        results.append(SearchResult(chunk=candidate, score=score, reasons=reasons))

    return results


def _is_call_path_candidate(
    chunk: CodeChunk, *, signals: FailureSignals, depth: int
) -> bool:
    tokens = chunk_signal_tokens(chunk)
    if signals.did_not_raise:
        return _is_expected_exception_call_path_candidate(chunk, signals=signals, tokens=tokens)

    strong_signal = (
        defines_expected_exception(chunk, signals)
        or raises_expected_exception(chunk.source_code, signals)
        or contains_expected_exception(chunk.source_code, signals)
        or has_validation_name(chunk.name)
        or calls_validation_helper(chunk)
        or bool(tokens & VALIDATION_WORDS)
    )
    if strong_signal:
        return True

    if chunk.chunk_type not in {"function", "method"}:
        return False

    if depth != 1:
        return False

    failure_terms = set(signals.operation_words) | set(signals.behavioral_words)
    return bool(tokens & failure_terms)


def _is_expected_exception_call_path_candidate(
    chunk: CodeChunk, *, signals: FailureSignals, tokens: set[str]
) -> bool:
    expected_exception_evidence = (
        defines_expected_exception(chunk, signals)
        or raises_expected_exception(chunk.source_code, signals)
        or contains_expected_exception(chunk.source_code, signals)
    )
    if expected_exception_evidence:
        return True

    relevant_exception_overlap = bool(relevant_exception_symbol_matches(tokens, signals))
    if has_validation_name(chunk.name) and relevant_exception_overlap:
        return True

    return calls_relevant_validation_helper(chunk, signals)


def _call_path_reasons(
    *,
    failure: TestFailure,
    source: CodeChunk,
    candidate: CodeChunk,
    dependency_name: str,
    top_seed_id: str,
) -> tuple[str, ...]:
    signals = extract_failure_signals(failure)
    reasons: list[str] = []

    if _is_traceback_source(failure, source):
        reasons.append("called by traceback source")
    if source.id == top_seed_id:
        reasons.append("called by top source chunk")

    if has_validation_name(candidate.name):
        reasons.append("validation helper on call path")

    if (
        defines_expected_exception(candidate, signals)
        or raises_expected_exception(candidate.source_code, signals)
        or contains_expected_exception(candidate.source_code, signals)
    ):
        reasons.append("exception logic on call path")

    if calls_validation_helper(candidate):
        reasons.append("calls validation helper")

    if not reasons:
        reasons.append(f"called via {dependency_name}")

    return _dedupe(reasons)


def _call_path_score(
    *,
    source_score: float,
    candidate: CodeChunk,
    signals: FailureSignals,
    depth: int,
    reasons: tuple[str, ...],
) -> float:
    source_component = source_score
    if signals.did_not_raise and not _has_expected_exception_or_validation_evidence(
        candidate, signals=signals
    ):
        source_component = min(source_component, 0.5)

    score = max(source_component - (0.25 * depth), 0.0)
    score += max(0.2, 0.9 - (0.2 * (depth - 1)))

    if defines_expected_exception(candidate, signals):
        score += 2.4
    if raises_expected_exception(candidate.source_code, signals):
        score += 2.0
    elif contains_expected_exception(candidate.source_code, signals):
        score += 1.2
    if has_validation_name(candidate.name):
        score += 0.9
    if calls_validation_helper(candidate):
        score += 0.5
    if candidate.chunk_type in {"function", "method"}:
        score += 0.2
    elif candidate.chunk_type == "class" and not defines_expected_exception(candidate, signals):
        score -= 1.0

    if "called by traceback source" in reasons:
        score += 0.4
    if "called by top source chunk" in reasons:
        score += 0.2

    return score


def _has_expected_exception_or_validation_evidence(
    chunk: CodeChunk, *, signals: FailureSignals
) -> bool:
    tokens = chunk_signal_tokens(chunk)
    return (
        defines_expected_exception(chunk, signals)
        or raises_expected_exception(chunk.source_code, signals)
        or contains_expected_exception(chunk.source_code, signals)
        or calls_relevant_validation_helper(chunk, signals)
        or (
            has_validation_name(chunk.name)
            and bool(relevant_exception_symbol_matches(tokens, signals))
        )
    )


def _update_best(best_by_id: dict[str, SearchResult], candidate: SearchResult) -> None:
    existing = best_by_id.get(candidate.chunk.id)
    if existing is None:
        best_by_id[candidate.chunk.id] = candidate
        return

    if candidate.score > existing.score:
        best_by_id[candidate.chunk.id] = candidate
        return

    if candidate.score == existing.score and candidate.reasons < existing.reasons:
        best_by_id[candidate.chunk.id] = candidate


def _is_traceback_source(failure: TestFailure, chunk: CodeChunk) -> bool:
    if is_test_path(chunk.file_path):
        return False
    chunk_path = normalize_path(chunk.file_path)
    for hint in _traceback_file_hints(failure):
        if chunk_path == hint or chunk_path.endswith(f"/{hint}") or hint.endswith(f"/{chunk_path}"):
            return True
    return False


def _traceback_file_hints(failure: TestFailure) -> list[str]:
    hints: list[str] = []
    for raw in failure.traceback.splitlines():
        line = raw.strip()
        for marker in ('File "', "File '"):
            if marker not in line:
                continue
            start = line.find(marker) + len(marker)
            end_quote = '"' if marker.endswith('"') else "'"
            end = line.find(end_quote, start)
            if end > start:
                hints.append(normalize_path(line[start:end]))
        if ".py:" in line:
            path = line.split(".py:", 1)[0] + ".py"
            hints.append(normalize_path(path.strip()))
    return _dedupe(hints)


def _call_path_sort_key(result: SearchResult) -> tuple[float, str, int, str]:
    return (
        -result.score,
        normalize_path(result.chunk.file_path),
        result.chunk.start_line,
        result.chunk.id,
    )


def _dedupe(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return tuple(result)
