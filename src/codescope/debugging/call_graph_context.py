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
    identifier_tokens,
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
    max_reverse_seed_chunks: int = 4,
    max_reverse_depth: int = 2,
    max_reverse_callers_per_seed: int = 3,
    max_reverse_candidates: int = 6,
) -> list[SearchResult]:
    if max_candidates <= 0:
        return []

    signals = extract_failure_signals(failure)
    seeds = _select_seed_results(seed_results, max_seed_chunks=max_seed_chunks)
    reverse_seeds = _select_reverse_seed_results(
        seed_results,
        signals=signals,
        max_seed_chunks=max_reverse_seed_chunks,
    )
    if not seeds and not reverse_seeds:
        return []

    top_seed_id = _top_seed_id(seeds, reverse_seeds)
    best_by_id: dict[str, SearchResult] = {}
    visited: set[str] = {seed.chunk.id for seed in seeds}

    if max_seed_chunks > 0 and max_depth > 0:
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

    if max_reverse_seed_chunks > 0 and max_reverse_depth > 0 and max_reverse_candidates > 0:
        reverse_context = _expand_reverse_context(
            failure=failure,
            signals=signals,
            seeds=reverse_seeds,
            graph=graph,
            max_depth=max_reverse_depth,
            callers_per_seed=max_reverse_callers_per_seed,
            max_candidates=max_reverse_candidates,
        )
        for result in reverse_context:
            _update_best(best_by_id, result)

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


def _select_reverse_seed_results(
    seed_results: list[SearchResult], *, signals: FailureSignals, max_seed_chunks: int
) -> list[SearchResult]:
    if not signals.did_not_raise and not signals.expected_exceptions:
        return []

    seeds: list[SearchResult] = []
    for result in seed_results:
        if is_test_path(result.chunk.file_path):
            continue
        if not _is_reverse_seed(result.chunk, signals=signals):
            continue
        seeds.append(result)
        if len(seeds) >= max_seed_chunks:
            break
    return seeds


def _is_reverse_seed(chunk: CodeChunk, *, signals: FailureSignals) -> bool:
    tokens = chunk_signal_tokens(chunk)
    if signals.did_not_raise:
        return (
            defines_expected_exception(chunk, signals)
            or raises_expected_exception(chunk.source_code, signals)
            or contains_expected_exception(chunk.source_code, signals)
            or (
                has_validation_name(chunk.name)
                and bool(relevant_exception_symbol_matches(tokens, signals))
            )
        )

    return (
        defines_expected_exception(chunk, signals)
        or raises_expected_exception(chunk.source_code, signals)
        or contains_expected_exception(chunk.source_code, signals)
        or has_validation_name(chunk.name)
    )


def _top_seed_id(
    forward_seeds: list[SearchResult], reverse_seeds: list[SearchResult]
) -> str:
    if forward_seeds:
        return forward_seeds[0].chunk.id
    if reverse_seeds:
        return reverse_seeds[0].chunk.id
    return ""


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


def _expand_reverse_context(
    *,
    failure: TestFailure,
    signals: FailureSignals,
    seeds: list[SearchResult],
    graph: DependencyGraph,
    max_depth: int,
    callers_per_seed: int,
    max_candidates: int,
) -> list[SearchResult]:
    best_by_id: dict[str, SearchResult] = {}

    for seed in seeds:
        frontier: list[SearchResult] = [seed]
        visited: set[str] = {seed.chunk.id}
        for depth in range(1, max_depth + 1):
            next_frontier: list[SearchResult] = []
            for target in frontier:
                expanded = _expand_reverse_source(
                    failure=failure,
                    signals=signals,
                    target=target,
                    graph=graph,
                    depth=depth,
                    visited=visited,
                )
                expanded.sort(key=_call_path_sort_key)
                for result in expanded[:callers_per_seed]:
                    visited.add(result.chunk.id)
                    _update_best(best_by_id, result)
                    if _is_reverse_seed(result.chunk, signals=signals):
                        next_frontier.append(result)
            if not next_frontier:
                break
            frontier = next_frontier

    results = list(best_by_id.values())
    results.sort(key=_call_path_sort_key)
    return results[:max_candidates]


def _expand_reverse_source(
    *,
    failure: TestFailure,
    signals: FailureSignals,
    target: SearchResult,
    graph: DependencyGraph,
    depth: int,
    visited: set[str],
) -> list[SearchResult]:
    results: list[SearchResult] = []

    for dependency_name, caller in graph.reverse_candidates(target.chunk):
        if caller.id in visited:
            continue
        if caller.id == target.chunk.id:
            continue
        if is_test_path(caller.file_path):
            continue
        if not _is_reverse_call_path_candidate(
            caller=caller,
            target=target.chunk,
            signals=signals,
            depth=depth,
        ):
            continue

        reasons = _reverse_call_path_reasons(
            failure=failure,
            target=target.chunk,
            caller=caller,
            dependency_name=dependency_name,
        )
        score = _reverse_call_path_score(
            source_score=target.score,
            caller=caller,
            target=target.chunk,
            signals=signals,
            depth=depth,
            reasons=reasons,
        )
        results.append(SearchResult(chunk=caller, score=score, reasons=reasons))

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


def _is_reverse_call_path_candidate(
    *, caller: CodeChunk, target: CodeChunk, signals: FailureSignals, depth: int
) -> bool:
    if caller.chunk_type not in {"function", "method"}:
        return False

    tokens = chunk_signal_tokens(caller)
    failure_terms = (
        set(signals.behavioral_words)
        | set(signals.operation_words)
        | set(signals.domain_words)
    )

    if calls_relevant_validation_helper(caller, signals):
        return True

    if calls_validation_helper(caller) and (
        depth == 1 or bool(tokens & failure_terms) or _has_business_source_role(caller)
    ):
        return True

    if _has_expected_exception_or_validation_evidence(target, signals=signals):
        if bool(tokens & failure_terms):
            return True
        if _has_business_source_role(caller):
            return True

    return False


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


def _reverse_call_path_reasons(
    *,
    failure: TestFailure,
    target: CodeChunk,
    caller: CodeChunk,
    dependency_name: str,
) -> tuple[str, ...]:
    signals = extract_failure_signals(failure)
    reasons: list[str] = ["reverse call-path context"]

    if has_validation_name(target.name) or raises_expected_exception(target.source_code, signals):
        reasons.append("caller of validation helper")

    if (
        defines_expected_exception(target, signals)
        or raises_expected_exception(target.source_code, signals)
        or contains_expected_exception(target.source_code, signals)
    ):
        reasons.append("caller of expected-exception logic")

    if calls_validation_helper(caller):
        reasons.append("calls validation helper")

    if _has_business_source_role(caller) and not (
        has_validation_name(caller.name) or raises_expected_exception(caller.source_code, signals)
    ):
        reasons.append("business caller context")

    if len(reasons) == 1:
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


def _reverse_call_path_score(
    *,
    source_score: float,
    caller: CodeChunk,
    target: CodeChunk,
    signals: FailureSignals,
    depth: int,
    reasons: tuple[str, ...],
) -> float:
    score = max(min(source_score, 2.0) - (0.45 * depth), 0.0)

    if calls_relevant_validation_helper(caller, signals):
        score += 0.8
    elif calls_validation_helper(caller):
        score += 0.35

    if raises_expected_exception(caller.source_code, signals):
        score += 1.2
    elif contains_expected_exception(caller.source_code, signals):
        score += 0.6

    if has_validation_name(caller.name):
        score += 0.2

    tokens = chunk_signal_tokens(caller)
    behavioral_matches = set(signals.behavioral_words) & tokens
    operation_matches = set(signals.operation_words) & tokens
    score += min(0.3, len(behavioral_matches) * 0.10)
    score += min(0.25, len(operation_matches) * 0.08)

    if _has_business_source_role(caller):
        score += 0.2

    if _has_expected_exception_or_validation_evidence(target, signals=signals):
        score += 0.15

    if "caller of expected-exception logic" in reasons:
        score += 0.1
    if "caller of validation helper" in reasons:
        score += 0.1

    max_score = max(source_score - 0.1, 0.0)
    if (
        defines_expected_exception(caller, signals)
        or raises_expected_exception(caller.source_code, signals)
        or contains_expected_exception(caller.source_code, signals)
    ):
        return min(score, max_score, 3.0)
    return min(score, max_score, 2.4)


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


def _has_business_source_role(chunk: CodeChunk) -> bool:
    role_words = {
        "api",
        "controller",
        "endpoint",
        "handler",
        "interactor",
        "route",
        "routes",
        "service",
        "usecase",
        "view",
    }
    path_tokens: set[str] = set()
    for part in normalize_path(chunk.file_path).replace("/", "_").split("_"):
        path_tokens.update(identifier_tokens(part))

    name_tokens = chunk_signal_tokens(chunk)
    return bool((path_tokens | name_tokens) & role_words)


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
