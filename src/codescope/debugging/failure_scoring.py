from __future__ import annotations

import re

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
from codescope.models.code_chunk import CodeChunk
from codescope.models.test_failure import TestFailure
from codescope.utils.path_utils import is_test_path, normalize_path
from codescope.vectorstore.memory_store import SearchResult

MAX_TRACEBACK_SYMBOLS = 12
MAX_SOURCE_HINTS = 10

FILE_LINE_RE = re.compile(
    r"""File\s+["'](?P<path>[^"']+?\.py)["'],\s+line\s+(?P<line>\d+)(?:,\s+in\s+(?P<func>[A-Za-z_]\w*))?"""
)
PYTEST_LOCATION_RE = re.compile(r"""(?P<path>\S+?\.py):(?P<line>\d+):""")
IN_FUNCTION_RE = re.compile(r"""\bin\s+(?P<func>[A-Za-z_]\w*)\b""")
CALL_SYMBOL_RE = re.compile(r"""\b(?P<name>[A-Za-z_]\w*)\s*\(""")
HINT_PATH_RE = re.compile(r"""^(?P<path>.+?\.py)(?::(?P<line>\d+))?$""")

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


def extract_traceback_hints(
    traceback_text: str,
    *,
    max_symbols: int = MAX_TRACEBACK_SYMBOLS,
    max_source_hints: int = MAX_SOURCE_HINTS,
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

        file_match = FILE_LINE_RE.search(line)
        if file_match:
            add_hint(file_match.group("path"), int(file_match.group("line")))
            func = file_match.group("func")
            if func:
                add_symbol(func)

        loc_match = PYTEST_LOCATION_RE.search(line)
        if loc_match:
            add_hint(loc_match.group("path"), int(loc_match.group("line")))

        func_match = IN_FUNCTION_RE.search(line)
        if func_match:
            add_symbol(func_match.group("func"))

        for call in CALL_SYMBOL_RE.finditer(line):
            add_symbol(call.group("name"))

        if len(symbols) >= max_symbols and len(source_hints) >= max_source_hints:
            break

    return (symbols, source_hints)


def extract_message_symbols(message: str) -> set[str]:
    symbols: set[str] = set()
    for match in CALL_SYMBOL_RE.finditer(message):
        symbols.add(match.group("name"))
    return symbols


def hint_to_normalized_path(hint: str) -> str:
    match = HINT_PATH_RE.match(hint.strip())
    if match is None:
        return ""
    return normalize_path(match.group("path"))


def score_failure_chunk(
    *,
    chunk: CodeChunk,
    base_score: float,
    failure: TestFailure,
    signals: FailureSignals | None = None,
) -> float:
    """Heuristic reranking score used during failure-aware retrieval."""
    score = float(base_score)
    failure_signals = signals or extract_failure_signals(failure)

    message_symbols = extract_message_symbols(failure.message)
    traceback_symbols, source_hints = extract_traceback_hints(failure.traceback)
    traceback_symbol_set = {symbol for symbol in traceback_symbols}

    if is_test_path(chunk.file_path):
        score -= _TEST_CHUNK_PENALTY
    else:
        score += _NON_TEST_CHUNK_BOOST

    if chunk.name in message_symbols:
        score += _MESSAGE_SYMBOL_BOOST

    if chunk.name in traceback_symbol_set:
        score += _TRACEBACK_SYMBOL_BOOST

    normalized_chunk_path = normalize_path(chunk.file_path)
    hint_files = [hint_to_normalized_path(hint) for hint in source_hints]
    hint_files = [hint for hint in hint_files if hint]

    non_test_hint_files = [hint for hint in hint_files if not is_test_path(hint)]
    hint_dirs = {hint.rsplit("/", 1)[0] for hint in non_test_hint_files if "/" in hint}

    for hint_file in non_test_hint_files:
        if normalized_chunk_path == hint_file or normalized_chunk_path.endswith(
            "/" + hint_file
        ):
            score += _SAME_FILE_HINT_BOOST
            break

    for hint_dir in hint_dirs:
        if normalized_chunk_path.startswith(hint_dir + "/") or normalized_chunk_path.endswith(
            "/" + hint_dir
        ):
            score += _SAME_DIR_HINT_BOOST
            break

    signal_score = _score_structured_failure_signals(chunk=chunk, signals=failure_signals)
    if is_test_path(chunk.file_path):
        signal_score *= 0.25
    score += signal_score

    return score


def rerank_semantic_results_for_failure(
    *, failure: TestFailure, semantic_results: list[SearchResult]
) -> list[SearchResult]:
    signals = extract_failure_signals(failure)
    scored: list[tuple[float, SearchResult]] = []
    for result in semantic_results:
        adjusted = score_failure_chunk(
            chunk=result.chunk,
            base_score=result.score,
            failure=failure,
            signals=signals,
        )
        scored.append((adjusted, result))

    scored.sort(key=lambda item: failure_result_sort_key(item[0], item[1]))
    return [
        SearchResult(chunk=item[1].chunk, score=item[0], reasons=item[1].reasons)
        for item in scored
    ]


def select_semantic_results_for_failure(
    *, failure: TestFailure, ranked_results: list[SearchResult], top_k: int
) -> list[SearchResult]:
    """Select final diagnose semantic roots with source chunks before test chunks.

    Failing tests often have the strongest raw semantic match because the
    failure query contains the test name and assertion text. For diagnosis,
    once strong source evidence exists, implementation chunks are better
    semantic roots for dependency expansion; the failing test remains visible
    in the failure summary instead of competing with likely root-cause code.
    """
    if top_k <= 0:
        return []

    if _failure_mentions_test_infrastructure(failure):
        return ranked_results[:top_k]

    source_results = [
        result for result in ranked_results if not is_test_path(result.chunk.file_path)
    ]
    if not source_results:
        return ranked_results[:top_k]

    strong_source_exists = any(
        _is_strong_failure_source(failure=failure, result=result)
        for result in source_results
    )
    if not strong_source_exists:
        return ranked_results[:top_k]

    selected: list[SearchResult] = []
    selected_ids: set[str] = set()

    for result in source_results:
        selected.append(result)
        selected_ids.add(result.chunk.id)
        if len(selected) >= top_k:
            return selected

    for result in ranked_results:
        if result.chunk.id in selected_ids:
            continue
        selected.append(result)
        selected_ids.add(result.chunk.id)
        if len(selected) >= top_k:
            break

    return selected


def failure_result_sort_key(
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
        score += _EXPECTED_EXCEPTION_BOOST

    if defines_exception:
        score += _EXPECTED_EXCEPTION_DEFINITION_BOOST

    exception_matches = set(signals.exception_symbols) & tokens
    weak_signal_score = min(
        _MAX_EXCEPTION_SYMBOL_BOOST,
        len(exception_matches) * _EXCEPTION_SYMBOL_BOOST,
    )

    behavioral_matches = set(signals.behavioral_words) & tokens
    weak_signal_score += min(
        _MAX_BEHAVIORAL_WORD_BOOST,
        len(behavioral_matches) * _BEHAVIORAL_WORD_BOOST,
    )

    operation_matches = set(signals.operation_words) & tokens
    weak_signal_score += min(
        _MAX_OPERATION_WORD_BOOST,
        len(operation_matches) * _OPERATION_WORD_BOOST,
    )

    domain_matches = set(signals.domain_words) & tokens
    weak_signal_score += min(
        _MAX_DOMAIN_WORD_BOOST,
        len(domain_matches) * _DOMAIN_WORD_BOOST,
    )

    if not signals.did_not_raise:
        score += weak_signal_score
        return score

    if raises_exception:
        score += _RAISE_EXPECTED_EXCEPTION_BOOST
    elif contains_raise_statement(source_lower) and relevant_exception_matches:
        score += _DID_NOT_RAISE_RAISE_BOOST
    elif contains_raise_statement(source_lower):
        score += _DID_NOT_RAISE_GENERIC_RAISE_BOOST

    if expected_exception_in_source and has_validation_text:
        score += _EXPECTED_EXCEPTION_VALIDATION_TEXT_BOOST

    if has_guard_name and (expected_exception_in_source or relevant_exception_matches):
        score += _DID_NOT_RAISE_GUARD_NAME_BOOST
    elif has_guard_name:
        score += _DID_NOT_RAISE_GENERIC_GUARD_NAME_BOOST

    if calls_relevant_guard:
        score += _DID_NOT_RAISE_VALIDATION_CALL_BOOST
    elif calls_guard:
        score += _DID_NOT_RAISE_GENERIC_VALIDATION_CALL_BOOST

    if has_validation_text and (expected_exception_in_source or relevant_exception_matches):
        score += _DID_NOT_RAISE_VALIDATION_TEXT_BOOST
    elif has_validation_text:
        score += _DID_NOT_RAISE_GENERIC_VALIDATION_TEXT_BOOST

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
            _DID_NOT_RAISE_GENERIC_SIGNAL_CAP,
        )

    score += weak_signal_score

    if has_strong_validation_signal and chunk.chunk_type in {"function", "method"}:
        score += _DID_NOT_RAISE_SPECIFIC_CHUNK_BOOST

    if chunk.chunk_type == "class" and not defines_exception:
        score *= _DID_NOT_RAISE_NON_EXCEPTION_CLASS_SCALE

    return score


def _failure_mentions_test_infrastructure(failure: TestFailure) -> bool:
    text = " ".join(
        [
            failure.test_name,
            failure.error_type or "",
            failure.message,
            failure.traceback,
        ]
    ).lower()
    test_infra_terms = {
        "capfd",
        "caplog",
        "capsys",
        "fixture",
        "monkeypatch",
        "mock",
        "tmp_path",
        "unittest",
    }
    return any(term in text for term in test_infra_terms)


def _is_strong_failure_source(*, failure: TestFailure, result: SearchResult) -> bool:
    chunk = result.chunk
    if is_test_path(chunk.file_path):
        return False

    if result.reasons:
        return True

    signals = extract_failure_signals(failure)
    tokens = chunk_signal_tokens(chunk)
    source_lower = chunk.source_code.lower()
    text_lower = chunk_signal_text(chunk).lower()
    message_symbols = extract_message_symbols(failure.message)
    traceback_symbols, _source_hints = extract_traceback_hints(failure.traceback)

    if chunk.name in message_symbols or chunk.name in set(traceback_symbols):
        return True

    if (
        defines_expected_exception(chunk, signals)
        or raises_expected_exception(chunk.source_code, signals)
        or contains_expected_exception(text_lower, signals)
        or relevant_exception_symbol_matches(tokens, signals)
    ):
        return True

    if has_validation_name(chunk.name):
        return True

    if calls_relevant_validation_helper(chunk, signals):
        return True

    if calls_validation_helper(chunk):
        failure_terms = (
            set(signals.behavioral_words)
            | set(signals.operation_words)
            | set(signals.domain_words)
        )
        if tokens & failure_terms:
            return True

    return contains_raise_statement(source_lower) and bool(tokens & VALIDATION_WORDS)
