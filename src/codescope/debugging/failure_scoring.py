from __future__ import annotations

import re
from dataclasses import dataclass

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
    identifier_tokens,
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
_BUSINESS_OPERATION_BOOST = 0.75
_DID_NOT_RAISE_BUSINESS_OPERATION_BOOST = 0.45
_STATE_UPDATE_LOGIC_BOOST = 0.55
_FILTERING_LOGIC_BOOST = 0.65
_GENERIC_DATA_ACCESS_PENALTY = -0.90
_ROUTE_OR_CONTROLLER_PENALTY = -0.35
_CONSTRUCTOR_OR_INIT_PENALTY = -0.35
_UNRELATED_EXCEPTION_CLASS_PENALTY = -0.35

_BUSINESS_ROLE_WORDS = frozenset(
    {
        "business",
        "domain",
        "engine",
        "interactor",
        "manager",
        "policy",
        "policies",
        "processor",
        "rule",
        "rules",
        "search",
        "service",
        "usecase",
        "workflow",
    }
)
_ROUTE_OR_CONTROLLER_WORDS = frozenset(
    {
        "api",
        "controller",
        "controllers",
        "endpoint",
        "handler",
        "route",
        "routes",
        "view",
    }
)
_FILTERING_WORDS = frozenset(
    {
        "criteria",
        "filter",
        "list",
        "match",
        "query",
        "rank",
        "rating",
        "result",
        "results",
        "search",
        "sort",
    }
)
_STATE_UPDATE_WORDS = frozenset(
    {
        "add",
        "append",
        "apply",
        "assign",
        "cancel",
        "change",
        "charge",
        "clear",
        "complete",
        "credit",
        "debit",
        "delete",
        "extend",
        "insert",
        "move",
        "record",
        "remove",
        "reserve",
        "save",
        "set",
        "ship",
        "submit",
        "transfer",
        "update",
    }
)


@dataclass(frozen=True, slots=True)
class ScoreComponent:
    name: str
    value: float
    details: tuple[str, ...] = ()
    contributes_to_score: bool = True


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    components: tuple[ScoreComponent, ...]

    @property
    def final_score(self) -> float:
        return sum(
            component.value
            for component in self.components
            if component.contributes_to_score
        )

    def by_name(self, name: str) -> tuple[ScoreComponent, ...]:
        return tuple(component for component in self.components if component.name == name)


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
    return build_score_breakdown(
        chunk=chunk,
        base_score=base_score,
        failure=failure,
        signals=signals,
    ).final_score


def build_score_breakdown(
    *,
    chunk: CodeChunk,
    base_score: float,
    failure: TestFailure,
    signals: FailureSignals | None = None,
    extra_reasons: tuple[str, ...] = (),
) -> ScoreBreakdown:
    failure_signals = signals or extract_failure_signals(failure)
    components: list[ScoreComponent] = [
        ScoreComponent(name="semantic_base_score", value=float(base_score)),
    ]

    message_symbols = extract_message_symbols(failure.message)
    traceback_symbols, source_hints = extract_traceback_hints(failure.traceback)
    traceback_symbol_set = {symbol for symbol in traceback_symbols}

    if is_test_path(chunk.file_path):
        components.append(
            ScoreComponent(name="test_chunk_penalty", value=-_TEST_CHUNK_PENALTY)
        )
    else:
        components.append(
            ScoreComponent(name="source_chunk_boost", value=_NON_TEST_CHUNK_BOOST)
        )

    if chunk.name in message_symbols:
        components.append(
            ScoreComponent(
                name="message_symbol_match",
                value=_MESSAGE_SYMBOL_BOOST,
                details=(chunk.name,),
            )
        )

    if chunk.name in traceback_symbol_set:
        components.append(
            ScoreComponent(
                name="traceback_symbol_match",
                value=_TRACEBACK_SYMBOL_BOOST,
                details=(chunk.name,),
            )
        )

    normalized_chunk_path = normalize_path(chunk.file_path)
    hint_files = [hint_to_normalized_path(hint) for hint in source_hints]
    hint_files = [hint for hint in hint_files if hint]

    non_test_hint_files = [hint for hint in hint_files if not is_test_path(hint)]
    hint_dirs = {hint.rsplit("/", 1)[0] for hint in non_test_hint_files if "/" in hint}

    for hint_file in non_test_hint_files:
        if normalized_chunk_path == hint_file or normalized_chunk_path.endswith(
            "/" + hint_file
        ):
            components.append(
                ScoreComponent(
                    name="same_file_source_hint",
                    value=_SAME_FILE_HINT_BOOST,
                    details=(hint_file,),
                )
            )
            break

    for hint_dir in hint_dirs:
        if normalized_chunk_path.startswith(hint_dir + "/") or normalized_chunk_path.endswith(
            "/" + hint_dir
        ):
            components.append(
                ScoreComponent(
                    name="same_directory_source_hint",
                    value=_SAME_DIR_HINT_BOOST,
                    details=(hint_dir,),
                )
            )
            break

    signal_components = list(
        _structured_failure_signal_components(
            chunk=chunk,
            signals=failure_signals,
            primary_terms=_primary_failure_terms(failure),
        )
    )
    if is_test_path(chunk.file_path):
        signal_components = _scale_scoring_components(signal_components, 0.25)
    components.extend(signal_components)

    component_names = {component.name for component in components}
    if "validation_helper_name" not in component_names and has_validation_name(chunk.name):
        components.append(
            ScoreComponent(
                name="validation_helper_name",
                value=0.0,
                details=(chunk.name,),
                contributes_to_score=False,
            )
        )

    if "calls_validation_helper" not in component_names and calls_validation_helper(chunk):
        components.append(
            ScoreComponent(
                name="calls_validation_helper",
                value=0.0,
                details=("observed_non_scoring_signal",),
                contributes_to_score=False,
            )
        )

    if _is_generic_crud_or_data_access_chunk(chunk):
        components.append(
            ScoreComponent(
                name="generic_crud_or_data_access_penalty",
                value=0.0,
                details=("observed_non_scoring_signal",),
                contributes_to_score=False,
            )
        )

    if _has_source_first_selection_signal(
        failure=failure,
        chunk=chunk,
        signals=failure_signals,
        message_symbols=message_symbols,
        traceback_symbols=traceback_symbol_set,
        extra_reasons=extra_reasons,
    ):
        components.append(
            ScoreComponent(
                name="source_first_selection_signal",
                value=0.0,
                details=("observed_non_scoring_signal",),
                contributes_to_score=False,
            )
        )

    if extra_reasons:
        call_path_reasons = tuple(
            reason for reason in extra_reasons if "call path" in reason or "called by" in reason
        )
        if call_path_reasons:
            components.append(
                ScoreComponent(
                    name="call_path_boost",
                    value=0.0,
                    details=call_path_reasons,
                    contributes_to_score=False,
                )
            )

    return ScoreBreakdown(components=tuple(components))


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
    return sum(
        component.value
        for component in _structured_failure_signal_components(
            chunk=chunk,
            signals=signals,
            primary_terms=set(signals.behavioral_words)
            | set(signals.operation_words)
            | set(signals.domain_words),
        )
        if component.contributes_to_score
    )


def _structured_failure_signal_components(
    *, chunk: CodeChunk, signals: FailureSignals, primary_terms: set[str]
) -> tuple[ScoreComponent, ...]:
    text = chunk_signal_text(chunk)
    text_lower = text.lower()
    source_lower = chunk.source_code.lower()
    tokens = chunk_signal_tokens(chunk)
    components: list[ScoreComponent] = []

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
        components.append(
            ScoreComponent(
                name="contains_expected_exception",
                value=_EXPECTED_EXCEPTION_BOOST,
                details=tuple(sorted(signals.expected_exceptions)),
            )
        )

    if defines_exception:
        components.append(
            ScoreComponent(
                name="expected_exception_definition",
                value=_EXPECTED_EXCEPTION_DEFINITION_BOOST,
                details=(chunk.name,),
            )
        )

    exception_matches = set(signals.exception_symbols) & tokens
    weak_components: list[ScoreComponent] = []
    exception_symbol_score = min(
        _MAX_EXCEPTION_SYMBOL_BOOST,
        len(exception_matches) * _EXCEPTION_SYMBOL_BOOST,
    )
    if exception_symbol_score:
        weak_components.append(
            ScoreComponent(
                name="exception_symbol_overlap",
                value=exception_symbol_score,
                details=tuple(sorted(exception_matches)),
            )
        )

    behavioral_matches = set(signals.behavioral_words) & tokens
    behavioral_score = min(
        _MAX_BEHAVIORAL_WORD_BOOST,
        len(behavioral_matches) * _BEHAVIORAL_WORD_BOOST,
    )
    if behavioral_score:
        weak_components.append(
            ScoreComponent(
                name="behavioral_keyword_overlap",
                value=behavioral_score,
                details=tuple(sorted(behavioral_matches)),
            )
        )

    operation_matches = set(signals.operation_words) & tokens
    operation_score = min(
        _MAX_OPERATION_WORD_BOOST,
        len(operation_matches) * _OPERATION_WORD_BOOST,
    )
    if operation_score:
        weak_components.append(
            ScoreComponent(
                name="operation_keyword_overlap",
                value=operation_score,
                details=tuple(sorted(operation_matches)),
            )
        )

    domain_matches = set(signals.domain_words) & tokens
    domain_score = min(
        _MAX_DOMAIN_WORD_BOOST,
        len(domain_matches) * _DOMAIN_WORD_BOOST,
    )
    if domain_score:
        weak_components.append(
            ScoreComponent(
                name="domain_keyword_overlap",
                value=domain_score,
                details=tuple(sorted(domain_matches)),
            )
        )

    business_components = _business_behavior_components(
        chunk=chunk,
        signals=signals,
        tokens=tokens,
        source_lower=source_lower,
        primary_terms=primary_terms,
    )

    if not signals.did_not_raise:
        components.extend(weak_components)
        components.extend(business_components)
        return tuple(components)

    if raises_exception:
        components.append(
            ScoreComponent(
                name="raises_expected_exception",
                value=_RAISE_EXPECTED_EXCEPTION_BOOST,
                details=tuple(sorted(signals.expected_exceptions)),
            )
        )
    elif contains_raise_statement(source_lower) and relevant_exception_matches:
        components.append(
            ScoreComponent(
                name="validation_raise_logic",
                value=_DID_NOT_RAISE_RAISE_BOOST,
                details=tuple(sorted(relevant_exception_matches)),
            )
        )
    elif contains_raise_statement(source_lower):
        components.append(
            ScoreComponent(
                name="generic_raise_statement",
                value=_DID_NOT_RAISE_GENERIC_RAISE_BOOST,
            )
        )

    if expected_exception_in_source and has_validation_text:
        components.append(
            ScoreComponent(
                name="validation_raise_logic",
                value=_EXPECTED_EXCEPTION_VALIDATION_TEXT_BOOST,
                details=tuple(sorted(signals.expected_exceptions)),
            )
        )

    if has_guard_name and (expected_exception_in_source or relevant_exception_matches):
        components.append(
            ScoreComponent(
                name="validation_helper_name",
                value=_DID_NOT_RAISE_GUARD_NAME_BOOST,
                details=(chunk.name,),
            )
        )
    elif has_guard_name:
        components.append(
            ScoreComponent(
                name="validation_helper_name",
                value=_DID_NOT_RAISE_GENERIC_GUARD_NAME_BOOST,
                details=(chunk.name,),
            )
        )

    if calls_relevant_guard:
        components.append(
            ScoreComponent(
                name="calls_validation_helper",
                value=_DID_NOT_RAISE_VALIDATION_CALL_BOOST,
            )
        )
    elif calls_guard:
        components.append(
            ScoreComponent(
                name="calls_validation_helper",
                value=_DID_NOT_RAISE_GENERIC_VALIDATION_CALL_BOOST,
            )
        )

    if has_validation_text and (expected_exception_in_source or relevant_exception_matches):
        components.append(
            ScoreComponent(
                name=(
                    "validation_raise_logic"
                    if expected_exception_in_source
                    else "validation_signal_overlap"
                ),
                value=_DID_NOT_RAISE_VALIDATION_TEXT_BOOST,
                details=tuple(sorted(relevant_exception_matches)),
            )
        )
    elif has_validation_text:
        components.append(
            ScoreComponent(
                name="validation_signal_overlap",
                value=_DID_NOT_RAISE_GENERIC_VALIDATION_TEXT_BOOST,
            )
        )

    has_strong_validation_signal = (
        defines_exception
        or raises_exception
        or expected_exception_in_source
        or (has_guard_name and bool(relevant_exception_matches))
        or calls_relevant_guard
    )
    if not has_strong_validation_signal:
        weak_components = _cap_components(
            weak_components,
            _DID_NOT_RAISE_GENERIC_SIGNAL_CAP,
        )

    components.extend(weak_components)

    did_not_raise_business_components = [
        component
        for component in business_components
        if component.name == "business_operation"
    ]
    components.extend(
        _scale_named_components(
            did_not_raise_business_components,
            factor=_DID_NOT_RAISE_BUSINESS_OPERATION_BOOST / _BUSINESS_OPERATION_BOOST,
            new_name="business_operation",
        )
    )

    if has_strong_validation_signal and chunk.chunk_type in {"function", "method"}:
        components.append(
            ScoreComponent(
                name="specific_source_chunk_boost",
                value=_DID_NOT_RAISE_SPECIFIC_CHUNK_BOOST,
            )
        )

    if chunk.chunk_type == "class" and not defines_exception:
        components = _scale_scoring_components(
            components,
            _DID_NOT_RAISE_NON_EXCEPTION_CLASS_SCALE,
        )

    return tuple(components)


def _scale_scoring_components(
    components: list[ScoreComponent], factor: float
) -> list[ScoreComponent]:
    scaled: list[ScoreComponent] = []
    for component in components:
        if not component.contributes_to_score:
            scaled.append(component)
            continue
        scaled.append(
            ScoreComponent(
                name=component.name,
                value=component.value * factor,
                details=component.details,
                contributes_to_score=component.contributes_to_score,
            )
        )
    return scaled


def _cap_components(components: list[ScoreComponent], max_total: float) -> list[ScoreComponent]:
    total = sum(component.value for component in components if component.contributes_to_score)
    if total <= max_total or total <= 0:
        return components
    return _scale_scoring_components(components, max_total / total)


def _scale_named_components(
    components: list[ScoreComponent], *, factor: float, new_name: str
) -> list[ScoreComponent]:
    scaled: list[ScoreComponent] = []
    for component in components:
        scaled.append(
            ScoreComponent(
                name=new_name,
                value=component.value * factor,
                details=component.details,
                contributes_to_score=component.contributes_to_score,
            )
        )
    return scaled


def _primary_failure_terms(failure: TestFailure) -> set[str]:
    text = "\n".join(
        [
            failure.test_name,
            failure.file_path,
            failure.error_type or "",
            failure.message,
        ]
    )
    return identifier_tokens(text)


def _business_behavior_components(
    *,
    chunk: CodeChunk,
    signals: FailureSignals,
    tokens: set[str],
    source_lower: str,
    primary_terms: set[str],
) -> list[ScoreComponent]:
    if chunk.chunk_type not in {"function", "method"}:
        return _non_operation_penalty_components(
            chunk=chunk,
            signals=signals,
            tokens=tokens,
            primary_terms=primary_terms,
        )

    components: list[ScoreComponent] = []
    name_terms = identifier_tokens(chunk.name)
    if chunk.parent:
        name_terms.update(identifier_tokens(chunk.parent))

    meaningful_primary_terms = _meaningful_failure_terms(primary_terms, signals)
    name_matches = sorted(name_terms & meaningful_primary_terms)
    source_matches = sorted((tokens & meaningful_primary_terms) - set(name_matches))
    action_terms = _primary_action_terms(primary_terms=primary_terms, signals=signals)
    action_name_matches = sorted(name_terms & action_terms)
    has_business_role = _has_business_role(chunk)
    has_state_logic = _has_state_update_logic(chunk, source_lower=source_lower)
    has_filter_logic = _has_filtering_logic(chunk, tokens=tokens, source_lower=source_lower)
    is_data_access = _is_generic_crud_or_data_access_chunk(chunk)
    is_route_passthrough = _is_route_or_controller_passthrough(chunk)
    is_constructor = _is_constructor_or_initializer(chunk)

    if action_name_matches and not is_data_access and not is_constructor and (
        has_business_role or has_state_logic or has_filter_logic
    ):
        components.append(
            ScoreComponent(
                name="business_operation",
                value=_BUSINESS_OPERATION_BOOST,
                details=tuple(action_name_matches[:3]),
            )
        )

    if (
        has_state_logic
        and (name_matches or source_matches)
        and not is_data_access
        and not is_constructor
    ):
        components.append(
            ScoreComponent(
                name="state_update_logic",
                value=_STATE_UPDATE_LOGIC_BOOST,
                details=tuple((name_matches or source_matches)[:3]),
            )
        )

    if has_filter_logic and (name_matches or source_matches) and not is_constructor:
        components.append(
            ScoreComponent(
                name="filtering_logic",
                value=_FILTERING_LOGIC_BOOST,
                details=tuple((name_matches or source_matches)[:3]),
            )
        )

    if _has_business_failure_context(primary_terms=primary_terms, signals=signals):
        if is_data_access:
            components.append(
                ScoreComponent(
                    name="generic_crud_or_data_access_penalty",
                    value=_GENERIC_DATA_ACCESS_PENALTY,
                )
            )
        if is_route_passthrough:
            components.append(
                ScoreComponent(
                    name="route_or_controller_passthrough_penalty",
                    value=_ROUTE_OR_CONTROLLER_PENALTY,
                )
            )
        if is_constructor:
            components.append(
                ScoreComponent(
                    name="constructor_or_init_penalty",
                    value=_CONSTRUCTOR_OR_INIT_PENALTY,
                )
            )

    return components


def _non_operation_penalty_components(
    *,
    chunk: CodeChunk,
    signals: FailureSignals,
    tokens: set[str],
    primary_terms: set[str],
) -> list[ScoreComponent]:
    if not _has_business_failure_context(primary_terms=primary_terms, signals=signals):
        return []

    if chunk.chunk_type == "class" and _looks_like_exception_class(chunk):
        if not defines_expected_exception(chunk, signals) and not contains_expected_exception(
            chunk_signal_text(chunk).lower(), signals
        ):
            return [
                ScoreComponent(
                    name="unrelated_exception_class_penalty",
                    value=_UNRELATED_EXCEPTION_CLASS_PENALTY,
                    details=tuple(sorted(tokens & {"error", "exception"})),
                )
            ]

    return []


def _meaningful_failure_terms(primary_terms: set[str], signals: FailureSignals) -> set[str]:
    return (
        primary_terms
        | set(signals.operation_words)
        | set(signals.behavioral_words)
        | set(signals.domain_words)
    ) - {
        "assert",
        "assertion",
        "error",
        "failed",
        "failure",
        "false",
        "none",
        "test",
        "tests",
        "true",
    }


def _has_business_failure_context(*, primary_terms: set[str], signals: FailureSignals) -> bool:
    terms = _meaningful_failure_terms(primary_terms, signals)
    return bool(
        terms
        & (
            set(signals.operation_words)
            | set(signals.behavioral_words)
            | _FILTERING_WORDS
            | _STATE_UPDATE_WORDS
        )
    )


def _primary_action_terms(*, primary_terms: set[str], signals: FailureSignals) -> set[str]:
    return (
        primary_terms
        & (
            set(signals.operation_words)
            | set(signals.behavioral_words)
            | _FILTERING_WORDS
            | _STATE_UPDATE_WORDS
        )
    )


def _has_business_role(chunk: CodeChunk) -> bool:
    return bool(_chunk_role_terms(chunk) & _BUSINESS_ROLE_WORDS)


def _is_route_or_controller_passthrough(chunk: CodeChunk) -> bool:
    if not (_chunk_role_terms(chunk) & _ROUTE_OR_CONTROLLER_WORDS):
        return False
    source = chunk.source_code.lower()
    if _has_state_update_logic(chunk, source_lower=source):
        return False
    if _has_filtering_logic(chunk, tokens=chunk_signal_tokens(chunk), source_lower=source):
        return False
    return "return " in source and len(source.splitlines()) <= 20


def _has_state_update_logic(chunk: CodeChunk, *, source_lower: str) -> bool:
    assignment_pattern = (
        r"\b(?:self\.)?[A-Za-z_]\w*(?:\.[A-Za-z_]\w+|\[[^\]]+\])\s*"
        r"(?:\+=|-=|(?<![=!<>])=(?!=))"
    )
    if re.search(assignment_pattern, source_lower):
        return True

    simple_assignment_pattern = r"\b[A-Za-z_]\w*\s*(?:\+=|-=|(?<![=!<>])=(?!=))"
    if re.search(simple_assignment_pattern, source_lower):
        return bool(chunk_signal_tokens(chunk) & _STATE_UPDATE_WORDS)

    method_call_match = re.search(
        r"\.(?P<name>add|append|apply|assign|cancel|change|charge|clear|complete|credit|"
        r"debit|delete|extend|insert|move|record|remove|reserve|save|set|ship|submit|"
        r"transfer|update)\s*\(",
        source_lower,
    )
    return method_call_match is not None


def _has_filtering_logic(chunk: CodeChunk, *, tokens: set[str], source_lower: str) -> bool:
    if not tokens & _FILTERING_WORDS:
        return False

    filter_patterns = (
        r"\bfor\b.+\bif\b",
        r"\bfilter\s*\(",
        r"\bsorted\s*\(",
        r"\bwhere\b",
        r"\bmatches\b",
    )
    if any(re.search(pattern, source_lower, flags=re.DOTALL) for pattern in filter_patterns):
        return True

    comparison_patterns = (">=", "<=", "==", "!=", " in ", " not in ")
    return any(pattern in source_lower for pattern in comparison_patterns)


def _is_constructor_or_initializer(chunk: CodeChunk) -> bool:
    name = chunk.name.lower()
    return name in {"__init__", "init", "initialize"}


def _looks_like_exception_class(chunk: CodeChunk) -> bool:
    name_tokens = identifier_tokens(chunk.name)
    source_tokens = identifier_tokens(chunk.source_code)
    return bool((name_tokens | source_tokens) & {"error", "exception"})


def _chunk_role_terms(chunk: CodeChunk) -> set[str]:
    terms: set[str] = set()
    terms.update(identifier_tokens(chunk.file_path))
    terms.update(identifier_tokens(chunk.name))
    if chunk.parent:
        terms.update(identifier_tokens(chunk.parent))
    return terms


def _is_generic_crud_or_data_access_chunk(chunk: CodeChunk) -> bool:
    path = normalize_path(chunk.file_path)
    path_parts = set(path.split("/"))
    parent = (chunk.parent or "").lower()
    name = chunk.name.lower().lstrip("_")

    data_access_path = bool(
        path_parts
        & {
            "dao",
            "data",
            "database",
            "db",
            "queries",
            "repositories",
            "repository",
            "storage",
            "stores",
        }
    )
    data_access_parent = any(
        marker in parent for marker in ("repository", "store", "storage", "dao")
    )
    crud_prefixes = (
        "add",
        "create",
        "delete",
        "fetch",
        "find",
        "get",
        "insert",
        "list",
        "load",
        "read",
        "record",
        "remove",
        "save",
        "select",
        "update",
        "write",
    )
    return (data_access_path or data_access_parent) and (
        name in crud_prefixes or name.startswith(tuple(f"{prefix}_" for prefix in crud_prefixes))
    )


def _has_source_first_selection_signal(
    *,
    failure: TestFailure,
    chunk: CodeChunk,
    signals: FailureSignals,
    message_symbols: set[str],
    traceback_symbols: set[str],
    extra_reasons: tuple[str, ...],
) -> bool:
    if is_test_path(chunk.file_path):
        return False
    if extra_reasons:
        return True
    if chunk.name in message_symbols or chunk.name in traceback_symbols:
        return True

    tokens = chunk_signal_tokens(chunk)
    text_lower = chunk_signal_text(chunk).lower()
    source_lower = chunk.source_code.lower()

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

    signals = extract_failure_signals(failure)
    traceback_symbols, _source_hints = extract_traceback_hints(failure.traceback)
    return _has_source_first_selection_signal(
        failure=failure,
        chunk=chunk,
        signals=signals,
        message_symbols=extract_message_symbols(failure.message),
        traceback_symbols=set(traceback_symbols),
        extra_reasons=result.reasons,
    )
