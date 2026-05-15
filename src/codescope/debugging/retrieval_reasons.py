from __future__ import annotations

import re

from codescope.debugging.failure_signals import (
    calls_validation_helper,
    chunk_signal_tokens,
    contains_expected_exception,
    defines_expected_exception,
    extract_failure_signals,
    has_validation_name,
    raises_expected_exception,
)
from codescope.models.code_chunk import CodeChunk
from codescope.models.test_failure import TestFailure
from codescope.utils.path_utils import is_test_path, normalize_path

_FILE_LINE_RE = re.compile(
    r"""File\s+["'](?P<path>[^"']+?\.py)["'],\s+line\s+(?P<line>\d+)"""
)
_PYTEST_LOCATION_RE = re.compile(r"""(?P<path>\S+?\.py):(?P<line>\d+):""")


def build_retrieval_reasons(
    failure: TestFailure,
    chunk: CodeChunk,
    *,
    extra_reasons: tuple[str, ...] = (),
    limit: int = 4,
) -> list[str]:
    signals = extract_failure_signals(failure)
    tokens = chunk_signal_tokens(chunk)
    reasons: list[str] = list(extra_reasons)

    if defines_expected_exception(chunk, signals):
        reasons.append("defines expected exception")

    if raises_expected_exception(chunk.source_code, signals):
        reasons.append("raises expected exception")
    elif contains_expected_exception(chunk.source_code, signals):
        reasons.append("contains expected exception")

    if has_validation_name(chunk.name):
        reasons.append("validation helper name")

    if calls_validation_helper(chunk):
        reasons.append("calls validation helper")

    if _is_source_traceback_file(failure, chunk):
        reasons.append("source chunk from traceback file")

    behavioral_overlap = _keyword_overlap(signals.behavioral_words, tokens)
    if behavioral_overlap:
        reasons.append(f"behavioral keyword overlap: {', '.join(behavioral_overlap)}")

    operation_overlap = _keyword_overlap(signals.operation_words, tokens)
    if operation_overlap:
        reasons.append(f"operation keyword overlap: {', '.join(operation_overlap)}")

    deduped = _dedupe(reasons)
    if deduped:
        return deduped[:limit]
    return ["semantic similarity"]


def format_retrieval_reasons(
    failure: TestFailure, chunk: CodeChunk, *, extra_reasons: tuple[str, ...] = ()
) -> str:
    return "; ".join(build_retrieval_reasons(failure, chunk, extra_reasons=extra_reasons))


def _keyword_overlap(terms: tuple[str, ...], tokens: set[str], *, limit: int = 3) -> list[str]:
    overlap = sorted(set(terms) & tokens)
    return overlap[:limit]


def _is_source_traceback_file(failure: TestFailure, chunk: CodeChunk) -> bool:
    if is_test_path(chunk.file_path):
        return False

    chunk_path = normalize_path(chunk.file_path)
    for hint_path in _traceback_file_paths(failure.traceback):
        if _paths_match(chunk_path, normalize_path(hint_path)):
            return True
    return False


def _traceback_file_paths(traceback: str) -> list[str]:
    paths: list[str] = []
    for line in traceback.splitlines():
        file_match = _FILE_LINE_RE.search(line)
        if file_match:
            paths.append(file_match.group("path"))

        location_match = _PYTEST_LOCATION_RE.search(line)
        if location_match:
            paths.append(location_match.group("path"))
    return _dedupe(paths)


def _paths_match(chunk_path: str, hint_path: str) -> bool:
    if not chunk_path or not hint_path:
        return False
    return (
        chunk_path == hint_path
        or chunk_path.endswith(f"/{hint_path}")
        or hint_path.endswith(f"/{chunk_path}")
    )


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
