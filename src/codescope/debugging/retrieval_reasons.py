from __future__ import annotations

import re

from codescope.debugging.failure_scoring import (
    ScoreBreakdown,
    build_score_breakdown,
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
    breakdown = build_score_breakdown(
        chunk=chunk,
        base_score=0.0,
        failure=failure,
        extra_reasons=extra_reasons,
    )
    reasons: list[str] = list(extra_reasons)
    reasons.extend(_reasons_from_score_breakdown(breakdown))

    if _is_source_traceback_file(failure, chunk):
        reasons.append("source chunk from traceback file")

    deduped = _dedupe([_public_reason(reason) for reason in reasons])
    if deduped:
        return deduped[:limit]
    return ["semantic match"]


def format_retrieval_reasons(
    failure: TestFailure, chunk: CodeChunk, *, extra_reasons: tuple[str, ...] = ()
) -> str:
    return "; ".join(build_retrieval_reasons(failure, chunk, extra_reasons=extra_reasons))


def _reasons_from_score_breakdown(breakdown: ScoreBreakdown) -> list[str]:
    reasons: list[str] = []

    if breakdown.by_name("expected_exception_definition"):
        reasons.append("defines expected exception")

    if breakdown.by_name("raises_expected_exception"):
        reasons.append("raises expected exception")
    elif breakdown.by_name("contains_expected_exception"):
        reasons.append("references expected exception")

    if breakdown.by_name("validation_helper_name"):
        reasons.append("validation logic")

    if breakdown.by_name("calls_validation_helper"):
        reasons.append("calls validation logic")

    if breakdown.by_name("same_file_source_hint"):
        reasons.append("traceback file match")

    for component_name, label in (
        ("behavioral_keyword_overlap", "keyword match"),
        ("operation_keyword_overlap", "operation match"),
    ):
        values = _component_details(breakdown, component_name)
        if values:
            reasons.append(f"{label}: {', '.join(values)}")

    if breakdown.by_name("validation_raise_logic"):
        reasons.append("validation logic")

    if breakdown.by_name("test_chunk_penalty"):
        reasons.append("test context")

    if breakdown.by_name("generic_crud_or_data_access_penalty"):
        reasons.append("data-access context")

    if breakdown.by_name("business_operation"):
        reasons.append("business operation")

    if breakdown.by_name("state_update_logic"):
        reasons.append("state update logic")

    if breakdown.by_name("filtering_logic"):
        reasons.append("filtering logic")

    return reasons


def _public_reason(reason: str) -> str:
    normalized = " ".join(reason.strip().split())
    if not normalized:
        return ""

    exact_mappings = {
        "called by top source chunk": "call path match",
        "called by traceback source": "call path match",
        "reverse call-path context": "reverse call path match",
        "caller of validation helper": "calls validation logic",
        "caller of expected-exception logic": "exception-related call path",
        "exception logic on call path": "exception-related call path",
        "validation helper on call path": "validation logic",
        "validation helper name": "validation logic",
        "calls validation helper": "calls validation logic",
        "contains expected exception": "references expected exception",
        "source chunk from traceback file": "traceback file match",
        "generic data-access signal": "data-access context",
        "semantic similarity": "semantic match",
        "validation raise logic": "validation logic",
        "business caller context": "call path match",
    }
    mapped = exact_mappings.get(normalized)
    if mapped:
        return mapped

    prefix_mappings = (
        ("behavioral keyword overlap:", "keyword match:"),
        ("operation keyword overlap:", "operation match:"),
        ("called via ", "call path match"),
    )
    for prefix, replacement in prefix_mappings:
        if normalized.startswith(prefix):
            suffix = normalized.removeprefix(prefix).strip()
            if suffix and replacement.endswith(":"):
                return f"{replacement} {suffix}"
            return replacement

    return normalized


def _component_details(
    breakdown: ScoreBreakdown, component_name: str, *, limit: int = 3
) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for component in breakdown.by_name(component_name):
        for detail in component.details:
            key = detail.lower()
            if key in seen:
                continue
            seen.add(key)
            values.append(detail)
            if len(values) >= limit:
                return values
    return values


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
