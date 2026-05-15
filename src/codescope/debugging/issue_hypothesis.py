from __future__ import annotations

import re

from codescope.models.code_chunk import CodeChunk
from codescope.models.test_failure import TestFailure
from codescope.retrieval.dependency_aware import RetrievalResult
from codescope.utils.path_utils import is_test_path

_BOOLEAN_NAME_RE = re.compile(r"^(validate|check|is_|has_|authorize)", re.IGNORECASE)
_RETURNED_INSTEAD_RE = re.compile(r"\breturned\s+.+?\s+instead\s+of\s+.+", re.IGNORECASE)
_COMPARISON_RE = re.compile(r"(?<![=!<>])(?:<=|>=|<|>)(?!=)")
_BOUNDARY_TERMS = {"expired", "boundary", "limit", "threshold", "edge"}


def build_issue_hypothesis(
    failure: TestFailure, results: list[RetrievalResult]
) -> str | None:
    source = _top_source_chunk(results)
    if source is None:
        return None

    bullets: list[str] = []
    message = failure.message.strip()
    message_lower = message.lower()

    if _has_boolean_mismatch(message_lower) and _BOOLEAN_NAME_RE.match(source.name):
        bullets.append(
            f"{_chunk_name(source)} may contain boolean validation logic returning the "
            "opposite truth value from what the test expects."
        )

    if _RETURNED_INSTEAD_RE.search(message):
        bullets.append(f"{_chunk_name(source)} may compute an incorrect value.")

    if _has_boundary_signal(failure) and _COMPARISON_RE.search(source.source_code):
        bullets.append(
            "The retrieved source includes comparison logic and the failing test appears "
            "boundary-related."
        )

    if not bullets:
        return None

    lines = ["Possible issue:"]
    lines.extend(f"- {bullet}" for bullet in _dedupe(bullets))
    lines.append(
        "- This is a hypothesis based on the failure signal and retrieved code, "
        "not a proven root cause."
    )
    return "\n".join(lines)


def _top_source_chunk(results: list[RetrievalResult]) -> CodeChunk | None:
    for result in results:
        if not is_test_path(result.chunk.file_path):
            return result.chunk
    return None


def _has_boolean_mismatch(message_lower: str) -> bool:
    return "assert true is false" in message_lower or "assert false is true" in message_lower


def _has_boundary_signal(failure: TestFailure) -> bool:
    text = f"{failure.test_name} {failure.message} {failure.traceback}".lower()
    return any(term in text for term in _BOUNDARY_TERMS)


def _chunk_name(chunk: CodeChunk) -> str:
    if chunk.parent:
        return f"{chunk.parent}.{chunk.name}"
    return chunk.name


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
