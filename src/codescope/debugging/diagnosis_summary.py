from __future__ import annotations

import re
from pathlib import Path

from codescope.models.code_chunk import CodeChunk
from codescope.models.test_failure import TestFailure
from codescope.retrieval.dependency_aware import RetrievalResult


def build_diagnosis_summary(failure: TestFailure, results: list[RetrievalResult]) -> str:
    test_name = _short_test_name(failure.test_name)
    signal = _failure_signal(failure)
    source = _top_source_result(results)
    related_context = _related_context_names(results)
    why = _why_relevant(failure=failure, source=source.chunk if source else None, results=results)

    lines = [
        "Diagnosis summary:",
        f"- Failing test: {test_name}",
        f"- Failure signal: {signal}",
    ]

    if source is None:
        lines.append("- Most relevant source chunk: <none>")
    else:
        lines.append(
            f"- Most relevant source chunk: {_chunk_display_name(source.chunk)} "
            f"in {_display_path(source.chunk.file_path)}"
        )

    if related_context:
        lines.append(f"- Related context: {', '.join(related_context)}")
    else:
        lines.append("- Related context: <none>")

    lines.append(f"- Why: {why}")
    return "\n".join(lines)


def _short_test_name(test_name: str) -> str:
    value = test_name.strip()
    if "::" in value:
        return value.rsplit("::", 1)[-1] or value
    return value or "<unknown>"


def _failure_signal(failure: TestFailure) -> str:
    error = (failure.error_type or "").strip()
    message = failure.message.strip()

    if error and message:
        return f"{error}, {_truncate(message)}"
    if error:
        return error
    if message:
        return _truncate(message)
    return "<none>"


def _top_source_result(results: list[RetrievalResult]) -> RetrievalResult | None:
    for result in results:
        if not _is_test_chunk(result.chunk):
            return result
    return results[0] if results else None


def _related_context_names(results: list[RetrievalResult], *, limit: int = 5) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for result in results:
        if result.kind != "related":
            continue
        name = _chunk_display_name(result.chunk)
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(name)
        if len(names) >= limit:
            break
    return names


def _why_relevant(
    *, failure: TestFailure, source: CodeChunk | None, results: list[RetrievalResult]
) -> str:
    if not results:
        return "no retrieved chunks were available to explain."

    failure_symbols = _failure_symbols(failure)
    chunk_names = {_chunk_display_name(result.chunk) for result in results}
    short_chunk_names = {result.chunk.name for result in results}
    dependencies = {
        dependency
        for result in results
        for dependency in result.chunk.dependencies
        if dependency
    }

    if failure_symbols & (chunk_names | short_chunk_names):
        return "failure/query symbols overlap with retrieved source/context chunks."

    if source is not None and set(source.dependencies) & (chunk_names | short_chunk_names):
        return "the top source chunk depends on retrieved related context."

    if dependencies:
        return "retrieved chunks share dependency names that provide nearby context."

    return "semantic retrieval ranked these chunks closest to the failure details."


def _failure_symbols(failure: TestFailure) -> set[str]:
    text = "\n".join([failure.test_name, failure.message, failure.traceback])
    return {match.group(0) for match in re.finditer(r"\b[A-Za-z_]\w*\b", text)}


def _chunk_display_name(chunk: CodeChunk) -> str:
    if chunk.parent:
        return f"{chunk.parent}.{chunk.name}"
    return chunk.name


def _display_path(file_path: str) -> str:
    return Path(file_path).as_posix()


def _is_test_chunk(chunk: CodeChunk) -> bool:
    path = chunk.file_path.replace("\\", "/").lower().strip("/")
    wrapped = f"/{path}/"
    if "/tests/" in wrapped:
        return True

    file_name = path.rsplit("/", 1)[-1]
    return file_name == "conftest.py" or file_name.startswith("test_") or file_name.endswith(
        "_test.py"
    )


def _truncate(value: str, *, limit: int = 140) -> str:
    clean = " ".join(value.strip().split())
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "..."
