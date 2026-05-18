from __future__ import annotations

from dataclasses import dataclass

from codescope.debugging.retrieval_reasons import build_retrieval_reasons
from codescope.models.test_failure import TestFailure
from codescope.retrieval.dependency_aware import RetrievalResult


@dataclass(frozen=True, slots=True)
class LLMFailureContext:
    test_name: str
    file_path: str
    line_number: int | None
    error_type: str | None
    message: str
    traceback_excerpt: str


@dataclass(frozen=True, slots=True)
class LLMChunkContext:
    rank: int
    source_label: str
    chunk_type: str
    name: str
    file_path: str
    start_line: int
    end_line: int
    score: float | None
    reasons: tuple[str, ...]
    dependencies: tuple[str, ...]
    code_snippet: str


@dataclass(frozen=True, slots=True)
class LLMDiagnosisContext:
    failure: LLMFailureContext
    diagnosis_summary: str
    possible_issue: str | None
    chunks: tuple[LLMChunkContext, ...]


def build_llm_diagnosis_context(
    *,
    failure: TestFailure,
    diagnosis_summary: str,
    possible_issue: str | None,
    retrieval_results: list[RetrievalResult],
    max_semantic_chunks: int = 5,
    max_related_chunks: int = 5,
    max_traceback_chars: int = 1200,
    max_code_snippet_chars: int = 3000,
    max_code_snippet_lines: int = 80,
    max_dependencies: int = 12,
    max_reasons: int = 6,
    max_total_context_chars: int | None = None,
) -> LLMDiagnosisContext:
    """Build a deterministic, bounded packet for optional LLM diagnosis.

    The packet intentionally contains only failure metadata and already-retrieved
    chunks. It does not read additional files, inspect environment variables, or
    include repository/index contents outside the deterministic diagnose result.
    """

    selected = _select_results(
        retrieval_results,
        max_semantic_chunks=max_semantic_chunks,
        max_related_chunks=max_related_chunks,
    )
    chunks = tuple(
        _build_chunk_context(
            rank=rank,
            failure=failure,
            result=result,
            max_code_snippet_chars=max_code_snippet_chars,
            max_code_snippet_lines=max_code_snippet_lines,
            max_dependencies=max_dependencies,
            max_reasons=max_reasons,
        )
        for rank, result in enumerate(selected, start=1)
    )

    context = LLMDiagnosisContext(
        failure=LLMFailureContext(
            test_name=failure.test_name,
            file_path=failure.file_path,
            line_number=failure.line_number,
            error_type=failure.error_type,
            message=failure.message,
            traceback_excerpt=_truncate_text(
                failure.traceback.strip(),
                max_chars=max_traceback_chars,
            ),
        ),
        diagnosis_summary=diagnosis_summary,
        possible_issue=possible_issue,
        chunks=chunks,
    )

    if max_total_context_chars is None:
        return context

    return _apply_total_context_cap(context, max_chars=max_total_context_chars)


def _select_results(
    results: list[RetrievalResult], *, max_semantic_chunks: int, max_related_chunks: int
) -> list[RetrievalResult]:
    selected: list[RetrievalResult] = []
    semantic_count = 0
    related_count = 0
    semantic_limit = max(max_semantic_chunks, 0)
    related_limit = max(max_related_chunks, 0)

    for result in results:
        if result.kind == "semantic":
            if semantic_count >= semantic_limit:
                continue
            semantic_count += 1
            selected.append(result)
            continue

        if result.kind == "related":
            if related_count >= related_limit:
                continue
            related_count += 1
            selected.append(result)

    return selected


def _build_chunk_context(
    *,
    rank: int,
    failure: TestFailure,
    result: RetrievalResult,
    max_code_snippet_chars: int,
    max_code_snippet_lines: int,
    max_dependencies: int,
    max_reasons: int,
) -> LLMChunkContext:
    chunk = result.chunk
    reasons = build_retrieval_reasons(
        failure,
        chunk,
        extra_reasons=result.reasons,
        limit=max(max_reasons, 0),
    )

    return LLMChunkContext(
        rank=rank,
        source_label=result.kind,
        chunk_type=chunk.chunk_type,
        name=_chunk_display_name(result),
        file_path=chunk.file_path,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        score=result.score,
        reasons=tuple(reasons),
        dependencies=tuple(chunk.dependencies[: max(max_dependencies, 0)]),
        code_snippet=_truncate_code(
            chunk.source_code,
            max_chars=max_code_snippet_chars,
            max_lines=max_code_snippet_lines,
        ),
    )


def _chunk_display_name(result: RetrievalResult) -> str:
    chunk = result.chunk
    if chunk.chunk_type == "method" and chunk.parent:
        return f"{chunk.parent}.{chunk.name}"
    return chunk.name


def _truncate_code(source: str, *, max_chars: int, max_lines: int) -> str:
    if max_lines <= 0 or max_chars <= 0:
        return ""

    lines = source.rstrip().splitlines()
    truncated_by_line = len(lines) > max_lines
    snippet = "\n".join(lines[:max_lines])
    if truncated_by_line:
        snippet = snippet.rstrip() + "\n..."

    return _truncate_text(snippet, max_chars=max_chars)


def _truncate_text(value: str, *, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return "." * max_chars
    return value[: max_chars - 3].rstrip() + "..."


def _apply_total_context_cap(
    context: LLMDiagnosisContext, *, max_chars: int
) -> LLMDiagnosisContext:
    if max_chars <= 0:
        return LLMDiagnosisContext(
            failure=context.failure,
            diagnosis_summary="",
            possible_issue=None,
            chunks=(),
        )

    chunks = list(context.chunks)
    while chunks and _context_size(context, tuple(chunks)) > max_chars:
        chunks.pop()

    return LLMDiagnosisContext(
        failure=context.failure,
        diagnosis_summary=_truncate_text(context.diagnosis_summary, max_chars=max_chars),
        possible_issue=(
            _truncate_text(context.possible_issue, max_chars=max_chars)
            if context.possible_issue is not None
            else None
        ),
        chunks=tuple(chunks),
    )


def _context_size(context: LLMDiagnosisContext, chunks: tuple[LLMChunkContext, ...]) -> int:
    total = (
        len(context.failure.test_name)
        + len(context.failure.file_path)
        + len(context.failure.error_type or "")
        + len(context.failure.message)
        + len(context.failure.traceback_excerpt)
        + len(context.diagnosis_summary)
        + len(context.possible_issue or "")
    )
    for chunk in chunks:
        total += (
            len(chunk.source_label)
            + len(chunk.chunk_type)
            + len(chunk.name)
            + len(chunk.file_path)
            + sum(len(reason) for reason in chunk.reasons)
            + sum(len(dependency) for dependency in chunk.dependencies)
            + len(chunk.code_snippet)
        )
    return total
