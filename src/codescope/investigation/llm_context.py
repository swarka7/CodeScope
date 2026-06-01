from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from codescope.investigation.investigator import InvestigationCodeResult
from codescope.llm.context_safety import (
    DEFAULT_MAX_TOTAL_CONTEXT_CHARS,
    fit_items_to_context_cap,
    redact_sensitive_text,
    truncate_code,
    truncate_text,
)


@dataclass(frozen=True, slots=True)
class LLMInvestigationChunkContext:
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
class LLMInvestigationContext:
    repo_path: str
    query: str
    likely_relevant_code: tuple[LLMInvestigationChunkContext, ...]
    related_context: tuple[LLMInvestigationChunkContext, ...]


def build_llm_investigation_context(
    *,
    repo_path: Path,
    query: str,
    likely_relevant_code: Sequence[InvestigationCodeResult],
    related_context: Sequence[InvestigationCodeResult],
    max_likely_chunks: int = 5,
    max_related_chunks: int = 5,
    max_query_chars: int = 1200,
    max_code_snippet_chars: int = 3000,
    max_code_snippet_lines: int = 80,
    max_dependencies: int = 12,
    max_reasons: int = 6,
    max_total_context_chars: int | None = DEFAULT_MAX_TOTAL_CONTEXT_CHARS,
) -> LLMInvestigationContext:
    """Build a deterministic, bounded packet for optional LLM investigation.

    The packet is built only from the user description and already-retrieved
    CodeScope results. It does not inspect additional files, environment
    variables, index internals, or unrelated repository contents.
    """

    context = LLMInvestigationContext(
        repo_path=redact_sensitive_text(repo_path.as_posix()),
        query=truncate_text(
            redact_sensitive_text(" ".join((query or "").split())),
            max_chars=max_query_chars,
        ),
        likely_relevant_code=tuple(
            _build_chunk_context(
                result,
                max_code_snippet_chars=max_code_snippet_chars,
                max_code_snippet_lines=max_code_snippet_lines,
                max_dependencies=max_dependencies,
                max_reasons=max_reasons,
            )
            for result in likely_relevant_code[: max(max_likely_chunks, 0)]
        ),
        related_context=tuple(
            _build_chunk_context(
                result,
                max_code_snippet_chars=max_code_snippet_chars,
                max_code_snippet_lines=max_code_snippet_lines,
                max_dependencies=max_dependencies,
                max_reasons=max_reasons,
            )
            for result in related_context[: max(max_related_chunks, 0)]
        ),
    )

    if max_total_context_chars is None:
        return context
    return _apply_total_context_cap(context, max_chars=max_total_context_chars)


def _build_chunk_context(
    result: InvestigationCodeResult,
    *,
    max_code_snippet_chars: int,
    max_code_snippet_lines: int,
    max_dependencies: int,
    max_reasons: int,
) -> LLMInvestigationChunkContext:
    return LLMInvestigationChunkContext(
        rank=result.rank,
        source_label=result.source,
        chunk_type=result.kind,
        name=redact_sensitive_text(result.name),
        file_path=redact_sensitive_text(result.file_path),
        start_line=result.start_line,
        end_line=result.end_line,
        score=result.score,
        reasons=tuple(
            redact_sensitive_text(reason) for reason in result.reasons[: max(max_reasons, 0)]
        ),
        dependencies=tuple(
            redact_sensitive_text(dependency)
            for dependency in result.dependencies[: max(max_dependencies, 0)]
        ),
        code_snippet=truncate_code(
            result.chunk.source_code,
            max_chars=max_code_snippet_chars,
            max_lines=max_code_snippet_lines,
        ),
    )


def _apply_total_context_cap(
    context: LLMInvestigationContext, *, max_chars: int
) -> LLMInvestigationContext:
    if max_chars <= 0:
        return LLMInvestigationContext(
            repo_path="",
            query="",
            likely_relevant_code=(),
            related_context=(),
        )

    if _context_size(context) <= max_chars:
        return context

    related_context = fit_items_to_context_cap(
        context.related_context,
        fixed_size=_context_size(
            LLMInvestigationContext(
                repo_path=context.repo_path,
                query=context.query,
                likely_relevant_code=context.likely_relevant_code,
                related_context=(),
            )
        ),
        max_chars=max_chars,
        item_size=_chunk_context_size,
    )
    context = LLMInvestigationContext(
        repo_path=context.repo_path,
        query=context.query,
        likely_relevant_code=context.likely_relevant_code,
        related_context=related_context,
    )
    if _context_size(context) <= max_chars:
        return context

    likely_relevant_code = fit_items_to_context_cap(
        context.likely_relevant_code,
        fixed_size=_context_size(
            LLMInvestigationContext(
                repo_path=context.repo_path,
                query=context.query,
                likely_relevant_code=(),
                related_context=context.related_context,
            )
        ),
        max_chars=max_chars,
        item_size=_chunk_context_size,
    )
    context = LLMInvestigationContext(
        repo_path=context.repo_path,
        query=context.query,
        likely_relevant_code=likely_relevant_code,
        related_context=context.related_context,
    )
    if _context_size(context) <= max_chars:
        return context

    query_budget = max(max_chars - len(context.repo_path), 0)
    context = LLMInvestigationContext(
        repo_path=context.repo_path,
        query=truncate_text(context.query, max_chars=query_budget),
        likely_relevant_code=context.likely_relevant_code,
        related_context=context.related_context,
    )
    if _context_size(context) <= max_chars:
        return context

    repo_budget = max(max_chars - len(context.query), 0)
    return LLMInvestigationContext(
        repo_path=truncate_text(context.repo_path, max_chars=repo_budget),
        query=context.query,
        likely_relevant_code=context.likely_relevant_code,
        related_context=context.related_context,
    )


def _context_size(context: LLMInvestigationContext) -> int:
    return (
        len(context.repo_path)
        + len(context.query)
        + sum(_chunk_context_size(chunk) for chunk in context.likely_relevant_code)
        + sum(_chunk_context_size(chunk) for chunk in context.related_context)
    )


def _chunk_context_size(chunk: LLMInvestigationChunkContext) -> int:
    return (
        len(chunk.source_label)
        + len(chunk.chunk_type)
        + len(chunk.name)
        + len(chunk.file_path)
        + sum(len(reason) for reason in chunk.reasons)
        + sum(len(dependency) for dependency in chunk.dependencies)
        + len(chunk.code_snippet)
    )
