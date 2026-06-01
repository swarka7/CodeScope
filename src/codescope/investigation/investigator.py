from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from codescope.debugging.failure_signals import (
    calls_validation_helper,
    has_validation_name,
    identifier_tokens,
)
from codescope.debugging.paired_operations import (
    called_paired_operation_terms,
    chunk_defines_paired_operation,
    counterpart_terms_for_called_operations,
)
from codescope.embeddings.embedder import Embedder
from codescope.graph.dependency_graph import DependencyGraph
from codescope.indexing.index_compatibility import EMPTY_INDEX_MESSAGE, check_index_compatibility
from codescope.indexing.index_store import IndexStore
from codescope.models.code_chunk import CodeChunk
from codescope.retrieval.dependency_aware import RetrievalResult, enrich_with_related
from codescope.utils.path_utils import display_path, is_test_path, normalize_path
from codescope.vectorstore.memory_store import MemoryStore, SearchResult

InvestigationSource = Literal["semantic", "related"]

_INVESTIGATION_INTENT = (
    "likely source code, business logic, validation, state update, "
    "filtering, related dependencies"
)
_CANDIDATE_MULTIPLIER = 8
_MIN_CANDIDATES = 30
_DEFAULT_RELATED_LIMIT = 5

_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "does",
        "for",
        "from",
        "i",
        "in",
        "into",
        "is",
        "it",
        "its",
        "not",
        "of",
        "on",
        "or",
        "the",
        "to",
        "when",
        "with",
    }
)

_BUSINESS_ROLE_TERMS = frozenset(
    {
        "business",
        "domain",
        "engine",
        "flow",
        "manager",
        "policy",
        "processor",
        "search",
        "service",
        "workflow",
    }
)

_FILTERING_TERMS = frozenset(
    {
        "criteria",
        "filter",
        "filtered",
        "genre",
        "list",
        "match",
        "query",
        "rank",
        "rating",
        "result",
        "results",
        "search",
        "sort",
        "year",
    }
)

_STATE_UPDATE_TERMS = frozenset(
    {
        "add",
        "amount",
        "balance",
        "cancel",
        "change",
        "credit",
        "debit",
        "increase",
        "move",
        "quantity",
        "record",
        "reserve",
        "save",
        "ship",
        "state",
        "status",
        "stock",
        "transfer",
        "update",
    }
)

_DATA_ACCESS_QUERY_TERMS = frozenset(
    {"database", "db", "repository", "storage", "store", "persistence"}
)

_DATA_ACCESS_TERMS = frozenset(
    {
        "dao",
        "database",
        "db",
        "find",
        "get",
        "list",
        "load",
        "repository",
        "save",
        "storage",
        "store",
    }
)

_GENERIC_DATA_ACCESS_NAMES = frozenset(
    {
        "add",
        "create",
        "delete",
        "find",
        "get",
        "list",
        "load",
        "read",
        "record",
        "save",
        "write",
    }
)

_INVESTIGATION_VALIDATION_PREFIXES = ("require", "enforce")

_ATTRIBUTE_MUTATION_RE = re.compile(
    r"\b(?:self|cls|[A-Za-z_]\w*)\.[A-Za-z_]\w*\s*(?:[+\-*/]?=)"
)
_COMPARISON_OR_FILTER_RE = re.compile(r"\bif\b|\bfor\b|\bsorted\s*\(|\bfilter\s*\(|[<>]=?|==|!=")


@dataclass(frozen=True, slots=True)
class InvestigationCodeResult:
    rank: int
    name: str
    kind: str
    file_path: str
    start_line: int
    end_line: int
    source: InvestigationSource
    score: float | None
    reasons: tuple[str, ...]
    dependencies: tuple[str, ...]
    chunk: CodeChunk = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class InvestigationResult:
    query: str
    likely_relevant_code: tuple[InvestigationCodeResult, ...]
    related_context: tuple[InvestigationCodeResult, ...]


@dataclass(frozen=True, slots=True)
class InvestigationScoredResult:
    chunk: CodeChunk
    score: float
    reasons: tuple[str, ...]
    semantic_score: float


class Investigator:
    """Retrieval-first investigation for natural-language bug descriptions."""

    def __init__(
        self,
        repo_path: Path,
        *,
        index_store: IndexStore | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._repo_path = Path(repo_path)
        self._index_store = index_store or IndexStore(self._repo_path)
        self._embedder = embedder or Embedder()

    def investigate(self, description: str, *, top_k: int = 5) -> InvestigationResult:
        cleaned_description = _clean_description(description)
        if not cleaned_description:
            raise ValueError("Bug description must not be empty")

        compatibility = check_index_compatibility(
            index_store=self._index_store,
            embedding_model_name=self._embedder.model_name,
        )
        if not compatibility.compatible:
            raise ValueError(compatibility.message)

        chunks, embeddings, _metadata = self._index_store.load()
        if not chunks:
            raise ValueError(EMPTY_INDEX_MESSAGE)

        query = build_investigation_query(cleaned_description)
        query_embedding = self._embedder.embed_text(query)

        store = MemoryStore()
        store.add(chunks, embeddings)
        candidate_limit = _candidate_limit(top_k=top_k, total_chunks=len(chunks))
        semantic_candidates = store.search(query_embedding, top_k=candidate_limit)
        reranked = rerank_investigation_results(
            description=cleaned_description,
            results=semantic_candidates,
        )

        semantic_results = [
            SearchResult(chunk=item.chunk, score=item.score, reasons=item.reasons)
            for item in reranked[: max(top_k, 0)]
        ]

        graph = DependencyGraph(chunks)
        enriched = enrich_with_related(
            query=cleaned_description,
            semantic_results=semantic_results,
            graph=graph,
            max_related=_DEFAULT_RELATED_LIMIT,
        )
        return _to_investigation_result(
            query=cleaned_description,
            results=enriched,
            repo_path=self._repo_path,
        )


def build_investigation_query(description: str) -> str:
    cleaned_description = _clean_description(description)
    return (
        "Bug description:\n"
        f"{cleaned_description}\n\n"
        "Investigation intent:\n"
        f"{_INVESTIGATION_INTENT}\n"
    )


def rerank_investigation_results(
    *, description: str, results: list[SearchResult]
) -> list[InvestigationScoredResult]:
    scored = [
        score_investigation_result(description=description, result=result)
        for result in results
    ]
    scored.sort(key=_investigation_sort_key)
    return scored


def score_investigation_result(
    *, description: str, result: SearchResult
) -> InvestigationScoredResult:
    chunk = result.chunk
    query_terms = _description_terms(description)
    chunk_terms = _chunk_terms(chunk)
    matched_terms = tuple(sorted(query_terms & chunk_terms))
    reasons = ["semantic match"]
    score = float(result.score)

    is_test_chunk = is_test_path(chunk.file_path)
    if not is_test_chunk:
        score += 0.25
    else:
        score -= 1.25
        reasons.append("test context")

    if matched_terms:
        score += min(1.2, 0.12 * len(matched_terms))
        reasons.append(f"operation match: {', '.join(matched_terms[:4])}")

    if is_test_chunk:
        return InvestigationScoredResult(
            chunk=chunk,
            score=score,
            reasons=_dedupe_reasons(reasons),
            semantic_score=result.score,
        )

    if _is_business_operation(chunk, query_terms=query_terms, chunk_terms=chunk_terms):
        score += 0.7
        reasons.append("business operation")

    if _has_filtering_logic(chunk, query_terms=query_terms, chunk_terms=chunk_terms):
        score += 0.65
        reasons.append("filtering logic")

    if _has_state_update_logic(chunk, query_terms=query_terms, chunk_terms=chunk_terms):
        score += 0.6
        reasons.append("state update logic")

    if _has_validation_logic(chunk):
        score += 0.55
        reasons.append("validation logic")

    paired_reason = _paired_operation_reason(
        chunk, query_terms=query_terms, chunk_terms=chunk_terms
    )
    if paired_reason is not None:
        score += 0.45
        reasons.append(paired_reason)

    if _is_generic_data_access(chunk) and not (query_terms & _DATA_ACCESS_QUERY_TERMS):
        score -= 0.5
        reasons.append("data-access context")

    if chunk.chunk_type == "class" and not _class_has_specific_business_signal(
        chunk, query_terms=query_terms, chunk_terms=chunk_terms
    ):
        score -= 0.25

    return InvestigationScoredResult(
        chunk=chunk,
        score=score,
        reasons=_dedupe_reasons(reasons),
        semantic_score=result.score,
    )


def _to_investigation_result(
    *, query: str, results: list[RetrievalResult], repo_path: Path
) -> InvestigationResult:
    semantic = [result for result in results if result.kind == "semantic"]
    related = [result for result in results if result.kind == "related"]
    return InvestigationResult(
        query=query,
        likely_relevant_code=tuple(
            _to_code_result(rank, result, repo_path=repo_path)
            for rank, result in enumerate(semantic, start=1)
        ),
        related_context=tuple(
            _to_code_result(rank, result, repo_path=repo_path)
            for rank, result in enumerate(related, start=1)
        ),
    )


def _to_code_result(
    rank: int, result: RetrievalResult, *, repo_path: Path
) -> InvestigationCodeResult:
    chunk = result.chunk
    reasons = result.reasons
    if result.kind == "related" and not reasons:
        reasons = ("dependency context",)

    return InvestigationCodeResult(
        rank=rank,
        name=_chunk_display_name(chunk),
        kind=chunk.chunk_type,
        file_path=_chunk_file_path(chunk, repo_path=repo_path),
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        source=result.kind,
        score=result.score,
        reasons=tuple(reasons),
        dependencies=tuple(chunk.dependencies),
        chunk=chunk,
    )


def _candidate_limit(*, top_k: int, total_chunks: int) -> int:
    if top_k <= 0:
        return 0
    return min(max(top_k * _CANDIDATE_MULTIPLIER, _MIN_CANDIDATES), total_chunks)


def _clean_description(description: str) -> str:
    return " ".join((description or "").split())


def _description_terms(description: str) -> set[str]:
    return _meaningful_terms(description)


def _chunk_terms(chunk: CodeChunk) -> set[str]:
    text = "\n".join(
        [
            chunk.name,
            chunk.parent or "",
            chunk.file_path,
            chunk.source_code,
            *chunk.dependencies,
            *chunk.decorators,
        ]
    )
    return _meaningful_terms(text)


def _meaningful_terms(text: str) -> set[str]:
    terms = set(identifier_tokens(text))
    expanded = set(terms)
    for term in terms:
        expanded.update(_term_variants(term))
    return {term for term in expanded if len(term) > 1 and term not in _STOP_WORDS}


def _term_variants(term: str) -> set[str]:
    variants: set[str] = set()
    if term.endswith("ies") and len(term) > 4:
        variants.add(f"{term[:-3]}y")
    if term.endswith("ing") and len(term) > 5:
        root = term[:-3]
        variants.add(root)
        variants.add(f"{root}e")
        if len(root) > 2 and root[-1] == root[-2]:
            variants.add(root[:-1])
    if term.endswith("ed") and len(term) > 4:
        variants.add(term[:-2])
    if term.endswith("s") and len(term) > 3:
        variants.add(term[:-1])
    return variants


def _is_business_operation(
    chunk: CodeChunk, *, query_terms: set[str], chunk_terms: set[str]
) -> bool:
    if chunk.chunk_type not in {"function", "method"}:
        return False
    if chunk.name.startswith("__") and chunk.name.endswith("__"):
        return False

    role_terms = set(identifier_tokens(chunk.parent or "")) | set(
        identifier_tokens(Path(chunk.file_path).stem)
    )
    if not (role_terms & _BUSINESS_ROLE_TERMS):
        return False

    return bool(query_terms & chunk_terms) or _contains_filtering_logic(chunk)


def _has_filtering_logic(
    chunk: CodeChunk, *, query_terms: set[str], chunk_terms: set[str]
) -> bool:
    if chunk.chunk_type not in {"function", "method"}:
        return False
    if chunk.name.startswith("__") and chunk.name.endswith("__"):
        return False
    if not (query_terms & _FILTERING_TERMS):
        return False
    if not (chunk_terms & _FILTERING_TERMS):
        return False
    return _COMPARISON_OR_FILTER_RE.search(chunk.source_code) is not None


def _has_state_update_logic(
    chunk: CodeChunk, *, query_terms: set[str], chunk_terms: set[str]
) -> bool:
    if chunk.chunk_type not in {"function", "method"}:
        return False
    if chunk.name.startswith("__") and chunk.name.endswith("__"):
        return False
    if not (query_terms & _STATE_UPDATE_TERMS):
        return False
    if not (chunk_terms & _STATE_UPDATE_TERMS):
        return False
    if _contains_attribute_mutation(chunk):
        return True

    called_terms = set(called_paired_operation_terms(chunk))
    name_terms = set(identifier_tokens(chunk.name))
    operation_name_terms = name_terms & _STATE_UPDATE_TERMS
    return bool(
        (
            _calls_domain_paired_operation(chunk)
            and query_terms & (called_terms | operation_name_terms)
        )
        or (called_terms and query_terms & (called_terms | operation_name_terms))
    )


def _has_validation_logic(chunk: CodeChunk) -> bool:
    lower_name = chunk.name.lower().lstrip("_")
    return (
        has_validation_name(chunk.name)
        or lower_name.startswith(_INVESTIGATION_VALIDATION_PREFIXES)
        or calls_validation_helper(chunk)
    )


def _paired_operation_reason(
    chunk: CodeChunk, *, query_terms: set[str], chunk_terms: set[str]
) -> str | None:
    if chunk.chunk_type not in {"function", "method"}:
        return None
    if chunk.name.startswith("__") and chunk.name.endswith("__"):
        return None

    called_terms = set(called_paired_operation_terms(chunk))
    counterpart_terms = set(counterpart_terms_for_called_operations(chunk))
    if called_terms and counterpart_terms and (
        query_terms & (called_terms | counterpart_terms)
        or (
            _calls_domain_paired_operation(chunk)
            and query_terms
            & (called_terms | (set(identifier_tokens(chunk.name)) & _STATE_UPDATE_TERMS))
        )
    ):
        return "possible missing counterpart operation"

    defined_operation_terms = set(identifier_tokens(chunk.name)) & set(
        called_paired_operation_terms(chunk)
    )
    if not defined_operation_terms:
        defined_operation_terms = set(identifier_tokens(chunk.name)) & chunk_terms
    if (
        chunk_defines_paired_operation(chunk)
        and _contains_attribute_mutation(chunk)
        and query_terms & (defined_operation_terms | chunk_terms)
    ):
        return "paired state operation"

    return None


def _is_generic_data_access(chunk: CodeChunk) -> bool:
    name_terms = set(identifier_tokens(chunk.name))
    parent_terms = set(identifier_tokens(chunk.parent or ""))
    path_terms = set(identifier_tokens(Path(chunk.file_path).stem))
    if not ((parent_terms | path_terms) & _DATA_ACCESS_TERMS):
        return False
    return bool(name_terms & _GENERIC_DATA_ACCESS_NAMES)


def _class_has_specific_business_signal(
    chunk: CodeChunk, *, query_terms: set[str], chunk_terms: set[str]
) -> bool:
    return bool(query_terms & chunk_terms & (_FILTERING_TERMS | _STATE_UPDATE_TERMS))


def _contains_filtering_logic(chunk: CodeChunk) -> bool:
    return _COMPARISON_OR_FILTER_RE.search(chunk.source_code) is not None


def _contains_attribute_mutation(chunk: CodeChunk) -> bool:
    return _ATTRIBUTE_MUTATION_RE.search(chunk.source_code) is not None


def _calls_domain_paired_operation(chunk: CodeChunk) -> bool:
    ignored_heads = {"repository", "repo", "storage", "store", "database", "db", "self", "cls"}
    for dependency in chunk.dependencies:
        parts = [part for part in dependency.split(".") if part]
        if not parts:
            continue
        if not (set(identifier_tokens(parts[-1])) & set(called_paired_operation_terms(chunk))):
            continue
        if len(parts) == 1:
            return True
        head_terms = set(identifier_tokens(parts[0]))
        if head_terms & ignored_heads:
            continue
        return True
    return False


def _investigation_sort_key(
    item: InvestigationScoredResult,
) -> tuple[float, bool, str, int, str, str]:
    chunk = item.chunk
    return (
        -item.score,
        is_test_path(chunk.file_path),
        normalize_path(chunk.file_path),
        chunk.start_line,
        chunk.id,
        chunk.name,
    )


def _chunk_display_name(chunk: CodeChunk) -> str:
    if chunk.chunk_type == "method" and chunk.parent:
        return f"{chunk.parent}.{chunk.name}"
    return chunk.name


def _chunk_file_path(chunk: CodeChunk, *, repo_path: Path) -> str:
    try:
        return Path(chunk.file_path).relative_to(repo_path).as_posix()
    except ValueError:
        return display_path(chunk.file_path)


def _dedupe_reasons(reasons: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for reason in reasons:
        if reason in seen:
            continue
        seen.add(reason)
        result.append(reason)
    return tuple(result)
