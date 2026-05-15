from __future__ import annotations

from dataclasses import dataclass

from codescope.graph.symbol_resolver import SymbolResolver
from codescope.models.code_chunk import CodeChunk
from codescope.utils.path_utils import normalize_path


@dataclass(frozen=True, slots=True)
class DependencyGraph:
    """A minimal, name-based dependency graph over extracted code chunks.

    This graph prefers basic static symbol resolution and falls back to exact name matching.
    """

    _chunks_by_name: dict[str, list[CodeChunk]]
    _resolver: SymbolResolver

    def __init__(self, chunks: list[CodeChunk]) -> None:
        chunks_by_name: dict[str, list[CodeChunk]] = {}
        for chunk in chunks:
            chunks_by_name.setdefault(chunk.name, []).append(chunk)
            if chunk.parent:
                chunks_by_name.setdefault(f"{chunk.parent}.{chunk.name}", []).append(chunk)

        object.__setattr__(self, "_chunks_by_name", chunks_by_name)
        object.__setattr__(self, "_resolver", SymbolResolver(chunks))

    def related_chunks(self, chunk: CodeChunk) -> list[CodeChunk]:
        related: list[CodeChunk] = []
        seen_ids: set[str] = set()

        for _, candidate in self.related_candidates(chunk):
            if candidate.id == chunk.id:
                continue
            if candidate.id in seen_ids:
                continue
            seen_ids.add(candidate.id)
            related.append(candidate)

        return related

    def related_candidates(self, chunk: CodeChunk) -> list[tuple[str, CodeChunk]]:
        """Return (dependency_name, candidate_chunk) pairs for a given chunk."""

        candidates: list[tuple[str, CodeChunk]] = []
        for dep in chunk.dependencies:
            resolved = self._resolver.resolve(dep, source_chunk=chunk)
            if resolved:
                for item in resolved:
                    if item.chunk.id == chunk.id:
                        continue
                    candidates.append((item.matched_name, item.chunk))
                continue

            for candidate in _safe_exact_fallback_candidates(
                dep, source_chunk=chunk, chunks_by_name=self._chunks_by_name
            ):
                if candidate.id == chunk.id:
                    continue
                candidates.append((dep, candidate))
        return candidates


def _safe_exact_fallback_candidates(
    dependency_name: str,
    *,
    source_chunk: CodeChunk,
    chunks_by_name: dict[str, list[CodeChunk]],
) -> list[CodeChunk]:
    candidates = chunks_by_name.get(dependency_name, [])
    if len(candidates) <= 1:
        return candidates

    source_path = normalize_path(source_chunk.file_path)
    same_file = [chunk for chunk in candidates if normalize_path(chunk.file_path) == source_path]
    if same_file:
        return same_file

    return []
