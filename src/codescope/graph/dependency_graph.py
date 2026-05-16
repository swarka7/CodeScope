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
    _chunks: tuple[CodeChunk, ...]

    def __init__(self, chunks: list[CodeChunk]) -> None:
        chunks_by_name: dict[str, list[CodeChunk]] = {}
        for chunk in chunks:
            chunks_by_name.setdefault(chunk.name, []).append(chunk)
            if chunk.parent:
                chunks_by_name.setdefault(f"{chunk.parent}.{chunk.name}", []).append(chunk)

        object.__setattr__(self, "_chunks_by_name", chunks_by_name)
        object.__setattr__(self, "_resolver", SymbolResolver(chunks))
        object.__setattr__(self, "_chunks", tuple(chunks))

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

    def chunks(self) -> tuple[CodeChunk, ...]:
        return self._chunks

    def related_candidates(self, chunk: CodeChunk) -> list[tuple[str, CodeChunk]]:
        """Return (dependency_name, candidate_chunk) pairs for a given chunk."""

        candidates: list[tuple[str, CodeChunk]] = []
        for dep in chunk.dependencies:
            for matched_name, candidate in self._resolve_dependency(dep, source_chunk=chunk):
                if candidate.id == chunk.id:
                    continue
                candidates.append((matched_name, candidate))
        return candidates

    def reverse_candidates(self, chunk: CodeChunk) -> list[tuple[str, CodeChunk]]:
        """Return (dependency_name, caller_chunk) pairs that resolve to the given chunk."""

        callers: list[tuple[str, CodeChunk]] = []
        seen: set[tuple[str, str]] = set()
        for source in self._chunks:
            if source.id == chunk.id:
                continue
            for dep in source.dependencies:
                for matched_name, candidate in self._resolve_dependency(dep, source_chunk=source):
                    if candidate.id != chunk.id:
                        continue
                    key = (matched_name, source.id)
                    if key in seen:
                        continue
                    seen.add(key)
                    callers.append((matched_name, source))

        callers.sort(
            key=lambda item: (
                normalize_path(item[1].file_path),
                item[1].chunk_type,
                item[1].parent or "",
                item[1].name,
                item[0],
            )
        )
        return callers

    def _resolve_dependency(
        self, dependency_name: str, *, source_chunk: CodeChunk
    ) -> list[tuple[str, CodeChunk]]:
        resolved = self._resolver.resolve(dependency_name, source_chunk=source_chunk)
        if resolved:
            return [(item.matched_name, item.chunk) for item in resolved]

        return [
            (dependency_name, candidate)
            for candidate in _safe_exact_fallback_candidates(
                dependency_name,
                source_chunk=source_chunk,
                chunks_by_name=self._chunks_by_name,
            )
        ]


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
