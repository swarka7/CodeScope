from __future__ import annotations

from dataclasses import dataclass

from codescope.models.code_chunk import CodeChunk


@dataclass(frozen=True, slots=True)
class DependencyGraph:
    """A minimal, name-based dependency graph over extracted code chunks.

    This graph intentionally uses exact name matching only (no symbol resolution).
    """

    _chunks_by_name: dict[str, list[CodeChunk]]

    def __init__(self, chunks: list[CodeChunk]) -> None:
        chunks_by_name: dict[str, list[CodeChunk]] = {}
        for chunk in chunks:
            chunks_by_name.setdefault(chunk.name, []).append(chunk)
            if chunk.parent:
                chunks_by_name.setdefault(f"{chunk.parent}.{chunk.name}", []).append(chunk)

        object.__setattr__(self, "_chunks_by_name", chunks_by_name)

    def related_chunks(self, chunk: CodeChunk) -> list[CodeChunk]:
        related: list[CodeChunk] = []
        seen_ids: set[str] = set()

        for dep in chunk.dependencies:
            for candidate in self._chunks_by_name.get(dep, []):
                if candidate.id == chunk.id:
                    continue
                if candidate.id in seen_ids:
                    continue
                seen_ids.add(candidate.id)
                related.append(candidate)

        return related

