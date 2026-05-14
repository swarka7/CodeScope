from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from codescope.embeddings.embedder import Embedder
from codescope.indexing.index_store import IndexStore
from codescope.models.code_chunk import CodeChunk
from codescope.parser.ast_parser import AstParser
from codescope.parser.chunker import Chunker
from codescope.scanner.repo_scanner import RepoScanner


@dataclass(frozen=True, slots=True)
class IndexUpdateSummary:
    indexed_files: int
    reused_files: int
    removed_files: int
    total_chunks: int


class Indexer:
    def __init__(
        self,
        repo_path: Path,
        *,
        store: IndexStore | None = None,
        scanner: RepoScanner | None = None,
        parser: AstParser | None = None,
        chunker: Chunker | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._repo_path = Path(repo_path)
        self._store = store or IndexStore(self._repo_path)
        self._scanner = scanner or RepoScanner()
        self._parser = parser or AstParser()
        self._chunker = chunker or Chunker()
        self._embedder = embedder or Embedder()

    def index(self) -> IndexUpdateSummary:
        files = self._scanner.scan(self._repo_path)
        file_stats_by_path = {
            file_path.relative_to(self._repo_path).as_posix(): _file_stat_fingerprint(file_path)
            for file_path in files
        }

        previous_file_meta: dict[str, dict[str, Any]] = {}
        previous_chunks: list[CodeChunk] = []
        previous_embeddings: list[list[float]] = []
        previous_metadata: dict[str, Any] = {}

        if self._store.exists():
            previous_chunks, previous_embeddings, previous_metadata = self._store.load()
            previous_file_meta = _metadata_files_by_path(previous_metadata)

        deleted_paths = set(previous_file_meta) - set(file_stats_by_path)

        unchanged_paths: set[str] = set()
        changed_paths: set[str] = set()
        updated_file_meta_by_path = dict(previous_file_meta)

        for rel_path, stat_fp in file_stats_by_path.items():
            old = previous_file_meta.get(rel_path)
            if old is None:
                changed_paths.add(rel_path)
                continue

            old_sha256 = old.get("sha256")
            if isinstance(old_sha256, str) and old_sha256:
                current_sha256 = _sha256(self._repo_path / rel_path)
                if current_sha256 == old_sha256:
                    unchanged_paths.add(rel_path)
                    updated_file_meta_by_path[rel_path] = {
                        "path": rel_path,
                        "mtime_ns": stat_fp["mtime_ns"],
                        "size": stat_fp["size"],
                        "sha256": old_sha256,
                    }
                else:
                    changed_paths.add(rel_path)
                continue

            # Backward-compatibility: older indexes may not include sha256 per file.
            # Reindex these files once to populate the content hash and ensure correctness.
            changed_paths.add(rel_path)

        for rel_path in deleted_paths:
            updated_file_meta_by_path.pop(rel_path, None)

        old_by_file = _group_by_file(
            previous_chunks, previous_embeddings, repo_path=self._repo_path
        )

        new_by_file: dict[str, list[tuple[CodeChunk, list[float]]]] = {}
        if changed_paths:
            changed_files = [self._repo_path / rel_path for rel_path in sorted(changed_paths)]
            new_chunks: list[CodeChunk] = []
            new_chunk_paths: list[str] = []
            for file_path in changed_files:
                parsed = self._parser.parse_file(file_path)
                file_chunks = self._chunker.extract_chunks(parsed)
                new_chunks.extend(file_chunks)
                rel_path = file_path.relative_to(self._repo_path).as_posix()
                new_chunk_paths.extend([rel_path] * len(file_chunks))
                stat_fp = file_stats_by_path[rel_path]
                updated_file_meta_by_path[rel_path] = {
                    "path": rel_path,
                    "mtime_ns": stat_fp["mtime_ns"],
                    "size": stat_fp["size"],
                    "sha256": _sha256(file_path),
                }

            embeddings: list[list[float]] = []
            if new_chunks:
                embeddings = self._embedder.embed_chunks(new_chunks)

            for chunk, embedding, rel_path in zip(
                new_chunks, embeddings, new_chunk_paths, strict=True
            ):
                new_by_file.setdefault(rel_path, []).append((chunk, embedding))

            # Handle files that produced no chunks but were changed.
            for rel_path in changed_paths:
                new_by_file.setdefault(rel_path, [])
                if rel_path not in updated_file_meta_by_path:
                    stat_fp = file_stats_by_path[rel_path]
                    updated_file_meta_by_path[rel_path] = {
                        "path": rel_path,
                        "mtime_ns": stat_fp["mtime_ns"],
                        "size": stat_fp["size"],
                        "sha256": _sha256(self._repo_path / rel_path),
                    }

        combined: list[tuple[CodeChunk, list[float]]] = []
        for file_path in files:
            rel_path = file_path.relative_to(self._repo_path).as_posix()
            if rel_path in unchanged_paths:
                combined.extend(old_by_file.get(rel_path, []))
            else:
                combined.extend(new_by_file.get(rel_path, []))

        combined_chunks = [item[0] for item in combined]
        combined_embeddings = [item[1] for item in combined]

        now = datetime.now(UTC).isoformat()
        created_at = previous_metadata.get("created_at") if previous_metadata else None

        metadata: dict[str, Any] = {
            "schema_version": 2,
            "created_at": created_at or now,
            "updated_at": now,
            "files_indexed": len(files),
            "chunks_indexed": len(combined_chunks),
            "files": _sorted_file_metadata(updated_file_meta_by_path),
        }

        self._store.save(chunks=combined_chunks, embeddings=combined_embeddings, metadata=metadata)

        return IndexUpdateSummary(
            indexed_files=len(changed_paths),
            reused_files=len(unchanged_paths),
            removed_files=len(deleted_paths),
            total_chunks=len(combined_chunks),
        )


def _file_stat_fingerprint(file_path: Path) -> dict[str, int]:
    stat = file_path.stat()
    return {
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
    }


def _sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 64), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def _metadata_files_by_path(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    files = metadata.get("files")
    if not isinstance(files, list):
        return {}

    by_path: dict[str, dict[str, Any]] = {}
    for entry in files:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        if isinstance(path, str) and path:
            by_path[path] = entry
    return by_path


def _sorted_file_metadata(file_meta_by_path: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [file_meta_by_path[path] for path in sorted(file_meta_by_path)]


def _fingerprints_match(old: dict[str, Any], current: dict[str, Any]) -> bool:
    return (
        isinstance(old.get("mtime_ns"), int)
        and isinstance(old.get("size"), int)
        and old.get("mtime_ns") == current.get("mtime_ns")
        and old.get("size") == current.get("size")
    )


def _group_by_file(
    chunks: list[CodeChunk], embeddings: list[list[float]], *, repo_path: Path
) -> dict[str, list[tuple[CodeChunk, list[float]]]]:
    grouped: dict[str, list[tuple[CodeChunk, list[float]]]] = {}
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        rel_path = _chunk_relative_path(chunk.file_path, repo_path=repo_path)
        grouped.setdefault(rel_path, []).append((chunk, embedding))
    return grouped


def _chunk_relative_path(chunk_file_path: str, *, repo_path: Path) -> str:
    path = Path(chunk_file_path)
    try:
        return path.relative_to(repo_path).as_posix()
    except ValueError:
        pass

    try:
        return path.resolve().relative_to(repo_path.resolve()).as_posix()
    except (OSError, RuntimeError, ValueError):
        return path.as_posix()
