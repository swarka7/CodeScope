from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from codescope.models.code_chunk import CodeChunk


class IndexStore:
    """Persists and loads a local CodeScope index under `<repo_path>/.codescope/`.

    Storage is intentionally simple JSON for now (readable and easy to debug).
    """

    index_dir_name = ".codescope"
    chunks_file_name = "chunks.json"
    embeddings_file_name = "embeddings.json"
    metadata_file_name = "index_metadata.json"

    def __init__(self, repo_path: Path) -> None:
        self._repo_path = Path(repo_path)
        self._index_dir = self._repo_path / self.index_dir_name

    @property
    def index_dir(self) -> Path:
        return self._index_dir

    def exists(self) -> bool:
        return (
            self._index_dir.is_dir()
            and (self._index_dir / self.chunks_file_name).is_file()
            and (self._index_dir / self.embeddings_file_name).is_file()
            and (self._index_dir / self.metadata_file_name).is_file()
        )

    def save(
        self, *, chunks: list[CodeChunk], embeddings: list[list[float]], metadata: dict[str, Any]
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        self._index_dir.mkdir(parents=True, exist_ok=True)

        chunk_dicts = [asdict(chunk) for chunk in chunks]
        embedding_rows = [[float(x) for x in row] for row in embeddings]

        writes = [
            (self._index_dir / self.chunks_file_name, self._json_text(chunk_dicts)),
            (self._index_dir / self.embeddings_file_name, self._json_text(embedding_rows)),
            (self._index_dir / self.metadata_file_name, self._json_text(metadata)),
        ]
        self._write_files_atomically(writes)

    def load(self) -> tuple[list[CodeChunk], list[list[float]], dict[str, Any]]:
        if not self.exists():
            raise FileNotFoundError(f"No CodeScope index found at: {self._index_dir}")

        chunks_data = self._read_json(self._index_dir / self.chunks_file_name)
        embeddings_data = self._read_json(self._index_dir / self.embeddings_file_name)
        metadata = self._read_json(self._index_dir / self.metadata_file_name)

        if not isinstance(chunks_data, list):
            raise ValueError("chunks.json must contain a JSON array")
        if not isinstance(embeddings_data, list):
            raise ValueError("embeddings.json must contain a JSON array")
        if not isinstance(metadata, dict):
            raise ValueError("index_metadata.json must contain a JSON object")

        chunks = [_chunk_from_mapping(_expect_mapping(item)) for item in chunks_data]
        embeddings: list[list[float]] = []
        for row in embeddings_data:
            if not isinstance(row, list):
                raise ValueError("embeddings.json must contain a list of lists")
            embeddings.append([float(x) for x in row])

        if len(chunks) != len(embeddings):
            raise ValueError("chunks.json and embeddings.json length mismatch")

        return chunks, embeddings, metadata

    def load_metadata(self) -> dict[str, Any]:
        if not self.exists():
            raise FileNotFoundError(f"No CodeScope index found at: {self._index_dir}")

        metadata = self._read_json(self._index_dir / self.metadata_file_name)
        if not isinstance(metadata, dict):
            raise ValueError("index_metadata.json must contain a JSON object")
        return metadata

    @classmethod
    def _json_text(cls, data: Any) -> str:
        text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
        return text + "\n"

    @classmethod
    def _write_files_atomically(cls, writes: list[tuple[Path, str]]) -> None:
        temp_paths: list[Path] = []
        backup_paths: list[tuple[Path, Path]] = []
        backup_candidates: list[Path] = []
        replaced_without_backup: list[Path] = []

        try:
            for final_path, text in writes:
                temp_path = _temp_path_for(final_path)
                _remove_if_exists(temp_path)
                cls._write_text_fsync(temp_path, text)
                temp_paths.append(temp_path)

            for final_path, _ in writes:
                if not final_path.exists():
                    continue
                backup_path = _backup_path_for(final_path)
                backup_candidates.append(backup_path)
                _remove_if_exists(backup_path)
                cls._copy_file_fsync(final_path, backup_path)
                backup_paths.append((final_path, backup_path))

            backed_up = {final_path for final_path, _ in backup_paths}
            for final_path, _ in writes:
                temp_path = _temp_path_for(final_path)
                os.replace(temp_path, final_path)
                if final_path not in backed_up:
                    replaced_without_backup.append(final_path)

        except BaseException:
            cls._rollback_atomic_write(
                backup_paths=backup_paths,
                replaced_without_backup=replaced_without_backup,
            )
            for backup_path in backup_candidates:
                _remove_if_exists(backup_path)
            raise
        finally:
            for temp_path in temp_paths:
                _remove_if_exists(temp_path)

        for _, backup_path in backup_paths:
            _remove_if_exists(backup_path)

    @classmethod
    def _rollback_atomic_write(
        cls, *, backup_paths: list[tuple[Path, Path]], replaced_without_backup: list[Path]
    ) -> None:
        for final_path in reversed(replaced_without_backup):
            _remove_if_exists(final_path)

        for final_path, backup_path in reversed(backup_paths):
            if backup_path.exists():
                os.replace(backup_path, final_path)

    @staticmethod
    def _write_text_fsync(path: Path, text: str) -> None:
        with path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _copy_file_fsync(source: Path, target: Path) -> None:
        with source.open("rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())

    @staticmethod
    def _read_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))


def _expect_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise ValueError(f"Expected object entries, got: {type(value)!r}")


def _chunk_from_mapping(value: dict[str, Any]) -> CodeChunk:
    data = dict(value)
    data.setdefault("decorators", [])
    return CodeChunk(**data)


def _temp_path_for(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _backup_path_for(path: Path) -> Path:
    return path.with_name(f".{path.name}.bak")


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
