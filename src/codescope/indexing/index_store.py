from __future__ import annotations

import json
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

        self._write_json(self._index_dir / self.chunks_file_name, chunk_dicts)
        self._write_json(self._index_dir / self.embeddings_file_name, embedding_rows)
        self._write_json(self._index_dir / self.metadata_file_name, metadata)

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

        chunks = [CodeChunk(**_expect_mapping(item)) for item in chunks_data]
        embeddings: list[list[float]] = []
        for row in embeddings_data:
            if not isinstance(row, list):
                raise ValueError("embeddings.json must contain a list of lists")
            embeddings.append([float(x) for x in row])

        if len(chunks) != len(embeddings):
            raise ValueError("chunks.json and embeddings.json length mismatch")

        return chunks, embeddings, metadata

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
        path.write_text(text + "\n", encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))


def _expect_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise ValueError(f"Expected object entries, got: {type(value)!r}")

