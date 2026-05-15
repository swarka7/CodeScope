from __future__ import annotations

from pathlib import Path

from codescope.indexing.index_compatibility import (
    MISSING_INDEX_MESSAGE,
    OUTDATED_INDEX_MESSAGE,
    check_index_compatibility,
)
from codescope.indexing.index_store import IndexStore
from codescope.indexing.index_versions import EMBEDDING_TEXT_VERSION, INDEX_SCHEMA_VERSION
from codescope.models.code_chunk import CodeChunk


def test_missing_index_is_incompatible(tmp_path: Path) -> None:
    result = check_index_compatibility(
        index_store=IndexStore(tmp_path),
        embedding_model_name="all-MiniLM-L6-v2",
    )

    assert result.compatible is False
    assert result.reason == "missing"
    assert result.message == MISSING_INDEX_MESSAGE
    assert result.requires_rebuild is False


def test_compatible_index_is_accepted(tmp_path: Path) -> None:
    store = _write_index(tmp_path, _metadata())

    result = check_index_compatibility(
        index_store=store,
        embedding_model_name="all-MiniLM-L6-v2",
    )

    assert result.compatible is True
    assert result.reason == "compatible"
    assert result.message == ""
    assert result.requires_rebuild is False


def test_stale_schema_version_requires_rebuild(tmp_path: Path) -> None:
    store = _write_index(tmp_path, _metadata(index_schema_version=1))

    result = check_index_compatibility(
        index_store=store,
        embedding_model_name="all-MiniLM-L6-v2",
    )

    assert result.compatible is False
    assert result.reason == "outdated"
    assert result.message == OUTDATED_INDEX_MESSAGE
    assert result.requires_rebuild is True


def test_stale_embedding_text_version_requires_rebuild(tmp_path: Path) -> None:
    store = _write_index(tmp_path, _metadata(embedding_text_version=EMBEDDING_TEXT_VERSION - 1))

    result = check_index_compatibility(
        index_store=store,
        embedding_model_name="all-MiniLM-L6-v2",
    )

    assert result.compatible is False
    assert result.reason == "outdated"
    assert result.message == OUTDATED_INDEX_MESSAGE
    assert result.requires_rebuild is True


def test_embedding_model_mismatch_requires_rebuild(tmp_path: Path) -> None:
    store = _write_index(tmp_path, _metadata(embedding_model_name="old-model"))

    result = check_index_compatibility(
        index_store=store,
        embedding_model_name="all-MiniLM-L6-v2",
    )

    assert result.compatible is False
    assert result.reason == "outdated"
    assert result.message == OUTDATED_INDEX_MESSAGE
    assert result.requires_rebuild is True


def _write_index(repo_path: Path, metadata: dict[str, object]) -> IndexStore:
    repo_path.mkdir(exist_ok=True)
    store = IndexStore(repo_path)
    store.save(chunks=[_chunk()], embeddings=[[1.0, 0.0]], metadata=metadata)
    return store


def _metadata(
    *,
    index_schema_version: int = INDEX_SCHEMA_VERSION,
    embedding_text_version: int = EMBEDDING_TEXT_VERSION,
    embedding_model_name: str = "all-MiniLM-L6-v2",
) -> dict[str, object]:
    return {
        "index_schema_version": index_schema_version,
        "embedding_text_version": embedding_text_version,
        "embedding_model_name": embedding_model_name,
        "chunks_indexed": 1,
        "files_indexed": 1,
    }


def _chunk() -> CodeChunk:
    return CodeChunk(
        id="chunk",
        file_path="app.py",
        chunk_type="function",
        name="run",
        parent=None,
        start_line=1,
        end_line=2,
        source_code="def run() -> None:\n    pass\n",
        imports=[],
        dependencies=[],
    )
