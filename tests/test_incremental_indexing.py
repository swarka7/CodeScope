from __future__ import annotations

import os
from pathlib import Path

from codescope.embeddings.embedder import Embedder
from codescope.indexing.index_store import IndexStore
from codescope.indexing.indexer import Indexer


class _CountingModel:
    def __init__(self) -> None:
        self.document_calls = 0

    def encode_document(self, texts: list[str], *, normalize_embeddings: bool) -> list[list[float]]:
        assert normalize_embeddings is True
        self.document_calls += 1
        return [[float(len(text)), 0.0, 0.0] for text in texts]


def test_first_full_index_builds_index_and_saves_file_metadata(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "a.py").write_text("def foo() -> int:\n    return 1\n", encoding="utf-8")

    model = _CountingModel()
    embedder = Embedder(model=model)

    summary = Indexer(repo_path, embedder=embedder).index()

    assert summary.indexed_files == 1
    assert summary.reused_files == 0
    assert summary.removed_files == 0
    assert summary.total_chunks == 1
    assert model.document_calls == 1

    chunks, embeddings, metadata = IndexStore(repo_path).load()
    assert len(chunks) == 1
    assert len(embeddings) == 1
    assert metadata["files_indexed"] == 1
    assert metadata["chunks_indexed"] == 1
    assert isinstance(metadata.get("files"), list)

    file_meta = metadata["files"][0]
    assert file_meta["path"] == "a.py"
    assert isinstance(file_meta["mtime_ns"], int)
    assert isinstance(file_meta["size"], int)
    assert isinstance(file_meta["sha256"], str)


def test_second_unchanged_index_reuses_existing_embeddings(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "a.py").write_text("def foo() -> int:\n    return 1\n", encoding="utf-8")

    model = _CountingModel()
    embedder = Embedder(model=model)
    indexer = Indexer(repo_path, embedder=embedder)

    first = indexer.index()
    second = indexer.index()

    assert first.total_chunks == 1
    assert second.indexed_files == 0
    assert second.reused_files == 1
    assert second.removed_files == 0
    assert second.total_chunks == 1
    assert model.document_calls == 1


def test_changed_file_gets_reindexed(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    file_path = repo_path / "a.py"
    file_path.write_text("def foo() -> int:\n    return 1\n", encoding="utf-8")

    model = _CountingModel()
    embedder = Embedder(model=model)
    indexer = Indexer(repo_path, embedder=embedder)

    first = indexer.index()
    before = file_path.stat()
    file_path.write_text("def foo() -> int:\n    return 2\n", encoding="utf-8")
    # Some Windows filesystems have coarse mtime resolution; keep the fingerprint stable.
    os.utime(file_path, ns=(before.st_atime_ns, before.st_mtime_ns))
    second = indexer.index()

    assert first.total_chunks == 1
    assert second.indexed_files == 1
    assert second.reused_files == 0
    assert second.total_chunks == 1
    assert model.document_calls == 2


def test_deleted_file_chunks_are_removed(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "a.py").write_text("def foo() -> int:\n    return 1\n", encoding="utf-8")
    file_b = repo_path / "b.py"
    file_b.write_text("def bar() -> int:\n    return 2\n", encoding="utf-8")

    model = _CountingModel()
    embedder = Embedder(model=model)
    indexer = Indexer(repo_path, embedder=embedder)

    first = indexer.index()
    file_b.unlink()
    second = indexer.index()

    assert first.total_chunks == 2
    assert second.indexed_files == 0
    assert second.reused_files == 1
    assert second.removed_files == 1
    assert second.total_chunks == 1
    assert model.document_calls == 1


def test_new_file_is_added(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "a.py").write_text("def foo() -> int:\n    return 1\n", encoding="utf-8")

    model = _CountingModel()
    embedder = Embedder(model=model)
    indexer = Indexer(repo_path, embedder=embedder)

    first = indexer.index()
    (repo_path / "b.py").write_text("def bar() -> int:\n    return 2\n", encoding="utf-8")
    second = indexer.index()

    assert first.total_chunks == 1
    assert second.indexed_files == 1
    assert second.reused_files == 1
    assert second.removed_files == 0
    assert second.total_chunks == 2
    assert model.document_calls == 2

