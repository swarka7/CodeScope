from __future__ import annotations

import json
import os
from pathlib import Path

import codescope.cli as cli_module
from codescope.embeddings.embedder import Embedder
from codescope.indexing.index_store import IndexStore
from codescope.indexing.index_versions import EMBEDDING_TEXT_VERSION, INDEX_SCHEMA_VERSION
from codescope.indexing.indexer import Indexer, IndexUpdateSummary


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
    assert metadata["index_schema_version"] == INDEX_SCHEMA_VERSION
    assert metadata["embedding_text_version"] == EMBEDDING_TEXT_VERSION
    assert metadata["embedding_model_name"] == "all-MiniLM-L6-v2"
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
    assert second.rebuilt_full_index is False


def test_old_missing_index_version_triggers_full_rebuild(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "a.py").write_text("def foo() -> int:\n    return 1\n", encoding="utf-8")

    model = _CountingModel()
    embedder = Embedder(model=model)
    indexer = Indexer(repo_path, embedder=embedder)

    first = indexer.index()
    _rewrite_metadata(repo_path, remove_keys=["index_schema_version", "embedding_text_version"])
    second = indexer.index()

    assert first.total_chunks == 1
    assert second.rebuilt_full_index is True
    assert second.indexed_files == 1
    assert second.reused_files == 0
    assert second.total_chunks == 1
    assert model.document_calls == 2


def test_embedding_model_mismatch_triggers_full_rebuild(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "a.py").write_text("def foo() -> int:\n    return 1\n", encoding="utf-8")

    first_model = _CountingModel()
    second_model = _CountingModel()

    first = Indexer(
        repo_path,
        embedder=Embedder(model_name="local-model-a", model=first_model),
    ).index()
    second = Indexer(
        repo_path,
        embedder=Embedder(model_name="local-model-b", model=second_model),
    ).index()

    assert first.total_chunks == 1
    assert second.rebuilt_full_index is True
    assert second.indexed_files == 1
    assert second.reused_files == 0
    assert first_model.document_calls == 1
    assert second_model.document_calls == 1


def test_cli_index_reports_outdated_rebuild(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    class _FakeIndexer:
        def __init__(self, repo_path: Path) -> None:
            self.repo_path = repo_path

        def index(self) -> IndexUpdateSummary:
            return IndexUpdateSummary(
                indexed_files=3,
                reused_files=0,
                removed_files=0,
                total_chunks=7,
                rebuilt_full_index=True,
            )

    monkeypatch.setattr(cli_module, "Indexer", _FakeIndexer)

    exit_code = cli_module.main(["index", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Existing index is outdated; rebuilding full index." in captured.out


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


def _rewrite_metadata(
    repo_path: Path,
    *,
    remove_keys: list[str] | None = None,
    updates: dict[str, object] | None = None,
) -> None:
    metadata_path = repo_path / ".codescope" / "index_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key in remove_keys or []:
        metadata.pop(key, None)
    metadata.update(updates or {})
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", "utf-8")

