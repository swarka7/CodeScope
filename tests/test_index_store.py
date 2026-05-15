from __future__ import annotations

import json
from pathlib import Path

import pytest

from codescope.cli import main as cli_main
from codescope.indexing.index_store import IndexStore
from codescope.indexing.index_versions import EMBEDDING_TEXT_VERSION, INDEX_SCHEMA_VERSION
from codescope.models.code_chunk import CodeChunk


def test_index_store_detects_missing_index(tmp_path: Path) -> None:
    store = IndexStore(tmp_path)

    assert store.exists() is False
    with pytest.raises(FileNotFoundError):
        store.load()


def test_index_store_saves_and_loads_index(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    store = IndexStore(repo_path)
    file_a = (repo_path / "a.py").as_posix()
    file_b = (repo_path / "b.py").as_posix()

    chunks = [
        _chunk(
            id="a",
            file_path=file_a,
            chunk_type="function",
            name="main",
            dependencies=["helper", "repo.save"],
            imports=["import os"],
        ),
        _chunk(
            id="b",
            file_path=file_b,
            chunk_type="class",
            name="Repo",
            dependencies=[],
            imports=[],
        ),
    ]
    embeddings = [
        [1.0, 2.5, 0.0],
        [0.0, 1.0, 3.0],
    ]
    metadata = {"schema_version": 1, "chunks_indexed": 2, "files_indexed": 2}

    store.save(chunks=chunks, embeddings=embeddings, metadata=metadata)
    assert store.exists() is True

    loaded_chunks, loaded_embeddings, loaded_metadata = store.load()

    assert loaded_chunks == chunks
    assert loaded_embeddings == embeddings
    assert loaded_metadata == metadata
    assert loaded_chunks[0].dependencies == ["helper", "repo.save"]


def test_index_store_loads_old_chunks_without_decorators(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    store = IndexStore(repo_path)
    store.index_dir.mkdir()
    (store.index_dir / "chunks.json").write_text(
        json.dumps(
            [
                {
                    "id": "old",
                    "file_path": "app.py",
                    "chunk_type": "function",
                    "name": "route",
                    "parent": None,
                    "start_line": 1,
                    "end_line": 2,
                    "source_code": "def route():\n    return None\n",
                    "imports": [],
                    "dependencies": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    (store.index_dir / "embeddings.json").write_text("[[1.0, 0.0]]\n", encoding="utf-8")
    (store.index_dir / "index_metadata.json").write_text(
        '{"schema_version": 1}\n',
        encoding="utf-8",
    )

    chunks, embeddings, metadata = store.load()

    assert chunks[0].decorators == []
    assert embeddings == [[1.0, 0.0]]
    assert metadata == {"schema_version": 1}


def test_search_requires_existing_index(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "a.py").write_text("def x():\n    return 1\n", encoding="utf-8")

    exit_code = cli_main(["search", str(repo_path), "query"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.err.strip() == (
        "No CodeScope index found. Run: python -m codescope.cli index <repo_path>"
    )


def test_search_rejects_index_with_missing_versions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    store = IndexStore(repo_path)
    store.save(
        chunks=[
            _chunk(
                id="route",
                file_path=(repo_path / "app.py").as_posix(),
                chunk_type="function",
                name="route",
                dependencies=[],
                imports=[],
            )
        ],
        embeddings=[[1.0, 0.0]],
        metadata={"schema_version": 1, "chunks_indexed": 1, "files_indexed": 1},
    )

    exit_code = cli_main(["search", str(repo_path), "route"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.err.strip() == (
        "Index is outdated. Run: python -m codescope.cli index <repo_path>"
    )


def test_search_rejects_embedding_model_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    store = IndexStore(repo_path)
    store.save(
        chunks=[
            _chunk(
                id="route",
                file_path=(repo_path / "app.py").as_posix(),
                chunk_type="function",
                name="route",
                dependencies=[],
                imports=[],
            )
        ],
        embeddings=[[1.0, 0.0]],
        metadata={
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "embedding_text_version": EMBEDDING_TEXT_VERSION,
            "embedding_model_name": "different-model",
            "chunks_indexed": 1,
            "files_indexed": 1,
        },
    )

    exit_code = cli_main(["search", str(repo_path), "route"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.err.strip() == (
        "Index is outdated. Run: python -m codescope.cli index <repo_path>"
    )


def _chunk(
    *,
    id: str,
    file_path: str,
    chunk_type: str,
    name: str,
    dependencies: list[str],
    imports: list[str],
) -> CodeChunk:
    return CodeChunk(
        id=id,
        file_path=file_path,
        chunk_type=chunk_type,  # type: ignore[arg-type]
        name=name,
        parent=None,
        start_line=1,
        end_line=1,
        source_code="pass\n",
        imports=imports,
        dependencies=dependencies,
    )
