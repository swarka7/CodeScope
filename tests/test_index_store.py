from __future__ import annotations

from pathlib import Path

import pytest

from codescope.cli import main as cli_main
from codescope.indexing.index_store import IndexStore
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

