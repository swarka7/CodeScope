from __future__ import annotations

import json
from pathlib import Path

import pytest

import codescope.cli as cli_module
import codescope.indexing.indexer as indexer_module
from codescope.models.code_chunk import CodeChunk


def test_cli_index_then_search_smoke_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_embedder(monkeypatch)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "api.py").write_text(
        "\n".join(
            [
                "from fastapi import FastAPI",
                "",
                "app = FastAPI()",
                "",
                '@app.post("/todos")',
                "def create_todo(title: str) -> dict[str, str]:",
                "    return save_todo(title)",
                "",
                "def save_todo(title: str) -> dict[str, str]:",
                "    return {'title': title}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    index_exit = cli_module.main(["index", str(repo_path)])
    index_output = capsys.readouterr()

    assert index_exit == 0
    assert "Indexed 1 new/changed files" in index_output.out
    assert "Total chunks: 2" in index_output.out

    search_exit = cli_module.main(
        ["search", str(repo_path), "todo creation endpoint", "--top-k", "1"]
    )
    search_output = capsys.readouterr()

    assert search_exit == 0
    assert "[semantic]" in search_output.out
    assert "create_todo" in search_output.out
    assert "[related]" in search_output.out
    assert "save_todo" in search_output.out


def test_cli_index_rebuilds_outdated_real_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_embedder(monkeypatch)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "app.py").write_text("def run() -> str:\n    return 'ok'\n", encoding="utf-8")

    first_exit = cli_module.main(["index", str(repo_path)])
    capsys.readouterr()
    assert first_exit == 0

    metadata_path = repo_path / ".codescope" / "index_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("embedding_text_version", None)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", "utf-8")

    second_exit = cli_module.main(["index", str(repo_path)])
    output = capsys.readouterr()

    assert second_exit == 0
    assert "Existing index is outdated; rebuilding full index." in output.out
    assert "Indexed 1 new/changed files" in output.out
    assert "Reused 0 unchanged files" in output.out


def test_diagnose_no_tests_does_not_require_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "app.py").write_text("def run() -> str:\n    return 'ok'\n", encoding="utf-8")

    exit_code = cli_module.main(["diagnose", str(repo_path)])
    captured = capsys.readouterr()
    output = captured.out + captured.err

    assert exit_code != 0
    assert "No CodeScope index found" not in output
    assert "no tests ran" in output.lower() or "collected 0 items" in output.lower()


def _patch_fake_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "Embedder", _FakeEmbedder)
    monkeypatch.setattr(indexer_module, "Embedder", _FakeEmbedder)


class _FakeEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", model: object | None = None) -> None:
        _ = model
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_text(self, text: str) -> list[float]:
        return _vector(text)

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float]]:
        return [_vector(chunk.source_code + "\n" + "\n".join(chunk.decorators)) for chunk in chunks]


def _vector(text: str) -> list[float]:
    lower = text.lower()
    return [
        1.0 if "todo" in lower else 0.0,
        1.0 if "creat" in lower or "post" in lower or "endpoint" in lower else 0.0,
        1.0 if "save" in lower else 0.0,
    ]
