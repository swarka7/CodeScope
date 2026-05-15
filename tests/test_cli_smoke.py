from __future__ import annotations

import json
from pathlib import Path

import pytest

import codescope.cli as cli_module
import codescope.debugging.failure_retriever as failure_retriever_module
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


def test_diagnose_auth_service_output_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_embedder(monkeypatch)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "auth_service.py").write_text(
        "\n".join(
            [
                "def validate_token(token: str) -> bool:",
                "    return True",
                "",
            ]
        ),
        encoding="utf-8",
    )
    tests_dir = repo_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_auth_service.py").write_text(
        "\n".join(
            [
                "from auth_service import validate_token",
                "",
                "def test_expired_token_is_rejected() -> None:",
                "    result = validate_token('expired')",
                "    assert result is False",
                "",
            ]
        ),
        encoding="utf-8",
    )

    index_exit = cli_module.main(["index", str(repo_path)])
    capsys.readouterr()
    assert index_exit == 0

    diagnose_exit = cli_module.main(["diagnose", str(repo_path)])
    captured = capsys.readouterr()

    assert diagnose_exit == 1
    assert "CodeScope Diagnose" in captured.out
    assert "Status" in captured.out
    assert "Tests failed" in captured.out
    assert "Failing test" in captured.out
    assert "[FAIL] tests/test_auth_service.py::test_expired_token_is_rejected" in captured.out
    assert "Failure signal" in captured.out
    assert "Diagnosis summary:" in captured.out
    assert "Most relevant source chunk:" in captured.out
    assert "Possible issue:" in captured.out
    assert "Likely relevant code:" in captured.out
    assert "Related context:" in captured.out
    assert "1. validate_token" in captured.out
    assert "Kind: function" in captured.out
    assert "Location: auth_service.py:1-2" in captured.out
    assert "Source: semantic" in captured.out
    assert "Score:" in captured.out
    assert "reasons=" in captured.out
    assert "validation logic" in captured.out
    assert "validate_token" in captured.out


def _patch_fake_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "Embedder", _FakeEmbedder)
    monkeypatch.setattr(failure_retriever_module, "Embedder", _FakeEmbedder)
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
        1.0
        if "creat" in lower
        or "post" in lower
        or "endpoint" in lower
        or "validate_token" in lower
        else 0.0,
        1.0 if "save" in lower else 0.0,
    ]
