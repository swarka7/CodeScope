from __future__ import annotations

from pathlib import Path

import pytest

import codescope.cli as cli_module
import codescope.investigation.investigator as investigator_module
from codescope.indexing.index_store import IndexStore
from codescope.indexing.index_versions import EMBEDDING_TEXT_VERSION, INDEX_SCHEMA_VERSION
from codescope.models.code_chunk import CodeChunk


def test_cli_investigate_reports_missing_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    exit_code = cli_module.main(["investigate", str(repo_path), "something is broken"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "No CodeScope index found" in captured.err


def test_cli_investigate_prints_readable_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(investigator_module, "Embedder", _FakeEmbedder)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    service = _chunk(
        repo_path,
        name="transfer",
        parent="LedgerService",
        chunk_type="method",
        file_path="app/service.py",
        source=(
            "def transfer(self, sender, receiver, amount):\n"
            "    sender.debit(amount)\n"
            "    self.validate_transfer(sender, receiver, amount)\n"
        ),
        dependencies=["validate_transfer"],
    )
    validator = _chunk(
        repo_path,
        name="validate_transfer",
        file_path="app/validators.py",
        source="def validate_transfer(sender, receiver, amount):\n    return amount > 0\n",
    )
    _save_index(repo_path, [service, validator])

    exit_code = cli_module.main(
        [
            "investigate",
            str(repo_path),
            "transferring money does not update the receiver balance",
            "--top-k",
            "1",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "CodeScope Investigate" in captured.out
    assert "Query:" in captured.out
    assert "transferring money does not update the receiver balance" in captured.out
    assert "Likely relevant code to inspect:" in captured.out
    assert "1. LedgerService.transfer" in captured.out
    assert "Kind: method" in captured.out
    assert "Location: app/service.py:1-5" in captured.out
    assert "Source: semantic" in captured.out
    assert "Score:" in captured.out
    assert "reasons=" in captured.out
    assert "semantic match" in captured.out
    assert "Related context:" in captured.out
    assert "validate_transfer" in captured.out


def test_cli_investigate_top_k_limits_likely_relevant_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(investigator_module, "Embedder", _FakeEmbedder)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    _save_index(
        repo_path,
        [
            _chunk(repo_path, name="operation_one", file_path="app/one.py"),
            _chunk(repo_path, name="operation_two", file_path="app/two.py"),
        ],
    )

    exit_code = cli_module.main(["investigate", str(repo_path), "operation", "--top-k", "1"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "1. operation_one" in captured.out
    assert "2. operation_two" not in captured.out


def _save_index(repo_path: Path, chunks: list[CodeChunk]) -> None:
    IndexStore(repo_path).save(
        chunks=chunks,
        embeddings=[[1.0] for _ in chunks],
        metadata={
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "embedding_text_version": EMBEDDING_TEXT_VERSION,
            "embedding_model_name": "fake-investigator",
        },
    )


def _chunk(
    repo_path: Path,
    *,
    name: str,
    file_path: str,
    source: str | None = None,
    parent: str | None = None,
    chunk_type: str = "function",
    dependencies: list[str] | None = None,
) -> CodeChunk:
    return CodeChunk(
        id=f"{file_path}:{parent or ''}:{name}",
        file_path=str(repo_path / file_path),
        chunk_type=chunk_type,  # type: ignore[arg-type]
        name=name,
        parent=parent,
        start_line=1,
        end_line=5,
        source_code=source or f"def {name}():\n    return None\n",
        imports=[],
        dependencies=dependencies or [],
    )


class _FakeEmbedder:
    @property
    def model_name(self) -> str:
        return "fake-investigator"

    def embed_text(self, text: str) -> list[float]:
        _ = text
        return [1.0]
