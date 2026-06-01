from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import codescope.cli as cli_module
import codescope.investigation.investigator as investigator_module
from codescope.indexing.index_store import IndexStore
from codescope.indexing.index_versions import EMBEDDING_TEXT_VERSION, INDEX_SCHEMA_VERSION
from codescope.llm.config import LLMConfig
from codescope.llm.providers import LLMRequest, LLMResponse
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


def test_cli_investigate_json_outputs_valid_machine_readable_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(investigator_module, "Embedder", _NoisyFakeEmbedder)
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
            "--json",
        ]
    )
    captured = capfd.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.out.startswith("{")
    assert "CodeScope Investigate" not in captured.out
    assert "Warning:" not in captured.out
    assert "Loading weights" not in captured.out
    assert "Warning:" in captured.err
    assert "Loading weights" in captured.err
    assert payload["schema_version"] == 1
    assert payload["status"] == "ok"
    assert payload["query"] == "transferring money does not update the receiver balance"
    assert payload["likely_relevant_code"][0]["name"] == "LedgerService.transfer"
    assert payload["likely_relevant_code"][0]["source"] == "semantic"
    assert isinstance(payload["likely_relevant_code"][0]["reasons"], list)
    assert isinstance(payload["likely_relevant_code"][0]["dependencies"], list)
    assert payload["related_context"][0]["name"] == "validate_transfer"
    assert isinstance(payload["related_context"][0]["reasons"], list)
    assert isinstance(payload["related_context"][0]["dependencies"], list)


def test_cli_investigate_json_top_k_limits_semantic_results(
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

    exit_code = cli_module.main(
        ["investigate", str(repo_path), "operation", "--top-k", "1", "--json"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert [item["name"] for item in payload["likely_relevant_code"]] == ["operation_one"]


def test_cli_investigate_json_missing_index_returns_error_object(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    exit_code = cli_module.main(
        ["investigate", str(repo_path), "something is broken", "--json"]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 2
    assert captured.out.startswith("{")
    assert "CodeScope Investigate" not in captured.out
    assert payload == {
        "schema_version": 1,
        "status": "error",
        "repo": str(repo_path).replace("\\", "/"),
        "query": "something is broken",
        "message": "No CodeScope index found. Run: python -m codescope.cli index <repo_path>",
        "likely_relevant_code": [],
        "related_context": [],
    }


def test_cli_investigate_json_empty_index_returns_error_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(investigator_module, "Embedder", _FakeEmbedder)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    _save_index(repo_path, [])

    exit_code = cli_module.main(["investigate", str(repo_path), "bug", "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 2
    assert captured.out.startswith("{")
    assert "Warning:" not in captured.out
    assert payload == {
        "schema_version": 1,
        "status": "error",
        "repo": str(repo_path).replace("\\", "/"),
        "query": "bug",
        "message": (
            "CodeScope index is empty. Run: "
            "python -m codescope.cli index <repo_path> --rebuild"
        ),
        "likely_relevant_code": [],
        "related_context": [],
    }


def test_cli_investigate_json_llm_with_fake_provider_returns_completed_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(investigator_module, "Embedder", _NoisyFakeEmbedder)
    monkeypatch.setenv("CODESCOPE_LLM_PROVIDER", "fake")
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    _save_index(
        repo_path,
        [
            _chunk(
                repo_path,
                name="transfer",
                parent="LedgerService",
                chunk_type="method",
                file_path="app/service.py",
                source="def transfer(self):\n    return self.repository.save()\n",
            )
        ],
    )

    exit_code = cli_module.main(
        ["investigate", str(repo_path), "transfer does not save", "--json", "--llm"]
    )
    captured = capfd.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.out.startswith("{")
    assert "Warning:" not in captured.out
    assert "Loading weights" not in captured.out
    assert "Warning:" in captured.err
    assert "Loading weights" in captured.err
    assert payload["status"] == "ok"
    assert payload["likely_relevant_code"][0]["name"] == "LedgerService.transfer"
    assert payload["llm"]["enabled"] is True
    assert payload["llm"]["provider"] == "fake"
    assert payload["llm"]["status"] == "completed"
    assert "Fake LLM diagnosis based on provided CodeScope context." in payload["llm"]["text"]


def test_cli_investigate_json_llm_without_provider_returns_skipped_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(investigator_module, "Embedder", _FakeEmbedder)
    monkeypatch.delenv("CODESCOPE_LLM_PROVIDER", raising=False)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    _save_index(repo_path, [_chunk(repo_path, name="operation", file_path="app/service.py")])

    exit_code = cli_module.main(["investigate", str(repo_path), "operation", "--json", "--llm"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.out.startswith("{")
    assert payload["status"] == "ok"
    assert payload["llm"] == {
        "enabled": True,
        "provider": None,
        "model": None,
        "status": "skipped",
        "message": "no LLM provider configured",
    }


def test_cli_investigate_json_llm_provider_failure_returns_safe_error_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(investigator_module, "Embedder", _FakeEmbedder)
    monkeypatch.setattr(cli_module, "load_llm_config", lambda: LLMConfig(provider="fake"))
    monkeypatch.setattr(cli_module, "load_llm_provider", lambda config: _FailingProvider())
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    _save_index(repo_path, [_chunk(repo_path, name="operation", file_path="app/service.py")])

    exit_code = cli_module.main(["investigate", str(repo_path), "operation", "--json", "--llm"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.out.startswith("{")
    assert payload["status"] == "ok"
    assert payload["llm"] == {
        "enabled": True,
        "provider": "failing",
        "model": None,
        "status": "error",
        "message": "LLM provider unavailable",
    }
    assert "sk-secret" not in captured.out
    assert "Traceback (most recent call last)" not in captured.out


def test_cli_investigate_json_llm_empty_index_skips_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(investigator_module, "Embedder", _FakeEmbedder)
    monkeypatch.setattr(cli_module, "load_llm_config", _fail_if_provider_loaded)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    _save_index(repo_path, [])

    exit_code = cli_module.main(["investigate", str(repo_path), "bug", "--json", "--llm"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 2
    assert captured.out.startswith("{")
    assert payload["status"] == "error"
    assert payload["message"] == (
        "CodeScope index is empty. Run: "
        "python -m codescope.cli index <repo_path> --rebuild"
    )
    assert payload["likely_relevant_code"] == []
    assert payload["related_context"] == []
    assert payload["llm"] == {
        "enabled": True,
        "provider": None,
        "model": None,
        "status": "skipped",
        "message": "deterministic investigation failed; LLM not run",
    }


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


class _NoisyFakeEmbedder(_FakeEmbedder):
    def embed_text(self, text: str) -> list[float]:
        print("Warning: fake model loading progress")
        os.write(1, b"Loading weights: fake progress\n")
        return super().embed_text(text)


class _FailingProvider:
    name = "failing"

    def diagnose(self, request: LLMRequest) -> LLMResponse:
        _ = request
        raise RuntimeError("provider unavailable for OPENAI_API_KEY=sk-secret")


def _fail_if_provider_loaded() -> LLMConfig:
    raise AssertionError("LLM provider should not be loaded")
