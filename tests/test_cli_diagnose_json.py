from __future__ import annotations

import json
from pathlib import Path

import pytest

import codescope.cli as cli_module
import codescope.debugging.failure_retriever as failure_retriever_module
import codescope.indexing.indexer as indexer_module
from codescope.indexing.index_store import IndexStore
from codescope.indexing.index_versions import EMBEDDING_TEXT_VERSION, INDEX_SCHEMA_VERSION
from codescope.llm.config import LLMConfig
from codescope.llm.providers import LLMRequest, LLMResponse
from codescope.models.code_chunk import CodeChunk
from codescope.retrieval.dependency_aware import RetrievalResult


def test_diagnose_json_outputs_valid_machine_readable_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_embedder(monkeypatch)
    _patch_fake_retriever(monkeypatch)
    repo_path = _write_failing_project(tmp_path)
    assert cli_module.main(["index", str(repo_path)]) == 0
    capsys.readouterr()

    exit_code = cli_module.main(["diagnose", str(repo_path), "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert captured.err == ""
    assert captured.out.startswith("{")
    assert "CodeScope Diagnose" not in captured.out
    assert "Warning:" not in captured.out
    assert "Loading weights" not in captured.out
    assert payload["schema_version"] == 1
    assert payload["status"] == "failed"
    assert payload["diagnose_exit_code"] == 1
    assert payload["repo"] == repo_path.as_posix()

    failure = payload["failures"][0]
    assert failure["test_name"] == "tests/test_auth_service.py::test_expired_token_is_rejected"
    assert failure["file_path"] == "tests/test_auth_service.py"
    assert failure["line_number"] == 5
    assert failure["error_type"] == "AssertionError"
    assert "assert True is False" in failure["message"]
    assert "Diagnosis summary:" in failure["diagnosis_summary"]
    assert isinstance(failure["possible_issue"], str)


def test_diagnose_json_includes_retrieval_results_reasons_and_related_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_embedder(monkeypatch)
    _patch_fake_retriever(monkeypatch)
    repo_path = _write_failing_project(tmp_path)
    assert cli_module.main(["index", str(repo_path)]) == 0
    capsys.readouterr()

    assert cli_module.main(["diagnose", str(repo_path), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    failure = payload["failures"][0]

    likely = failure["likely_relevant_code"]
    assert likely[0]["rank"] == 1
    assert likely[0]["name"] == "validate_token"
    assert likely[0]["kind"] == "function"
    assert likely[0]["file_path"] == "auth_service.py"
    assert likely[0]["start_line"] == 1
    assert likely[0]["end_line"] == 3
    assert likely[0]["source"] == "semantic"
    assert likely[0]["score"] == 2.5
    assert isinstance(likely[0]["reasons"], list)
    assert "validation logic" in likely[0]["reasons"]
    assert likely[0]["dependencies"] == ["decode_token"]

    related = failure["related_context"]
    assert related[0]["rank"] == 1
    assert related[0]["name"] == "decode_token"
    assert related[0]["source"] == "related"
    assert related[0]["score"] is None
    assert isinstance(related[0]["reasons"], list)


def test_diagnose_json_passing_tests_returns_passed_without_loading_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_embedder(monkeypatch)

    def fail_if_called() -> LLMConfig:
        raise AssertionError("LLM config should not load when tests pass")

    monkeypatch.setattr(cli_module, "load_llm_config", fail_if_called)
    repo_path = _write_passing_project(tmp_path)

    exit_code = cli_module.main(["diagnose", str(repo_path), "--json", "--llm"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload == {
        "diagnose_exit_code": 0,
        "failures": [],
        "repo": repo_path.as_posix(),
        "schema_version": 1,
        "status": "passed",
    }


def test_diagnose_json_llm_fake_provider_includes_completed_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_embedder(monkeypatch)
    _patch_fake_retriever(monkeypatch)
    monkeypatch.setenv("CODESCOPE_LLM_PROVIDER", "fake")
    repo_path = _write_failing_project(tmp_path)
    assert cli_module.main(["index", str(repo_path)]) == 0
    capsys.readouterr()

    exit_code = cli_module.main(["diagnose", str(repo_path), "--json", "--llm"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    llm = payload["failures"][0]["llm"]

    assert exit_code == 1
    assert captured.out.startswith("{")
    assert "LLM Diagnosis" not in captured.out
    assert "Warning:" not in captured.out
    assert "Loading weights" not in captured.out
    assert llm["enabled"] is True
    assert llm["provider"] == "fake"
    assert llm["status"] == "completed"
    assert llm["text"] == "Fake LLM diagnosis based on provided CodeScope context."


def test_diagnose_json_llm_without_provider_includes_skipped_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_embedder(monkeypatch)
    _patch_fake_retriever(monkeypatch)
    monkeypatch.delenv("CODESCOPE_LLM_PROVIDER", raising=False)
    repo_path = _write_failing_project(tmp_path)
    assert cli_module.main(["index", str(repo_path)]) == 0
    capsys.readouterr()

    exit_code = cli_module.main(["diagnose", str(repo_path), "--json", "--llm"])
    payload = json.loads(capsys.readouterr().out)
    llm = payload["failures"][0]["llm"]

    assert exit_code == 1
    assert llm == {
        "enabled": True,
        "message": "no LLM provider configured",
        "model": None,
        "provider": None,
        "status": "skipped",
    }


def test_diagnose_json_llm_provider_failure_has_no_stack_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_embedder(monkeypatch)
    _patch_fake_retriever(monkeypatch)
    monkeypatch.setattr(cli_module, "load_llm_config", lambda: LLMConfig(provider="fake"))
    monkeypatch.setattr(cli_module, "load_llm_provider", lambda config: _FailingProvider())
    repo_path = _write_failing_project(tmp_path)
    assert cli_module.main(["index", str(repo_path)]) == 0
    capsys.readouterr()

    exit_code = cli_module.main(["diagnose", str(repo_path), "--json", "--llm"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    llm = payload["failures"][0]["llm"]

    assert exit_code == 1
    assert llm["status"] == "error"
    assert llm["message"] == "LLM provider unavailable"
    assert "Traceback (most recent call last)" not in captured.out
    assert "Traceback (most recent call last)" not in captured.err


def test_diagnose_json_missing_index_outputs_error_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_embedder(monkeypatch)
    repo_path = _write_failing_project(tmp_path)

    exit_code = cli_module.main(["diagnose", str(repo_path), "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 2
    assert captured.err == ""
    assert captured.out.startswith("{")
    assert payload["status"] == "error"
    assert payload["diagnose_exit_code"] == 2
    assert payload["failures"] == []
    assert "No CodeScope index found" in payload["message"]


def test_diagnose_json_empty_index_outputs_error_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_embedder(monkeypatch)
    repo_path = _write_failing_project(tmp_path)
    IndexStore(repo_path).save(
        chunks=[],
        embeddings=[],
        metadata={
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "embedding_text_version": EMBEDDING_TEXT_VERSION,
            "embedding_model_name": "all-MiniLM-L6-v2",
            "chunks_indexed": 0,
            "files_indexed": 1,
        },
    )

    exit_code = cli_module.main(["diagnose", str(repo_path), "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 2
    assert captured.err == ""
    assert captured.out.startswith("{")
    assert "Warning:" not in captured.out
    assert payload["status"] == "error"
    assert payload["diagnose_exit_code"] == 2
    assert payload["failures"] == []
    assert payload["message"] == (
        "CodeScope index is empty. Run: python -m codescope.cli index <repo_path> --rebuild"
    )


def test_diagnose_json_redirects_incidental_stdout_to_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_embedder(monkeypatch)
    repo_path = _write_failing_project(tmp_path)
    assert cli_module.main(["index", str(repo_path)]) == 0
    capsys.readouterr()

    class NoisyRetriever:
        def __init__(self, repo_path: Path) -> None:
            _ = repo_path

        def retrieve(self, failure: object, *, top_k: int = 5) -> list[RetrievalResult]:
            _ = failure
            _ = top_k
            print("Warning: noisy third-party stdout")
            print("Loading weights: noisy progress")
            return _FakeRetriever(repo_path).retrieve(failure, top_k=top_k)

    monkeypatch.setattr(cli_module, "FailureRetriever", NoisyRetriever)

    exit_code = cli_module.main(["diagnose", str(repo_path), "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert captured.out.startswith("{")
    assert "Warning:" not in captured.out
    assert "Loading weights" not in captured.out
    assert "Warning: noisy third-party stdout" in captured.err
    assert "Loading weights: noisy progress" in captured.err
    assert payload["status"] == "failed"


def test_diagnose_without_json_keeps_human_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_embedder(monkeypatch)
    _patch_fake_retriever(monkeypatch)
    repo_path = _write_failing_project(tmp_path)
    assert cli_module.main(["index", str(repo_path)]) == 0
    capsys.readouterr()

    exit_code = cli_module.main(["diagnose", str(repo_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "CodeScope Diagnose" in captured.out
    assert "Likely relevant code:" in captured.out


class _FailingProvider:
    name = "failing"

    def diagnose(self, request: LLMRequest) -> LLMResponse:
        _ = request
        raise RuntimeError("provider exploded")


class _FakeRetriever:
    def __init__(self, repo_path: Path) -> None:
        _ = repo_path

    def retrieve(self, failure: object, *, top_k: int = 5) -> list[RetrievalResult]:
        _ = failure
        _ = top_k
        return [
            RetrievalResult(
                kind="semantic",
                chunk=_chunk(
                    id="validate",
                    name="validate_token",
                    file_path="auth_service.py",
                    source_code=(
                        "def validate_token(token: str) -> bool:\n"
                        "    return decode_token(token).is_valid\n"
                    ),
                    dependencies=["decode_token"],
                ),
                score=2.5,
                reasons=("validation logic",),
            ),
            RetrievalResult(
                kind="related",
                chunk=_chunk(
                    id="decode",
                    name="decode_token",
                    file_path="token_manager.py",
                    source_code="def decode_token(token: str):\n    return token\n",
                ),
                score=None,
                reasons=("semantic match",),
            ),
        ]


def _write_failing_project(tmp_path: Path) -> Path:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "auth_service.py").write_text(
        "def validate_token(token: str) -> bool:\n    return True\n",
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
    return repo_path


def _write_passing_project(tmp_path: Path) -> Path:
    repo_path = tmp_path / "passing_repo"
    repo_path.mkdir()
    tests_dir = repo_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_ok.py").write_text("def test_ok() -> None:\n    assert True\n")
    return repo_path


def _patch_fake_retriever(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "FailureRetriever", _FakeRetriever)


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
        lower = text.lower()
        return [1.0 if "validate_token" in lower else 0.0, 1.0]

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float]]:
        return [self.embed_text(chunk.source_code) for chunk in chunks]


def _chunk(
    *,
    id: str,
    name: str,
    file_path: str,
    source_code: str,
    dependencies: list[str] | None = None,
) -> CodeChunk:
    return CodeChunk(
        id=id,
        file_path=file_path,
        chunk_type="function",
        name=name,
        parent=None,
        start_line=1,
        end_line=3,
        source_code=source_code,
        imports=[],
        dependencies=dependencies or [],
    )
