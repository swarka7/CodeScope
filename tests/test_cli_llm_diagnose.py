from __future__ import annotations

from pathlib import Path

import pytest

import codescope.cli as cli_module
import codescope.debugging.failure_retriever as failure_retriever_module
import codescope.indexing.indexer as indexer_module
from codescope.llm.config import LLMConfig
from codescope.llm.providers import LLMRequest, LLMResponse
from codescope.models.code_chunk import CodeChunk


def test_diagnose_without_llm_does_not_print_llm_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_embedder(monkeypatch)
    repo_path = _write_failing_project(tmp_path)
    assert cli_module.main(["index", str(repo_path)]) == 0
    capsys.readouterr()

    exit_code = cli_module.main(["diagnose", str(repo_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "CodeScope Diagnose" in captured.out
    assert "Likely relevant code:" in captured.out
    assert "LLM Diagnosis" not in captured.out


def test_diagnose_llm_without_provider_prints_skipped_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_embedder(monkeypatch)
    monkeypatch.delenv("CODESCOPE_LLM_PROVIDER", raising=False)
    repo_path = _write_failing_project(tmp_path)
    assert cli_module.main(["index", str(repo_path)]) == 0
    capsys.readouterr()

    exit_code = cli_module.main(["diagnose", str(repo_path), "--llm"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Likely relevant code:" in captured.out
    assert "LLM Diagnosis" in captured.out
    assert "- Skipped: no LLM provider configured." in captured.out
    assert "Set CODESCOPE_LLM_PROVIDER=fake" in captured.out


def test_diagnose_llm_with_fake_provider_prints_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_embedder(monkeypatch)
    monkeypatch.setenv("CODESCOPE_LLM_PROVIDER", "fake")
    repo_path = _write_failing_project(tmp_path)
    assert cli_module.main(["index", str(repo_path)]) == 0
    capsys.readouterr()

    exit_code = cli_module.main(["diagnose", str(repo_path), "--llm"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "LLM Diagnosis" in captured.out
    assert "AI-generated reasoning based only on retrieved CodeScope context." in captured.out
    assert "Fake LLM diagnosis based on provided CodeScope context." in captured.out


def test_diagnose_llm_provider_receives_prompt_with_retrieved_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_embedder(monkeypatch)
    provider = _CapturingProvider()
    monkeypatch.setattr(cli_module, "load_llm_config", lambda: LLMConfig(provider="fake"))
    monkeypatch.setattr(cli_module, "load_llm_provider", lambda config: provider)
    repo_path = _write_failing_project(tmp_path)
    assert cli_module.main(["index", str(repo_path)]) == 0
    capsys.readouterr()

    exit_code = cli_module.main(["diagnose", str(repo_path), "--llm"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "captured provider response" in captured.out
    assert provider.latest_request is not None
    assert "validate_token" in provider.latest_request.prompt
    assert "tests/test_auth_service.py::test_expired_token_is_rejected" in (
        provider.latest_request.prompt
    )
    assert "Retrieved chunks:" in provider.latest_request.prompt
    assert "reasons" in provider.latest_request.prompt.lower()


def test_diagnose_llm_provider_failure_prints_unavailable_without_stack_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_embedder(monkeypatch)
    monkeypatch.setattr(cli_module, "load_llm_config", lambda: LLMConfig(provider="fake"))
    monkeypatch.setattr(cli_module, "load_llm_provider", lambda config: _FailingProvider())
    repo_path = _write_failing_project(tmp_path)
    assert cli_module.main(["index", str(repo_path)]) == 0
    capsys.readouterr()

    exit_code = cli_module.main(["diagnose", str(repo_path), "--llm"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "LLM Diagnosis" in captured.out
    assert "- Unavailable: LLM diagnosis provider failed." in captured.out
    assert "- Reason: provider unavailable" in captured.out
    assert "Traceback (most recent call last)" not in captured.out
    assert "Traceback (most recent call last)" not in captured.err


def test_diagnose_llm_tests_pass_does_not_load_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_embedder(monkeypatch)

    def fail_if_called() -> LLMConfig:
        raise AssertionError("LLM config should not load when tests pass")

    monkeypatch.setattr(cli_module, "load_llm_config", fail_if_called)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    tests_dir = repo_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_ok.py").write_text(
        "def test_ok() -> None:\n    assert True\n",
        encoding="utf-8",
    )

    exit_code = cli_module.main(["diagnose", str(repo_path), "--llm"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Tests passed" in captured.out
    assert "LLM Diagnosis" not in captured.out


class _CapturingProvider:
    name = "capturing"

    def __init__(self) -> None:
        self.latest_request: LLMRequest | None = None

    def diagnose(self, request: LLMRequest) -> LLMResponse:
        self.latest_request = request
        return LLMResponse(
            text="captured provider response",
            provider=self.name,
            model=request.model,
        )


class _FailingProvider:
    name = "failing"

    def diagnose(self, request: LLMRequest) -> LLMResponse:
        _ = request
        raise RuntimeError("provider unavailable")


def _write_failing_project(tmp_path: Path) -> Path:
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
    return repo_path


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
