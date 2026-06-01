from __future__ import annotations

from pathlib import Path

import pytest

import codescope.cli as cli_module
import codescope.investigation.investigator as investigator_module
from codescope.indexing.index_store import IndexStore
from codescope.indexing.index_versions import EMBEDDING_TEXT_VERSION, INDEX_SCHEMA_VERSION
from codescope.llm.config import LLMConfig
from codescope.llm.providers import LLMRequest, LLMResponse
from codescope.models.code_chunk import CodeChunk


def test_investigate_without_llm_does_not_print_llm_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_embedder(monkeypatch)
    monkeypatch.setattr(cli_module, "load_llm_config", _fail_if_provider_loaded)
    repo_path = _write_indexed_project(tmp_path)

    exit_code = cli_module.main(
        [
            "investigate",
            str(repo_path),
            "transferring money does not update receiver balance",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "CodeScope Investigate" in captured.out
    assert "Likely relevant code to inspect:" in captured.out
    assert "LLM Investigation" not in captured.out


def test_investigate_llm_without_provider_prints_skipped_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_embedder(monkeypatch)
    monkeypatch.delenv("CODESCOPE_LLM_PROVIDER", raising=False)
    repo_path = _write_indexed_project(tmp_path)

    exit_code = cli_module.main(
        [
            "investigate",
            str(repo_path),
            "transferring money does not update receiver balance",
            "--llm",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "CodeScope Investigate" in captured.out
    assert "Likely relevant code to inspect:" in captured.out
    assert "PaymentService.transfer" in captured.out
    assert "LLM Investigation" in captured.out
    assert "- Skipped: no LLM provider configured." in captured.out
    assert "Set CODESCOPE_LLM_PROVIDER=fake" in captured.out


def test_investigate_llm_with_fake_provider_prints_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_embedder(monkeypatch)
    monkeypatch.setenv("CODESCOPE_LLM_PROVIDER", "fake")
    repo_path = _write_indexed_project(tmp_path)

    exit_code = cli_module.main(
        [
            "investigate",
            str(repo_path),
            "transferring money does not update receiver balance",
            "--llm",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "LLM Investigation" in captured.out
    assert "AI-generated reasoning based only on retrieved CodeScope context." in captured.out
    assert "Fake LLM diagnosis based on provided CodeScope context." in captured.out


def test_investigate_llm_provider_receives_prompt_with_retrieved_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_embedder(monkeypatch)
    provider = _CapturingProvider()
    monkeypatch.setattr(cli_module, "load_llm_config", lambda: LLMConfig(provider="fake"))
    monkeypatch.setattr(cli_module, "load_llm_provider", lambda config: provider)
    repo_path = _write_indexed_project(tmp_path)

    exit_code = cli_module.main(
        [
            "investigate",
            str(repo_path),
            "transferring money does not update receiver balance",
            "--llm",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "captured investigation response" in captured.out
    assert provider.latest_request is not None
    assert "Investigation query:" in provider.latest_request.prompt
    assert "transferring money does not update receiver balance" in (
        provider.latest_request.prompt
    )
    assert "PaymentService.transfer" in provider.latest_request.prompt
    assert "business operation" in provider.latest_request.prompt
    assert "Do not include a title or repeat the `LLM Investigation` heading." in (
        provider.latest_request.prompt
    )


def test_investigate_llm_provider_failure_prints_safe_error_without_stack_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_embedder(monkeypatch)
    monkeypatch.setattr(cli_module, "load_llm_config", lambda: LLMConfig(provider="fake"))
    monkeypatch.setattr(cli_module, "load_llm_provider", lambda config: _FailingProvider())
    repo_path = _write_indexed_project(tmp_path)

    exit_code = cli_module.main(
        [
            "investigate",
            str(repo_path),
            "transferring money does not update receiver balance",
            "--llm",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "LLM Investigation" in captured.out
    assert "- Unavailable: LLM investigation provider failed." in captured.out
    assert "- Reason: LLM provider unavailable." in captured.out
    assert "sk-secret" not in captured.out
    assert "sk-secret" not in captured.err
    assert "Traceback (most recent call last)" not in captured.out
    assert "Traceback (most recent call last)" not in captured.err


def test_investigate_llm_setup_error_does_not_load_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_embedder(monkeypatch)
    monkeypatch.setattr(cli_module, "load_llm_config", _fail_if_provider_loaded)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    exit_code = cli_module.main(["investigate", str(repo_path), "bug", "--llm"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "No CodeScope index found" in captured.err
    assert "LLM Investigation" not in captured.out


class _CapturingProvider:
    name = "capturing"

    def __init__(self) -> None:
        self.latest_request: LLMRequest | None = None

    def diagnose(self, request: LLMRequest) -> LLMResponse:
        self.latest_request = request
        return LLMResponse(
            text="captured investigation response",
            provider=self.name,
            model=request.model,
        )


class _FailingProvider:
    name = "failing"

    def diagnose(self, request: LLMRequest) -> LLMResponse:
        _ = request
        raise RuntimeError("provider unavailable for OPENAI_API_KEY=sk-secret")


def _fail_if_provider_loaded() -> LLMConfig:
    raise AssertionError("LLM provider should not be loaded")


def _write_indexed_project(tmp_path: Path) -> Path:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    service = _chunk(
        repo_path,
        name="transfer",
        parent="PaymentService",
        chunk_type="method",
        file_path="app/service.py",
        source=(
            "def transfer(self, sender, receiver, amount):\n"
            "    sender.debit(amount)\n"
            "    self.repository.save(sender)\n"
            "    self.repository.save(receiver)\n"
        ),
        dependencies=["sender.debit", "repository.save"],
    )
    credit = _chunk(
        repo_path,
        name="credit",
        parent="Wallet",
        chunk_type="method",
        file_path="app/models.py",
        source="def credit(self, amount):\n    self.balance += amount\n",
    )
    _save_index(repo_path, [service, credit])
    return repo_path


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
    source: str,
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
        source_code=source,
        imports=[],
        dependencies=dependencies or [],
    )


def _patch_fake_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(investigator_module, "Embedder", _FakeEmbedder)


class _FakeEmbedder:
    @property
    def model_name(self) -> str:
        return "fake-investigator"

    def embed_text(self, text: str) -> list[float]:
        _ = text
        return [1.0]
