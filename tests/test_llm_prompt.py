from __future__ import annotations

from codescope.debugging.llm_context import (
    LLMChunkContext,
    LLMDiagnosisContext,
    LLMFailureContext,
)
from codescope.debugging.llm_prompt import build_llm_diagnosis_prompt


def test_prompt_includes_grounding_and_safety_rules() -> None:
    prompt = build_llm_diagnosis_prompt(_context())

    assert "Use only the provided CodeScope context." in prompt
    assert "Do not invent files, functions, classes, tests, dependencies" in prompt
    assert "Do not claim certainty" in prompt
    assert "Do not generate patches or diffs." in prompt
    assert "Do not modify files." in prompt
    assert "Suggest only a high-level fix direction." in prompt
    assert "Cite or mention chunk names and paths" in prompt


def test_prompt_includes_failure_summary_and_possible_issue() -> None:
    prompt = build_llm_diagnosis_prompt(_context())

    assert "tests/test_transfers.py::test_successful_transfer_moves_money" in prompt
    assert "assert receiver balance changed" in prompt
    assert "Diagnosis summary:" in prompt
    assert "Most relevant source chunk: TransferService.transfer" in prompt
    assert "Rule-based possible issue:" in prompt
    assert "possible missing counterpart operation" in prompt


def test_prompt_includes_chunk_metadata_reasons_dependencies_and_code() -> None:
    prompt = build_llm_diagnosis_prompt(_context())

    assert "1. TransferService.transfer" in prompt
    assert "Path: app/service.py:22-45" in prompt
    assert "Kind: method" in prompt
    assert "Source: semantic" in prompt
    assert "Score: 3.95" in prompt
    assert "- business operation" in prompt
    assert "- state update logic" in prompt
    assert "- debit" in prompt
    assert "```python" in prompt
    assert "sender.debit(amount)" in prompt


def test_prompt_preserves_chunk_ordering() -> None:
    prompt = build_llm_diagnosis_prompt(_context())

    transfer_index = prompt.index("1. TransferService.transfer")
    credit_index = prompt.index("2. Account.credit")
    repository_index = prompt.index("3. BankRepository.save_account")

    assert transfer_index < credit_index < repository_index


def test_prompt_is_valid_without_possible_issue() -> None:
    context = _context(possible_issue=None)
    prompt = build_llm_diagnosis_prompt(context)

    assert "Rule-based possible issue:" not in prompt
    assert "Diagnosis summary:" in prompt
    assert "Retrieved chunks:" in prompt
    assert "LLM Diagnosis" in prompt


def test_prompt_includes_output_format() -> None:
    prompt = build_llm_diagnosis_prompt(_context())

    assert "Write the answer in this exact structure:" in prompt
    assert "- Likely root cause:" in prompt
    assert "- Inspect first:" in prompt
    assert "- Why these chunks matter:" in prompt
    assert "- Possible fix direction:" in prompt
    assert "- Uncertainty:" in prompt


def test_prompt_does_not_include_unrelated_text() -> None:
    prompt = build_llm_diagnosis_prompt(_context())

    assert "OPENAI_API_KEY" not in prompt
    assert ".codescope/index_metadata.json" not in prompt
    assert "unrelated_file.py" not in prompt
    assert "automatic patch" not in prompt.lower()


def test_prompt_handles_empty_chunks() -> None:
    context = LLMDiagnosisContext(
        failure=_failure(),
        diagnosis_summary="Diagnosis summary:\n- Most relevant source chunk: <none>",
        possible_issue=None,
        chunks=(),
    )

    prompt = build_llm_diagnosis_prompt(context)

    assert "Retrieved chunks:" in prompt
    assert "- <none>" in prompt


_DEFAULT_POSSIBLE_ISSUE = (
            "Possible issue:\n"
            "- TransferService.transfer may be missing a counterpart operation.\n"
            "- Account.credit is relevant context for the possible missing counterpart operation."
        )


def _context(*, possible_issue: str | None = _DEFAULT_POSSIBLE_ISSUE) -> LLMDiagnosisContext:
    return LLMDiagnosisContext(
        failure=_failure(),
        diagnosis_summary=(
            "Diagnosis summary:\n"
            "- Failing test: test_successful_transfer_moves_money\n"
            "- Most relevant source chunk: TransferService.transfer in app/service.py"
        ),
        possible_issue=possible_issue,
        chunks=(
            LLMChunkContext(
                rank=1,
                source_label="semantic",
                chunk_type="method",
                name="TransferService.transfer",
                file_path="app/service.py",
                start_line=22,
                end_line=45,
                score=3.95,
                reasons=("business operation", "state update logic"),
                dependencies=("debit", "save_account"),
                code_snippet=(
                    "def transfer(self, sender, receiver, amount):\n"
                    "    sender.debit(amount)\n"
                    "    self.repository.save_account(sender)\n"
                    "    self.repository.save_account(receiver)"
                ),
            ),
            LLMChunkContext(
                rank=2,
                source_label="semantic",
                chunk_type="method",
                name="Account.credit",
                file_path="app/models.py",
                start_line=19,
                end_line=20,
                score=3.90,
                reasons=("paired state operation", "possible missing counterpart operation"),
                dependencies=(),
                code_snippet="def credit(self, amount):\n    self.balance += amount",
            ),
            LLMChunkContext(
                rank=3,
                source_label="related",
                chunk_type="method",
                name="BankRepository.save_account",
                file_path="app/repository.py",
                start_line=23,
                end_line=24,
                score=None,
                reasons=("data-access context",),
                dependencies=(),
                code_snippet=(
                    "def save_account(self, account):\n"
                    "    self.accounts[account.id] = account"
                ),
            ),
        ),
    )


def _failure() -> LLMFailureContext:
    return LLMFailureContext(
        test_name="tests/test_transfers.py::test_successful_transfer_moves_money",
        file_path="tests/test_transfers.py",
        line_number=52,
        error_type="AssertionError",
        message="assert receiver balance changed",
        traceback_excerpt="E   At index 1 diff: Decimal('20.00') != Decimal('45.00')",
    )
