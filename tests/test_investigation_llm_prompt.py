from __future__ import annotations

from codescope.investigation.llm_context import (
    LLMInvestigationChunkContext,
    LLMInvestigationContext,
)
from codescope.investigation.llm_prompt import build_llm_investigation_prompt


def test_prompt_includes_grounding_and_safety_rules() -> None:
    prompt = build_llm_investigation_prompt(_context())

    assert "Use only the provided CodeScope context." in prompt
    assert "Do not invent files, functions, classes, dependencies" in prompt
    assert "Do not claim certainty" in prompt
    assert "Do not generate patches or diffs." in prompt
    assert "Do not modify files." in prompt
    assert "Suggest only a high-level debugging direction." in prompt
    assert "Cite or mention chunk names and paths" in prompt


def test_prompt_includes_query_chunks_reasons_dependencies_and_code() -> None:
    prompt = build_llm_investigation_prompt(_context())

    assert "Investigation query:" in prompt
    assert "receiver balance does not increase" in prompt
    assert "Likely relevant code:" in prompt
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
    assert "Related context:" in prompt
    assert "1. Account.credit" in prompt


def test_prompt_preserves_likely_and_related_ordering() -> None:
    prompt = build_llm_investigation_prompt(_context())

    transfer_index = prompt.index("1. TransferService.transfer")
    repository_index = prompt.index("2. BankRepository.save_account")
    credit_index = prompt.index("1. Account.credit")

    assert transfer_index < repository_index < credit_index


def test_prompt_includes_output_format_without_repeating_heading() -> None:
    prompt = build_llm_investigation_prompt(_context())

    assert "Write the answer in this exact structure:" in prompt
    assert "Do not include a title or repeat the `LLM Investigation` heading." in prompt
    assert "Start directly with these bullet sections:" in prompt
    assert "- Likely relevant area:" in prompt
    assert "- Inspect first:" in prompt
    assert "- Why these chunks matter:" in prompt
    assert "- Possible debugging direction:" in prompt
    assert "- Uncertainty:" in prompt


def test_prompt_handles_empty_context_sections() -> None:
    context = LLMInvestigationContext(
        repo_path="repo",
        query="bug",
        likely_relevant_code=(),
        related_context=(),
    )

    prompt = build_llm_investigation_prompt(context)

    assert "Likely relevant code:" in prompt
    assert "Related context:" in prompt
    assert prompt.count("- <none>") == 2


def test_prompt_does_not_include_unrelated_text() -> None:
    prompt = build_llm_investigation_prompt(_context())

    assert "OPENAI_API_KEY" not in prompt
    assert ".codescope/index_metadata.json" not in prompt
    assert "unrelated_file.py" not in prompt
    assert "automatic patch" not in prompt.lower()


def _context() -> LLMInvestigationContext:
    return LLMInvestigationContext(
        repo_path="examples/realistic_bugs/banking_app",
        query="receiver balance does not increase",
        likely_relevant_code=(
            LLMInvestigationChunkContext(
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
            LLMInvestigationChunkContext(
                rank=2,
                source_label="semantic",
                chunk_type="method",
                name="BankRepository.save_account",
                file_path="app/repository.py",
                start_line=23,
                end_line=24,
                score=2.1,
                reasons=("data-access context",),
                dependencies=(),
                code_snippet=(
                    "def save_account(self, account):\n"
                    "    self.accounts[account.id] = account"
                ),
            ),
        ),
        related_context=(
            LLMInvestigationChunkContext(
                rank=1,
                source_label="related",
                chunk_type="method",
                name="Account.credit",
                file_path="app/models.py",
                start_line=19,
                end_line=20,
                score=None,
                reasons=("paired state operation",),
                dependencies=(),
                code_snippet="def credit(self, amount):\n    self.balance += amount",
            ),
        ),
    )
