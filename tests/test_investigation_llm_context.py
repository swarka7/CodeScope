from __future__ import annotations

from pathlib import Path

from codescope.investigation.investigator import InvestigationCodeResult
from codescope.investigation.llm_context import (
    LLMInvestigationContext,
    build_llm_investigation_context,
)
from codescope.investigation.llm_prompt import build_llm_investigation_prompt
from codescope.llm.context_safety import (
    DEFAULT_MAX_TOTAL_CONTEXT_CHARS,
    REDACTION_PLACEHOLDER,
)
from codescope.models.code_chunk import CodeChunk


def test_context_includes_query_likely_code_and_related_context() -> None:
    context = build_llm_investigation_context(
        repo_path=Path("repo"),
        query="receiver balance does not increase",
        likely_relevant_code=[
            _result(
                rank=1,
                name="TransferService.transfer",
                source="semantic",
                reasons=("business operation", "state update logic"),
                dependencies=("debit", "save_account"),
                source_code="def transfer(self):\n    sender.debit(amount)\n",
            )
        ],
        related_context=[
            _result(
                rank=1,
                name="Account.credit",
                source="related",
                reasons=("paired state operation",),
                source_code="def credit(self, amount):\n    self.balance += amount\n",
            )
        ],
    )

    assert context.repo_path == "repo"
    assert context.query == "receiver balance does not increase"
    assert [chunk.name for chunk in context.likely_relevant_code] == [
        "TransferService.transfer"
    ]
    assert context.likely_relevant_code[0].source_label == "semantic"
    assert context.likely_relevant_code[0].reasons == (
        "business operation",
        "state update logic",
    )
    assert context.likely_relevant_code[0].dependencies == ("debit", "save_account")
    assert "sender.debit" in context.likely_relevant_code[0].code_snippet
    assert [chunk.name for chunk in context.related_context] == ["Account.credit"]
    assert "self.balance += amount" in context.related_context[0].code_snippet


def test_context_preserves_ordering_and_applies_chunk_caps() -> None:
    likely = [
        _result(rank=index, name=f"likely_{index}", source="semantic")
        for index in range(1, 4)
    ]
    related = [
        _result(rank=index, name=f"related_{index}", source="related")
        for index in range(1, 3)
    ]

    context = build_llm_investigation_context(
        repo_path=Path("repo"),
        query="bug",
        likely_relevant_code=likely,
        related_context=related,
        max_likely_chunks=2,
        max_related_chunks=1,
    )

    assert [chunk.name for chunk in context.likely_relevant_code] == [
        "likely_1",
        "likely_2",
    ]
    assert [chunk.name for chunk in context.related_context] == ["related_1"]


def test_context_redacts_query_dependencies_reasons_and_snippets() -> None:
    context = build_llm_investigation_context(
        repo_path=Path("repo"),
        query="api_key = 'query-secret' causes Authorization: Bearer query-token",
        likely_relevant_code=[
            _result(
                rank=1,
                name="load_secret",
                source="semantic",
                reasons=("token = 'reason-secret'",),
                dependencies=("client_secret = 'dependency-secret'",),
                source_code="\n".join(
                    [
                        "OPENAI_API_KEY = 'code-secret'",
                        "password = 'code-password'",
                        "Authorization: Bearer code-token",
                        "visible_value = 1",
                    ]
                ),
            )
        ],
        related_context=[],
    )

    context_text = _context_text(context)

    assert REDACTION_PLACEHOLDER in context_text
    assert "query-secret" not in context_text
    assert "query-token" not in context_text
    assert "reason-secret" not in context_text
    assert "dependency-secret" not in context_text
    assert "code-secret" not in context_text
    assert "code-password" not in context_text
    assert "code-token" not in context_text
    assert "visible_value = 1" in context_text


def test_code_snippet_is_bounded_by_lines_and_characters() -> None:
    source = "\n".join(f"line_{index} = '{'x' * 40}'" for index in range(10))

    context = build_llm_investigation_context(
        repo_path=Path("repo"),
        query="bug",
        likely_relevant_code=[
            _result(rank=1, name="large_chunk", source="semantic", source_code=source)
        ],
        related_context=[],
        max_code_snippet_lines=3,
        max_code_snippet_chars=55,
    )

    snippet = context.likely_relevant_code[0].code_snippet
    assert len(snippet) <= 55
    assert snippet.endswith("...")
    assert "line_0" in snippet
    assert "line_4" not in snippet


def test_total_context_cap_drops_chunks_deterministically() -> None:
    large_source = "x = '" + ("x" * 3000) + "'"
    likely = [
        _result(
            rank=index,
            name=f"large_{index}",
            source="semantic",
            source_code=large_source,
        )
        for index in range(5)
    ]

    context = build_llm_investigation_context(
        repo_path=Path("repo"),
        query="large investigation",
        likely_relevant_code=likely,
        related_context=[],
        max_total_context_chars=500,
    )

    assert _context_size(context) <= 500
    assert [chunk.name for chunk in context.likely_relevant_code] in (
        [],
        ["large_0"],
    )


def test_default_total_context_cap_is_applied() -> None:
    large_source = "x = '" + ("x" * 4000) + "'"
    likely = [
        _result(
            rank=index,
            name=f"large_{index}",
            source="semantic",
            source_code=large_source,
        )
        for index in range(5)
    ]

    context = build_llm_investigation_context(
        repo_path=Path("repo"),
        query="large investigation",
        likely_relevant_code=likely,
        related_context=[],
    )

    assert _context_size(context) <= DEFAULT_MAX_TOTAL_CONTEXT_CHARS
    assert len(context.likely_relevant_code) < len(likely)


def test_redacted_text_reaches_prompt() -> None:
    context = build_llm_investigation_context(
        repo_path=Path("repo"),
        query="token = 'prompt-secret'",
        likely_relevant_code=[
            _result(
                rank=1,
                name="helper",
                source="semantic",
                source_code="password = 'prompt-password'\nprint('safe')\n",
            )
        ],
        related_context=[],
    )

    prompt = build_llm_investigation_prompt(context)

    assert REDACTION_PLACEHOLDER in prompt
    assert "prompt-secret" not in prompt
    assert "prompt-password" not in prompt
    assert "print('safe')" in prompt


def _context_text(context: LLMInvestigationContext) -> str:
    values = [context.repo_path, context.query]
    for chunk in (*context.likely_relevant_code, *context.related_context):
        values.extend(
            [
                chunk.name,
                chunk.file_path,
                " ".join(chunk.reasons),
                " ".join(chunk.dependencies),
                chunk.code_snippet,
            ]
        )
    return "\n".join(values)


def _context_size(context: LLMInvestigationContext) -> int:
    return len(_context_text(context))


def _result(
    *,
    rank: int,
    name: str,
    source: str,
    source_code: str = "def helper():\n    return None\n",
    reasons: tuple[str, ...] = (),
    dependencies: tuple[str, ...] = (),
) -> InvestigationCodeResult:
    return InvestigationCodeResult(
        rank=rank,
        name=name,
        kind="method" if "." in name else "function",
        file_path="app/service.py",
        start_line=10,
        end_line=20,
        source="related" if source == "related" else "semantic",
        score=None if source == "related" else 1.5,
        reasons=reasons,
        dependencies=dependencies,
        chunk=CodeChunk(
            id=f"app/service.py:{name}",
            file_path="app/service.py",
            chunk_type="method" if "." in name else "function",
            name=name.rsplit(".", 1)[-1],
            parent=name.rsplit(".", 1)[0] if "." in name else None,
            start_line=10,
            end_line=20,
            source_code=source_code,
            imports=[],
            dependencies=list(dependencies),
        ),
    )
