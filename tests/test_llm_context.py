from __future__ import annotations

from codescope.debugging.llm_context import build_llm_diagnosis_context
from codescope.models.code_chunk import CodeChunk
from codescope.models.test_failure import TestFailure
from codescope.retrieval.dependency_aware import RetrievalResult


def test_context_includes_failure_summary_issue_chunks_and_reasons() -> None:
    context = build_llm_diagnosis_context(
        failure=_failure(message="assert True is False"),
        diagnosis_summary="Diagnosis summary:\n- Most relevant source chunk: validate_token",
        possible_issue="Possible issue:\n- validate_token may return the wrong boolean.",
        retrieval_results=[
            _result(
                _chunk(
                    id="validate",
                    name="validate_token",
                    dependencies=["decode_token"],
                    source_code="def validate_token(token):\n    return True\n",
                ),
                score=1.2,
                reasons=("called by top source chunk",),
            )
        ],
    )

    assert context.failure.test_name == "tests/test_auth.py::test_expired_token_is_rejected"
    assert context.failure.message == "assert True is False"
    assert "Most relevant source chunk" in context.diagnosis_summary
    assert context.possible_issue is not None
    assert "wrong boolean" in context.possible_issue

    assert len(context.chunks) == 1
    chunk = context.chunks[0]
    assert chunk.rank == 1
    assert chunk.source_label == "semantic"
    assert chunk.chunk_type == "function"
    assert chunk.name == "validate_token"
    assert chunk.file_path == "src/auth.py"
    assert chunk.start_line == 10
    assert chunk.end_line == 12
    assert chunk.score == 1.2
    assert chunk.dependencies == ("decode_token",)
    assert "call path match" in chunk.reasons
    assert "validation logic" in chunk.reasons
    assert "def validate_token" in chunk.code_snippet


def test_traceback_excerpt_is_bounded() -> None:
    context = build_llm_diagnosis_context(
        failure=_failure(traceback="x" * 200),
        diagnosis_summary="summary",
        possible_issue=None,
        retrieval_results=[],
        max_traceback_chars=25,
    )

    assert len(context.failure.traceback_excerpt) <= 25
    assert context.failure.traceback_excerpt.endswith("...")


def test_code_snippet_is_bounded_by_lines_and_characters() -> None:
    source = "\n".join(f"line_{index} = '{'x' * 40}'" for index in range(20))
    context = build_llm_diagnosis_context(
        failure=_failure(),
        diagnosis_summary="summary",
        possible_issue=None,
        retrieval_results=[_result(_chunk(source_code=source))],
        max_code_snippet_lines=3,
        max_code_snippet_chars=55,
    )

    snippet = context.chunks[0].code_snippet
    assert len(snippet) <= 55
    assert snippet.endswith("...")
    assert "line_0" in snippet
    assert "line_4" not in snippet


def test_semantic_and_related_chunks_are_capped_and_extra_chunks_excluded() -> None:
    results = [
        _result(_chunk(id="semantic_1", name="semantic_1"), kind="semantic"),
        _result(_chunk(id="semantic_2", name="semantic_2"), kind="semantic"),
        _result(_chunk(id="semantic_3", name="semantic_3"), kind="semantic"),
        _result(_chunk(id="related_1", name="related_1"), kind="related"),
        _result(_chunk(id="related_2", name="related_2"), kind="related"),
        _result(_chunk(id="related_3", name="related_3"), kind="related"),
    ]

    context = build_llm_diagnosis_context(
        failure=_failure(),
        diagnosis_summary="summary",
        possible_issue=None,
        retrieval_results=results,
        max_semantic_chunks=2,
        max_related_chunks=1,
    )

    assert [chunk.name for chunk in context.chunks] == [
        "semantic_1",
        "semantic_2",
        "related_1",
    ]
    assert "semantic_3" not in {chunk.name for chunk in context.chunks}
    assert "related_2" not in {chunk.name for chunk in context.chunks}


def test_context_preserves_result_ordering_after_caps() -> None:
    results = [
        _result(_chunk(id="s1", name="first_semantic"), kind="semantic"),
        _result(_chunk(id="r1", name="first_related"), kind="related"),
        _result(_chunk(id="s2", name="second_semantic"), kind="semantic"),
        _result(_chunk(id="r2", name="second_related"), kind="related"),
    ]

    context = build_llm_diagnosis_context(
        failure=_failure(),
        diagnosis_summary="summary",
        possible_issue=None,
        retrieval_results=results,
        max_semantic_chunks=2,
        max_related_chunks=2,
    )

    assert [chunk.name for chunk in context.chunks] == [
        "first_semantic",
        "first_related",
        "second_semantic",
        "second_related",
    ]
    assert [chunk.rank for chunk in context.chunks] == [1, 2, 3, 4]


def test_dependencies_and_reasons_are_bounded() -> None:
    dependencies = [f"dependency_{index}" for index in range(10)]
    context = build_llm_diagnosis_context(
        failure=_failure(),
        diagnosis_summary="summary",
        possible_issue=None,
        retrieval_results=[
            _result(
                _chunk(name="validate_order", dependencies=dependencies),
                reasons=(
                    "called by top source chunk",
                    "validation helper on call path",
                    "caller of expected-exception logic",
                ),
            )
        ],
        max_dependencies=3,
        max_reasons=2,
    )

    assert context.chunks[0].dependencies == ("dependency_0", "dependency_1", "dependency_2")
    assert len(context.chunks[0].reasons) == 2


def test_optional_total_context_cap_drops_chunks_deterministically() -> None:
    context = build_llm_diagnosis_context(
        failure=_failure(message="failure message"),
        diagnosis_summary="summary",
        possible_issue="possible issue",
        retrieval_results=[
            _result(_chunk(id="one", name="one", source_code="x" * 80)),
            _result(_chunk(id="two", name="two", source_code="y" * 80)),
        ],
        max_total_context_chars=120,
    )

    assert [chunk.name for chunk in context.chunks] in ([], ["one"])


def _failure(
    *,
    message: str = "assert value",
    traceback: str = "",
) -> TestFailure:
    return TestFailure(
        test_name="tests/test_auth.py::test_expired_token_is_rejected",
        file_path="tests/test_auth.py",
        line_number=42,
        error_type="AssertionError",
        message=message,
        traceback=traceback,
    )


def _result(
    chunk: CodeChunk,
    *,
    kind: str = "semantic",
    score: float | None = 0.8,
    reasons: tuple[str, ...] = (),
) -> RetrievalResult:
    if kind == "related":
        return RetrievalResult(kind="related", chunk=chunk, score=None, reasons=reasons)
    return RetrievalResult(kind="semantic", chunk=chunk, score=score, reasons=reasons)


def _chunk(
    *,
    id: str = "chunk",
    name: str = "helper",
    file_path: str = "src/auth.py",
    chunk_type: str = "function",
    parent: str | None = None,
    start_line: int = 10,
    end_line: int = 12,
    source_code: str = "def helper():\n    return None\n",
    dependencies: list[str] | None = None,
) -> CodeChunk:
    return CodeChunk(
        id=id,
        file_path=file_path,
        chunk_type=chunk_type,  # type: ignore[arg-type]
        name=name,
        parent=parent,
        start_line=start_line,
        end_line=end_line,
        source_code=source_code,
        imports=[],
        dependencies=dependencies or [],
    )
