from __future__ import annotations

from codescope.debugging.diagnosis_summary import build_diagnosis_summary
from codescope.models.code_chunk import CodeChunk
from codescope.models.test_failure import TestFailure
from codescope.retrieval.dependency_aware import RetrievalResult


def test_summary_includes_failed_test_name() -> None:
    summary = build_diagnosis_summary(_failure(), [_result(_chunk(name="validate_token"))])

    assert "Diagnosis summary:" in summary
    assert "- Failing test: test_expired_token_is_rejected" in summary


def test_summary_includes_error_and_message() -> None:
    summary = build_diagnosis_summary(_failure(), [_result(_chunk(name="validate_token"))])

    assert "- Failure signal: AssertionError, assert True is False" in summary


def test_summary_chooses_top_non_test_source_chunk() -> None:
    test_chunk = _chunk(
        name="test_expired_token_is_rejected",
        file_path="tests/test_auth_service.py",
        chunk_type="function",
    )
    source_chunk = _chunk(name="validate_token", file_path="auth_service.py")

    summary = build_diagnosis_summary(
        _failure(),
        [
            _result(test_chunk, kind="semantic"),
            _result(source_chunk, kind="semantic"),
        ],
    )

    assert "- Most relevant source chunk: validate_token in auth_service.py" in summary


def test_summary_includes_related_context_names() -> None:
    summary = build_diagnosis_summary(
        _failure(),
        [
            _result(_chunk(name="validate_token"), kind="semantic"),
            _result(_chunk(name="decode_token"), kind="related"),
            _result(_chunk(name="TokenPayload"), kind="related"),
        ],
    )

    assert "- Related context: decode_token, TokenPayload" in summary


def test_summary_handles_empty_retrieval() -> None:
    summary = build_diagnosis_summary(_failure(), [])

    assert "- Most relevant source chunk: <none>" in summary
    assert "- Related context: <none>" in summary
    assert "- Why: no retrieved chunks were available to explain." in summary


def _failure() -> TestFailure:
    return TestFailure(
        test_name="tests/test_auth_service.py::test_expired_token_is_rejected",
        file_path="tests/test_auth_service.py",
        line_number=12,
        error_type="AssertionError",
        message="assert True is False",
        traceback=">   assert validate_token(expired_token) is False",
    )


def _result(chunk: CodeChunk, *, kind: str = "semantic") -> RetrievalResult:
    if kind == "related":
        return RetrievalResult(kind="related", chunk=chunk, score=None)
    return RetrievalResult(kind="semantic", chunk=chunk, score=0.8)


def _chunk(
    *,
    name: str,
    file_path: str = "auth_service.py",
    chunk_type: str = "function",
    dependencies: list[str] | None = None,
) -> CodeChunk:
    return CodeChunk(
        id=f"{chunk_type}:{name}:{file_path}",
        file_path=file_path,
        chunk_type=chunk_type,  # type: ignore[arg-type]
        name=name,
        parent=None,
        start_line=1,
        end_line=2,
        source_code="pass\n",
        imports=[],
        dependencies=dependencies or [],
    )
