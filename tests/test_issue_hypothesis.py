from __future__ import annotations

from codescope.debugging.issue_hypothesis import build_issue_hypothesis
from codescope.models.code_chunk import CodeChunk
from codescope.models.test_failure import TestFailure
from codescope.retrieval.dependency_aware import RetrievalResult


def test_boolean_validation_hypothesis() -> None:
    hypothesis = build_issue_hypothesis(
        _failure(message="assert True is False"),
        [
            _result(
                _chunk(
                    name="validate_token",
                    source_code="def validate_token():\n    return True\n",
                )
            )
        ],
    )

    assert hypothesis is not None
    assert "Possible issue:" in hypothesis
    assert "validate_token may contain boolean validation logic" in hypothesis
    assert "not a proven root cause" in hypothesis


def test_returned_value_hypothesis() -> None:
    hypothesis = build_issue_hypothesis(
        _failure(message="calculate_discount(100, 10) returned -900 instead of 90"),
        [_result(_chunk(name="calculate_discount"))],
    )

    assert hypothesis is not None
    assert "calculate_discount may compute an incorrect value." in hypothesis


def test_boundary_condition_hypothesis() -> None:
    hypothesis = build_issue_hypothesis(
        _failure(
            test_name="tests/test_auth.py::test_expired_token_is_rejected",
            message="assert True is False",
        ),
        [
            _result(
                _chunk(
                    name="validate_token",
                    source_code=(
                        "def validate_token(payload):\n"
                        "    return payload.expires_at < now\n"
                    ),
                )
            )
        ],
    )

    assert hypothesis is not None
    assert "comparison logic" in hypothesis
    assert "boundary-related" in hypothesis


def test_no_hypothesis_when_signal_is_too_weak() -> None:
    hypothesis = build_issue_hypothesis(
        _failure(message="unexpected error"),
        [_result(_chunk(name="run", source_code="def run():\n    return None\n"))],
    )

    assert hypothesis is None


def _failure(
    *,
    test_name: str = "tests/test_auth.py::test_invalid_token",
    message: str,
) -> TestFailure:
    return TestFailure(
        test_name=test_name,
        file_path="tests/test_auth.py",
        line_number=10,
        error_type="AssertionError",
        message=message,
        traceback=f">   {message}",
    )


def _result(chunk: CodeChunk) -> RetrievalResult:
    return RetrievalResult(kind="semantic", chunk=chunk, score=0.8)


def _chunk(
    *,
    name: str,
    source_code: str = "def function() -> int:\n    return 1\n",
) -> CodeChunk:
    return CodeChunk(
        id=name,
        file_path="auth_service.py",
        chunk_type="function",
        name=name,
        parent=None,
        start_line=1,
        end_line=2,
        source_code=source_code,
        imports=[],
        dependencies=[],
    )
