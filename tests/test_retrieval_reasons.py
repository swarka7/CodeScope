from __future__ import annotations

from codescope.debugging.retrieval_reasons import build_retrieval_reasons
from codescope.models.code_chunk import CodeChunk
from codescope.models.test_failure import TestFailure


def test_reasons_explain_expected_exception_validator() -> None:
    reasons = build_retrieval_reasons(
        _did_not_raise_failure(),
        _chunk(
            name="validate_status_transition",
            source_code=(
                "def validate_status_transition(current, requested):\n"
                "    if requested not in allowed:\n"
                "        raise InvalidStatusTransitionError('invalid transition')\n"
            ),
            dependencies=["InvalidStatusTransitionError"],
        ),
    )

    assert "raises expected exception" in reasons
    assert "validation helper name" in reasons
    assert any(reason.startswith("behavioral keyword overlap") for reason in reasons)


def test_reasons_explain_expected_exception_class() -> None:
    reasons = build_retrieval_reasons(
        _did_not_raise_failure(),
        _chunk(
            name="InvalidStatusTransitionError",
            chunk_type="class",
            source_code="class InvalidStatusTransitionError(ValueError):\n    pass\n",
        ),
    )

    assert "defines expected exception" in reasons
    assert "contains expected exception" in reasons


def test_reasons_explain_validation_caller_and_operation_overlap() -> None:
    failure = TestFailure(
        test_name="tests/test_transfers.py::test_transfer_over_limit_should_be_rejected",
        file_path="tests/test_transfers.py",
        line_number=10,
        error_type="Failed",
        message="Failed: DID NOT RAISE <class 'banking.rules.DailyLimitExceeded'>",
        traceback="",
    )

    reasons = build_retrieval_reasons(
        failure,
        _chunk(
            name="transfer",
            parent="TransferService",
            source_code=(
                "def transfer(self, account, amount):\n"
                "    validate_daily_limit(account, amount)\n"
                "    return self.repository.record_transfer(account, amount)\n"
            ),
            dependencies=["validate_daily_limit"],
        ),
    )

    assert "calls validation helper" in reasons
    assert any(reason.startswith("operation keyword overlap") for reason in reasons)


def test_reasons_explain_source_traceback_file() -> None:
    failure = TestFailure(
        test_name="tests/test_auth.py::test_invalid_token",
        file_path="tests/test_auth.py",
        line_number=8,
        error_type="AssertionError",
        message="assert True is False",
        traceback='File "src/auth/service.py", line 12, in validate_token',
    )

    reasons = build_retrieval_reasons(
        failure,
        _chunk(name="validate_token", file_path="src/auth/service.py"),
    )

    assert "source chunk from traceback file" in reasons
    assert "validation helper name" in reasons


def test_reasons_keep_source_logic_before_test_context() -> None:
    reasons = build_retrieval_reasons(
        _did_not_raise_failure(),
        _chunk(
            name="test_archived_task_cannot_be_marked_done",
            file_path="tests/test_tasks.py",
            source_code=(
                "def test_archived_task_cannot_be_marked_done():\n"
                "    with pytest.raises(InvalidStatusTransitionError):\n"
                "        service.complete_task(task.id)\n"
            ),
        ),
        limit=6,
    )

    assert "contains expected exception" in reasons
    assert "test context" in reasons
    assert reasons.index("contains expected exception") < reasons.index("test context")


def test_reasons_preserve_call_path_context() -> None:
    reasons = build_retrieval_reasons(
        _did_not_raise_failure(),
        _chunk(
            name="validate_status_transition",
            source_code=(
                "def validate_status_transition(current, requested):\n"
                "    raise InvalidStatusTransitionError()\n"
            ),
        ),
        extra_reasons=("called by top source chunk", "validation helper on call path"),
    )

    assert reasons[:2] == ["called by top source chunk", "validation helper on call path"]
    assert "raises expected exception" in reasons


def test_reasons_include_generic_data_access_signal_when_present() -> None:
    reasons = build_retrieval_reasons(
        _did_not_raise_failure(),
        _chunk(
            name="get_task",
            file_path="tasks/repository.py",
            parent="TaskRepository",
            source_code="def get_task(self, task_id):\n    return self.tasks[task_id]\n",
        ),
        limit=6,
    )

    assert "generic data-access signal" in reasons


def test_reasons_fall_back_to_semantic_similarity() -> None:
    reasons = build_retrieval_reasons(
        _did_not_raise_failure(),
        _chunk(name="render_dashboard", source_code="def render_dashboard():\n    return None\n"),
    )

    assert reasons == ["semantic similarity"]


def _did_not_raise_failure() -> TestFailure:
    return TestFailure(
        test_name="tests/test_tasks.py::test_archived_task_cannot_be_marked_done",
        file_path="tests/test_tasks.py",
        line_number=42,
        error_type="Failed",
        message="Failed: DID NOT RAISE <class 'app.validators.InvalidStatusTransitionError'>",
        traceback="",
    )


def _chunk(
    *,
    name: str,
    file_path: str = "app/validators.py",
    chunk_type: str = "function",
    source_code: str = "def helper():\n    pass\n",
    dependencies: list[str] | None = None,
    parent: str | None = None,
) -> CodeChunk:
    return CodeChunk(
        id=f"{file_path}:{chunk_type}:{parent or ''}:{name}",
        file_path=file_path,
        chunk_type=chunk_type,  # type: ignore[arg-type]
        name=name,
        parent=parent,
        start_line=1,
        end_line=3,
        source_code=source_code,
        imports=[],
        dependencies=dependencies or [],
    )
