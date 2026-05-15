from __future__ import annotations

import pytest

from codescope.debugging.failure_scoring import (
    ScoreBreakdown,
    build_score_breakdown,
    score_failure_chunk,
)
from codescope.models.code_chunk import CodeChunk
from codescope.models.test_failure import TestFailure


def test_breakdown_records_expected_exception_definition() -> None:
    breakdown = build_score_breakdown(
        chunk=_chunk(
            id="limit_error",
            file_path="payments/rules.py",
            chunk_type="class",
            name="TransferLimitExceeded",
            source_code="class TransferLimitExceeded(ValueError):\n    pass\n",
        ),
        base_score=0.8,
        failure=_did_not_raise("payments.rules.TransferLimitExceeded"),
    )

    assert _component_names(breakdown) >= {
        "semantic_base_score",
        "source_chunk_boost",
        "contains_expected_exception",
        "expected_exception_definition",
    }


def test_breakdown_records_raise_expected_exception() -> None:
    breakdown = build_score_breakdown(
        chunk=_chunk(
            id="validator",
            file_path="payments/rules.py",
            chunk_type="function",
            name="validate_transfer_limit",
            source_code=(
                "def validate_transfer_limit(account, amount):\n"
                "    if amount > account.limit:\n"
                "        raise TransferLimitExceeded(account.id)\n"
            ),
        ),
        base_score=0.7,
        failure=_did_not_raise("payments.rules.TransferLimitExceeded"),
    )

    assert breakdown.by_name("raises_expected_exception")
    assert breakdown.by_name("validation_helper_name")


def test_breakdown_records_validation_helper_name() -> None:
    breakdown = build_score_breakdown(
        chunk=_chunk(
            id="guard",
            file_path="security/permissions.py",
            chunk_type="function",
            name="authorize_document_update",
            source_code="def authorize_document_update(document, user_id):\n    return True\n",
        ),
        base_score=0.6,
        failure=_did_not_raise("security.permissions.PermissionDenied"),
    )

    assert breakdown.by_name("validation_helper_name")


def test_breakdown_records_test_chunk_penalty() -> None:
    breakdown = build_score_breakdown(
        chunk=_chunk(
            id="test",
            file_path="tests/test_transfers.py",
            chunk_type="function",
            name="test_transfer_limit",
            source_code="def test_transfer_limit():\n    assert False\n",
        ),
        base_score=1.0,
        failure=_failure(message="assert False"),
    )

    penalty = breakdown.by_name("test_chunk_penalty")
    assert penalty
    assert penalty[0].value == pytest.approx(-0.35)


def test_breakdown_records_non_scoring_generic_data_access_signal() -> None:
    breakdown = build_score_breakdown(
        chunk=_chunk(
            id="repo_get",
            file_path="payments/repository.py",
            chunk_type="method",
            parent="PaymentRepository",
            name="get_payment",
            source_code=(
                "def get_payment(self, payment_id):\n"
                "    return self.payments[payment_id]\n"
            ),
        ),
        base_score=1.0,
        failure=_failure(message="assert False"),
    )

    generic_component = breakdown.by_name("generic_crud_or_data_access_penalty")
    assert generic_component
    assert generic_component[0].contributes_to_score is False
    assert generic_component[0].value == 0.0
    assert breakdown.final_score == pytest.approx(1.1)


def test_breakdown_final_score_matches_score_failure_chunk() -> None:
    chunk = _chunk(
        id="plain",
        file_path="app/service.py",
        chunk_type="function",
        name="run",
        source_code="def run():\n    return None\n",
    )
    failure = _failure(message="assert False")

    breakdown = build_score_breakdown(chunk=chunk, base_score=1.0, failure=failure)

    assert breakdown.final_score == pytest.approx(1.1)
    assert breakdown.final_score == pytest.approx(
        score_failure_chunk(chunk=chunk, base_score=1.0, failure=failure)
    )


def test_breakdown_records_business_state_update_logic() -> None:
    breakdown = build_score_breakdown(
        chunk=_chunk(
            id="service_transfer",
            file_path="payments/service.py",
            chunk_type="method",
            parent="PaymentService",
            name="transfer_funds",
            source_code=(
                "def transfer_funds(self, sender, recipient, amount):\n"
                "    sender.debit(amount)\n"
                "    recipient.credit(amount)\n"
                "    self.ledger.record(sender, recipient, amount)\n"
            ),
        ),
        base_score=0.8,
        failure=_failure(
            message="assert recipient balance was updated",
            test_name="tests/test_payments.py::test_transfer_moves_money_between_accounts",
        ),
    )

    assert breakdown.by_name("business_operation")
    assert breakdown.by_name("state_update_logic")


def test_breakdown_records_filtering_logic() -> None:
    breakdown = build_score_breakdown(
        chunk=_chunk(
            id="catalog_search",
            file_path="catalog/search.py",
            chunk_type="method",
            parent="CatalogSearch",
            name="apply_filters",
            source_code=(
                "def apply_filters(self, criteria):\n"
                "    results = [item for item in self.items if item.rating >= criteria.rating]\n"
                "    return sorted(results, key=lambda item: item.title)\n"
            ),
        ),
        base_score=0.7,
        failure=_failure(
            message="assert search results match genre and rating filters",
            test_name="tests/test_catalog.py::test_combined_filters_apply_all_criteria",
        ),
    )

    assert breakdown.by_name("business_operation")
    assert breakdown.by_name("filtering_logic")


def test_business_failure_penalizes_data_access_and_init_chunks() -> None:
    repository_breakdown = build_score_breakdown(
        chunk=_chunk(
            id="repo_get",
            file_path="orders/repository.py",
            chunk_type="method",
            parent="OrderRepository",
            name="get_order",
            source_code="def get_order(self, order_id):\n    return self.orders[order_id]\n",
        ),
        base_score=1.0,
        failure=_failure(
            message="assert order total was updated",
            test_name="tests/test_orders.py::test_update_order_total",
        ),
    )
    init_breakdown = build_score_breakdown(
        chunk=_chunk(
            id="service_init",
            file_path="orders/service.py",
            chunk_type="method",
            parent="OrderService",
            name="__init__",
            source_code="def __init__(self, repository):\n    self.repository = repository\n",
        ),
        base_score=1.0,
        failure=_failure(
            message="assert order total was updated",
            test_name="tests/test_orders.py::test_update_order_total",
        ),
    )

    assert repository_breakdown.by_name("generic_crud_or_data_access_penalty")
    assert init_breakdown.by_name("constructor_or_init_penalty")
    assert repository_breakdown.final_score < 1.1
    assert init_breakdown.final_score < 1.1


def _did_not_raise(exception_name: str) -> TestFailure:
    return _failure(
        error_type="Failed",
        message=f"Failed: DID NOT RAISE <class '{exception_name}'>",
    )


def _failure(
    *,
    message: str,
    error_type: str = "AssertionError",
    test_name: str = "tests/test_example.py::test_example",
) -> TestFailure:
    return TestFailure(
        test_name=test_name,
        file_path="tests/test_example.py",
        line_number=10,
        error_type=error_type,
        message=message,
        traceback="",
    )


def _chunk(
    *,
    id: str,
    file_path: str,
    chunk_type: str,
    name: str,
    source_code: str,
    parent: str | None = None,
) -> CodeChunk:
    return CodeChunk(
        id=id,
        file_path=file_path,
        chunk_type=chunk_type,  # type: ignore[arg-type]
        name=name,
        parent=parent,
        start_line=1,
        end_line=3,
        source_code=source_code,
        imports=[],
        dependencies=[],
    )


def _component_names(breakdown: ScoreBreakdown) -> set[str]:
    return {component.name for component in breakdown.components}
