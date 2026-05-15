from __future__ import annotations

from codescope.debugging.failure_retriever import FailureRetriever
from codescope.models.code_chunk import CodeChunk
from codescope.models.test_failure import TestFailure
from codescope.vectorstore.memory_store import SearchResult


def test_banking_limit_failure_prefers_validator_over_test_and_repository() -> None:
    failure = _failure(
        test_name="tests/test_transfers.py::test_daily_transfer_limit_is_rejected",
        message="Failed: DID NOT RAISE <class 'payments.rules.DailyTransferLimitExceeded'>",
    )
    selected, ranked = _rank_and_select(
        failure,
        [
            SearchResult(
                chunk=_chunk(
                    id="test",
                    file_path="tests/test_transfers.py",
                    chunk_type="function",
                    name="test_daily_transfer_limit_is_rejected",
                    dependencies=["transfer", "DailyTransferLimitExceeded"],
                    source_code=(
                        "def test_daily_transfer_limit_is_rejected():\n"
                        "    with pytest.raises(DailyTransferLimitExceeded):\n"
                        "        service.transfer(account, amount)\n"
                    ),
                ),
                score=1.0,
            ),
            SearchResult(
                chunk=_chunk(
                    id="repo",
                    file_path="payments/repository.py",
                    chunk_type="method",
                    parent="AccountRepository",
                    name="get_account",
                    source_code=(
                        "def get_account(self, account_id):\n"
                        "    return self.accounts[account_id]\n"
                    ),
                ),
                score=0.98,
            ),
            SearchResult(
                chunk=_chunk(
                    id="transfer",
                    file_path="payments/service.py",
                    chunk_type="method",
                    parent="TransferService",
                    name="transfer",
                    dependencies=["get_account", "validate_daily_transfer_limit"],
                    source_code=(
                        "def transfer(self, account_id, amount):\n"
                        "    account = self.repository.get_account(account_id)\n"
                        "    validate_daily_transfer_limit(account, amount)\n"
                        "    return self.repository.record_transfer(account, amount)\n"
                    ),
                ),
                score=0.84,
            ),
            SearchResult(
                chunk=_chunk(
                    id="validator",
                    file_path="payments/rules.py",
                    chunk_type="function",
                    name="validate_daily_transfer_limit",
                    dependencies=["DailyTransferLimitExceeded"],
                    source_code=(
                        "def validate_daily_transfer_limit(account, amount):\n"
                        "    if account.daily_total + amount > account.daily_limit:\n"
                        "        raise DailyTransferLimitExceeded(account.id)\n"
                    ),
                ),
                score=0.76,
            ),
            SearchResult(
                chunk=_chunk(
                    id="exception",
                    file_path="payments/rules.py",
                    chunk_type="class",
                    name="DailyTransferLimitExceeded",
                    source_code="class DailyTransferLimitExceeded(ValueError):\n    pass\n",
                ),
                score=0.72,
            ),
        ],
        top_k=4,
    )

    ranked_names = _names(ranked)
    selected_names = _names(selected)
    assert ranked_names.index("validate_daily_transfer_limit") < ranked_names.index("get_account")
    assert ranked_names.index("transfer") < ranked_names.index("get_account")
    assert "test_daily_transfer_limit_is_rejected" not in selected_names
    assert selected_names[:3] == [
        "validate_daily_transfer_limit",
        "DailyTransferLimitExceeded",
        "transfer",
    ]


def test_subscription_expiration_prefers_source_validator_over_failing_test() -> None:
    failure = _failure(
        test_name="tests/test_subscriptions.py::test_expired_subscription_is_not_accepted",
        error_type="AssertionError",
        message="assert True is False",
        traceback=(
            "def test_expired_subscription_is_not_accepted():\n"
            "    result = service.activate_subscription(expired_subscription)\n"
            ">   assert result is False\n"
            "E   assert True is False\n"
        ),
    )
    selected, _ranked = _rank_and_select(
        failure,
        [
            SearchResult(
                chunk=_chunk(
                    id="test",
                    file_path="tests/test_subscriptions.py",
                    chunk_type="function",
                    name="test_expired_subscription_is_not_accepted",
                    dependencies=["activate_subscription"],
                    source_code=(
                        "def test_expired_subscription_is_not_accepted():\n"
                        "    result = service.activate_subscription(expired_subscription)\n"
                        "    assert result is False\n"
                    ),
                ),
                score=1.0,
            ),
            SearchResult(
                chunk=_chunk(
                    id="validator",
                    file_path="billing/expiration.py",
                    chunk_type="function",
                    name="validate_subscription_active",
                    source_code=(
                        "def validate_subscription_active(subscription, now):\n"
                        "    return subscription.expires_at > now\n"
                    ),
                ),
                score=0.78,
            ),
            SearchResult(
                chunk=_chunk(
                    id="caller",
                    file_path="billing/service.py",
                    chunk_type="method",
                    parent="SubscriptionService",
                    name="activate_subscription",
                    dependencies=["validate_subscription_active"],
                    source_code=(
                        "def activate_subscription(self, subscription):\n"
                        "    return validate_subscription_active(subscription, self.clock.now())\n"
                    ),
                ),
                score=0.77,
            ),
        ],
        top_k=3,
    )

    selected_names = _names(selected)
    assert selected_names.index("validate_subscription_active") < selected_names.index(
        "test_expired_subscription_is_not_accepted"
    )
    assert selected_names.index("activate_subscription") < selected_names.index(
        "test_expired_subscription_is_not_accepted"
    )


def test_inventory_transition_prioritizes_expected_exception_over_unrelated_exception() -> None:
    failure = _failure(
        test_name="tests/test_inventory.py::test_reserved_item_cannot_be_discarded",
        message="Failed: DID NOT RAISE <class 'warehouse.rules.InventoryTransitionRejected'>",
    )
    selected, ranked = _rank_and_select(
        failure,
        [
            SearchResult(
                chunk=_chunk(
                    id="test",
                    file_path="tests/test_inventory.py",
                    chunk_type="function",
                    name="test_reserved_item_cannot_be_discarded",
                    dependencies=["InventoryTransitionRejected"],
                    source_code=(
                        "def test_reserved_item_cannot_be_discarded():\n"
                        "    with pytest.raises(InventoryTransitionRejected):\n"
                        "        service.discard_item(item.id)\n"
                    ),
                ),
                score=1.0,
            ),
            SearchResult(
                chunk=_chunk(
                    id="unrelated_exception",
                    file_path="warehouse/repository.py",
                    chunk_type="class",
                    name="InventoryNotFoundError",
                    source_code="class InventoryNotFoundError(LookupError):\n    pass\n",
                ),
                score=0.95,
            ),
            SearchResult(
                chunk=_chunk(
                    id="validator",
                    file_path="warehouse/rules.py",
                    chunk_type="function",
                    name="validate_inventory_transition",
                    dependencies=["InventoryTransitionRejected"],
                    source_code=(
                        "def validate_inventory_transition(current, requested):\n"
                        "    if requested not in ALLOWED_TRANSITIONS[current]:\n"
                        "        raise InventoryTransitionRejected(current, requested)\n"
                    ),
                ),
                score=0.74,
            ),
            SearchResult(
                chunk=_chunk(
                    id="expected_exception",
                    file_path="warehouse/rules.py",
                    chunk_type="class",
                    name="InventoryTransitionRejected",
                    source_code="class InventoryTransitionRejected(ValueError):\n    pass\n",
                ),
                score=0.72,
            ),
        ],
        top_k=3,
    )

    ranked_names = _names(ranked)
    selected_names = _names(selected)
    assert ranked_names.index("InventoryTransitionRejected") < ranked_names.index(
        "InventoryNotFoundError"
    )
    assert ranked_names.index("validate_inventory_transition") < ranked_names.index(
        "InventoryNotFoundError"
    )
    assert "test_reserved_item_cannot_be_discarded" not in selected_names


def test_authorization_failure_prefers_guard_over_generic_update_and_save_wrappers() -> None:
    failure = _failure(
        test_name="tests/test_documents.py::test_unauthorized_editor_cannot_update_document",
        message="Failed: DID NOT RAISE <class 'security.permissions.UnauthorizedUpdateError'>",
    )
    selected, ranked = _rank_and_select(
        failure,
        [
            SearchResult(
                chunk=_chunk(
                    id="test",
                    file_path="tests/test_documents.py",
                    chunk_type="function",
                    name="test_unauthorized_editor_cannot_update_document",
                    dependencies=["update_document", "UnauthorizedUpdateError"],
                    source_code=(
                        "def test_unauthorized_editor_cannot_update_document():\n"
                        "    with pytest.raises(UnauthorizedUpdateError):\n"
                        "        service.update_document(document.id, editor.id, payload)\n"
                    ),
                ),
                score=1.0,
            ),
            SearchResult(
                chunk=_chunk(
                    id="save",
                    file_path="documents/repository.py",
                    chunk_type="method",
                    parent="DocumentRepository",
                    name="save",
                    source_code=(
                        "def save(self, document):\n"
                        "    self.documents[document.id] = document\n"
                    ),
                ),
                score=0.96,
            ),
            SearchResult(
                chunk=_chunk(
                    id="update",
                    file_path="documents/service.py",
                    chunk_type="method",
                    parent="DocumentService",
                    name="update_document",
                    dependencies=["authorize_document_update", "save"],
                    source_code=(
                        "def update_document(self, document_id, user_id, payload):\n"
                        "    document = self.repository.get(document_id)\n"
                        "    authorize_document_update(document, user_id)\n"
                        "    document.title = payload.title\n"
                        "    return self.repository.save(document)\n"
                    ),
                ),
                score=0.82,
            ),
            SearchResult(
                chunk=_chunk(
                    id="guard",
                    file_path="security/permissions.py",
                    chunk_type="function",
                    name="authorize_document_update",
                    dependencies=["UnauthorizedUpdateError"],
                    source_code=(
                        "def authorize_document_update(document, user_id):\n"
                        "    if document.owner_id != user_id:\n"
                        "        raise UnauthorizedUpdateError(document.id)\n"
                    ),
                ),
                score=0.75,
            ),
        ],
        top_k=3,
    )

    ranked_names = _names(ranked)
    selected_names = _names(selected)
    assert ranked_names.index("authorize_document_update") < ranked_names.index("save")
    assert ranked_names.index("update_document") < ranked_names.index("save")
    assert selected_names == [
        "authorize_document_update",
        "update_document",
        "save",
    ]


def _rank_and_select(
    failure: TestFailure, semantic_results: list[SearchResult], *, top_k: int
) -> tuple[list[SearchResult], list[SearchResult]]:
    ranked = FailureRetriever.rerank_semantic_results_for_failure(
        failure=failure,
        semantic_results=semantic_results,
    )
    selected = FailureRetriever.select_semantic_results_for_failure(
        failure=failure,
        ranked_results=ranked,
        top_k=top_k,
    )
    return selected, ranked


def _failure(
    *,
    test_name: str,
    message: str,
    error_type: str = "Failed",
    traceback: str = "",
) -> TestFailure:
    return TestFailure(
        test_name=test_name,
        file_path=test_name.split("::", 1)[0],
        line_number=12,
        error_type=error_type,
        message=message,
        traceback=traceback,
    )


def _chunk(
    *,
    id: str,
    file_path: str,
    chunk_type: str,
    name: str,
    source_code: str,
    parent: str | None = None,
    dependencies: list[str] | None = None,
) -> CodeChunk:
    return CodeChunk(
        id=id,
        file_path=file_path,
        chunk_type=chunk_type,  # type: ignore[arg-type]
        name=name,
        parent=parent,
        start_line=1,
        end_line=5,
        source_code=source_code,
        imports=[],
        dependencies=dependencies or [],
    )


def _names(results: list[SearchResult]) -> list[str]:
    return [result.chunk.name for result in results]
