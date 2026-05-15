from __future__ import annotations

from codescope.debugging.call_graph_context import expand_failure_call_path_context
from codescope.graph.dependency_graph import DependencyGraph
from codescope.models.code_chunk import CodeChunk
from codescope.models.test_failure import TestFailure
from codescope.vectorstore.memory_store import SearchResult


def test_call_path_expands_service_to_imported_validator_and_exception() -> None:
    service = _chunk(
        id="service:update",
        file_path="app/service.py",
        chunk_type="method",
        parent="TaskService",
        name="update_status",
        imports=["from app.validators import validate_status_transition"],
        dependencies=["validate_status_transition"],
        source_code=(
            "def update_status(self, task, requested):\n"
            "    validate_status_transition(task.status, requested)\n"
            "    task.status = requested\n"
        ),
    )
    validator = _chunk(
        id="validators:validate",
        file_path="app/validators.py",
        chunk_type="function",
        name="validate_status_transition",
        dependencies=["InvalidStatusTransitionError"],
        source_code=(
            "def validate_status_transition(current, requested):\n"
            "    if requested not in ALLOWED_TRANSITIONS[current]:\n"
            "        raise InvalidStatusTransitionError('invalid transition')\n"
        ),
    )
    exception = _chunk(
        id="validators:error",
        file_path="app/validators.py",
        chunk_type="class",
        name="InvalidStatusTransitionError",
        source_code="class InvalidStatusTransitionError(ValueError):\n    pass\n",
    )

    results = expand_failure_call_path_context(
        failure=_did_not_raise_failure(),
        seed_results=[SearchResult(chunk=service, score=1.0)],
        graph=DependencyGraph([service, validator, exception]),
    )

    names = [result.chunk.name for result in results]
    assert "validate_status_transition" in names
    assert "InvalidStatusTransitionError" in names
    validator_result = _find(results, "validate_status_transition")
    assert "called by top source chunk" in validator_result.reasons
    assert "validation helper on call path" in validator_result.reasons


def test_call_path_resolves_validator_through_module_alias_import() -> None:
    service = _chunk(
        id="service:update",
        file_path="app/service.py",
        chunk_type="function",
        name="update_status",
        imports=["import app.validators as v"],
        dependencies=["v.check_status", "check_status"],
        source_code=(
            "def update_status(task, requested):\n"
            "    v.check_status(task.status, requested)\n"
            "    task.status = requested\n"
        ),
    )
    validator = _chunk(
        id="app:validator",
        file_path="app/validators.py",
        chunk_type="function",
        name="check_status",
        dependencies=["InvalidStatusTransitionError"],
        source_code=(
            "def check_status(current, requested):\n"
            "    raise InvalidStatusTransitionError()\n"
        ),
    )
    unrelated_validator = _chunk(
        id="admin:validator",
        file_path="admin/validators.py",
        chunk_type="function",
        name="check_status",
        dependencies=["InvalidStatusTransitionError"],
        source_code=(
            "def check_status(current, requested):\n"
            "    raise InvalidStatusTransitionError()\n"
        ),
    )

    results = expand_failure_call_path_context(
        failure=_did_not_raise_failure(),
        seed_results=[SearchResult(chunk=service, score=1.0)],
        graph=DependencyGraph([service, validator, unrelated_validator]),
    )

    assert validator.id in [result.chunk.id for result in results]
    assert unrelated_validator.id not in [result.chunk.id for result in results]


def test_call_path_resolves_validator_through_from_import_alias() -> None:
    service = _chunk(
        id="service:update",
        file_path="app/service.py",
        chunk_type="function",
        name="update_status",
        imports=["from app.validators import check_status as check"],
        dependencies=["check"],
        source_code=(
            "def update_status(task, requested):\n"
            "    check(task.status, requested)\n"
            "    task.status = requested\n"
        ),
    )
    validator = _chunk(
        id="app:validator",
        file_path="app/validators.py",
        chunk_type="function",
        name="check_status",
        dependencies=["InvalidStatusTransitionError"],
        source_code=(
            "def check_status(current, requested):\n"
            "    raise InvalidStatusTransitionError()\n"
        ),
    )
    unrelated_validator = _chunk(
        id="admin:validator",
        file_path="admin/validators.py",
        chunk_type="function",
        name="check_status",
        dependencies=["InvalidStatusTransitionError"],
        source_code=(
            "def check_status(current, requested):\n"
            "    raise InvalidStatusTransitionError()\n"
        ),
    )

    results = expand_failure_call_path_context(
        failure=_did_not_raise_failure(),
        seed_results=[SearchResult(chunk=service, score=1.0)],
        graph=DependencyGraph([service, validator, unrelated_validator]),
    )

    assert validator.id in [result.chunk.id for result in results]
    assert unrelated_validator.id not in [result.chunk.id for result in results]


def test_call_path_follows_controller_to_service_to_relative_imported_validator() -> None:
    route = _chunk(
        id="route:update",
        file_path="app/routes.py",
        chunk_type="function",
        name="update_task_status",
        imports=["from app.service import update_status"],
        dependencies=["update_status"],
        source_code=(
            "def update_task_status(task, requested):\n"
            "    return update_status(task, requested)\n"
        ),
    )
    service = _chunk(
        id="service:update",
        file_path="app/service.py",
        chunk_type="function",
        name="update_status",
        imports=["from .validators import check_status"],
        dependencies=["check_status"],
        source_code=(
            "def update_status(task, requested):\n"
            "    check_status(task.status, requested)\n"
            "    task.status = requested\n"
        ),
    )
    validator = _chunk(
        id="app:validator",
        file_path="app/validators.py",
        chunk_type="function",
        name="check_status",
        dependencies=["InvalidStatusTransitionError"],
        source_code=(
            "def check_status(current, requested):\n"
            "    raise InvalidStatusTransitionError()\n"
        ),
    )

    results = expand_failure_call_path_context(
        failure=_did_not_raise_failure(),
        seed_results=[SearchResult(chunk=route, score=1.0)],
        graph=DependencyGraph([route, service, validator]),
    )

    assert [result.chunk.id for result in results[:2]] == [validator.id, service.id]


def test_call_path_does_not_select_ambiguous_same_name_validator() -> None:
    service = _chunk(
        id="service:update",
        file_path="app/service.py",
        chunk_type="function",
        name="update_status",
        dependencies=["check_status"],
        source_code=(
            "def update_status(task, requested):\n"
            "    check_status(task.status, requested)\n"
            "    task.status = requested\n"
        ),
    )
    app_validator = _chunk(
        id="app:validator",
        file_path="app/validators.py",
        chunk_type="function",
        name="check_status",
        dependencies=["InvalidStatusTransitionError"],
        source_code=(
            "def check_status(current, requested):\n"
            "    raise InvalidStatusTransitionError()\n"
        ),
    )
    admin_validator = _chunk(
        id="admin:validator",
        file_path="admin/validators.py",
        chunk_type="function",
        name="check_status",
        dependencies=["InvalidStatusTransitionError"],
        source_code=(
            "def check_status(current, requested):\n"
            "    raise InvalidStatusTransitionError()\n"
        ),
    )

    results = expand_failure_call_path_context(
        failure=_did_not_raise_failure(),
        seed_results=[SearchResult(chunk=service, score=1.0)],
        graph=DependencyGraph([service, app_validator, admin_validator]),
    )

    assert app_validator.id not in [result.chunk.id for result in results]
    assert admin_validator.id not in [result.chunk.id for result in results]


def test_call_path_resolves_self_helper_method() -> None:
    service = _chunk(
        id="service:archive",
        file_path="app/service.py",
        chunk_type="method",
        parent="TaskService",
        name="archive_task",
        dependencies=["self._ensure_owner"],
        source_code=(
            "def archive_task(self, task, user_id):\n"
            "    self._ensure_owner(task, user_id)\n"
            "    task.status = 'archived'\n"
        ),
    )
    helper = _chunk(
        id="service:ensure",
        file_path="app/service.py",
        chunk_type="method",
        parent="TaskService",
        name="_ensure_owner",
        dependencies=["PermissionDenied"],
        source_code=(
            "def _ensure_owner(self, task, user_id):\n"
            "    if task.owner_id != user_id:\n"
            "        raise PermissionDenied()\n"
        ),
    )

    results = expand_failure_call_path_context(
        failure=_permission_failure(),
        seed_results=[SearchResult(chunk=service, score=1.0)],
        graph=DependencyGraph([service, helper]),
    )

    helper_result = _find(results, "_ensure_owner")
    assert helper_result.chunk.parent == "TaskService"
    assert "validation helper on call path" in helper_result.reasons
    assert "exception logic on call path" in helper_result.reasons


def test_call_path_follows_route_to_service_to_validator() -> None:
    route = _chunk(
        id="routes:update",
        file_path="app/routes.py",
        chunk_type="function",
        name="update_task_status",
        dependencies=["update_status"],
        source_code=(
            "def update_task_status(task_id, payload, service):\n"
            "    return service.update_status(task_id, payload['status'])\n"
        ),
    )
    service = _chunk(
        id="service:update",
        file_path="app/service.py",
        chunk_type="method",
        parent="TaskService",
        name="update_status",
        dependencies=["validate_status_transition"],
        source_code=(
            "def update_status(self, task_id, status):\n"
            "    validate_status_transition(task.status, status)\n"
            "    return task\n"
        ),
    )
    validator = _chunk(
        id="validators:validate",
        file_path="app/validators.py",
        chunk_type="function",
        name="validate_status_transition",
        dependencies=["InvalidStatusTransitionError"],
        source_code=(
            "def validate_status_transition(current, requested):\n"
            "    raise InvalidStatusTransitionError()\n"
        ),
    )

    results = expand_failure_call_path_context(
        failure=_did_not_raise_failure(),
        seed_results=[SearchResult(chunk=route, score=1.0)],
        graph=DependencyGraph([route, service, validator]),
    )

    names = [result.chunk.name for result in results]
    assert "update_status" in names
    assert "validate_status_transition" in names
    assert "called by top source chunk" in _find(results, "update_status").reasons
    assert "validation helper on call path" in _find(results, "validate_status_transition").reasons


def test_call_path_does_not_boost_unrelated_helpers() -> None:
    service = _chunk(
        id="service:transfer",
        file_path="banking/service.py",
        chunk_type="method",
        parent="TransferService",
        name="transfer",
        dependencies=["validate_daily_limit", "format_receipt"],
        source_code=(
            "def transfer(self, account, amount):\n"
            "    validate_daily_limit(account, amount)\n"
            "    format_receipt(account)\n"
        ),
    )
    validator = _chunk(
        id="rules:validate",
        file_path="banking/rules.py",
        chunk_type="function",
        name="validate_daily_limit",
        dependencies=["DailyLimitExceeded"],
        source_code=(
            "def validate_daily_limit(account, amount):\n"
            "    if amount > account.daily_limit:\n"
            "        raise DailyLimitExceeded()\n"
        ),
    )
    unrelated = _chunk(
        id="format",
        file_path="banking/formatting.py",
        chunk_type="function",
        name="format_receipt",
        source_code="def format_receipt(account):\n    return str(account.id)\n",
    )

    results = expand_failure_call_path_context(
        failure=_daily_limit_failure(),
        seed_results=[SearchResult(chunk=service, score=1.0)],
        graph=DependencyGraph([service, validator, unrelated]),
    )

    names = [result.chunk.name for result in results]
    assert "validate_daily_limit" in names
    assert "format_receipt" not in names


def test_expected_exception_validator_outranks_repository_call_path_helper() -> None:
    service = _chunk(
        id="service:update",
        file_path="app/service.py",
        chunk_type="method",
        parent="TaskService",
        name="update_status",
        dependencies=["validate_status_transition", "_get_owned_task"],
        source_code=(
            "def update_status(self, task_id, status):\n"
            "    task = self._get_owned_task(task_id)\n"
            "    validate_status_transition(task.status, status)\n"
        ),
    )
    validator = _chunk(
        id="validators:validate",
        file_path="app/validators.py",
        chunk_type="function",
        name="validate_status_transition",
        dependencies=["InvalidStatusTransitionError"],
        source_code=(
            "def validate_status_transition(current, requested):\n"
            "    raise InvalidStatusTransitionError()\n"
        ),
    )
    owned_helper = _chunk(
        id="service:get_owned",
        file_path="app/service.py",
        chunk_type="method",
        parent="TaskService",
        name="_get_owned_task",
        dependencies=["get"],
        source_code=(
            "def _get_owned_task(self, task_id):\n"
            "    return self.repository.get(task_id)\n"
        ),
    )
    repository_get = _chunk(
        id="repo:get",
        file_path="app/repository.py",
        chunk_type="method",
        parent="InMemoryTaskRepository",
        name="get",
        dependencies=["TaskNotFoundError"],
        source_code=(
            "def get(self, task_id):\n"
            "    if task_id not in self.tasks:\n"
            "        raise TaskNotFoundError(task_id)\n"
        ),
    )

    results = expand_failure_call_path_context(
        failure=_did_not_raise_failure(),
        seed_results=[SearchResult(chunk=service, score=9.0)],
        graph=DependencyGraph([service, validator, owned_helper, repository_get]),
    )
    names = [result.chunk.name for result in results]

    assert "validate_status_transition" in names
    assert "get" not in names
    assert "_get_owned_task" not in names


def test_unrelated_exception_class_does_not_outrank_expected_validator() -> None:
    service = _chunk(
        id="service:update",
        file_path="app/service.py",
        chunk_type="method",
        parent="TaskService",
        name="update_status",
        dependencies=["validate_status_transition", "_get_owned_task"],
        source_code=(
            "def update_status(self, task_id, status):\n"
            "    self._get_owned_task(task_id)\n"
            "    validate_status_transition(task.status, status)\n"
        ),
    )
    validator = _chunk(
        id="validators:validate",
        file_path="app/validators.py",
        chunk_type="function",
        name="validate_status_transition",
        dependencies=["InvalidStatusTransitionError"],
        source_code=(
            "def validate_status_transition(current, requested):\n"
            "    raise InvalidStatusTransitionError()\n"
        ),
    )
    unrelated_exception = _chunk(
        id="repo:not_found",
        file_path="app/repository.py",
        chunk_type="class",
        name="TaskNotFoundError",
        source_code="class TaskNotFoundError(LookupError):\n    pass\n",
    )
    owned_helper = _chunk(
        id="service:get_owned",
        file_path="app/service.py",
        chunk_type="method",
        parent="TaskService",
        name="_get_owned_task",
        dependencies=["TaskNotFoundError"],
        source_code="def _get_owned_task(self, task_id):\n    raise TaskNotFoundError(task_id)\n",
    )

    results = expand_failure_call_path_context(
        failure=_did_not_raise_failure(),
        seed_results=[SearchResult(chunk=service, score=9.0)],
        graph=DependencyGraph([service, validator, owned_helper, unrelated_exception]),
    )
    names = [result.chunk.name for result in results]

    assert names[0] == "validate_status_transition"
    assert "TaskNotFoundError" not in names


def test_business_operation_caller_stays_near_validator() -> None:
    route = _chunk(
        id="routes:update",
        file_path="app/routes.py",
        chunk_type="function",
        name="update_task_status",
        dependencies=["update_status"],
        source_code=(
            "def update_task_status(task_id, payload, service):\n"
            "    return service.update_status(task_id, payload['status'])\n"
        ),
    )
    service = _chunk(
        id="service:update",
        file_path="app/service.py",
        chunk_type="method",
        parent="TaskService",
        name="update_status",
        dependencies=["validate_status_transition"],
        source_code=(
            "def update_status(self, task_id, status):\n"
            "    validate_status_transition(task.status, status)\n"
            "    return task\n"
        ),
    )
    validator = _chunk(
        id="validators:validate",
        file_path="app/validators.py",
        chunk_type="function",
        name="validate_status_transition",
        dependencies=["InvalidStatusTransitionError"],
        source_code=(
            "def validate_status_transition(current, requested):\n"
            "    raise InvalidStatusTransitionError()\n"
        ),
    )
    repository_get = _chunk(
        id="repo:get",
        file_path="app/repository.py",
        chunk_type="method",
        parent="InMemoryTaskRepository",
        name="get",
        dependencies=["TaskNotFoundError"],
        source_code="def get(self, task_id):\n    raise TaskNotFoundError(task_id)\n",
    )

    results = expand_failure_call_path_context(
        failure=_did_not_raise_failure(),
        seed_results=[SearchResult(chunk=route, score=9.0)],
        graph=DependencyGraph([route, service, validator, repository_get]),
    )
    names = [result.chunk.name for result in results]

    assert names[:2] == ["validate_status_transition", "update_status"]
    assert "get" not in names


def _find(results: list[SearchResult], name: str) -> SearchResult:
    for result in results:
        if result.chunk.name == name:
            return result
    raise AssertionError(f"Missing result: {name}")


def _did_not_raise_failure() -> TestFailure:
    return TestFailure(
        test_name="tests/test_tasks.py::test_archived_task_cannot_be_marked_done",
        file_path="tests/test_tasks.py",
        line_number=42,
        error_type="Failed",
        message="Failed: DID NOT RAISE <class 'app.validators.InvalidStatusTransitionError'>",
        traceback="",
    )


def _permission_failure() -> TestFailure:
    return TestFailure(
        test_name="tests/test_tasks.py::test_unauthorized_user_cannot_archive_task",
        file_path="tests/test_tasks.py",
        line_number=42,
        error_type="Failed",
        message="Failed: DID NOT RAISE <class 'app.service.PermissionDenied'>",
        traceback="",
    )


def _daily_limit_failure() -> TestFailure:
    return TestFailure(
        test_name="tests/test_transfers.py::test_transfer_over_daily_limit_should_be_rejected",
        file_path="tests/test_transfers.py",
        line_number=42,
        error_type="Failed",
        message="Failed: DID NOT RAISE <class 'banking.rules.DailyLimitExceeded'>",
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
    dependencies: list[str] | None = None,
    imports: list[str] | None = None,
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
        imports=imports or [],
        dependencies=dependencies or [],
    )
