from __future__ import annotations

import pytest
from app.models import TaskStatus
from app.repository import InMemoryTaskRepository
from app.service import PermissionDeniedError, TaskService
from app.validators import InvalidStatusTransitionError, InvalidTaskDataError


def test_create_task_defaults_to_pending_status() -> None:
    service = make_service()

    task = service.create_task(owner_id="alice", title="Write release notes")

    assert task.title == "Write release notes"
    assert task.owner_id == "alice"
    assert task.status is TaskStatus.PENDING


def test_blank_title_is_rejected() -> None:
    service = make_service()

    with pytest.raises(InvalidTaskDataError):
        service.create_task(owner_id="alice", title="   ")


def test_owner_can_move_task_to_in_progress() -> None:
    service = make_service()
    task = service.create_task(owner_id="alice", title="Review pull request")

    updated = service.start_task(owner_id="alice", task_id=task.id)

    assert updated.status is TaskStatus.IN_PROGRESS


def test_user_cannot_update_another_users_task() -> None:
    service = make_service()
    task = service.create_task(owner_id="alice", title="Deploy backend")

    with pytest.raises(PermissionDeniedError):
        service.start_task(owner_id="bob", task_id=task.id)


def test_done_task_can_be_archived() -> None:
    service = make_service()
    task = service.create_task(owner_id="alice", title="Close support ticket")

    service.start_task(owner_id="alice", task_id=task.id)
    service.complete_task(owner_id="alice", task_id=task.id)
    archived = service.archive_task(owner_id="alice", task_id=task.id)

    assert archived.status is TaskStatus.ARCHIVED
    assert archived.archived_at is not None


def test_archived_task_cannot_be_marked_done() -> None:
    service = make_service()
    task = service.create_task(owner_id="alice", title="Clean up stale task")
    service.archive_task(owner_id="alice", task_id=task.id)

    with pytest.raises(InvalidStatusTransitionError):
        service.complete_task(owner_id="alice", task_id=task.id)


def make_service() -> TaskService:
    return TaskService(InMemoryTaskRepository())
