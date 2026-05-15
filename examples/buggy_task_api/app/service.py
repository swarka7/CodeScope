from __future__ import annotations

from datetime import UTC, datetime

from app.models import Task, TaskStatus
from app.repository import InMemoryTaskRepository
from app.validators import validate_status_transition, validate_task_title


class PermissionDeniedError(PermissionError):
    pass


class TaskService:
    def __init__(self, repository: InMemoryTaskRepository) -> None:
        self._repository = repository

    def create_task(self, *, owner_id: str, title: str) -> Task:
        title = validate_task_title(title)
        return self._repository.create(title=title, owner_id=owner_id)

    def list_tasks(self, *, owner_id: str) -> list[Task]:
        return self._repository.list_for_user(owner_id)

    def start_task(self, *, owner_id: str, task_id: int) -> Task:
        return self.update_status(owner_id=owner_id, task_id=task_id, status=TaskStatus.IN_PROGRESS)

    def complete_task(self, *, owner_id: str, task_id: int) -> Task:
        return self.update_status(owner_id=owner_id, task_id=task_id, status=TaskStatus.DONE)

    def archive_task(self, *, owner_id: str, task_id: int) -> Task:
        task = self.update_status(owner_id=owner_id, task_id=task_id, status=TaskStatus.ARCHIVED)
        task.archived_at = datetime.now(UTC)
        return self._repository.save(task)

    def update_status(self, *, owner_id: str, task_id: int, status: TaskStatus) -> Task:
        task = self._get_owned_task(owner_id=owner_id, task_id=task_id)
        validate_status_transition(task.status, status)
        task.status = status
        task.touch()
        return self._repository.save(task)

    def rename_task(self, *, owner_id: str, task_id: int, title: str) -> Task:
        task = self._get_owned_task(owner_id=owner_id, task_id=task_id)
        task.title = validate_task_title(title)
        task.touch()
        return self._repository.save(task)

    def _get_owned_task(self, *, owner_id: str, task_id: int) -> Task:
        task = self._repository.get(task_id)
        if task.owner_id != owner_id:
            raise PermissionDeniedError("Only the task owner can update this task")
        return task
