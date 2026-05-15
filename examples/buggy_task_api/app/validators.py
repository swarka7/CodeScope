from __future__ import annotations

from app.models import TaskStatus


class InvalidTaskDataError(ValueError):
    pass


class InvalidStatusTransitionError(ValueError):
    pass


ALLOWED_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.IN_PROGRESS, TaskStatus.ARCHIVED},
    TaskStatus.IN_PROGRESS: {TaskStatus.DONE, TaskStatus.ARCHIVED},
    TaskStatus.DONE: {TaskStatus.ARCHIVED},
    TaskStatus.ARCHIVED: {TaskStatus.DONE},
}


def validate_task_title(title: str) -> str:
    cleaned = title.strip()
    if not cleaned:
        raise InvalidTaskDataError("Task title is required")
    if len(cleaned) > 120:
        raise InvalidTaskDataError("Task title must be 120 characters or fewer")
    return cleaned


def validate_status_transition(current: TaskStatus, requested: TaskStatus) -> None:
    if current == requested:
        return

    allowed = ALLOWED_TRANSITIONS.get(current, set())
    if requested not in allowed:
        raise InvalidStatusTransitionError(
            f"Cannot change task status from {current.value} to {requested.value}"
        )
