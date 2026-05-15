from __future__ import annotations

from app.models import Task


class TaskNotFoundError(LookupError):
    pass


class InMemoryTaskRepository:
    def __init__(self) -> None:
        self._tasks: dict[int, Task] = {}
        self._next_id = 1

    def create(self, *, title: str, owner_id: str) -> Task:
        task = Task(id=self._next_id, title=title, owner_id=owner_id)
        self._next_id += 1
        self._tasks[task.id] = task
        return task

    def get(self, task_id: int) -> Task:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise TaskNotFoundError(f"Task not found: {task_id}") from exc

    def save(self, task: Task) -> Task:
        self._tasks[task.id] = task
        return task

    def list_for_user(self, owner_id: str) -> list[Task]:
        return [task for task in self._tasks.values() if task.owner_id == owner_id]
