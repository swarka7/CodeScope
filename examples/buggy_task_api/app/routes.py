from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from app.models import Task, TaskStatus
from app.service import TaskService

RouteHandler = Callable[..., Any]
RouteDecorator = Callable[[RouteHandler], RouteHandler]


class RouteRegistry:
    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], RouteHandler] = {}

    def get(self, path: str) -> RouteDecorator:
        return self._register("GET", path)

    def post(self, path: str) -> RouteDecorator:
        return self._register("POST", path)

    def patch(self, path: str) -> RouteDecorator:
        return self._register("PATCH", path)

    def _register(self, method: str, path: str) -> RouteDecorator:
        def decorator(handler: RouteHandler) -> RouteHandler:
            self.routes[(method, path)] = handler
            return handler

        return decorator


router = RouteRegistry()


@router.post("/tasks")
def create_task(payload: dict[str, str], *, user_id: str, service: TaskService) -> dict[str, Any]:
    task = service.create_task(owner_id=user_id, title=payload.get("title", ""))
    return serialize_task(task)


@router.get("/tasks")
def list_tasks(*, user_id: str, service: TaskService) -> list[dict[str, Any]]:
    return [serialize_task(task) for task in service.list_tasks(owner_id=user_id)]


@router.patch("/tasks/{task_id}/status")
def update_task_status(
    task_id: int, payload: dict[str, str], *, user_id: str, service: TaskService
) -> dict[str, Any]:
    status = TaskStatus(payload["status"])
    task = service.update_status(owner_id=user_id, task_id=task_id, status=status)
    return serialize_task(task)


def serialize_task(task: Task) -> dict[str, Any]:
    data = asdict(task)
    data["status"] = task.status.value
    data["created_at"] = task.created_at.isoformat()
    data["updated_at"] = task.updated_at.isoformat()
    data["archived_at"] = task.archived_at.isoformat() if task.archived_at else None
    return data
