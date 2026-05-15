from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4


class OrderStatus(StrEnum):
    DRAFT = "draft"
    RESERVED = "reserved"
    SHIPPED = "shipped"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class Product:
    sku: str
    name: str
    on_hand: int

    def reserve(self, quantity: int) -> None:
        self.on_hand -= quantity


@dataclass(frozen=True, slots=True)
class OrderLine:
    sku: str
    quantity: int


@dataclass(slots=True)
class Order:
    customer_id: str
    lines: list[OrderLine]
    order_id: str = field(default_factory=lambda: str(uuid4()))
    status: OrderStatus = OrderStatus.DRAFT
