from __future__ import annotations

from app.models import Order, OrderLine
from app.service import FulfillmentService


def create_order(
    service: FulfillmentService,
    *,
    customer_id: str,
    items: list[dict[str, int | str]],
) -> Order:
    lines = [
        OrderLine(sku=str(item["sku"]), quantity=int(item["quantity"]))
        for item in items
    ]
    return service.create_order(customer_id=customer_id, lines=lines)


def ship_order(service: FulfillmentService, *, order_id: str) -> dict[str, str]:
    order = service.ship_order(order_id)
    return {"order_id": order.order_id, "status": order.status.value}
