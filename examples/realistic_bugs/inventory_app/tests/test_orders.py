from __future__ import annotations

import pytest
from app.exceptions import (
    InvalidOrderStateError,
    InvalidQuantityError,
    OutOfStockError,
)
from app.models import OrderLine, OrderStatus
from app.repository import InventoryRepository
from app.routes import create_order, ship_order
from app.service import FulfillmentService


def test_create_order_accepts_available_products() -> None:
    service, _ = _build_service()
    service.add_product("SKU-1", "Keyboard", 5)

    order = create_order(
        service,
        customer_id="customer-1",
        items=[{"sku": "SKU-1", "quantity": 2}],
    )

    assert order.status == OrderStatus.DRAFT
    assert order.lines == [OrderLine(sku="SKU-1", quantity=2)]


def test_negative_order_quantity_is_rejected() -> None:
    service, _ = _build_service()
    service.add_product("SKU-1", "Keyboard", 5)

    with pytest.raises(InvalidQuantityError):
        service.create_order("customer-1", [OrderLine(sku="SKU-1", quantity=-1)])


def test_cannot_cancel_shipped_order() -> None:
    service, _ = _build_service()
    service.add_product("SKU-1", "Keyboard", 5)
    order = service.create_order("customer-1", [OrderLine(sku="SKU-1", quantity=1)])
    service.ship_order(order.order_id)

    with pytest.raises(InvalidOrderStateError):
        service.cancel_order(order.order_id)


def test_order_with_insufficient_stock_cannot_ship() -> None:
    service, repository = _build_service()
    service.add_product("SKU-1", "Keyboard", 1)
    order = service.create_order("customer-1", [OrderLine(sku="SKU-1", quantity=3)])

    with pytest.raises(OutOfStockError):
        ship_order(service, order_id=order.order_id)

    assert repository.get_product("SKU-1").on_hand == 1


def _build_service() -> tuple[FulfillmentService, InventoryRepository]:
    repository = InventoryRepository()
    return FulfillmentService(repository), repository
