from __future__ import annotations

from app.exceptions import (
    InvalidOrderStateError,
    InvalidQuantityError,
    OutOfStockError,
)
from app.models import Order, OrderStatus, Product


def require_positive_quantity(quantity: int) -> None:
    if quantity <= 0:
        raise InvalidQuantityError("quantity must be positive")


def require_open_order(order: Order) -> None:
    if order.status in {OrderStatus.SHIPPED, OrderStatus.CANCELLED}:
        raise InvalidOrderStateError(f"order is already {order.status}")


def require_available_stock(product: Product, quantity: int) -> None:
    if product.on_hand < quantity:
        raise OutOfStockError(f"{product.sku} has {product.on_hand} units available")
