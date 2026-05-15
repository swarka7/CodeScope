from __future__ import annotations

from app.exceptions import (
    OrderNotFoundError,
    ProductNotFoundError,
)
from app.models import Order, Product


class InventoryRepository:
    def __init__(self) -> None:
        self._products: dict[str, Product] = {}
        self._orders: dict[str, Order] = {}

    def add_product(self, product: Product) -> None:
        self._products[product.sku] = product

    def get_product(self, sku: str) -> Product:
        try:
            return self._products[sku]
        except KeyError as exc:
            raise ProductNotFoundError(sku) from exc

    def save_product(self, product: Product) -> None:
        self._products[product.sku] = product

    def add_order(self, order: Order) -> None:
        self._orders[order.order_id] = order

    def get_order(self, order_id: str) -> Order:
        try:
            return self._orders[order_id]
        except KeyError as exc:
            raise OrderNotFoundError(order_id) from exc

    def save_order(self, order: Order) -> None:
        self._orders[order.order_id] = order
