from __future__ import annotations

from app.models import Order, OrderLine, OrderStatus, Product
from app.policies import (
    require_open_order,
    require_positive_quantity,
)
from app.repository import InventoryRepository


class FulfillmentService:
    def __init__(self, repository: InventoryRepository) -> None:
        self._repository = repository

    def add_product(self, sku: str, name: str, on_hand: int) -> Product:
        require_positive_quantity(on_hand)
        product = Product(sku=sku, name=name, on_hand=on_hand)
        self._repository.add_product(product)
        return product

    def create_order(self, customer_id: str, lines: list[OrderLine]) -> Order:
        for line in lines:
            require_positive_quantity(line.quantity)
            self._repository.get_product(line.sku)

        order = Order(customer_id=customer_id, lines=list(lines))
        self._repository.add_order(order)
        return order

    def ship_order(self, order_id: str) -> Order:
        order = self._repository.get_order(order_id)
        require_open_order(order)

        for line in order.lines:
            product = self._repository.get_product(line.sku)
            product.reserve(line.quantity)
            self._repository.save_product(product)

        order.status = OrderStatus.SHIPPED
        self._repository.save_order(order)
        return order

    def cancel_order(self, order_id: str) -> Order:
        order = self._repository.get_order(order_id)
        require_open_order(order)
        order.status = OrderStatus.CANCELLED
        self._repository.save_order(order)
        return order
