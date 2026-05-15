from __future__ import annotations


class InventoryError(Exception):
    """Base exception for inventory workflow failures."""


class ProductNotFoundError(InventoryError):
    """Raised when a product does not exist."""


class OrderNotFoundError(InventoryError):
    """Raised when an order does not exist."""


class OutOfStockError(InventoryError):
    """Raised when an order cannot be fulfilled from available inventory."""


class InvalidQuantityError(InventoryError):
    """Raised when a requested quantity is invalid."""


class InvalidOrderStateError(InventoryError):
    """Raised when an order cannot move to a requested state."""
