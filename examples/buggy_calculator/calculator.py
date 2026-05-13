from __future__ import annotations


def calculate_discount(price: int, percent: int) -> int:
    """Return the discounted price.

    Bug (intentional): `percent` is treated as a whole number instead of a percentage.
    Example: 10% off 100 should be 90, but this returns -900.
    """

    if percent < 0 or percent > 100:
        raise ValueError("percent must be between 0 and 100")

    # BUG: should be `price * (percent / 100)`.
    discount_amount = price * percent
    return price - discount_amount

