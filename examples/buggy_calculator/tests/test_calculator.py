from __future__ import annotations

from calculator import calculate_discount


def test_calculate_discount_applies_percent() -> None:
    assert calculate_discount(100, 10) == 90


def test_calculate_discount_no_discount() -> None:
    assert calculate_discount(100, 0) == 100

