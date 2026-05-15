from __future__ import annotations

from decimal import Decimal

import pytest
from app.exceptions import InsufficientFundsError
from app.repository import BankRepository
from app.routes import create_transfer, get_account_activity
from app.service import TransferService


def test_open_account_stores_initial_balance() -> None:
    service, repository = _build_service()

    service.open_account("checking-1", "user-1", Decimal("125.00"))

    assert repository.get_account("checking-1").balance == Decimal("125.00")


def test_transfer_rejects_insufficient_funds() -> None:
    service, _ = _build_service()
    service.open_account("sender", "user-1", Decimal("10.00"))
    service.open_account("receiver", "user-2", Decimal("0.00"))

    with pytest.raises(InsufficientFundsError):
        service.transfer("sender", "receiver", Decimal("25.00"))


def test_route_records_transfer_activity() -> None:
    service, _ = _build_service()
    service.open_account("sender", "user-1", Decimal("40.00"))
    service.open_account("receiver", "user-2", Decimal("5.00"))

    response = create_transfer(
        service,
        sender_account_id="sender",
        receiver_account_id="receiver",
        amount="15.00",
    )

    assert response["status"] == "accepted"
    assert len(get_account_activity(service, account_id="sender")) == 1


def test_successful_transfer_moves_money_and_records_activity() -> None:
    service, repository = _build_service()
    service.open_account("sender", "user-1", Decimal("100.00"))
    service.open_account("receiver", "user-2", Decimal("20.00"))

    transfer_id = service.transfer("sender", "receiver", Decimal("25.00"))

    assert (
        repository.get_account("sender").balance,
        repository.get_account("receiver").balance,
        len(list(repository.list_transfers())),
        bool(transfer_id),
    ) == (Decimal("75.00"), Decimal("45.00"), 1, True)


def _build_service() -> tuple[TransferService, BankRepository]:
    repository = BankRepository()
    return TransferService(repository), repository
