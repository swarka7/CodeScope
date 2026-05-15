from __future__ import annotations

from decimal import Decimal

from app.models import TransferRecord
from app.service import TransferService


def create_transfer(
    service: TransferService,
    *,
    sender_account_id: str,
    receiver_account_id: str,
    amount: str,
) -> dict[str, str]:
    transfer_id = service.transfer(
        sender_account_id=sender_account_id,
        receiver_account_id=receiver_account_id,
        amount=Decimal(amount),
    )
    return {"transfer_id": transfer_id, "status": "accepted"}


def get_account_activity(service: TransferService, *, account_id: str) -> list[TransferRecord]:
    return service.transfers_for_account(account_id)
