from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4


@dataclass(slots=True)
class Account:
    account_id: str
    owner_id: str
    balance: Decimal
    currency: str = "USD"

    def debit(self, amount: Decimal) -> None:
        self.balance -= amount

    def credit(self, amount: Decimal) -> None:
        self.balance += amount


@dataclass(slots=True)
class TransferRecord:
    sender_account_id: str
    receiver_account_id: str
    amount: Decimal
    transfer_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
