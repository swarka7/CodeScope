from __future__ import annotations

from decimal import Decimal

from app.exceptions import (
    InsufficientFundsError,
    InvalidTransferAmountError,
)
from app.models import Account, TransferRecord
from app.repository import BankRepository


class TransferService:
    def __init__(self, repository: BankRepository) -> None:
        self._repository = repository

    def open_account(self, account_id: str, owner_id: str, balance: Decimal) -> Account:
        account = Account(account_id=account_id, owner_id=owner_id, balance=balance)
        self._repository.add_account(account)
        return account

    def transfer(self, sender_account_id: str, receiver_account_id: str, amount: Decimal) -> str:
        if amount <= Decimal("0"):
            raise InvalidTransferAmountError("transfer amount must be positive")

        sender = self._repository.get_account(sender_account_id)
        receiver = self._repository.get_account(receiver_account_id)

        if sender.currency != receiver.currency:
            raise InvalidTransferAmountError("currency mismatch")
        if sender.balance < amount:
            raise InsufficientFundsError("insufficient funds")

        sender.debit(amount)

        self._repository.save_account(sender)
        self._repository.save_account(receiver)

        transfer = TransferRecord(
            sender_account_id=sender.account_id,
            receiver_account_id=receiver.account_id,
            amount=amount,
        )
        self._repository.record_transfer(transfer)
        return transfer.transfer_id

    def transfers_for_account(self, account_id: str) -> list[TransferRecord]:
        return [
            transfer
            for transfer in self._repository.list_transfers()
            if account_id in {transfer.sender_account_id, transfer.receiver_account_id}
        ]
