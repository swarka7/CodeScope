from __future__ import annotations

from collections.abc import Iterable

from app.exceptions import AccountNotFoundError
from app.models import Account, TransferRecord


class BankRepository:
    def __init__(self) -> None:
        self._accounts: dict[str, Account] = {}
        self._transfers: list[TransferRecord] = []

    def add_account(self, account: Account) -> None:
        self._accounts[account.account_id] = account

    def get_account(self, account_id: str) -> Account:
        try:
            return self._accounts[account_id]
        except KeyError as exc:
            raise AccountNotFoundError(account_id) from exc

    def save_account(self, account: Account) -> None:
        self._accounts[account.account_id] = account

    def record_transfer(self, transfer: TransferRecord) -> None:
        self._transfers.append(transfer)

    def list_transfers(self) -> Iterable[TransferRecord]:
        return tuple(self._transfers)
