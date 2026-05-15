from __future__ import annotations


class BankingError(Exception):
    """Base exception for banking workflow failures."""


class AccountNotFoundError(BankingError):
    """Raised when an account cannot be found."""


class InsufficientFundsError(BankingError):
    """Raised when an account does not have enough available funds."""


class InvalidTransferAmountError(BankingError):
    """Raised when a transfer amount cannot be processed."""
